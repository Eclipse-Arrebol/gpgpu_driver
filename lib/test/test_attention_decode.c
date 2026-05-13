/*
 * test_attention_decode.c
 * decode attention 端到端测试
 *
 * pipeline: Q + K_cache + V_cache + cur_pos
 *   → qkt_decode    → scores [q_heads, max_seq]
 *   → softmax_decode → P     [q_heads, max_seq]  (in-place)
 *   → pv_decode     → O     [q_heads, head_dim]
 *
 * 4 档测试矩阵同 qkt_decode 测试
 */

#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- args 结构体 ---- */
struct qkt_decode_args {
    uint32_t Q_off;
    uint32_t K_cache_off;
    uint32_t scores_off;
    uint32_t cur_pos;
    uint32_t max_seq;
    uint32_t head_dim;
    uint32_t q_per_kv;
    uint32_t scale_bits;
};

struct softmax_decode_args {
    uint32_t scores_off;
    uint32_t cur_pos;
    uint32_t max_seq;
};

struct pv_decode_args {
    uint32_t P_off;
    uint32_t V_cache_off;
    uint32_t O_off;
    uint32_t cur_pos;
    uint32_t max_seq;
    uint32_t head_dim;
    uint32_t q_per_kv;
};

/* ---- 填充模式(沿用 qkt 测试) ---- */
enum kv_pattern {
    KV_J_PLUS_1,
    KV_GQA_SIGN_FLIP,
    KV_QWEN2_LIKE,
};

enum q_pattern {
    Q_ALL_ONE,
    Q_QWEN2_LIKE,
};

struct case_config {
    const char    *name;
    int            num_q_heads;
    int            num_kv_heads;
    int            head_dim;
    int            max_seq;
    int            cur_pos;
    float          scale;
    int            poison_kv_beyond_S;
    enum kv_pattern kvpat;
    enum q_pattern  qpat;
    float           tol;
};

static gpgpu_ctx *g_ctx;
static uint64_t   g_qkt_off;
static uint64_t   g_softmax_off;
static uint64_t   g_pv_off;

/* ---- 填充函数 ---- */
static float fill_q(enum q_pattern p, int qh, int k)
{
    switch (p) {
        case Q_ALL_ONE:    return 1.0f;
        case Q_QWEN2_LIKE: return (qh + 1) * 0.02f + k * 0.0001f;
    }
    return 0.0f;
}

static float fill_kv(enum kv_pattern p, int kh, int j, int k)
{
    switch (p) {
        case KV_J_PLUS_1:      return (float)(j + 1);
        case KV_GQA_SIGN_FLIP: return (kh == 0) ? (float)(j + 1) : -(float)(j + 1);
        case KV_QWEN2_LIKE:    return (j + 1) * (kh + 1) * 0.01f + k * 0.001f;
    }
    return 0.0f;
}

/* ---- host reference ---- */
static void attention_ref(
    const float *Q,        /* [num_q_heads, head_dim] */
    const float *K_cache,  /* [num_kv_heads, max_seq, head_dim] */
    const float *V_cache,  /* [num_kv_heads, max_seq, head_dim] */
    float       *O,        /* [num_q_heads, head_dim] */
    int num_q_heads, int num_kv_heads,
    int head_dim, int max_seq, int S,
    int q_per_kv, float scale)
{
    float *scores = calloc(num_q_heads * S, sizeof(float));

    /* QK^T * scale */
    for (int qh = 0; qh < num_q_heads; qh++) {
        int kh = qh / q_per_kv;
        for (int j = 0; j < S; j++) {
            float acc = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                acc += Q[qh * head_dim + k]
                     * K_cache[kh * max_seq * head_dim + j * head_dim + k];
            }
            scores[qh * S + j] = acc * scale;
        }
    }

    /* softmax per row */
    for (int qh = 0; qh < num_q_heads; qh++) {
        float *row = scores + qh * S;
        float maxv = -1e30f;
        for (int j = 0; j < S; j++)
            if (row[j] > maxv) maxv = row[j];
        float sum = 0.0f;
        for (int j = 0; j < S; j++) {
            row[j] = expf(row[j] - maxv);
            sum += row[j];
        }
        for (int j = 0; j < S; j++)
            row[j] /= sum;
    }

    /* P * V */
    for (int qh = 0; qh < num_q_heads; qh++) {
        int kh = qh / q_per_kv;
        for (int k = 0; k < head_dim; k++) {
            float acc = 0.0f;
            for (int j = 0; j < S; j++) {
                acc += scores[qh * S + j]
                     * V_cache[kh * max_seq * head_dim + j * head_dim + k];
            }
            O[qh * head_dim + k] = acc;
        }
    }

    free(scores);
}

/* ---- 单 case ---- */
static int run_case(const struct case_config *cfg)
{
    const int S        = cfg->cur_pos + 1;
    const int q_per_kv = cfg->num_q_heads / cfg->num_kv_heads;

    printf("\n========================================\n");
    printf("CASE: %s\n", cfg->name);
    printf("  q_heads=%d kv_heads=%d head_dim=%d max_seq=%d"
           " cur_pos=%d S=%d q_per_kv=%d scale=%f tol=%.1e\n",
           cfg->num_q_heads, cfg->num_kv_heads, cfg->head_dim,
           cfg->max_seq, cfg->cur_pos, S, q_per_kv, cfg->scale, cfg->tol);

    /* ---- 尺寸 ---- */
    size_t Q_size       = (size_t)cfg->num_q_heads  * cfg->head_dim * sizeof(float);
    size_t KV_cache_size= (size_t)cfg->num_kv_heads * cfg->max_seq  * cfg->head_dim * sizeof(float);
    size_t scores_size  = (size_t)cfg->num_q_heads  * cfg->max_seq  * sizeof(float);
    size_t O_size       = (size_t)cfg->num_q_heads  * cfg->head_dim * sizeof(float);

    /* ---- GPU 分配 ---- */
    uint64_t qkt_args_off     = gpuMalloc(g_ctx, sizeof(struct qkt_decode_args));
    uint64_t softmax_args_off = gpuMalloc(g_ctx, sizeof(struct softmax_decode_args));
    uint64_t pv_args_off      = gpuMalloc(g_ctx, sizeof(struct pv_decode_args));
    uint64_t Q_off            = gpuMalloc(g_ctx, Q_size);
    uint64_t K_cache_off      = gpuMalloc(g_ctx, KV_cache_size);
    uint64_t V_cache_off      = gpuMalloc(g_ctx, KV_cache_size);
    uint64_t scores_off       = gpuMalloc(g_ctx, scores_size);
    uint64_t O_off            = gpuMalloc(g_ctx, O_size);

    /* ---- host buffer ---- */
    float *Q_h       = malloc(Q_size);
    float *K_cache_h = malloc(KV_cache_size);
    float *V_cache_h = malloc(KV_cache_size);
    float *O_h       = calloc(cfg->num_q_heads * cfg->head_dim, sizeof(float));
    float *O_ref     = calloc(cfg->num_q_heads * cfg->head_dim, sizeof(float));

    /* ---- 填 Q ---- */
    for (int qh = 0; qh < cfg->num_q_heads; qh++)
        for (int k = 0; k < cfg->head_dim; k++)
            Q_h[qh * cfg->head_dim + k] = fill_q(cfg->qpat, qh, k);

    /* ---- 填 K_cache / V_cache ---- */
    for (int kh = 0; kh < cfg->num_kv_heads; kh++) {
        for (int j = 0; j < cfg->max_seq; j++) {
            for (int k = 0; k < cfg->head_dim; k++) {
                size_t idx = (size_t)kh * cfg->max_seq * cfg->head_dim
                           + (size_t)j  * cfg->head_dim + k;
                if (cfg->poison_kv_beyond_S && j >= S) {
                    K_cache_h[idx] = 999.0f;
                    V_cache_h[idx] = 999.0f;
                } else {
                    K_cache_h[idx] = fill_kv(cfg->kvpat, kh, j, k);
                    V_cache_h[idx] = fill_kv(cfg->kvpat, kh, j, k);
                }
            }
        }
    }

    /* ---- host reference ---- */
    attention_ref(Q_h, K_cache_h, V_cache_h, O_ref,
                  cfg->num_q_heads, cfg->num_kv_heads,
                  cfg->head_dim, cfg->max_seq, S,
                  q_per_kv, cfg->scale);

    /* ---- 上传数据 ---- */
    gpuMemcpy(g_ctx, Q_off,       Q_h,       Q_size,        0);
    gpuMemcpy(g_ctx, K_cache_off, K_cache_h, KV_cache_size, 0);
    gpuMemcpy(g_ctx, V_cache_off, V_cache_h, KV_cache_size, 0);
    gpuMemcpy(g_ctx, scores_off,  O_h,       scores_size,   0); /* 清零 */
    gpuMemcpy(g_ctx, O_off,       O_h,       O_size,        0); /* 清零 */

    /* ---- 上传 args ---- */
    uint32_t scale_bits;
    memcpy(&scale_bits, &cfg->scale, sizeof(float));

    struct qkt_decode_args qkt_args = {
        .Q_off       = (uint32_t)Q_off,
        .K_cache_off = (uint32_t)K_cache_off,
        .scores_off  = (uint32_t)scores_off,
        .cur_pos     = (uint32_t)cfg->cur_pos,
        .max_seq     = (uint32_t)cfg->max_seq,
        .head_dim    = (uint32_t)cfg->head_dim,
        .q_per_kv    = (uint32_t)q_per_kv,
        .scale_bits  = scale_bits,
    };
    gpuMemcpy(g_ctx, qkt_args_off, &qkt_args, sizeof(qkt_args), 0);

    struct softmax_decode_args sm_args = {
        .scores_off = (uint32_t)scores_off,
        .cur_pos    = (uint32_t)cfg->cur_pos,
        .max_seq    = (uint32_t)cfg->max_seq,
    };
    gpuMemcpy(g_ctx, softmax_args_off, &sm_args, sizeof(sm_args), 0);

    struct pv_decode_args pv_args = {
        .P_off       = (uint32_t)scores_off,   /* softmax 原地写回,scores即P */
        .V_cache_off = (uint32_t)V_cache_off,
        .O_off       = (uint32_t)O_off,
        .cur_pos     = (uint32_t)cfg->cur_pos,
        .max_seq     = (uint32_t)cfg->max_seq,
        .head_dim    = (uint32_t)cfg->head_dim,
        .q_per_kv    = (uint32_t)q_per_kv,
    };
    gpuMemcpy(g_ctx, pv_args_off, &pv_args, sizeof(pv_args), 0);

    /* ---- 启动三个 kernel ---- */
    uint32_t grid[3]  = {(uint32_t)cfg->num_q_heads, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    int ret;

    ret = gpuLaunchKernel(g_ctx, g_qkt_off, qkt_args_off, grid, block, 0);
    if (ret < 0) { fprintf(stderr, "qkt launch failed\n"); return -1; }

    ret = gpuLaunchKernel(g_ctx, g_softmax_off, softmax_args_off, grid, block, 0);
    if (ret < 0) { fprintf(stderr, "softmax launch failed\n"); return -1; }

    ret = gpuLaunchKernel(g_ctx, g_pv_off, pv_args_off, grid, block, 0);
    if (ret < 0) { fprintf(stderr, "pv launch failed\n"); return -1; }

    /* ---- 回读 O ---- */
    gpuMemcpy(g_ctx, O_off, O_h, O_size, 1);

    /* ---- 比对 ---- */
    float max_err = 0.0f;
    int   max_qh  = -1, max_k = -1;
    int   fail    = 0;

    for (int qh = 0; qh < cfg->num_q_heads; qh++) {
        for (int k = 0; k < cfg->head_dim; k++) {
            int   idx  = qh * cfg->head_dim + k;
            float got  = O_h[idx];
            float ref  = O_ref[idx];
            float diff = fabsf(got - ref);
            if (diff > max_err) {
                max_err = diff;
                max_qh  = qh;
                max_k   = k;
            }
            if (diff >= cfg->tol) {
                if (fail < 10)
                    printf("FAIL O[qh=%d, k=%d] got=%f ref=%f diff=%.3e\n",
                           qh, k, got, ref, diff);
                fail++;
            }
        }
    }

    /* ---- 采样输出 ---- */
    printf("---- RESULT ----\n");
    printf("max_err = %.3e at O[qh=%d, k=%d]\n", max_err, max_qh, max_k);
    printf("fail    = %d\n", fail);

    int probe_qhs[] = {0, cfg->num_q_heads - 1, cfg->num_q_heads / 2};
    int probe_ks[]  = {0, cfg->head_dim - 1, cfg->head_dim / 2};
    for (int pi = 0; pi < 3; pi++) {
        int qh = probe_qhs[pi];
        if (qh < 0 || qh >= cfg->num_q_heads) continue;
        for (int pk = 0; pk < 3; pk++) {
            int k = probe_ks[pk];
            if (k < 0 || k >= cfg->head_dim) continue;
            int idx = qh * cfg->head_dim + k;
            printf("O[qh=%d, k=%d] = %f (ref %f)\n",
                   qh, k, O_h[idx], O_ref[idx]);
        }
    }

    printf("%s\n", fail == 0 ? "PASS" : "FAIL");

    free(Q_h); free(K_cache_h); free(V_cache_h); free(O_h); free(O_ref);
    return fail;
}

/* ---- 加载单个 kernel binary ---- */
static uint64_t load_kernel(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return 0; }
    fseek(f, 0, SEEK_END);
    size_t ksize = ftell(f);
    rewind(f);
    uint8_t *buf = malloc(ksize);
    fread(buf, 1, ksize, f);
    fclose(f);
    uint64_t off = gpuMalloc(g_ctx, ksize);
    gpuMemcpy(g_ctx, off, buf, ksize, 0);
    free(buf);
    printf("[init] loaded %s, size=%zu\n", path, ksize);
    return off;
}

/* ---- main ---- */
int main(void)
{
    g_ctx = malloc(sizeof(gpgpu_ctx));
    int ret = gpuInit(g_ctx, "/dev/gpgpu", 64ULL * 1024 * 1024);
    if (ret < 0) { fprintf(stderr, "gpuInit failed\n"); return 1; }
    printf("[init] gpuInit OK\n");

    g_qkt_off    = load_kernel("/tmp/qkt_decode.bin");
    g_softmax_off = load_kernel("/tmp/softmax_decode.bin");
    g_pv_off     = load_kernel("/tmp/pv_decode.bin");
    if (!g_qkt_off || !g_softmax_off || !g_pv_off) return 1;

    struct case_config cases[] = {
        {
            .name             = "case0_baseline (S=max_seq=32, q_per_kv=1)",
            .num_q_heads      = 1,  .num_kv_heads = 1,
            .head_dim         = 32, .max_seq = 32, .cur_pos = 31,
            .scale            = 1.0f / sqrtf(32.0f),
            .poison_kv_beyond_S = 0,
            .kvpat = KV_J_PLUS_1,  .qpat = Q_ALL_ONE,
            .tol   = 1e-4f,
        },
        {
            .name             = "case1_S_lt_maxseq (S=16, max_seq=32, KV poisoned)",
            .num_q_heads      = 1,  .num_kv_heads = 1,
            .head_dim         = 32, .max_seq = 32, .cur_pos = 15,
            .scale            = 1.0f / sqrtf(32.0f),
            .poison_kv_beyond_S = 1,
            .kvpat = KV_J_PLUS_1,  .qpat = Q_ALL_ONE,
            .tol   = 1e-4f,
        },
        {
            .name             = "case2_GQA (q=4, kv=2, V sign-flipped)",
            .num_q_heads      = 4,  .num_kv_heads = 2,
            .head_dim         = 32, .max_seq = 32, .cur_pos = 15,
            .scale            = 1.0f / sqrtf(32.0f),
            .poison_kv_beyond_S = 1,
            .kvpat = KV_GQA_SIGN_FLIP, .qpat = Q_ALL_ONE,
            .tol   = 1e-4f,
        },
        {
            .name             = "case3_qwen2 (q=14, kv=2, hd=64, ms=128, cp=99)",
            .num_q_heads      = 14, .num_kv_heads = 2,
            .head_dim         = 64, .max_seq = 128, .cur_pos = 99,
            .scale            = 0.125f,
            .poison_kv_beyond_S = 1,
            .kvpat = KV_QWEN2_LIKE, .qpat = Q_QWEN2_LIKE,
            .tol   = 1e-3f,   /* softmax reduce 误差累积,适当放宽 */
        },
    };

    int total_fail = 0;
    int n      = (int)(sizeof(cases) / sizeof(cases[0]));
    int passed = 0;
    for (int i = 0; i < n; i++) {
        int f = run_case(&cases[i]);
        if (f != 0) {
            total_fail += f;
            printf("\n>>> Stop at case %d due to failure\n", i);
            break;
        }
        passed++;
    }

    printf("\n========== SUMMARY ==========\n");
    printf("Passed: %d / %d\n", passed, n);
    printf("Total failures: %d\n", total_fail);
    printf("=============================\n");

    gpuDestroy(g_ctx);
    free(g_ctx);
    return total_fail == 0 ? 0 : 1;
}