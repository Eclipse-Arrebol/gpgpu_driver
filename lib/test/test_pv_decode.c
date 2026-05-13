/*
 * test_pv_decode.c
 * 测试 pv_decode.S kernel
 *
 * 数学: O[qh, k] = sum_{j=0}^{S-1} P[qh, j] * V_cache[kh, j, k]
 *       kh = qh / q_per_kv
 *
 * 4 档测试矩阵:
 *   case0: baseline  (S=max_seq=32, q_per_kv=1)
 *   case1: S<max_seq (S=16, V_cache 后半毒数据, 检查 O 不被污染)
 *   case2: GQA       (q=4, kv=2, V 符号翻转)
 *   case3: Qwen2     (q=14, kv=2, head_dim=64, max_seq=128, cur_pos=99)
 */

#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- args 结构体,与 pv_decode.S 对齐 ---- */
struct pv_decode_args {
    uint32_t P_off;
    uint32_t V_cache_off;
    uint32_t O_off;
    uint32_t cur_pos;
    uint32_t max_seq;
    uint32_t head_dim;
    uint32_t q_per_kv;
};

/* ---- V_cache 填充模式(复用 qkt 测试的语义) ---- */
enum vcache_pattern {
    VCACHE_J_PLUS_1,      /* V[kh, j, k] = j+1            (档 0/1)    */
    VCACHE_GQA_SIGN_FLIP, /* kh=0: j+1, kh=1: -(j+1)     (档 2)      */
    VCACHE_QWEN2_LIKE,    /* (j+1)*(kh+1)*0.01 + k*0.001 (档 3)      */
};

/* ---- P 填充模式 ---- */
enum p_pattern {
    P_UNIFORM,      /* P[qh, j] = 1/S  (均匀 softmax, 行和=1)         */
    P_QWEN2_LIKE,   /* P[qh, j] = softmax of (qh+1)*0.1*(j+1)/S      */
};

struct case_config {
    const char      *name;
    int              num_q_heads;
    int              num_kv_heads;
    int              head_dim;
    int              max_seq;
    int              cur_pos;
    int              poison_vcache_beyond_S; /* 后半填 999.0 */
    enum vcache_pattern vpat;
    enum p_pattern      ppat;
    float            tol;
};

static gpgpu_ctx *g_ctx;
static uint64_t   g_kernel_off;

/* ---- 填充函数 ---- */
static float fill_v(enum vcache_pattern p, int kh, int j, int k)
{
    switch (p) {
        case VCACHE_J_PLUS_1:       return (float)(j + 1);
        case VCACHE_GQA_SIGN_FLIP:  return (kh == 0) ? (float)(j + 1)
                                                      : -(float)(j + 1);
        case VCACHE_QWEN2_LIKE:     return (j + 1) * (kh + 1) * 0.01f
                                         + k * 0.001f;
    }
    return 0.0f;
}

/*
 * 生成 P[qh, 0..S-1],满足行和 = 1.0
 * P_UNIFORM:    每格 1/S
 * P_QWEN2_LIKE: logit = (qh+1)*0.1*(j+1), 做 softmax
 */
static void fill_p_row(enum p_pattern ppat, int qh, int S,
                       float *row /* 长度 S */)
{
    if (ppat == P_UNIFORM) {
        float v = 1.0f / S;
        for (int j = 0; j < S; j++) row[j] = v;
        return;
    }
    /* P_QWEN2_LIKE: softmax */
    float maxv = -1e30f;
    for (int j = 0; j < S; j++) {
        float logit = (qh + 1) * 0.1f * (j + 1);
        if (logit > maxv) maxv = logit;
        row[j] = logit;
    }
    float sum = 0.0f;
    for (int j = 0; j < S; j++) {
        row[j] = expf(row[j] - maxv);
        sum += row[j];
    }
    for (int j = 0; j < S; j++) row[j] /= sum;
}

/* ---- 单 case 运行 ---- */
static int run_case(const struct case_config *cfg)
{
    const int S        = cfg->cur_pos + 1;
    const int q_per_kv = cfg->num_q_heads / cfg->num_kv_heads;

    printf("\n========================================\n");
    printf("CASE: %s\n", cfg->name);
    printf("  q_heads=%d kv_heads=%d head_dim=%d max_seq=%d"
           " cur_pos=%d S=%d q_per_kv=%d\n",
           cfg->num_q_heads, cfg->num_kv_heads, cfg->head_dim,
           cfg->max_seq, cfg->cur_pos, S, q_per_kv);
    printf("  tol=%.1e poison_V_beyond_S=%d\n",
           cfg->tol, cfg->poison_vcache_beyond_S);

    /* ---- 内存尺寸 ---- */
    size_t P_size       = (size_t)cfg->num_q_heads * cfg->max_seq  * sizeof(float);
    size_t V_cache_size = (size_t)cfg->num_kv_heads * cfg->max_seq * cfg->head_dim * sizeof(float);
    size_t O_size       = (size_t)cfg->num_q_heads  * cfg->head_dim * sizeof(float);

    /* ---- GPU 分配 ---- */
    uint64_t args_off    = gpuMalloc(g_ctx, sizeof(struct pv_decode_args));
    uint64_t P_off       = gpuMalloc(g_ctx, P_size);
    uint64_t V_cache_off = gpuMalloc(g_ctx, V_cache_size);
    uint64_t O_off       = gpuMalloc(g_ctx, O_size);

    /* ---- host buffer ---- */
    float *P_h       = malloc(P_size);
    float *V_cache_h = malloc(V_cache_size);
    float *O_h       = calloc(cfg->num_q_heads * cfg->head_dim, sizeof(float));
    float *O_ref     = calloc(cfg->num_q_heads * cfg->head_dim, sizeof(float));

    /* ---- 填 P ([num_q_heads, max_seq], j >= S 的位置填 0) ---- */
    memset(P_h, 0, P_size);
    for (int qh = 0; qh < cfg->num_q_heads; qh++) {
        float *row = P_h + (size_t)qh * cfg->max_seq;
        fill_p_row(cfg->ppat, qh, S, row); /* 只写 [0, S) */
    }

    /* ---- 填 V_cache ([num_kv_heads, max_seq, head_dim]) ---- */
    for (int kh = 0; kh < cfg->num_kv_heads; kh++) {
        for (int j = 0; j < cfg->max_seq; j++) {
            for (int k = 0; k < cfg->head_dim; k++) {
                size_t idx = (size_t)kh * cfg->max_seq * cfg->head_dim
                           + (size_t)j  * cfg->head_dim + k;
                if (cfg->poison_vcache_beyond_S && j >= S)
                    V_cache_h[idx] = 999.0f;
                else
                    V_cache_h[idx] = fill_v(cfg->vpat, kh, j, k);
            }
        }
    }

    /* ---- host reference: O[qh, k] = sum_j P[qh,j] * V[kh,j,k] ---- */
    for (int qh = 0; qh < cfg->num_q_heads; qh++) {
        int kh = qh / q_per_kv;
        for (int k = 0; k < cfg->head_dim; k++) {
            float acc = 0.0f;
            for (int j = 0; j < S; j++) {
                float p = P_h[(size_t)qh * cfg->max_seq + j];
                float v = V_cache_h[(size_t)kh * cfg->max_seq * cfg->head_dim
                                  + (size_t)j  * cfg->head_dim + k];
                acc += p * v;
            }
            O_ref[qh * cfg->head_dim + k] = acc;
        }
    }

    /* ---- 上传 ---- */
    gpuMemcpy(g_ctx, P_off,       P_h,       P_size,       0);
    gpuMemcpy(g_ctx, V_cache_off, V_cache_h, V_cache_size, 0);
    gpuMemcpy(g_ctx, O_off,       O_h,       O_size,       0); /* 初始化 0 */

    struct pv_decode_args args = {
        .P_off       = (uint32_t)P_off,
        .V_cache_off = (uint32_t)V_cache_off,
        .O_off       = (uint32_t)O_off,
        .cur_pos     = (uint32_t)cfg->cur_pos,
        .max_seq     = (uint32_t)cfg->max_seq,
        .head_dim    = (uint32_t)cfg->head_dim,
        .q_per_kv    = (uint32_t)q_per_kv,
    };
    gpuMemcpy(g_ctx, args_off, &args, sizeof(args), 0);

    /* ---- 启动 kernel ---- */
    uint32_t grid[3]  = {(uint32_t)cfg->num_q_heads, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    int ret = gpuLaunchKernel(g_ctx, g_kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return -1;
    }

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

    free(P_h);
    free(V_cache_h);
    free(O_h);
    free(O_ref);

    return fail;
}

/* ---- main ---- */
int main(void)
{
    int ret;

    g_ctx = malloc(sizeof(gpgpu_ctx));
    ret   = gpuInit(g_ctx, "/dev/gpgpu", 64ULL * 1024 * 1024);
    if (ret < 0) { fprintf(stderr, "gpuInit failed: %d\n", ret); return 1; }
    printf("[init] gpuInit OK\n");

    /* 加载 kernel binary */
    FILE *f = fopen("/tmp/pv_decode.bin", "rb");
    if (!f) { fprintf(stderr, "open pv_decode.bin failed\n"); return 1; }
    fseek(f, 0, SEEK_END);
    size_t ksize = ftell(f);
    rewind(f);
    uint8_t *kbuf = malloc(ksize);
    fread(kbuf, 1, ksize, f);
    fclose(f);

    g_kernel_off = gpuMalloc(g_ctx, ksize);
    gpuMemcpy(g_ctx, g_kernel_off, kbuf, ksize, 0);
    free(kbuf);
    printf("[init] kernel uploaded, size=%zu\n", ksize);

    /* ---- 4 档测试 ---- */
    struct case_config cases[] = {
        {
            .name            = "case0_baseline (S=max_seq=32, q_per_kv=1)",
            .num_q_heads     = 1, .num_kv_heads = 1,
            .head_dim        = 32, .max_seq = 32, .cur_pos = 31,
            .poison_vcache_beyond_S = 0,
            .vpat = VCACHE_J_PLUS_1,
            .ppat = P_UNIFORM,
            .tol  = 1e-5f,
        },
        {
            .name            = "case1_S_lt_maxseq (S=16, V poisoned beyond S)",
            .num_q_heads     = 1, .num_kv_heads = 1,
            .head_dim        = 32, .max_seq = 32, .cur_pos = 15,
            .poison_vcache_beyond_S = 1,
            .vpat = VCACHE_J_PLUS_1,
            .ppat = P_UNIFORM,
            .tol  = 1e-5f,
        },
        {
            .name            = "case2_GQA (q=4, kv=2, V sign-flipped)",
            .num_q_heads     = 4, .num_kv_heads = 2,
            .head_dim        = 32, .max_seq = 32, .cur_pos = 15,
            .poison_vcache_beyond_S = 1,
            .vpat = VCACHE_GQA_SIGN_FLIP,
            .ppat = P_UNIFORM,
            .tol  = 1e-5f,
        },
        {
            .name            = "case3_qwen2 (q=14, kv=2, hd=64, ms=128, cp=99)",
            .num_q_heads     = 14, .num_kv_heads = 2,
            .head_dim        = 64, .max_seq = 128, .cur_pos = 99,
            .poison_vcache_beyond_S = 1,
            .vpat = VCACHE_QWEN2_LIKE,
            .ppat = P_QWEN2_LIKE,
            .tol  = 1e-4f,  /* 无跨 lane reduce,理论 max_err=0,给一点编译器余量 */
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