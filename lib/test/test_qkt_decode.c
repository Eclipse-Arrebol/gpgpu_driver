#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* K_cache 填充模式 */
enum kcache_pattern {
    KCACHE_J_PLUS_1,            /* K[kh, j, k] = j+1,所有 kh 一样(档 0/1)*/
    KCACHE_GQA_SIGN_FLIP,       /* K[kh=0,...] = j+1,K[kh=1,...] = -(j+1)(档 2)*/
    KCACHE_QWEN2_LIKE,          /* K[kh, j, k] = (j+1) * (kh+1) * 0.01 + k*0.001(档 3)*/
};

enum q_pattern {
    Q_ALL_ONE,                  /* Q[qh, k] = 1.0(档 0/1/2)*/
    Q_QWEN2_LIKE,               /* Q[qh, k] = (qh+1) * 0.02 + k*0.0001(档 3)*/
};

struct case_config {
    const char *name;
    int   num_q_heads;
    int   num_kv_heads;
    int   head_dim;
    int   max_seq;
    int   cur_pos;
    float scale;
    int   poison_kcache_beyond_S;
    int   check_scores_beyond_S;
    enum kcache_pattern kpat;
    enum q_pattern      qpat;
    float tol;                  /* 容忍度 */
};

static gpgpu_ctx *g_ctx;
static uint64_t   g_kernel_off;

static float fill_q(enum q_pattern p, int qh, int k) {
    switch (p) {
        case Q_ALL_ONE:     return 1.0f;
        case Q_QWEN2_LIKE:  return (qh + 1) * 0.02f + k * 0.0001f;
    }
    return 0;
}

static float fill_k(enum kcache_pattern p, int kh, int j, int k) {
    switch (p) {
        case KCACHE_J_PLUS_1:        return (float)(j + 1);
        case KCACHE_GQA_SIGN_FLIP:   return (kh == 0) ? (float)(j + 1) : -(float)(j + 1);
        case KCACHE_QWEN2_LIKE:      return (j + 1) * (kh + 1) * 0.01f + k * 0.001f;
    }
    return 0;
}

int run_case(const struct case_config *cfg) {
    const int S = cfg->cur_pos + 1;
    const int q_per_kv = cfg->num_q_heads / cfg->num_kv_heads;

    printf("\n========================================\n");
    printf("CASE: %s\n", cfg->name);
    printf("  q_heads=%d kv_heads=%d head_dim=%d max_seq=%d cur_pos=%d S=%d q_per_kv=%d\n",
           cfg->num_q_heads, cfg->num_kv_heads, cfg->head_dim,
           cfg->max_seq, cfg->cur_pos, S, q_per_kv);
    printf("  scale=%f tol=%.1e poison_K_beyond_S=%d check_scores_beyond_S=%d\n",
           cfg->scale, cfg->tol,
           cfg->poison_kcache_beyond_S, cfg->check_scores_beyond_S);

    size_t Q_size       = cfg->num_q_heads  * cfg->head_dim * sizeof(float);
    size_t K_cache_size = cfg->num_kv_heads * cfg->max_seq  * cfg->head_dim * sizeof(float);
    size_t scores_size  = cfg->num_q_heads  * cfg->max_seq  * sizeof(float);

    uint64_t args_off    = gpuMalloc(g_ctx, sizeof(struct qkt_decode_args));
    uint64_t Q_off       = gpuMalloc(g_ctx, Q_size);
    uint64_t K_cache_off = gpuMalloc(g_ctx, K_cache_size);
    uint64_t scores_off  = gpuMalloc(g_ctx, scores_size);

    float *Q_h         = malloc(Q_size);
    float *K_cache_h   = malloc(K_cache_size);
    float *scores_h    = malloc(scores_size);
    float *scores_ref  = malloc(scores_size);
    float *scores_init = malloc(scores_size);

    /* 填 Q */
    for (int qh = 0; qh < cfg->num_q_heads; qh++)
        for (int k = 0; k < cfg->head_dim; k++)
            Q_h[qh * cfg->head_dim + k] = fill_q(cfg->qpat, qh, k);

    /* 填 K_cache */
    for (int kh = 0; kh < cfg->num_kv_heads; kh++) {
        for (int j = 0; j < cfg->max_seq; j++) {
            for (int k = 0; k < cfg->head_dim; k++) {
                int idx = kh * cfg->max_seq * cfg->head_dim
                        + j * cfg->head_dim + k;
                if (cfg->poison_kcache_beyond_S && j >= S) {
                    K_cache_h[idx] = 999.0f;
                } else {
                    K_cache_h[idx] = fill_k(cfg->kpat, kh, j, k);
                }
            }
        }
    }

    /* host reference */
    for (int j = 0; j < cfg->num_q_heads * cfg->max_seq; j++) {
        scores_ref[j]  = 0.0f;
        scores_init[j] = 0.0f;
    }
    for (int qh = 0; qh < cfg->num_q_heads; qh++) {
        int kh = qh / q_per_kv;
        for (int j = 0; j < S; j++) {
            float acc = 0.0f;
            for (int k = 0; k < cfg->head_dim; k++) {
                int q_idx = qh * cfg->head_dim + k;
                int k_idx = kh * cfg->max_seq * cfg->head_dim
                          + j * cfg->head_dim + k;
                acc += Q_h[q_idx] * K_cache_h[k_idx];
            }
            scores_ref[qh * cfg->max_seq + j] = acc * cfg->scale;
        }
    }

    /* 上传 */
    gpuMemcpy(g_ctx, Q_off, Q_h, Q_size, 0);
    gpuMemcpy(g_ctx, K_cache_off, K_cache_h, K_cache_size, 0);
    gpuMemcpy(g_ctx, scores_off, scores_init, scores_size, 0);

    uint32_t scale_bits;
    memcpy(&scale_bits, &cfg->scale, sizeof(float));
    struct qkt_decode_args args = {
        .Q_off       = (uint32_t)Q_off,
        .K_cache_off = (uint32_t)K_cache_off,
        .scores_off  = (uint32_t)scores_off,
        .cur_pos     = (uint32_t)cfg->cur_pos,
        .max_seq     = (uint32_t)cfg->max_seq,
        .head_dim    = (uint32_t)cfg->head_dim,
        .q_per_kv    = (uint32_t)q_per_kv,
        .scale_bits  = scale_bits,
    };
    gpuMemcpy(g_ctx, args_off, &args, sizeof(args), 0);

    uint32_t grid[3]  = {cfg->num_q_heads, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    int ret = gpuLaunchKernel(g_ctx, g_kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return -1;
    }

    gpuMemcpy(g_ctx, scores_off, scores_h, scores_size, 1);

    float max_err = 0.0f;
    int   max_qh  = -1, max_j = -1;
    int   fail = 0;
    int   over_write_fail = 0;

    int total_j = cfg->check_scores_beyond_S ? cfg->max_seq : S;
    for (int qh = 0; qh < cfg->num_q_heads; qh++) {
        for (int j = 0; j < total_j; j++) {
            int idx = qh * cfg->max_seq + j;
            float got = scores_h[idx];
            float ref = scores_ref[idx];
            float diff = fabsf(got - ref);
            if (diff > max_err) {
                max_err = diff;
                max_qh  = qh;
                max_j   = j;
            }
            if (diff >= cfg->tol) {
                if (fail < 10) {
                    if (j >= S)
                        printf("OVERWRITE scores[qh=%d, j=%d] got=%f (should be 0)\n",
                               qh, j, got);
                    else
                        printf("FAIL scores[qh=%d, j=%d] got=%f ref=%f diff=%.3e\n",
                               qh, j, got, ref, diff);
                }
                fail++;
                if (j >= S) over_write_fail++;
            }
        }
    }

    printf("---- RESULT ----\n");
    printf("max_err = %.3e at qh=%d, j=%d\n", max_err, max_qh, max_j);
    printf("fail    = %d  (overwrite_beyond_S = %d)\n", fail, over_write_fail);

    /* 几个采样点输出 */
    int probe_qhs[] = {0, cfg->num_q_heads - 1, cfg->num_q_heads / 2};
    int probe_js[]  = {0, S - 1, S / 2};
    for (int pi = 0; pi < 3; pi++) {
        int qh = probe_qhs[pi];
        if (qh < 0 || qh >= cfg->num_q_heads) continue;
        for (int pj = 0; pj < 3; pj++) {
            int j = probe_js[pj];
            if (j < 0 || j >= S) continue;
            int idx = qh * cfg->max_seq + j;
            printf("scores[qh=%d, j=%d] = %f (ref %f)\n",
                   qh, j, scores_h[idx], scores_ref[idx]);
        }
    }

    /* 越界采样 */
    if (cfg->check_scores_beyond_S && cfg->max_seq > S) {
        int qh = cfg->num_q_heads / 2;
        int j_oob = S;
        int idx = qh * cfg->max_seq + j_oob;
        printf("scores[qh=%d, j=%d] = %f (should be 0.0)\n",
               qh, j_oob, scores_h[idx]);
    }

    printf("%s\n", fail == 0 ? "PASS" : "FAIL");

    free(Q_h);
    free(K_cache_h);
    free(scores_h);
    free(scores_ref);
    free(scores_init);

    return fail;
}

int main() {
    int ret;

    g_ctx = malloc(sizeof(gpgpu_ctx));
    ret = gpuInit(g_ctx, "/dev/gpgpu", 64ULL * 1024 * 1024);
    if (ret < 0) { fprintf(stderr, "gpuInit failed: %d\n", ret); return 1; }
    printf("[init] gpuInit OK\n");

    FILE *f = fopen("/tmp/qkt_decode.bin", "rb");
    if (!f) { fprintf(stderr, "open qkt_decode.bin failed\n"); return 1; }
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

    struct case_config cases[] = {
        {
            .name = "case0_baseline (S=max_seq=32)",
            .num_q_heads = 1, .num_kv_heads = 1,
            .head_dim = 32, .max_seq = 32, .cur_pos = 31,
            .scale = 1.0f,
            .poison_kcache_beyond_S = 0,
            .check_scores_beyond_S  = 0,
            .kpat = KCACHE_J_PLUS_1,
            .qpat = Q_ALL_ONE,
            .tol  = 1e-4f,
        },
        {
            .name = "case1_S_lt_maxseq (S=16, max_seq=32, K poisoned beyond S)",
            .num_q_heads = 1, .num_kv_heads = 1,
            .head_dim = 32, .max_seq = 32, .cur_pos = 15,
            .scale = 1.0f,
            .poison_kcache_beyond_S = 1,
            .check_scores_beyond_S  = 1,
            .kpat = KCACHE_J_PLUS_1,
            .qpat = Q_ALL_ONE,
            .tol  = 1e-4f,
        },
        {
            .name = "case2_GQA (q_heads=4, kv_heads=2, q_per_kv=2, K sign-flipped)",
            .num_q_heads = 4, .num_kv_heads = 2,
            .head_dim = 32, .max_seq = 32, .cur_pos = 15,
            .scale = 1.0f,
            .poison_kcache_beyond_S = 1,
            .check_scores_beyond_S  = 1,
            .kpat = KCACHE_GQA_SIGN_FLIP,
            .qpat = Q_ALL_ONE,
            .tol  = 1e-4f,
        },
        {
            .name = "case3_qwen2_scale (q_heads=14, kv_heads=2, head_dim=64, max_seq=128, cur_pos=99)",
            .num_q_heads = 14, .num_kv_heads = 2,
            .head_dim = 64, .max_seq = 128, .cur_pos = 99,
            .scale = 0.125f,    /* 1/sqrt(64) */
            .poison_kcache_beyond_S = 1,
            .check_scores_beyond_S  = 1,
            .kpat = KCACHE_QWEN2_LIKE,
            .qpat = Q_QWEN2_LIKE,
            .tol  = 1e-3f,      /* 大规模浮点累积,放宽 */
        },
    };

    int total_fail = 0;
    int n = sizeof(cases) / sizeof(cases[0]);
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