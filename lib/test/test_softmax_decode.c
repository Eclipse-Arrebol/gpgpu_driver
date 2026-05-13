/*
 * test_softmax_decode.c
 * 测试 softmax_decode.S kernel
 *
 * 数学: scores[qh, 0..S-1] = softmax(scores[qh, 0..S-1])  in-place
 *       S = cur_pos + 1
 *
 * 4 档测试矩阵:
 *   case0: baseline  (S=max_seq=32, q_heads=1)
 *   case1: S<max_seq (S=16, max_seq=32, 检查 j>=S 位置不被写)
 *   case2: GQA尺度  (q_heads=4, S=16)
 *   case3: Qwen2    (q_heads=14, max_seq=128, cur_pos=99)
 */

#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct softmax_decode_args {
    uint32_t scores_off;
    uint32_t cur_pos;
    uint32_t max_seq;
};

struct case_config {
    const char *name;
    int         num_q_heads;
    int         max_seq;
    int         cur_pos;
    int         check_beyond_S;   /* 检查 j>=S 的位置是否被污染 */
    float       tol;
};

static gpgpu_ctx *g_ctx;
static uint64_t   g_kernel_off;

/* host reference softmax，只处理 [0, S) */
static void softmax_ref(const float *in, float *out, int S)
{
    float maxv = -1e30f;
    for (int j = 0; j < S; j++)
        if (in[j] > maxv) maxv = in[j];

    float sum = 0.0f;
    for (int j = 0; j < S; j++) {
        out[j] = expf(in[j] - maxv);
        sum += out[j];
    }
    for (int j = 0; j < S; j++)
        out[j] /= sum;
}

static int run_case(const struct case_config *cfg)
{
    const int S = cfg->cur_pos + 1;

    printf("\n========================================\n");
    printf("CASE: %s\n", cfg->name);
    printf("  q_heads=%d max_seq=%d cur_pos=%d S=%d check_beyond_S=%d tol=%.1e\n",
           cfg->num_q_heads, cfg->max_seq, cfg->cur_pos, S,
           cfg->check_beyond_S, cfg->tol);

    size_t scores_size = (size_t)cfg->num_q_heads * cfg->max_seq * sizeof(float);

    uint64_t args_off   = gpuMalloc(g_ctx, sizeof(struct softmax_decode_args));
    uint64_t scores_off = gpuMalloc(g_ctx, scores_size);

    float *scores_in  = malloc(scores_size);
    float *scores_out = malloc(scores_size);
    float *scores_ref = malloc(scores_size);

    /* 填输入：scores[qh, j] = (qh+1) * 0.1f * (j+1)，j>=S 填 -999 做毒数据 */
    for (int qh = 0; qh < cfg->num_q_heads; qh++) {
        for (int j = 0; j < cfg->max_seq; j++) {
            int idx = qh * cfg->max_seq + j;
            if (j < S)
                scores_in[idx] = (qh + 1) * 0.1f * (j + 1);
            else
                scores_in[idx] = -999.0f;   /* 毒数据，kernel 不应该碰 */
        }
    }

    /* host reference：每行独立 softmax */
    memset(scores_ref, 0, scores_size);
    for (int qh = 0; qh < cfg->num_q_heads; qh++) {
        softmax_ref(scores_in + qh * cfg->max_seq,
                    scores_ref + qh * cfg->max_seq, S);
        /* j>=S 保持 -999，用于越界检查 */
        for (int j = S; j < cfg->max_seq; j++)
            scores_ref[qh * cfg->max_seq + j] = -999.0f;
    }

    /* 上传 */
    gpuMemcpy(g_ctx, scores_off, scores_in, scores_size, 0);

    struct softmax_decode_args args = {
        .scores_off = (uint32_t)scores_off,
        .cur_pos    = (uint32_t)cfg->cur_pos,
        .max_seq    = (uint32_t)cfg->max_seq,
    };
    gpuMemcpy(g_ctx, args_off, &args, sizeof(args), 0);

    /* 启动 kernel */
    uint32_t grid[3]  = {(uint32_t)cfg->num_q_heads, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    int ret = gpuLaunchKernel(g_ctx, g_kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return -1;
    }

    /* 回读 */
    gpuMemcpy(g_ctx, scores_off, scores_out, scores_size, 1);

    /* 比对 */
    float max_err = 0.0f;
    int   max_qh  = -1, max_j = -1;
    int   fail    = 0;
    int   poison_fail = 0;

    for (int qh = 0; qh < cfg->num_q_heads; qh++) {
        /* [0, S) 区间：比对 softmax 结果 */
        for (int j = 0; j < S; j++) {
            int   idx  = qh * cfg->max_seq + j;
            float got  = scores_out[idx];
            float ref  = scores_ref[idx];
            float diff = fabsf(got - ref);
            if (diff > max_err) {
                max_err = diff;
                max_qh  = qh;
                max_j   = j;
            }
            if (diff >= cfg->tol) {
                if (fail < 10)
                    printf("FAIL scores[qh=%d, j=%d] got=%f ref=%f diff=%.3e\n",
                           qh, j, got, ref, diff);
                fail++;
            }
        }

        /* [S, max_seq) 区间：检查毒数据没被改写 */
        if (cfg->check_beyond_S) {
            for (int j = S; j < cfg->max_seq; j++) {
                int   idx = qh * cfg->max_seq + j;
                float got = scores_out[idx];
                if (fabsf(got - (-999.0f)) > 1e-3f) {
                    if (poison_fail < 5)
                        printf("OVERWRITE scores[qh=%d, j=%d] got=%f (should be -999)\n",
                               qh, j, got);
                    poison_fail++;
                    fail++;
                }
            }
        }
    }

    /* 额外验证：每行和应为 1.0 */
    for (int qh = 0; qh < cfg->num_q_heads; qh++) {
        float row_sum = 0.0f;
        for (int j = 0; j < S; j++)
            row_sum += scores_out[qh * cfg->max_seq + j];
        float sum_err = fabsf(row_sum - 1.0f);
        if (sum_err > cfg->tol * S) {
            printf("SUM_FAIL qh=%d row_sum=%f (expected 1.0, err=%.3e)\n",
                   qh, row_sum, sum_err);
            fail++;
        }
    }

    /* 采样输出 */
    printf("---- RESULT ----\n");
    printf("max_err     = %.3e at scores[qh=%d, j=%d]\n", max_err, max_qh, max_j);
    printf("fail        = %d  (overwrite_beyond_S=%d)\n", fail, poison_fail);

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
                   qh, j, scores_out[idx], scores_ref[idx]);
        }
    }

    /* 越界采样 */
    if (cfg->check_beyond_S && cfg->max_seq > S) {
        int qh  = cfg->num_q_heads / 2;
        int idx = qh * cfg->max_seq + S;
        printf("scores[qh=%d, j=%d] = %f (should be -999)\n",
               qh, S, scores_out[idx]);
    }

    printf("%s\n", fail == 0 ? "PASS" : "FAIL");

    free(scores_in);
    free(scores_out);
    free(scores_ref);

    return fail;
}

int main(void)
{
    int ret;

    g_ctx = malloc(sizeof(gpgpu_ctx));
    ret   = gpuInit(g_ctx, "/dev/gpgpu", 64ULL * 1024 * 1024);
    if (ret < 0) { fprintf(stderr, "gpuInit failed: %d\n", ret); return 1; }
    printf("[init] gpuInit OK\n");

    FILE *f = fopen("/tmp/softmax_decode.bin", "rb");
    if (!f) { fprintf(stderr, "open softmax_decode.bin failed\n"); return 1; }
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
            .name          = "case0_baseline (S=max_seq=32, q_heads=1)",
            .num_q_heads   = 1,
            .max_seq       = 32, .cur_pos = 31,
            .check_beyond_S = 0,
            .tol           = 1e-5f,
        },
        {
            .name          = "case1_S_lt_maxseq (S=16, max_seq=32)",
            .num_q_heads   = 1,
            .max_seq       = 32, .cur_pos = 15,
            .check_beyond_S = 1,
            .tol           = 1e-5f,
        },
        {
            .name          = "case2_multi_heads (q_heads=4, S=16, max_seq=32)",
            .num_q_heads   = 4,
            .max_seq       = 32, .cur_pos = 15,
            .check_beyond_S = 1,
            .tol           = 1e-5f,
        },
        {
            .name          = "case3_qwen2 (q_heads=14, max_seq=128, cur_pos=99)",
            .num_q_heads   = 14,
            .max_seq       = 128, .cur_pos = 99,
            .check_beyond_S = 1,
            .tol           = 1e-4f,   /* 跨 lane reduce，预期 1~2 ULP 误差 */
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
