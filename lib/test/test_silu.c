/*
 * test_silu.c  (M3.7.1 multi-block 改造)
 *
 * M3.7.1 改动:
 *   - silu.S 从单 warp 硬编码 D=896 改为 multi-block runtime D
 *   - dispatch 从 grid={1,1,1} 改为 calc_dispatch_1warp_per_block(D, grid, block)
 *   - 所有函数参数化 D
 *   - 新增 D=4864 (Qwen2 intermediate) 回归 case
 *   - PAT_LINEAR 用 D/2 替代硬编码 448
 *
 * 覆盖矩阵:
 *   D=896 (Qwen2 hidden_size):
 *     linear_baseline, all_positive, all_negative, near_zero,
 *     extreme_pos, extreme_neg, asymmetric  (7 case)
 *   D=4864 (Qwen2 intermediate_size):
 *     linear_baseline_4864, all_positive_4864, extreme_neg_4864  (3 case)
 *
 * 容差: max(ABS_TOL, REL_TOL * |ref|)
 */

#include "../libgpgpu.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_D 4864
#define SENTINEL_OUT 0xDEADBEEFu
#define ABS_TOL 1e-6f
#define REL_TOL 4e-6f
#define MAX_FAIL_PRINT 10

struct silu_args {
    uint32_t in_off;
    uint32_t out_off;
    uint32_t d;
};

typedef enum {
    PAT_LINEAR,
    PAT_ALL_POS,
    PAT_ALL_NEG,
    PAT_NEAR_ZERO,
    PAT_EXTREME_POS,
    PAT_EXTREME_NEG,
    PAT_ASYMMETRIC,
} Pattern;

typedef struct {
    const char *name;
    Pattern     pat;
    uint32_t    D;
} TestCase;

static const TestCase test_cases[] = {
    /* D=896 回归 */
    {"linear_baseline_896",  PAT_LINEAR,      896},
    {"all_positive_896",     PAT_ALL_POS,     896},
    {"all_negative_896",     PAT_ALL_NEG,     896},
    {"near_zero_896",        PAT_NEAR_ZERO,   896},
    {"extreme_pos_896",      PAT_EXTREME_POS, 896},
    {"extreme_neg_896",      PAT_EXTREME_NEG, 896},
    {"asymmetric_896",       PAT_ASYMMETRIC,  896},
    /* D=4864 新增 */
    {"linear_baseline_4864", PAT_LINEAR,      4864},
    {"all_positive_4864",    PAT_ALL_POS,     4864},
    {"extreme_neg_4864",     PAT_EXTREME_NEG, 4864},
};

#define NUM_CASES (sizeof(test_cases) / sizeof(test_cases[0]))

static inline void calc_dispatch_1warp_per_block(uint32_t total,
                                                 uint32_t grid[3],
                                                 uint32_t block[3]) {
    grid[0]  = (total + 31) / 32;
    grid[1]  = 1;
    grid[2]  = 1;
    block[0] = 32;
    block[1] = 1;
    block[2] = 1;
}

static float silu_ref(float x) { return x / (1.0f + expf(-x)); }

static void reference_self_check(void) {
    assert(fabsf(silu_ref(0.0f)) < 1e-7f);
    float eps    = 1e-3f;
    float approx = silu_ref(eps);
    assert(fabsf(approx - eps * 0.5f) < 1e-5f);
    assert(fabsf(silu_ref(30.0f) - 30.0f) < 1e-5f);
    assert(fabsf(silu_ref(-30.0f)) < 1e-10f);
    printf("[self-check] silu_ref passed 4 sanity tests\n");
}

static void fill_input(float *h_in, Pattern pat, uint32_t D) {
    for (uint32_t i = 0; i < D; i++) {
        switch (pat) {
        case PAT_LINEAR:
            h_in[i] = (float)((int)i - (int)(D / 2)) / 50.0f;
            break;
        case PAT_ALL_POS:
            h_in[i] = 0.1f * (float)i + 0.05f;
            break;
        case PAT_ALL_NEG:
            h_in[i] = -0.01f * (float)i - 0.05f;
            break;
        case PAT_NEAR_ZERO:
            h_in[i] = (float)((int)i - (int)(D / 2)) * 1e-4f;
            break;
        case PAT_EXTREME_POS:
            h_in[i] = (float)(i % 7) * 5.0f;
            break;
        case PAT_EXTREME_NEG:
            h_in[i] = -((float)(i % 7) * 2.0f);
            break;
        case PAT_ASYMMETRIC:
            h_in[i] = ((float)(i % 7) - 3.0f) * 1.7f + 0.13f;
            break;
        }
    }
}

static int compare_and_report(const float *h_in, const float *h_out,
                              const float *h_ref, uint32_t D) {
    int   num_fail = 0;
    float max_err = 0.0f, max_rel = 0.0f, sum_err = 0.0f;
    int   max_err_idx = -1;

    for (uint32_t i = 0; i < D; i++) {
        float err  = fabsf(h_out[i] - h_ref[i]);
        float tol  = ABS_TOL;
        float rtol = REL_TOL * fabsf(h_ref[i]);
        if (rtol > tol)
            tol = rtol;

        sum_err += err;
        if (err > max_err) {
            max_err     = err;
            max_err_idx = (int)i;
        }
        if (fabsf(h_ref[i]) > 1e-10f) {
            float rel = err / fabsf(h_ref[i]);
            if (rel > max_rel)
                max_rel = rel;
        }

        if (err > tol) {
            if (num_fail < MAX_FAIL_PRINT) {
                printf("    mismatch i=%-5u x=% .6f  dev=% .8e  ref=% .8e  "
                       "err=%.2e  tol=%.2e\n",
                       i, h_in[i], h_out[i], h_ref[i], err, tol);
            } else if (num_fail == MAX_FAIL_PRINT) {
                printf("    ... (more mismatches suppressed)\n");
            }
            num_fail++;
        }
    }

    int pass = (num_fail == 0);
    printf("    max_err=%.2e (at i=%d, x=%.4f, ref=%.6g)  "
           "max_rel=%.2e  mean_err=%.2e  %s  (%d/%u fail)\n",
           max_err, max_err_idx, max_err_idx >= 0 ? h_in[max_err_idx] : 0.0f,
           max_err_idx >= 0 ? h_ref[max_err_idx] : 0.0f, max_rel,
           sum_err / (float)D, pass ? "PASS" : "FAIL", num_fail, D);

    return pass;
}

static int run_one_test(gpgpu_ctx *ctx, uint64_t kernel_off, uint64_t args_off,
                        uint64_t in_off, uint64_t out_off,
                        const TestCase *tc) {
    printf("\n--- case: %-25s  D=%u ---\n", tc->name, tc->D);
    uint32_t D = tc->D;

    float *h_in  = malloc(D * 4);
    float *h_out = malloc(D * 4);
    float *h_ref = malloc(D * 4);
    assert(h_in && h_out && h_ref);

    fill_input(h_in, tc->pat, D);
    for (uint32_t i = 0; i < D; i++)
        h_ref[i] = silu_ref(h_in[i]);

    for (uint32_t i = 0; i < D; i++) {
        uint32_t s = SENTINEL_OUT;
        memcpy(&h_out[i], &s, 4);
    }
    gpuMemcpy(ctx, out_off, h_out, D * 4, 0);
    gpuMemcpy(ctx, in_off, h_in, D * 4, 0);

    struct silu_args args = {
        .in_off  = (uint32_t)in_off,
        .out_off = (uint32_t)out_off,
        .d       = D,
    };
    gpuMemcpy(ctx, args_off, &args, sizeof(args), 0);

    uint32_t grid[3], block[3];
    calc_dispatch_1warp_per_block(D, grid, block);
    printf("    grid=[%u,1,1] block=[32,1,1] total_threads=%u\n",
           grid[0], grid[0] * 32);

    int ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        printf("    gpuLaunchKernel failed: %d\n", ret);
        free(h_in);
        free(h_out);
        free(h_ref);
        return 0;
    }

    gpuMemcpy(ctx, out_off, h_out, D * 4, 1);

    int untouched = 0;
    for (uint32_t i = 0; i < D; i++) {
        uint32_t bits;
        memcpy(&bits, &h_out[i], 4);
        if (bits == SENTINEL_OUT)
            untouched++;
    }
    if (untouched > 0) {
        printf("    FAIL: %d / %u outputs untouched (sentinel intact)\n",
               untouched, D);
        free(h_in);
        free(h_out);
        free(h_ref);
        return 0;
    }

    int pass = compare_and_report(h_in, h_out, h_ref, D);
    free(h_in);
    free(h_out);
    free(h_ref);
    return pass;
}

int main(void) {
    int ret;
    printf("=== test_silu (M3.7.1 multi-block) ===\n");

    reference_self_check();

    gpgpu_ctx *ctx = malloc(sizeof(gpgpu_ctx));
    if (!ctx) {
        fprintf(stderr, "malloc ctx failed\n");
        return 1;
    }

    ret = gpuInit(ctx, "/dev/gpgpu", 256ULL * 1024 * 1024);
    if (ret < 0) {
        fprintf(stderr, "gpuInit failed: %d\n", ret);
        return 1;
    }
    printf("[init] gpuInit OK\n");

    FILE *f = fopen("/tmp/silu.bin", "rb");
    if (!f) {
        fprintf(stderr, "open silu.bin failed\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t ksize = ftell(f);
    rewind(f);
    uint8_t *kbuf = malloc(ksize);
    fread(kbuf, 1, ksize, f);
    fclose(f);

    uint64_t kernel_off = gpuMalloc(ctx, ksize);
    uint64_t args_off   = gpuMalloc(ctx, sizeof(struct silu_args));
    uint64_t in_off     = gpuMalloc(ctx, MAX_D * 4);
    uint64_t out_off    = gpuMalloc(ctx, MAX_D * 4);
    if (kernel_off == (uint64_t)-1 || args_off == (uint64_t)-1 ||
        in_off == (uint64_t)-1 || out_off == (uint64_t)-1) {
        fprintf(stderr, "gpuMalloc failed\n");
        return 1;
    }
    gpuMemcpy(ctx, kernel_off, kbuf, ksize, 0);
    free(kbuf);
    printf("[init] kernel uploaded (%zu bytes), MAX_D=%d\n", ksize, MAX_D);

    int total_pass = 0;
    int total      = (int)NUM_CASES;
    for (int i = 0; i < total; i++) {
        if (run_one_test(ctx, kernel_off, args_off, in_off, out_off,
                         &test_cases[i]))
            total_pass++;
    }

    printf("\n=== summary: %d / %d cases passed ===\n", total_pass, total);

    gpuDestroy(ctx);
    free(ctx);
    return (total_pass == total) ? 0 : 1;
}
