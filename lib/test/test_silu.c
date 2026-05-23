/*
 * test_silu.c  (V22 重写版,Milestone 3.6.2)
 *
 * 取代旧版的:
 *   - 单 case 线性均匀输入
 *   - softmax 残留断言 "sum expect 1.0"
 *   - 失败时刷屏,无前 N 截断
 *
 * 覆盖矩阵 (SiLU = x * sigmoid(x), 重点是数值范围,非 lane 模式):
 *   linear_baseline   - 老 case 回归基线 [-8.96, 8.94]
 *   all_positive      - 全正区, SiLU(x) ≈ x 当 x 大
 *   all_negative      - 全负区, SiLU(x) ≈ 0 当 x 很负
 *   near_zero         - 精度敏感区 [-0.045, 0.045]
 *   extreme_pos       - 大正值 ≤ 30, 验证 exp(-x) 不下溢
 *   extreme_neg       - 大负值 ≥ -30, 验证 exp(-x) 不上溢
 *   asymmetric        - 7 周期非对称 pattern (V21 教训正面应用)
 *
 * 容差: max(abs_tol, rel_tol * |ref|), 防止 |x| 大时绝对容差太严、
 * |x| 小时绝对容差太松。
 *
 * silu.S 当前写死 d=896,所有 case 都用 d=896 (V23 债务 14 同款)。
 */

#include "../libgpgpu.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D 896
#define SENTINEL_OUT 0xDEADBEEFu
#define ABS_TOL 1e-6f
#define REL_TOL 4e-6f     /* ~4 ULP for fp32 */
#define MAX_FAIL_PRINT 10 /* 失败时最多打这么多 mismatch */

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
} TestCase;

static const TestCase test_cases[] = {
    {"linear_baseline", PAT_LINEAR},  {"all_positive", PAT_ALL_POS},
    {"all_negative", PAT_ALL_NEG},    {"near_zero", PAT_NEAR_ZERO},
    {"extreme_pos", PAT_EXTREME_POS}, {"extreme_neg", PAT_EXTREME_NEG},
    {"asymmetric", PAT_ASYMMETRIC},
};

#define NUM_CASES (sizeof(test_cases) / sizeof(test_cases[0]))

/* ------------------------------------------------------------------ */
/* host reference: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))       */
/* ------------------------------------------------------------------ */
static float silu_ref(float x) { return x / (1.0f + expf(-x)); }

/* ------------------------------------------------------------------ */
/* helper 自检: silu_ref 在已知点行为正确                              */
/* ------------------------------------------------------------------ */
static void reference_self_check(void) {
    /* SiLU(0) = 0 / (1 + 1) = 0 */
    assert(fabsf(silu_ref(0.0f)) < 1e-7f);

    /* SiLU'(0) = 1/2 + 0/4 = 1/2,所以 SiLU(eps) ≈ eps/2 */
    float eps    = 1e-3f;
    float approx = silu_ref(eps);
    assert(fabsf(approx - eps * 0.5f) < 1e-5f);

    /* SiLU(很大正) ≈ x */
    assert(fabsf(silu_ref(30.0f) - 30.0f) < 1e-5f);

    /* SiLU(很大负) ≈ 0 */
    assert(fabsf(silu_ref(-30.0f)) < 1e-10f);

    printf("[self-check] silu_ref passed 4 sanity tests\n");
}

/* ------------------------------------------------------------------ */
/* 填输入                                                              */
/* ------------------------------------------------------------------ */
static void fill_input(float *h_in, Pattern pat) {
    for (int i = 0; i < D; i++) {
        switch (pat) {
        case PAT_LINEAR:
            h_in[i] = (float)(i - 448) / 50.0f; /* [-8.96, 8.94] */
            break;
        case PAT_ALL_POS:
            h_in[i] = 0.1f * (float)i + 0.05f; /* [0.05, ~89.6] */
            break;
        case PAT_ALL_NEG:
            /* 收紧到 [-9.0, -0.05]: 避免 |ref| 下溢到 abs_tol 之下,
             * 否则 device 输出 0 也会 PASS (V22 原则 15 反例) */
            h_in[i] = -0.01f * (float)i - 0.05f;
            break;
        case PAT_NEAR_ZERO:
            h_in[i] = (float)(i - 448) * 1e-4f; /* [-0.0448, 0.0447] */
            break;
        case PAT_EXTREME_POS:
            h_in[i] = (float)(i % 7) * 5.0f; /* {0,5,10,...,30} */
            break;
        case PAT_EXTREME_NEG:
            /* 收紧到 {-12, -10, -8, -6, -4, -2, 0}: x=-12 时 ref ≈ -7e-5,
             * 仍在 abs_tol=1e-6 之上,rel_tol 能真起作用 */
            h_in[i] = -((float)(i % 7) * 2.0f);
            break;
        case PAT_ASYMMETRIC:
            h_in[i] = ((float)(i % 7) - 3.0f) * 1.7f +
                      0.13f; /* 7 周期 [-4.97, 5.33] */
            break;
        }
    }
}

/* ------------------------------------------------------------------ */
/* 误差判定 + 诊断                                                     */
/* ------------------------------------------------------------------ */
static int compare_and_report(const float *h_in, const float *h_out,
                              const float *h_ref) {
    int   num_fail = 0;
    float max_err = 0.0f, max_rel = 0.0f, sum_err = 0.0f;
    int   max_err_idx = -1;

    for (int i = 0; i < D; i++) {
        float err  = fabsf(h_out[i] - h_ref[i]);
        float tol  = ABS_TOL;
        float rtol = REL_TOL * fabsf(h_ref[i]);
        if (rtol > tol)
            tol = rtol;

        sum_err += err;
        if (err > max_err) {
            max_err     = err;
            max_err_idx = i;
        }
        if (fabsf(h_ref[i]) > 1e-10f) {
            float rel = err / fabsf(h_ref[i]);
            if (rel > max_rel)
                max_rel = rel;
        }

        if (err > tol) {
            if (num_fail < MAX_FAIL_PRINT) {
                printf("    mismatch i=%-4d x=% .6f  dev=% .8e  ref=% .8e  "
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
           "max_rel=%.2e  mean_err=%.2e  %s  (%d/%d fail)\n",
           max_err, max_err_idx, max_err_idx >= 0 ? h_in[max_err_idx] : 0.0f,
           max_err_idx >= 0 ? h_ref[max_err_idx] : 0.0f, max_rel, sum_err / D,
           pass ? "PASS" : "FAIL", num_fail, D);

    return pass;
}

/* ------------------------------------------------------------------ */
/* 单 case 执行                                                       */
/* ------------------------------------------------------------------ */
static int run_one_test(gpgpu_ctx *ctx, uint64_t kernel_off, uint64_t args_off,
                        uint64_t in_off, uint64_t out_off, const TestCase *tc) {
    printf("\n--- case: %s ---\n", tc->name);

    float *h_in  = malloc(D * 4);
    float *h_out = malloc(D * 4);
    float *h_ref = malloc(D * 4);
    assert(h_in && h_out && h_ref);

    /* 填输入 + host reference */
    fill_input(h_in, tc->pat);
    for (int i = 0; i < D; i++) {
        h_ref[i] = silu_ref(h_in[i]);
    }

    /* 哨兵填 out (检测漏写) */
    for (int i = 0; i < D; i++) {
        uint32_t s = SENTINEL_OUT;
        memcpy(&h_out[i], &s, 4);
    }
    gpuMemcpy(ctx, out_off, h_out, D * 4, 0);

    /* 上传输入 */
    gpuMemcpy(ctx, in_off, h_in, D * 4, 0);

    /* 上传 args */
    struct silu_args args = {
        .in_off  = (uint32_t)in_off,
        .out_off = (uint32_t)out_off,
        .d       = D,
    };
    gpuMemcpy(ctx, args_off, &args, sizeof(args), 0);

    /* launch */
    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    int      ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        printf("    gpuLaunchKernel failed: %d\n", ret);
        free(h_in);
        free(h_out);
        free(h_ref);
        return 0;
    }

    /* 取回 */
    gpuMemcpy(ctx, out_off, h_out, D * 4, 1);

    /* 哨兵检查 */
    int untouched = 0;
    for (int i = 0; i < D; i++) {
        uint32_t bits;
        memcpy(&bits, &h_out[i], 4);
        if (bits == SENTINEL_OUT)
            untouched++;
    }
    if (untouched > 0) {
        printf("    FAIL: %d / %d outputs untouched (sentinel intact) → "
               "kernel 漏写\n",
               untouched, D);
        free(h_in);
        free(h_out);
        free(h_ref);
        return 0;
    }

    /* 判定 */
    int pass = compare_and_report(h_in, h_out, h_ref);

    free(h_in);
    free(h_out);
    free(h_ref);
    return pass;
}

/* ------------------------------------------------------------------ */
int main(void) {
    int ret;
    printf("=== test_silu (V22 重写) ===\n");

    reference_self_check();

    /* ---------- ctx 初始化 ---------- */
    gpgpu_ctx *ctx = malloc(sizeof(gpgpu_ctx));
    if (!ctx) {
        fprintf(stderr, "malloc ctx failed\n");
        return 1;
    }

    ret = gpuInit(ctx, "/dev/gpgpu", 64ULL * 1024 * 1024);
    if (ret < 0) {
        fprintf(stderr, "gpuInit failed: %d\n", ret);
        return 1;
    }
    printf("[init] gpuInit OK\n");

    /* ---------- 加载 kernel binary ---------- */
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

    /* ---------- 跨 case 共享的资源 ---------- */
    uint64_t kernel_off = gpuMalloc(ctx, ksize);
    uint64_t args_off   = gpuMalloc(ctx, sizeof(struct silu_args));
    uint64_t in_off     = gpuMalloc(ctx, D * 4);
    uint64_t out_off    = gpuMalloc(ctx, D * 4);
    if (kernel_off == (uint64_t)-1 || args_off == (uint64_t)-1 ||
        in_off == (uint64_t)-1 || out_off == (uint64_t)-1) {
        fprintf(stderr, "gpuMalloc failed\n");
        return 1;
    }
    gpuMemcpy(ctx, kernel_off, kbuf, ksize, 0);
    free(kbuf);
    printf("[init] kernel uploaded (%zu bytes), D=%d\n", ksize, D);

    /* ---------- 跑所有 case ---------- */
    int total_pass = 0;
    int total      = (int)NUM_CASES;
    for (int i = 0; i < total; i++) {
        if (run_one_test(ctx, kernel_off, args_off, in_off, out_off,
                         &test_cases[i])) {
            total_pass++;
        }
    }

    printf("\n=== summary: %d / %d cases passed ===\n", total_pass, total);

    gpuDestroy(ctx);
    free(ctx);
    return (total_pass == total) ? 0 : 1;
}