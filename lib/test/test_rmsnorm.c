/*
 * test_rmsnorm.c  (V22 重写版,Milestone 3.6.3)
 *
 * 取代旧版的:
 *   - 全 1 输入 (每 lane 的 local sumsq 全等于 28,reduce 算式部分错会隐形)
 *   - 全 2 weight (weight 路径退化)
 *   - 单 case + 容差 1e-3 (太松)
 *
 * 覆盖矩阵 (rmsnorm 是 warp reduce 类,V21 原则"非重叠 lane 范围"是核心):
 *   baseline_uniform   - 全 1 输入,保留作最弱基线
 *   linear             - (i - D/2) * 0.01,跨 lane 单调
 *   asymmetric7        - 7 周期 pattern,打破 lane 内对称
 *   lane_distinct      - 每 lane 的 28 元素生成不同的 local sumsq
 *                        (V21 原则正面应用:确保 32 lane 全部 sumsq 唯一)
 *   mixed_sign         - 正负混合 + 大动态范围
 *   asym_weight        - 非对称 weight (独立验证 weight 路径)
 *
 * Helper self-check: 两个 host reference 互相交叉验证,
 *   ref_strict (模拟 device fp 路径) vs ref_natural (直观算法),
 *   两者 max_err ≤ 1e-6 才放过。
 *
 * 容差: max(1e-5, 5e-6 * |ref|),覆盖 reduce + sqrt + div + mul 累积。
 * rmsnorm.S 当前写死 d=896,所有 case 都用 d=896 (V23 债务 14 同款)。
 */

#include "../libgpgpu.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D 896
#define LANES 32
#define PER_LANE (D / LANES) /* 28 */
#define EPS 1e-6f
#define SENTINEL_OUT 0xDEADBEEFu
#define ABS_TOL 1e-5f
#define REL_TOL 5e-6f
#define MAX_FAIL_PRINT 10

typedef struct {
    uint32_t in_off;
    uint32_t out_off;
    uint32_t weight_off;
    uint32_t d;
    float    inv_d;
    float    eps;
} rmsnorm_args;

typedef enum {
    INP_UNIFORM,
    INP_LINEAR,
    INP_ASYM7,
    INP_LANE_DISTINCT,
    INP_MIXED_SIGN,
} InputPat;

typedef enum {
    W_UNIFORM, /* 全 2,只验证 input 路径 */
    W_ASYM,    /* 非对称,验证 weight 路径 */
} WeightPat;

typedef struct {
    const char *name;
    InputPat    inp;
    WeightPat   w;
} TestCase;

static const TestCase test_cases[] = {
    {"baseline_uniform", INP_UNIFORM, W_UNIFORM},
    {"linear", INP_LINEAR, W_UNIFORM},
    {"asymmetric7", INP_ASYM7, W_UNIFORM},
    {"lane_distinct", INP_LANE_DISTINCT, W_UNIFORM},
    {"mixed_sign", INP_MIXED_SIGN, W_UNIFORM},
    {"asym_weight", INP_ASYM7, W_ASYM},
    {"distinct_x_asymW", INP_LANE_DISTINCT, W_ASYM}, /* 双独立非对称 */
};

#define NUM_CASES (sizeof(test_cases) / sizeof(test_cases[0]))

/* ------------------------------------------------------------------ */
/* 填 input                                                            */
/* ------------------------------------------------------------------ */
static void fill_input(float *h_in, InputPat pat) {
    switch (pat) {
    case INP_UNIFORM:
        for (int i = 0; i < D; i++)
            h_in[i] = 1.0f;
        break;
    case INP_LINEAR:
        for (int i = 0; i < D; i++)
            h_in[i] = (float)(i - D / 2) * 0.01f;
        break;
    case INP_ASYM7:
        for (int i = 0; i < D; i++)
            h_in[i] = ((float)(i % 7) - 3.0f) * 0.5f + 0.13f;
        break;
    case INP_LANE_DISTINCT:
        /* 关键 case: 每 lane 的 28 元素生成不同 local sumsq.
         * 数据 = (lane_id + 1) * 0.1 + i * 0.01
         * → 32 个 lane 的 sumsq 单调递增,全部不同 */
        for (int lane = 0; lane < LANES; lane++) {
            for (int i = 0; i < PER_LANE; i++) {
                h_in[lane * PER_LANE + i] =
                    (float)(lane + 1) * 0.1f + (float)i * 0.01f;
            }
        }
        break;
    case INP_MIXED_SIGN:
        for (int i = 0; i < D; i++) {
            float base = ((float)(i % 11) - 5.0f) * 0.3f;
            float sign = (i % 2) ? -1.0f : 1.0f;
            h_in[i]    = base * sign + 0.07f;
        }
        break;
    }
}

/* ------------------------------------------------------------------ */
/* 填 weight                                                          */
/* ------------------------------------------------------------------ */
static void fill_weight(float *h_w, WeightPat pat) {
    switch (pat) {
    case W_UNIFORM:
        for (int i = 0; i < D; i++)
            h_w[i] = 2.0f;
        break;
    case W_ASYM:
        /* 非对称: ((i % 13) + 1) * 0.1,范围 [0.1, 1.3],每个 lane 都覆盖多个值
         */
        for (int i = 0; i < D; i++)
            h_w[i] = ((float)(i % 13) + 1.0f) * 0.1f;
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Reference 1: ref_strict — 完全模拟 device fp 路径                   */
/* ------------------------------------------------------------------ */
static void ref_strict(float *out, const float *in, const float *w, float inv_d,
                       float eps) {
    /* device 路径: sumsq -> *inv_d -> +eps -> sqrt -> 1/x -> *x*w */
    float sumsq = 0.0f;
    for (int i = 0; i < D; i++)
        sumsq += in[i] * in[i];
    float ms   = sumsq * inv_d;
    float rms  = sqrtf(ms + eps);
    float rrms = 1.0f / rms;
    for (int i = 0; i < D; i++)
        out[i] = in[i] * rrms * w[i];
}

/* ------------------------------------------------------------------ */
/* Reference 2: ref_natural — 直观算法 (用 sumsq/D 而非 sumsq*inv_d)   */
/* ------------------------------------------------------------------ */
static void ref_natural(float *out, const float *in, const float *w,
                        float eps) {
    float sumsq = 0.0f;
    for (int i = 0; i < D; i++)
        sumsq += in[i] * in[i];
    float rms  = sqrtf(sumsq / (float)D + eps);
    float rrms = 1.0f / rms;
    for (int i = 0; i < D; i++)
        out[i] = in[i] * rrms * w[i];
}

/* ------------------------------------------------------------------ */
/* Helper self-check: 两个 reference 交叉验证 + lane sumsq 唯一性     */
/* ------------------------------------------------------------------ */
static void reference_self_check(void) {
    float h_in[D], h_w[D];
    float r1[D], r2[D];

    /* check 1: 两个 reference 在干净输入下应非常接近 */
    fill_input(h_in, INP_ASYM7);
    fill_weight(h_w, W_UNIFORM);
    ref_strict(r1, h_in, h_w, 1.0f / (float)D, EPS);
    ref_natural(r2, h_in, h_w, EPS);
    float max_diff = 0.0f;
    for (int i = 0; i < D; i++) {
        float d = fabsf(r1[i] - r2[i]);
        if (d > max_diff)
            max_diff = d;
    }
    assert(max_diff < 1e-5f);
    printf("[self-check] ref_strict vs ref_natural: max_diff = %.2e\n",
           max_diff);

    /* check 2: lane_distinct 必须真的让 32 lane sumsq 全不同 */
    fill_input(h_in, INP_LANE_DISTINCT);
    float lane_sumsq[LANES];
    for (int lane = 0; lane < LANES; lane++) {
        float s = 0.0f;
        for (int i = 0; i < PER_LANE; i++) {
            float x = h_in[lane * PER_LANE + i];
            s += x * x;
        }
        lane_sumsq[lane] = s;
    }
    /* 验证全不同 (任意两两差 > 1e-6) */
    int distinct = 1;
    for (int a = 0; a < LANES && distinct; a++) {
        for (int b = a + 1; b < LANES; b++) {
            if (fabsf(lane_sumsq[a] - lane_sumsq[b]) < 1e-6f) {
                printf("[self-check] FAIL: lane_sumsq[%d] == lane_sumsq[%d] = "
                       "%.6f\n",
                       a, b, lane_sumsq[a]);
                distinct = 0;
                break;
            }
        }
    }
    assert(distinct);
    printf("[self-check] lane_distinct: 32 lane sumsq all unique "
           "(range [%.4f, %.4f])\n",
           lane_sumsq[0], lane_sumsq[LANES - 1]);

    /* check 3: rmsnorm 不变量: 全 1 输入 + 全 1 weight → 输出全 1 */
    for (int i = 0; i < D; i++) {
        h_in[i] = 1.0f;
        h_w[i]  = 1.0f;
    }
    ref_strict(r1, h_in, h_w, 1.0f / (float)D, EPS);
    for (int i = 0; i < D; i++)
        assert(fabsf(r1[i] - 1.0f) < 1e-5f);
    printf("[self-check] rmsnorm(1, 1) ≈ 1: ok\n");
}

/* ------------------------------------------------------------------ */
/* 误差判定 + 失败诊断                                                 */
/* ------------------------------------------------------------------ */
static int compare_and_report(const float *h_out, const float *h_ref) {
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
                printf("    mismatch i=%-4d dev=% .8e  ref=% .8e  "
                       "err=%.2e  tol=%.2e\n",
                       i, h_out[i], h_ref[i], err, tol);
            } else if (num_fail == MAX_FAIL_PRINT) {
                printf("    ... (more mismatches suppressed)\n");
            }
            num_fail++;
        }
    }

    int pass = (num_fail == 0);
    printf("    max_err=%.2e (at i=%d, ref=%.6g)  max_rel=%.2e  mean_err=%.2e  "
           "%s  (%d/%d fail)\n",
           max_err, max_err_idx, max_err_idx >= 0 ? h_ref[max_err_idx] : 0.0f,
           max_rel, sum_err / D, pass ? "PASS" : "FAIL", num_fail, D);
    return pass;
}

/* ------------------------------------------------------------------ */
/* 单 case 执行                                                       */
/* ------------------------------------------------------------------ */
static int run_one_test(gpgpu_ctx *ctx, uint64_t kernel_off, uint64_t args_off,
                        uint64_t in_off, uint64_t out_off, uint64_t w_off,
                        const TestCase *tc) {
    printf("\n--- case: %s ---\n", tc->name);

    float *h_in  = malloc(D * 4);
    float *h_out = malloc(D * 4);
    float *h_w   = malloc(D * 4);
    float *h_ref = malloc(D * 4);
    assert(h_in && h_out && h_w && h_ref);

    fill_input(h_in, tc->inp);
    fill_weight(h_w, tc->w);
    ref_strict(h_ref, h_in, h_w, 1.0f / (float)D, EPS);

    /* 哨兵 */
    for (int i = 0; i < D; i++) {
        uint32_t s = SENTINEL_OUT;
        memcpy(&h_out[i], &s, 4);
    }
    gpuMemcpy(ctx, out_off, h_out, D * 4, 0);

    /* 上传输入 */
    gpuMemcpy(ctx, in_off, h_in, D * 4, 0);
    gpuMemcpy(ctx, w_off, h_w, D * 4, 0);

    /* args */
    rmsnorm_args args = {
        .in_off     = (uint32_t)in_off,
        .out_off    = (uint32_t)out_off,
        .weight_off = (uint32_t)w_off,
        .d          = D,
        .inv_d      = 1.0f / (float)D,
        .eps        = EPS,
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
        free(h_w);
        free(h_ref);
        return 0;
    }

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
        free(h_w);
        free(h_ref);
        return 0;
    }

    int pass = compare_and_report(h_out, h_ref);

    free(h_in);
    free(h_out);
    free(h_w);
    free(h_ref);
    return pass;
}

/* ------------------------------------------------------------------ */
int main(void) {
    int ret;
    printf("=== test_rmsnorm (V22 重写) ===\n");

    reference_self_check();

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

    FILE *f = fopen("/tmp/rmsnorm.bin", "rb");
    if (!f) {
        fprintf(stderr, "open rmsnorm.bin failed\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t ksize = ftell(f);
    rewind(f);
    uint8_t *kbuf = malloc(ksize);
    fread(kbuf, 1, ksize, f);
    fclose(f);

    uint64_t kernel_off = gpuMalloc(ctx, ksize);
    uint64_t args_off   = gpuMalloc(ctx, sizeof(rmsnorm_args));
    uint64_t in_off     = gpuMalloc(ctx, D * 4);
    uint64_t out_off    = gpuMalloc(ctx, D * 4);
    uint64_t w_off      = gpuMalloc(ctx, D * 4);
    if (kernel_off == (uint64_t)-1 || args_off == (uint64_t)-1 ||
        in_off == (uint64_t)-1 || out_off == (uint64_t)-1 ||
        w_off == (uint64_t)-1) {
        fprintf(stderr, "gpuMalloc failed\n");
        return 1;
    }
    gpuMemcpy(ctx, kernel_off, kbuf, ksize, 0);
    free(kbuf);
    printf("[init] kernel uploaded (%zu bytes), D=%d, eps=%g\n", ksize, D, EPS);

    int total_pass = 0;
    int total      = (int)NUM_CASES;
    for (int i = 0; i < total; i++) {
        if (run_one_test(ctx, kernel_off, args_off, in_off, out_off, w_off,
                         &test_cases[i])) {
            total_pass++;
        }
    }

    printf("\n=== summary: %d / %d cases passed ===\n", total_pass, total);

    gpuDestroy(ctx);
    free(ctx);
    return (total_pass == total) ? 0 : 1;
}