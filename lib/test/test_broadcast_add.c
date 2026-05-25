/*
 * test_broadcast_add.c  (Milestone 3.7.0)
 *
 * 测试 broadcast_add.S: y[m*N + n] = x[m*N + n] + b[n]
 *
 * 覆盖矩阵 (Qwen2-0.5B Q/K/V bias 加法路径):
 *   small_basic           - M=2,  N=4       肉眼可读,验证 broadcast 行为
 *   qwen2_kv_bias         - M=16, N=128     Qwen2 K/V bias prefill (std=28)
 *   qwen2_q_bias          - M=16, N=896     Qwen2 Q bias prefill (grid=448)
 *   qwen2_q_bias_decode   - M=1,  N=896     decode 单 token (grid=28)
 *   inplace               - M=4,  N=8       y_off == x_off 防回归
 *   non_multiple_of_32    - M=3,  N=5       M*N=15 < 32,验越界 thread 不乱写
 *
 * 容差: bit-exact (broadcast_add 只做 1 次 fadd.s,无 reduction,
 *       结果应当 deterministic;任何 ULP 差异都是 bug)
 *
 * 失败诊断: 前 10 个 mismatch + max_err 位置 + ULP 差 (区分小漂移与彻底错)
 *
 * 与 gemm test 的差异:
 *   - bit-exact 比对 (memcmp)
 *   - 新增 inplace case (y_off == x_off)
 *   - 新增 non-multiple-of-32 case (验越界检查)
 *   - bias 数据用 std≈28 大数值 (模拟 Qwen2 k_proj.bias 数值特征)
 */

#include "../libgpgpu.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SENTINEL_BITS 0xDEADBEEFu
#define MAX_FAIL_PRINT 10

typedef struct {
    uint32_t x_off;
    uint32_t y_off;
    uint32_t b_off;
    uint32_t M;
    uint32_t N;
} BcastAddArgs;

typedef struct {
    const char *name;
    uint32_t    M;
    uint32_t    N;
    int         inplace; /* 1 → y_off == x_off */
    int large_bias; /* 1 → bias 用 std≈28 大数值 (模拟 k_proj.bias) */
} TestCase;

static const TestCase test_cases[] = {
    {"small_basic", 2, 4, 0, 0},     {"qwen2_kv_bias", 16, 128, 0, 1},
    {"qwen2_q_bias", 16, 896, 0, 1}, {"qwen2_q_bias_decode", 1, 896, 0, 1},
    {"inplace", 4, 8, 1, 0},         {"non_multiple_of_32", 3, 5, 0, 0},
};

#define NUM_CASES (sizeof(test_cases) / sizeof(test_cases[0]))

/* ------------------------------------------------------------------ */
/* 派发: 每 block 1 warp                                              */
/* ------------------------------------------------------------------ */
static inline void calc_dispatch_1warp_per_block(uint32_t total_threads,
                                                 uint32_t grid[3],
                                                 uint32_t block[3]) {
    grid[0]  = (total_threads + 31) / 32;
    grid[1]  = 1;
    grid[2]  = 1;
    block[0] = 32;
    block[1] = 1;
    block[2] = 1;
}

/* ------------------------------------------------------------------ */
/* 填数据                                                              */
/* ------------------------------------------------------------------ */
static void fill_inputs(float *hx, float *hb, uint32_t M, uint32_t N,
                        int large_bias) {
    /* x 用普通 ±1 量级数据 */
    for (uint32_t i = 0; i < M * N; i++)
        hx[i] = ((float)((i % 7) - 3)) * 0.1f;

    /* bias: 默认 ±1 量级;large_bias=1 时放大到 std≈28 (模拟 Qwen2 k_proj.bias)
     * V26 §3.1 实测: k_proj.bias range [-152, 98], std 28.05
     * 这里 sin*30 大致覆盖 ±30 range */
    if (large_bias) {
        for (uint32_t n = 0; n < N; n++)
            hb[n] = 30.0f * sinf(0.7f * (float)n + 0.3f);
    } else {
        for (uint32_t n = 0; n < N; n++)
            hb[n] = ((float)((n % 5) - 2)) * 0.25f;
    }
}

/* ------------------------------------------------------------------ */
/* CPU reference: y[m,n] = x[m,n] + b[n]                              */
/* ------------------------------------------------------------------ */
static void bcast_add_reference(float *ref, const float *hx, const float *hb,
                                uint32_t M, uint32_t N) {
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++)
            ref[m * N + n] = hx[m * N + n] + hb[n];
}

/* ------------------------------------------------------------------ */
/* bit-exact 比对 + ULP 诊断                                          */
/* ------------------------------------------------------------------ */
static int ulp_diff(float a, float b) {
    uint32_t ua, ub;
    memcpy(&ua, &a, 4);
    memcpy(&ub, &b, 4);
    /* 简单 ULP: 不处理跨 0 情形,够诊断用 */
    return (ua > ub) ? (int)(ua - ub) : (int)(ub - ua);
}

static int compare_and_report_bit_exact(const float *hy, const float *ref,
                                        uint32_t out_count) {
    int   num_fail = 0, sentinel_hits = 0;
    float max_err     = 0.0f;
    int   max_err_idx = -1;
    int   max_ulp     = 0;

    for (uint32_t i = 0; i < out_count; i++) {
        uint32_t bits;
        memcpy(&bits, &hy[i], 4);
        if (bits == SENTINEL_BITS)
            sentinel_hits++;

        uint32_t hy_bits, ref_bits;
        memcpy(&hy_bits, &hy[i], 4);
        memcpy(&ref_bits, &ref[i], 4);
        int bit_diff = (hy_bits != ref_bits);

        float err = fabsf(hy[i] - ref[i]);
        int   uld = ulp_diff(hy[i], ref[i]);
        if (err > max_err) {
            max_err     = err;
            max_err_idx = (int)i;
        }
        if (uld > max_ulp)
            max_ulp = uld;

        if (bit_diff) {
            if (num_fail < MAX_FAIL_PRINT) {
                printf("    mismatch i=%-7u dev=% .9e (0x%08x)  "
                       "ref=% .9e (0x%08x)  ulp_diff=%d\n",
                       i, hy[i], hy_bits, ref[i], ref_bits, uld);
            } else if (num_fail == MAX_FAIL_PRINT) {
                printf("    ... (more mismatches suppressed)\n");
            }
            num_fail++;
        }
    }

    if (sentinel_hits > 0) {
        printf("    SENTINEL still: %d / %u outputs untouched (kernel 漏写)\n",
               sentinel_hits, out_count);
    }

    int pass = (num_fail == 0 && sentinel_hits == 0);
    printf("    max_err=%.2e (at i=%d, ref=%.6g)  max_ulp=%d  "
           "%s  (%d/%u bit-exact fail)\n",
           max_err, max_err_idx, max_err_idx >= 0 ? ref[max_err_idx] : 0.0f,
           max_ulp, pass ? "PASS" : "FAIL", num_fail, out_count);
    return pass;
}

/* ------------------------------------------------------------------ */
/* 单 case 执行                                                       */
/* ------------------------------------------------------------------ */
static int run_one_test(gpgpu_ctx *ctx, uint64_t kernel_off, uint64_t args_off,
                        const TestCase *tc) {
    printf("\n--- case: %-22s  M=%-3u N=%-4u  inplace=%d large_bias=%d ---\n",
           tc->name, tc->M, tc->N, tc->inplace, tc->large_bias);

    size_t xy_bytes = (size_t)tc->M * tc->N * 4;
    size_t b_bytes  = (size_t)tc->N * 4;
    printf("    sizes: X/Y=%zu B  B=%zu B\n", xy_bytes, b_bytes);

    /* alloc */
    uint64_t x_off = gpuMalloc(ctx, xy_bytes);
    uint64_t b_off = gpuMalloc(ctx, b_bytes);
    uint64_t y_off;
    if (tc->inplace) {
        y_off = x_off;
    } else {
        y_off = gpuMalloc(ctx, xy_bytes);
    }
    if (x_off == (uint64_t)-1 || b_off == (uint64_t)-1 ||
        (!tc->inplace && y_off == (uint64_t)-1)) {
        printf("    FAIL: gpuMalloc failed\n");
        return 0;
    }

    /* host buffer */
    float *hx  = malloc(xy_bytes);
    float *hb  = malloc(b_bytes);
    float *hy  = malloc(xy_bytes);
    float *ref = malloc(xy_bytes);
    if (!hx || !hb || !hy || !ref) {
        printf("    FAIL: host malloc failed\n");
        free(hx);
        free(hb);
        free(hy);
        free(ref);
        return 0;
    }

    /* fill */
    fill_inputs(hx, hb, tc->M, tc->N, tc->large_bias);

    /* 输出 buffer 预填哨兵
     * inplace 时 x 和 y 是同一块,先填 x 数据;非 inplace 时填 y 为哨兵 */
    if (!tc->inplace) {
        uint32_t sent_bits = SENTINEL_BITS;
        for (uint32_t i = 0; i < tc->M * tc->N; i++)
            memcpy(&hy[i], &sent_bits, 4);
    }

    /* 上传 */
    gpuMemcpy(ctx, x_off, hx, xy_bytes, GPU_MEMCPY_H2D);
    gpuMemcpy(ctx, b_off, hb, b_bytes, GPU_MEMCPY_H2D);
    if (!tc->inplace) {
        gpuMemcpy(ctx, y_off, hy, xy_bytes, GPU_MEMCPY_H2D);
    }

    /* args */
    BcastAddArgs args = {
        .x_off = (uint32_t)x_off,
        .y_off = (uint32_t)y_off,
        .b_off = (uint32_t)b_off,
        .M     = tc->M,
        .N     = tc->N,
    };
    gpuMemcpy(ctx, args_off, &args, sizeof(args), GPU_MEMCPY_H2D);

    /* dispatch */
    uint32_t total = tc->M * tc->N;
    uint32_t grid[3], block[3];
    calc_dispatch_1warp_per_block(total, grid, block);
    printf("    dispatch: %u threads, grid={%u,%u,%u} block={%u,%u,%u}\n",
           total, grid[0], grid[1], grid[2], block[0], block[1], block[2]);

    /* launch */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    if (ret < 0) {
        printf("    FAIL: gpuLaunchKernel failed: %d\n", ret);
        free(hx);
        free(hb);
        free(hy);
        free(ref);
        return 0;
    }
    printf("    launch time: %.4f s\n", elapsed);

    /* 取回 */
    gpuMemcpy(ctx, y_off, hy, xy_bytes, GPU_MEMCPY_D2H);

    /* host reference */
    bcast_add_reference(ref, hx, hb, tc->M, tc->N);

    int pass = compare_and_report_bit_exact(hy, ref, tc->M * tc->N);

    /* small_basic case 打印前几个值方便肉眼检查 */
    if (pass && strcmp(tc->name, "small_basic") == 0) {
        printf("    [hy first 8]:  ");
        for (uint32_t i = 0; i < 8 && i < tc->M * tc->N; i++)
            printf("%+.3f ", hy[i]);
        printf("\n    [ref first 8]: ");
        for (uint32_t i = 0; i < 8 && i < tc->M * tc->N; i++)
            printf("%+.3f ", ref[i]);
        printf("\n");
    }

    free(hx);
    free(hb);
    free(hy);
    free(ref);
    return pass;
}

/* ------------------------------------------------------------------ */
int main(void) {
    int ret;
    printf("=== test_broadcast_add (Milestone 3.7.0) ===\n");

    gpgpu_ctx *ctx = malloc(sizeof(gpgpu_ctx));
    if (!ctx) {
        fprintf(stderr, "malloc ctx failed\n");
        return 1;
    }

    /* VRAM: 各 case 累计远不到 100 MB,16 MB 足够 */
    ret = gpuInit(ctx, "/dev/gpgpu", 16ULL * 1024 * 1024);
    if (ret < 0) {
        fprintf(stderr, "gpuInit failed: %d\n", ret);
        return 1;
    }
    printf("[init] gpuInit OK, VRAM=16MB\n");

    /* kernel binary */
    FILE *f = fopen("/tmp/broadcast_add.bin", "rb");
    if (!f) {
        perror("fopen broadcast_add.bin");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t kernel_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *kbuf = malloc(kernel_size);
    fread(kbuf, 1, kernel_size, f);
    fclose(f);

    uint64_t kernel_off = gpuMalloc(ctx, kernel_size);
    uint64_t args_off   = gpuMalloc(ctx, sizeof(BcastAddArgs));
    if (kernel_off == (uint64_t)-1 || args_off == (uint64_t)-1) {
        fprintf(stderr, "gpuMalloc kernel/args failed\n");
        return 1;
    }
    gpuMemcpy(ctx, kernel_off, kbuf, kernel_size, GPU_MEMCPY_H2D);
    free(kbuf);
    printf("[init] kernel uploaded (%zu bytes)\n", kernel_size);

    /* 跑所有 case */
    int total_pass = 0;
    int total      = (int)NUM_CASES;
    for (int i = 0; i < total; i++) {
        if (run_one_test(ctx, kernel_off, args_off, &test_cases[i])) {
            total_pass++;
        } else {
            printf("    (continuing to next case despite failure)\n");
        }
    }

    printf("\n=== summary: %d / %d cases passed ===\n", total_pass, total);
    printf("\nbreakdown:\n");
    printf("  broadcast 语义 (b[n] 广播到所有 row): small_basic 通过即可\n");
    printf("  Qwen2 大数值 bias  (std≈28):        kv_bias / q_bias 通过即可\n");
    printf("  decode 单 token (M=1):              q_bias_decode 通过即可\n");
    printf("  in-place 支持 (y==x):               inplace 通过即可\n");
    printf(
        "  越界 thread 不写非法地址:           non_multiple_of_32 通过即可\n");
    printf("  bit-exact (max_ulp 应当为 0):       所有 case 都应是 0\n");

    gpuDestroy(ctx);
    free(ctx);
    return (total_pass == total) ? 0 : 1;
}