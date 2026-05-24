/*
 * test_gemm.c  (V22 重写版,Milestone 3.6.4)
 *
 * 取代旧版的:
 *   - 单 shape M=1, K=4864, N=896 (FFN down)
 *   - 容差 1e-3 太松
 *   - 没覆盖 LM head 大 N / K/V proj 小 N / FFN gate 大 N
 *
 * 覆盖矩阵 (Qwen2-0.5B decode 路径所有 gemm shape):
 *   kv_proj   - M=1, K=896,  N=128     (K/V projection)
 *   q_proj    - M=1, K=896,  N=896     (Q projection / O projection)
 *   ffn_down  - M=1, K=4864, N=896     (FFN down,老 case 回归)
 *   ffn_gate  - M=1, K=896,  N=4864    (FFN gate / FFN up)
 *   lm_head   - M=1, K=896,  N=151936  (LM head,最大 dispatch)
 *
 * 重大改动:
 *   - gemm.S 已改造为 "每 block 1 warp" multi-block dispatch
 *   - host 用 calc_dispatch_1warp_per_block: grid=(N/32,1,1) block=(32,1,1)
 *   - VRAM 提到 1GB (LM head 的 B 矩阵 = 519MB)
 *
 * 容差: max(1e-4, 1e-5 * |ref|) (覆盖 K=4864 时的累积误差)
 * 失败诊断: 前 10 个 mismatch + max_err 位置
 * V23 债务 (15): gemm.S 朴素 1 thread 1 output, prefill (M>1) 未覆盖
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
#define ABS_TOL 1e-4f
#define REL_TOL 1e-5f
#define MAX_FAIL_PRINT 10

typedef struct {
    uint32_t a_off;
    uint32_t b_off;
    uint32_t c_off;
    uint32_t M;
    uint32_t K;
    uint32_t N;
} GemmArgs;

typedef struct {
    const char *name;
    uint32_t    M;
    uint32_t    K;
    uint32_t    N;
} TestCase;

/* 按规模从小到大排,跑挂的话至少前面有结果 */
static const TestCase test_cases[] = {
    {"kv_proj", 1, 896, 128},    {"q_proj", 1, 896, 896},
    {"ffn_down", 1, 4864, 896},  {"ffn_gate", 1, 896, 4864},
    {"lm_head", 1, 896, 151936},
};

#define NUM_CASES (sizeof(test_cases) / sizeof(test_cases[0]))

/* ------------------------------------------------------------------ */
/* 派发:每 block 1 warp,grid[0] 扩展                                  */
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
static void fill_inputs(float *ha, float *hb, uint32_t M, uint32_t K,
                        uint32_t N) {
    for (uint32_t i = 0; i < M * K; i++)
        ha[i] = ((float)((i % 7) - 3)) * 0.1f;
    for (uint32_t i = 0; i < K * N; i++)
        hb[i] = ((float)((i % 11) - 5)) * 0.1f + 1e-7f;
}

/* ------------------------------------------------------------------ */
/* CPU reference                                                       */
/* ------------------------------------------------------------------ */
static void gemm_reference(float *ref, const float *ha, const float *hb,
                           uint32_t M, uint32_t K, uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++)
                sum += ha[i * K + k] * hb[k * N + j];
            ref[i * N + j] = sum;
        }
    }
}

/* ------------------------------------------------------------------ */
/* 误差判定 + 诊断                                                     */
/* ------------------------------------------------------------------ */
static int compare_and_report(const float *hc, const float *ref,
                              uint32_t out_count) {
    int   num_fail = 0, sentinel_hits = 0;
    float max_err = 0.0f, max_rel = 0.0f, sum_err = 0.0f;
    int   max_err_idx = -1;

    for (uint32_t i = 0; i < out_count; i++) {
        uint32_t bits;
        memcpy(&bits, &hc[i], 4);
        if (bits == SENTINEL_BITS)
            sentinel_hits++;

        float err  = fabsf(hc[i] - ref[i]);
        float tol  = ABS_TOL;
        float rtol = REL_TOL * fabsf(ref[i]);
        if (rtol > tol)
            tol = rtol;

        sum_err += err;
        if (err > max_err) {
            max_err     = err;
            max_err_idx = (int)i;
        }
        if (fabsf(ref[i]) > 1e-10f) {
            float rel = err / fabsf(ref[i]);
            if (rel > max_rel)
                max_rel = rel;
        }

        if (err > tol) {
            if (num_fail < MAX_FAIL_PRINT) {
                printf("    mismatch i=%-7u dev=% .8e  ref=% .8e  "
                       "err=%.2e  tol=%.2e\n",
                       i, hc[i], ref[i], err, tol);
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
    printf("    max_err=%.2e (at i=%d, ref=%.6g)  max_rel=%.2e  "
           "mean_err=%.2e  %s  (%d/%u fail)\n",
           max_err, max_err_idx, max_err_idx >= 0 ? ref[max_err_idx] : 0.0f,
           max_rel, sum_err / out_count, pass ? "PASS" : "FAIL", num_fail,
           out_count);
    return pass;
}

/* ------------------------------------------------------------------ */
/* 单 case 执行                                                       */
/* ------------------------------------------------------------------ */
static int run_one_test(gpgpu_ctx *ctx, uint64_t kernel_off, uint64_t args_off,
                        const TestCase *tc) {
    printf("\n--- case: %-10s  M=%u K=%-5u N=%-6u ---\n", tc->name, tc->M,
           tc->K, tc->N);

    size_t a_bytes = (size_t)tc->M * tc->K * 4;
    size_t b_bytes = (size_t)tc->K * tc->N * 4;
    size_t c_bytes = (size_t)tc->M * tc->N * 4;
    printf("    sizes: A=%zu KB  B=%zu KB  C=%zu KB  (total %zu KB)\n",
           a_bytes / 1024, b_bytes / 1024, c_bytes / 1024,
           (a_bytes + b_bytes + c_bytes) / 1024);

    /* 每 case 独立 alloc/free 大 buffer,避免一次性占太多 VRAM */
    uint64_t a_off = gpuMalloc(ctx, a_bytes);
    uint64_t b_off = gpuMalloc(ctx, b_bytes);
    uint64_t c_off = gpuMalloc(ctx, c_bytes);
    if (a_off == (uint64_t)-1 || b_off == (uint64_t)-1 ||
        c_off == (uint64_t)-1) {
        printf("    FAIL: gpuMalloc failed (case 占 %zu MB,"
               "ctx VRAM 可能不够)\n",
               (a_bytes + b_bytes + c_bytes) / (1024 * 1024));
        return 0;
    }

    /* host buffer */
    float *ha  = malloc(a_bytes);
    float *hb  = malloc(b_bytes);
    float *hc  = malloc(c_bytes);
    float *ref = malloc(c_bytes);
    if (!ha || !hb || !hc || !ref) {
        printf("    FAIL: host malloc failed\n");
        free(ha);
        free(hb);
        free(hc);
        free(ref);
        return 0;
    }

    /* 填数据 + 哨兵 */
    fill_inputs(ha, hb, tc->M, tc->K, tc->N);
    uint32_t sent_bits = SENTINEL_BITS;
    for (uint32_t i = 0; i < tc->M * tc->N; i++)
        memcpy(&hc[i], &sent_bits, 4);

    /* 上传 */
    gpuMemcpy(ctx, a_off, ha, a_bytes, GPU_MEMCPY_H2D);
    gpuMemcpy(ctx, b_off, hb, b_bytes, GPU_MEMCPY_H2D);
    gpuMemcpy(ctx, c_off, hc, c_bytes, GPU_MEMCPY_H2D);

    /* args */
    GemmArgs args = {
        .a_off = (uint32_t)a_off,
        .b_off = (uint32_t)b_off,
        .c_off = (uint32_t)c_off,
        .M     = tc->M,
        .K     = tc->K,
        .N     = tc->N,
    };
    gpuMemcpy(ctx, args_off, &args, sizeof(args), GPU_MEMCPY_H2D);

    /* dispatch */
    uint32_t total = tc->M * tc->N;
    uint32_t grid[3], block[3];
    calc_dispatch_1warp_per_block(total, grid, block);
    printf("    dispatch: %u threads, grid={%u,%u,%u} block={%u,%u,%u}\n",
           total, grid[0], grid[1], grid[2], block[0], block[1], block[2]);

    /* launch + 计时 */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    if (ret < 0) {
        printf("    FAIL: gpuLaunchKernel failed: %d\n", ret);
        goto cleanup_fail;
    }
    printf("    launch time: %.2f s\n", elapsed);

    /* 取回 */
    gpuMemcpy(ctx, c_off, hc, c_bytes, GPU_MEMCPY_D2H);

    /* host reference */
    gemm_reference(ref, ha, hb, tc->M, tc->K, tc->N);

    int pass = compare_and_report(hc, ref, tc->M * tc->N);

    free(ha);
    free(hb);
    free(hc);
    free(ref);
    /* 注: libgpgpu 当前的 buddy/slab allocator 暂没暴露 gpuFree,
     * 大 case 后续可能 OOM。如果有 gpuFree 在此调用。*/
    return pass;

cleanup_fail:
    free(ha);
    free(hb);
    free(hc);
    free(ref);
    return 0;
}

/* ------------------------------------------------------------------ */
int main(void) {
    int ret;
    printf("=== test_gemm (V22 重写, Milestone 3.6.4) ===\n");

    gpgpu_ctx *ctx = malloc(sizeof(gpgpu_ctx));
    if (!ctx) {
        fprintf(stderr, "malloc ctx failed\n");
        return 1;
    }

    /* VRAM: LM head 的 B 矩阵就 519MB,需要 1GB+ */
    ret = gpuInit(ctx, "/dev/gpgpu", 2048ULL * 1024 * 1024);
    if (ret < 0) {
        fprintf(stderr,
                "gpuInit failed: %d (尝试缩小 VRAM 或确认 device 容量)\n", ret);
        return 1;
    }
    printf("[init] gpuInit OK, VRAM=1GB\n");

    /* kernel binary */
    FILE *f = fopen("/tmp/gemm.bin", "rb");
    if (!f) {
        perror("fopen gemm.bin");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t kernel_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *kbuf = malloc(kernel_size);
    fread(kbuf, 1, kernel_size, f);
    fclose(f);

    uint64_t kernel_off = gpuMalloc(ctx, kernel_size);
    uint64_t args_off   = gpuMalloc(ctx, sizeof(GemmArgs));
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
    printf("  shape coverage:                查看每个 case 是否 PASS\n");
    printf("  LM head dispatch (4748 block): 查看 lm_head 是否 PASS,"
           "若 OOM 需检查 VRAM\n");
    printf("  V23 债务: gemm.S 朴素 1-thread-1-output, prefill (M>1) 未测\n");

    gpuDestroy(ctx);
    free(ctx);
    return (total_pass == total) ? 0 : 1;
}