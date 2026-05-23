/*
 * test_gemm_verify.c — 最小验证版
 *
 * 目的:验证 gemm.S 改造为 "每 block 1 warp" 后,旧 shape (M=1, K=4864, N=896)
 * 仍能正确计算。
 *
 * 唯一改动: calc_dispatch 改为多 block 派发,grid=(N/32, 1, 1), block=(32,1,1)。
 * 通过后再写完整 3.6.4 多 shape 测试 (含 LM head)。
 */

#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const uint32_t M = 1;
const uint32_t K = 4864;
const uint32_t N = 896;

typedef struct {
    uint32_t a_off;
    uint32_t b_off;
    uint32_t c_off;
    uint32_t M;
    uint32_t K;
    uint32_t N;
} GemmArgs;

/* 新派发: 每 block 1 warp (32 thread), 用 grid[0] 扩展到任意大小 */
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

int main() {
    int        ret;
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
    printf("[1] gpuInit OK\n");

    FILE *f = fopen("/tmp/gemm.bin", "rb");
    if (!f) {
        perror("fopen gemm.bin");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t kernel_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    printf("[1.5] gemm.bin size = %zu bytes\n", kernel_size);

    uint64_t kernel_off = gpuMalloc(ctx, kernel_size);
    uint64_t args_off   = gpuMalloc(ctx, sizeof(GemmArgs));
    uint64_t a_off      = gpuMalloc(ctx, M * K * 4);
    uint64_t b_off      = gpuMalloc(ctx, K * N * 4);
    uint64_t c_off      = gpuMalloc(ctx, M * N * 4);
    if (kernel_off == (uint64_t)-1 || args_off == (uint64_t)-1 ||
        a_off == (uint64_t)-1 || b_off == (uint64_t)-1 ||
        c_off == (uint64_t)-1) {
        fprintf(stderr, "gpuMalloc failed\n");
        return 1;
    }
    printf("[2] alloc OK\n");

    uint8_t *kbuf = malloc(kernel_size);
    fread(kbuf, 1, kernel_size, f);
    fclose(f);
    gpuMemcpy(ctx, kernel_off, kbuf, kernel_size, GPU_MEMCPY_H2D);
    free(kbuf);
    printf("[3] kernel uploaded\n");

    float *ha = malloc(M * K * sizeof(float));
    float *hb = malloc(K * N * sizeof(float));
    float *hc = malloc(M * N * sizeof(float));

    for (uint32_t i = 0; i < M * K; i++)
        ha[i] = ((float)((i % 7) - 3)) * 0.1f;
    for (uint32_t i = 0; i < K * N; i++)
        hb[i] = ((float)((i % 11) - 5)) * 0.1f + 1e-7f;

    /* 哨兵 */
    for (uint32_t i = 0; i < M * N; i++)
        hc[i] = -999999.0f;

    gpuMemcpy(ctx, a_off, ha, M * K * 4, GPU_MEMCPY_H2D);
    gpuMemcpy(ctx, b_off, hb, K * N * 4, GPU_MEMCPY_H2D);
    gpuMemcpy(ctx, c_off, hc, M * N * 4, GPU_MEMCPY_H2D);
    printf("[4] input + C sentinel uploaded\n");

    GemmArgs args = {
        .a_off = (uint32_t)a_off,
        .b_off = (uint32_t)b_off,
        .c_off = (uint32_t)c_off,
        .M     = M,
        .K     = K,
        .N     = N,
    };
    gpuMemcpy(ctx, args_off, &args, sizeof(args), GPU_MEMCPY_H2D);

    uint32_t total = M * N;
    uint32_t grid[3], block[3];
    calc_dispatch_1warp_per_block(total, grid, block);
    printf("[5] dispatch: %u threads, grid={%u,%u,%u} block={%u,%u,%u}\n",
           total, grid[0], grid[1], grid[2], block[0], block[1], block[2]);

    ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "launch failed: %d\n", ret);
        return 1;
    }
    printf("[6] kernel done\n");

    gpuMemcpy(ctx, c_off, hc, M * N * 4, GPU_MEMCPY_D2H);
    printf("[7] result read back\n");

    /* 参考 */
    float *ref = malloc(M * N * sizeof(float));
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++)
                sum += ha[i * K + k] * hb[k * N + j];
            ref[i * N + j] = sum;
        }
    }

    int   sentinel_hits = 0;
    int   exact_match   = 0;
    float max_err       = 0.0f;
    int   max_idx       = -1;
    for (uint32_t i = 0; i < M * N; i++) {
        if (hc[i] == -999999.0f)
            sentinel_hits++;
        if (hc[i] == ref[i])
            exact_match++;
        float err = fabsf(hc[i] - ref[i]);
        if (err > max_err) {
            max_err = err;
            max_idx = (int)i;
        }
    }
    printf("[8] exact match: %d / %u\n", exact_match, M * N);
    printf("    sentinel still: %d (should be 0)\n", sentinel_hits);
    printf("    max_err = %g at idx %d\n", max_err, max_idx);

    /* 失败诊断:打前几个 */
    if (sentinel_hits > 0 || max_err > 1e-3f) {
        printf("    FIRST 16 mismatch/sentinel:\n");
        int shown = 0;
        for (uint32_t i = 0; i < M * N && shown < 16; i++) {
            float err = fabsf(hc[i] - ref[i]);
            if (err > 1e-3f || hc[i] == -999999.0f) {
                printf("      [%u] got=%g  ref=%g  err=%g\n", i, hc[i], ref[i],
                       err);
                shown++;
            }
        }
    }

    int ok = (sentinel_hits == 0) && (max_err < 1e-3f);

    free(ha);
    free(hb);
    free(hc);
    free(ref);
    gpuDestroy(ctx);
    free(ctx);
    printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}