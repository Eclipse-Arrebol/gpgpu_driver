#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 32            /* 一个 warp，32个元素 */
#define M 4             /* 一个 warp，32个元素 */
#define K 8             /* 一个 warp，32个元素 */
#define KERNEL_SIZE 144 /* 一个 warp，32个元素 */

typedef struct {
    uint32_t a_off;
    uint32_t b_off;
    uint32_t c_off;
    uint32_t k;
    uint32_t n;
} VecAddArgs;

int main() {
    int ret;
    /* 1. 初始化 */
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
    printf("[1] gpuInit OK\n");

    uint64_t kernel_off = gpuMalloc(ctx, KERNEL_SIZE); /* kernel binary */
    uint64_t args_off   = gpuMalloc(ctx, sizeof(VecAddArgs));
    uint64_t a_off      = gpuMalloc(ctx, M * K * 4);
    uint64_t b_off      = gpuMalloc(ctx, K * N * 4);
    uint64_t c_off      = gpuMalloc(ctx, M * N * 4);

    printf("[2] alloc: kernel=0x%lx args=0x%lx A=0x%lx B=0x%lx C=0x%lx\n",
           kernel_off, args_off, a_off, b_off, c_off);

    FILE *f = fopen("/tmp/gemm.bin", "rb");
    if (!f) {
        fprintf(stderr, "open fmadd_test.bin failed\n");
        return 1;
    }

    uint8_t kbuf[KERNEL_SIZE];
    fread(kbuf, 1, KERNEL_SIZE, f);
    fclose(f);

    gpuMemcpy(ctx, kernel_off, kbuf, KERNEL_SIZE, 0); /* H2D */
    printf("[3] kernel uploaded\n");

    float ha[M * K], hb[N * K], hc[M * N];
    memset(hc, 0, sizeof(hc));

    for (int i = 0; i < M * K; i++) {
        ha[i] = i; /* A[i] = i */
    }
    for (int i = 0; i < N * K; i++) {
        hb[i] = i; /* A[i] = i */
    }
    gpuMemcpy(ctx, a_off, ha, M * K * 4, 0);
    gpuMemcpy(ctx, b_off, hb, N * K * 4, 0);
    gpuMemcpy(ctx, c_off, hc, M * N * 4, 0);
    printf("[4] input uploaded\n");

    VecAddArgs args = {.a_off = (uint32_t)a_off,
                       .b_off = (uint32_t)b_off,
                       .c_off = (uint32_t)c_off,
                       .k     = K,
                       .n     = N};
    gpuMemcpy(ctx, args_off, &args, sizeof(args), 0);
    printf("[5] args uploaded\n");

    printf("args: a_off=0x%x b_off=0x%x c_off=0x%x k=%d n=%d\n", args.a_off,
           args.b_off, args.c_off, args.k, args.n);

    /* 6. 启动 kernel */
    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {N, M, 1};
    ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return 1;
    }
    printf("[6] kernel done\n");

    gpuMemcpy(ctx, c_off, hc, M * N * 4, 1); /* D2H */
    printf("[7] result read back\n");

    float ref[M * N];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += ha[i * K + k] * hb[k * N + j];
            }
            ref[i * N + j] = sum;
        }
    }

    int ok = 1;
    for (int i = 0; i < M * N; i++) {
        if (fabsf(hc[i] - ref[i]) > 1e-3f) {
            printf("FAIL at [%d]: got %f, expected %f\n", i, hc[i], ref[i]);
            ok = 0;
        }
    }

    gpuDestroy(ctx);
    free(ctx);

    return ok ? 0 : 1;
}