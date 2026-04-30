#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 32 /* 一个 warp，32个元素 */
#define SHUFFLE_SIZE 80

typedef struct {
    uint32_t in_off;  // 输入数组 VRAM offset（32 个 float）
    uint32_t out_off; // 输出数组 VRAM offset（32 个 float）
} ShflArgs;

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

    uint64_t kernel_off = gpuMalloc(ctx, SHUFFLE_SIZE); /* kernel binary */
    uint64_t args_off   = gpuMalloc(ctx, sizeof(ShflArgs));
    uint64_t in_off     = gpuMalloc(ctx, N * 4);
    uint64_t out_off    = gpuMalloc(ctx, N * 4);

    printf("[2] alloc: kernel=0x%lx args=0x%lx A=0x%lx B=0x%lx \n", kernel_off,
           args_off, in_off, out_off);

    FILE *f = fopen("/tmp/shuffle.bin", "rb");
    if (!f) {
        fprintf(stderr, "open fmadd_test.bin failed\n");
        return 1;
    }
    uint8_t kbuf[SHUFFLE_SIZE];
    fread(kbuf, 1, SHUFFLE_SIZE, f);
    fclose(f);

    gpuMemcpy(ctx, kernel_off, kbuf, SHUFFLE_SIZE, 0); /* H2D */
    printf("[3] kernel uploaded\n");

    float ha[N], hb[N];

    for (int i = 0; i < N; i++) {
        ha[i] = i + 1.0f; /* A[i] = i */
        hb[i] = 0;        /* B[i] = N - i，所以 C[i] 应该全是 N=32 */
    }

    gpuMemcpy(ctx, in_off, ha, N * 4, 0);
    gpuMemcpy(ctx, out_off, hb, N * 4, 0);
    printf("[4] input uploaded\n");

    ShflArgs args = {
        .in_off  = (uint32_t)in_off,
        .out_off = (uint32_t)out_off,
    };
    gpuMemcpy(ctx, args_off, &args, sizeof(args), 0);
    printf("[5] args uploaded\n");

    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {N, 1, 1};
    ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return 1;
    }
    printf("[6] kernel done\n");

    gpuMemcpy(ctx, in_off, ha, N * 4, 1);  /* D2H */
    gpuMemcpy(ctx, out_off, hb, N * 4, 1); /* D2H */
    printf("[7] result read back\n");

    int ok = 1;
    for (int i = 0; i < N; i++) {
        if (fabsf(hb[i] - 528.0f) >= 1e-3f) {
            printf("  FAIL i=%d  expected=528.0  got=%.4f\n", i, hb[i]);
            ok = 0;
        }
    }

    if (ok)
        printf("[8] PASS: all %d elements correct (each = %d)\n", N, N);
    else
        printf("[8] FAIL\n");

    gpuDestroy(ctx);
    free(ctx);
    return ok ? 0 : 1;
}