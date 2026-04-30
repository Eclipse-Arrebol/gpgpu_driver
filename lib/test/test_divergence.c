#include "../libgpgpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 32 /* 一个 warp，32个元素 */
#define DIVER_SIZE 68

/* args 结构体，和 kernel 汇编里的 lw 偏移一一对应 */
typedef struct {
    uint32_t a_off;
    uint32_t b_off;
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

    uint64_t kernel_off = gpuMalloc(ctx, DIVER_SIZE); /* kernel binary */
    uint64_t args_off   = gpuMalloc(ctx, sizeof(VecAddArgs));
    uint64_t a_off      = gpuMalloc(ctx, N * 4);
    uint64_t b_off      = gpuMalloc(ctx, N * 4);

    printf("[2] alloc: kernel=0x%lx args=0x%lx A=0x%lx B=0x%lx \n", kernel_off,
           args_off, a_off, b_off);

    FILE *f = fopen("/tmp/divergence.bin", "rb");
    if (!f) {
        fprintf(stderr, "open fmadd_test.bin failed\n");
        return 1;
    }
    uint8_t kbuf[DIVER_SIZE];
    fread(kbuf, 1, DIVER_SIZE, f);
    fclose(f);

    gpuMemcpy(ctx, kernel_off, kbuf, DIVER_SIZE, 0); /* H2D */
    printf("[3] kernel uploaded\n");

    uint32_t ha[N], hb[N];

    for (int i = 0; i < N; i++) {
        ha[i] = 0; /* A[i] = i */
        hb[i] = 0; /* B[i] = N - i，所以 C[i] 应该全是 N=32 */
    }
    gpuMemcpy(ctx, a_off, ha, N * 4, 0);
    gpuMemcpy(ctx, b_off, hb, N * 4, 0);
    printf("[4] input uploaded\n");

    VecAddArgs args = {
        .a_off = (uint32_t)a_off,
        .b_off = (uint32_t)b_off,
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

    gpuMemcpy(ctx, a_off, ha, N * 4, 1); /* D2H */
    gpuMemcpy(ctx, b_off, hb, N * 4, 1); /* D2H */
    printf("[7] result read back\n");

    int ok = 1;
    for (int i = 0; i < N; i++) {
        if (i % 2) {
            // 奇数 lane 应该：hb[i] == 0xBBBB 且 ha[i] == 0
            if (hb[i] != 0xBBBB || ha[i] != 0) {
                printf("  FAIL i=%d ha=0x%x hb=0x%x\n", i, ha[i], hb[i]);
                ok = 0;
            }
        } else {
            // 偶数 lane 应该：ha[i] == 0xAAAA 且 hb[i] == 0
            if (ha[i] != 0xAAAA || hb[i] != 0) {
                printf("  FAIL i=%d ha=0x%x hb=0x%x\n", i, ha[i], hb[i]);
                ok = 0;
            }
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