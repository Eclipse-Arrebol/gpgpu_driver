#include "../libgpgpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 32 /* 一个 warp，32个元素 */

/* args 结构体，和 kernel 汇编里的 lw 偏移一一对应 */
typedef struct {
    uint32_t a_off;
    uint32_t b_off;
    uint32_t c_off;
} VecAddArgs;

int main(void) {
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

    /* 2. 分配 VRAM：kernel + args + A + B + C */
    uint64_t kernel_off = gpuMalloc(ctx, 56); /* kernel binary */
    uint64_t args_off   = gpuMalloc(ctx, sizeof(VecAddArgs));
    uint64_t a_off      = gpuMalloc(ctx, N * 4);
    uint64_t b_off      = gpuMalloc(ctx, N * 4);
    uint64_t c_off      = gpuMalloc(ctx, N * 4);

    printf("[2] alloc: kernel=0x%lx args=0x%lx A=0x%lx B=0x%lx C=0x%lx\n",
           kernel_off, args_off, a_off, b_off, c_off);

    /* 3. 上传 kernel binary */
    FILE *f = fopen("/tmp/vec_add.bin", "rb");
    if (!f) {
        fprintf(stderr, "open vec_add.bin failed\n");
        return 1;
    }
    uint8_t kbuf[56];
    fread(kbuf, 1, 56, f);
    fclose(f);

    gpuMemcpy(ctx, kernel_off, kbuf, 56, 0); /* H2D */
    printf("[3] kernel uploaded\n");

    /* 4. 准备输入数据 A B，上传 */
    int32_t ha[N], hb[N], hc[N];
    for (int i = 0; i < N; i++) {
        ha[i] = i;     /* A[i] = i */
        hb[i] = N - i; /* B[i] = N - i，所以 C[i] 应该全是 N=32 */
    }
    gpuMemcpy(ctx, a_off, ha, N * 4, 0);
    gpuMemcpy(ctx, b_off, hb, N * 4, 0);
    printf("[4] input uploaded\n");

    /* 5. 填写 args 并上传 */
    VecAddArgs args = {
        .a_off = (uint32_t)a_off,
        .b_off = (uint32_t)b_off,
        .c_off = (uint32_t)c_off,
    };
    gpuMemcpy(ctx, args_off, &args, sizeof(args), 0);
    printf("[5] args uploaded\n");

    /* 6. 启动 kernel */
    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {N, 1, 1};
    ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return 1;
    }
    printf("[6] kernel done\n");

    /* 7. 读回结果 */
    gpuMemcpy(ctx, c_off, hc, N * 4, 1); /* D2H */
    printf("[7] result read back\n");

    /* 8. 验证 */
    int ok = 1;
    for (int i = 0; i < N; i++) {
        int32_t expected = ha[i] + hb[i];
        if (hc[i] != expected) {
            printf("  FAIL at [%d]: got %d, expected %d\n", i, hc[i], expected);
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
