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

    uint64_t kernel_off = gpuMalloc(ctx, 60); /* kernel binary */
    uint64_t args_off   = gpuMalloc(ctx, sizeof(VecAddArgs));
    uint64_t a_off      = gpuMalloc(ctx, N * 4);
    uint64_t b_off      = gpuMalloc(ctx, N * 4);
    uint64_t c_off      = gpuMalloc(ctx, N * 4);

    printf("[2] alloc: kernel=0x%lx args=0x%lx A=0x%lx B=0x%lx C=0x%lx\n",
           kernel_off, args_off, a_off, b_off, c_off);

    FILE *f = fopen("/tmp/fmadd.bin", "rb");
    if (!f) {
        fprintf(stderr, "open fmadd_test.bin failed\n");
        return 1;
    }
    uint8_t kbuf[60];
    fread(kbuf, 1, 60, f);
    fclose(f);

    gpuMemcpy(ctx, kernel_off, kbuf, 60, 0); /* H2D */
    printf("[3] kernel uploaded\n");

    float ha[N], hb[N], hc[N];
    for (int i = 0; i < N; i++) {
        ha[i] = 2.0f; /* A[i] = i */
        hb[i] = 3.0f; /* B[i] = N - i，所以 C[i] 应该全是 N=32 */
        hc[i] = 1.0f;
    }
    gpuMemcpy(ctx, a_off, ha, N * 4, 0);
    gpuMemcpy(ctx, b_off, hb, N * 4, 0);
    gpuMemcpy(ctx, c_off, hc, N * 4, 0);
    printf("[4] input uploaded\n");

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
    printf("[6] kernel done\n");

    /* DEBUG: 直接读 c_off 处的 VRAM 内容(不经过 D2H)*/
    {
        float *vram_c = (float *)((uint8_t *)ctx->vram_mmap + c_off);
        printf("[DEBUG] vram[c_off] = ");
        for (int i = 0; i < 4; i++)
            printf("%f ", vram_c[i]);
        printf("\n");

        float *vram_a = (float *)((uint8_t *)ctx->vram_mmap + a_off);
        printf("[DEBUG] vram[a_off] = ");
        for (int i = 0; i < 4; i++)
            printf("%f ", vram_a[i]);
        printf("\n");
    }

    gpuMemcpy(ctx, c_off, hc, N * 4, 1);
    printf("[7] result read back\n");

    /* 8. 验证 */
    int ok = 1;
    for (int i = 0; i < N; i++) {

        if (hc[i] != 7.0f) {
            printf("  FAIL%f \n", hc[i]);
            ok = 0;
        }
    }
    if (ok)
        printf("[8] PASS: all %d elements correct (each = %d)\n", N, N);
    else
        printf("[8] FAIL\n");

    uint32_t *vram_a = (uint32_t *)((uint8_t *)ctx->vram_mmap + a_off);
    printf("[DEBUG] a3 (expect c_off=0x%x) = 0x%x\n", (uint32_t)c_off,
           vram_a[0]);
    printf("[DEBUG] t4 (expect c_off+0)    = 0x%x\n", vram_a[1]);
    printf("[DEBUG] a0 (expect args_off)   = 0x%x\n", vram_a[2]);

    gpuDestroy(ctx);
    free(ctx);
    return ok ? 0 : 1;
}