#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float    in;
    uint32_t out_off;

} vexp_args;

int main() {
    int        ret;
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

    /* 读 kernel binary */
    FILE *f = fopen("/tmp/vexp.bin", "rb");
    if (!f) {
        fprintf(stderr, "open vexp.bin failed\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t ksize = ftell(f);
    rewind(f);
    uint8_t *kbuf = malloc(ksize);
    fread(kbuf, 1, ksize, f);
    fclose(f);

    uint64_t kernel_off = gpuMalloc(ctx, ksize);
    printf("kernel_off = 0x%lx  ksize=%zu\n", kernel_off, ksize);

    uint64_t args_off = gpuMalloc(ctx, sizeof(vexp_args));
    printf("args_off   = 0x%lx  size=%zu\n", args_off, sizeof(vexp_args));

    uint64_t out_off = gpuMalloc(ctx, 4);
    printf("out_off    = 0x%lx  size=%d\n", out_off, 4);

    gpuMemcpy(ctx, kernel_off, kbuf, ksize, 0);
    free(kbuf);
    printf("[3] kernel uploaded\n");

    vexp_args args = {.in = 2.5f, .out_off = (uint32_t)out_off};

    gpuMemcpy(ctx, args_off, &args, sizeof(args), 0);
    printf("[5] args uploaded\n");

    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return 1;
    }
    printf("[6] kernel done\n");

    float h_out[1];
    gpuMemcpy(ctx, out_off, h_out, 4, 1);
    printf("[7] result read back\n");

    int ok = 1;
    if (fabsf(expf(2.5f) - h_out[0]) < 1e-3f) {
        printf("[8] test pass\n");
    } else {
        printf("[8] test fail reslut : %f\n", h_out[0]);
        ok = 0;
    }

    gpuDestroy(ctx);
    free(ctx);
    return ok ? 0 : 1;
}