#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D 896

struct silu_args {
    uint32_t in_off;
    uint32_t out_off;
    uint32_t d;
};

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
    FILE *f = fopen("/tmp/silu.bin", "rb");
    if (!f) {
        fprintf(stderr, "open silu.bin failed\n");
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

    uint64_t args_off = gpuMalloc(ctx, sizeof(struct silu_args));
    printf("args_off   = 0x%lx  size=%zu\n", args_off,
           sizeof(struct silu_args));

    uint64_t in_off = gpuMalloc(ctx, D * 4);
    printf("in_off     = 0x%lx  size=%d\n", in_off, D * 4);

    uint64_t out_off = gpuMalloc(ctx, D * 4);
    printf("out_off    = 0x%lx  size=%d\n", out_off, D * 4);

    printf("[2] alloc OK\n");

    gpuMemcpy(ctx, kernel_off, kbuf, ksize, 0);
    free(kbuf);
    printf("[3] kernel uploaded\n");

    float *h_in  = malloc(D * sizeof(float));
    float *h_out = malloc(D * sizeof(float));
    float *h_ref = malloc(D * sizeof(float));
    for (int i = 0; i < D; i++) {
        h_in[i] = (float)(i - 448) / 50.0f; /* 范围 [-8.96, 8.94] */
    }
    /* host reference */
    for (int i = 0; i < D; i++) {
        float x  = h_in[i];
        h_ref[i] = x / (1.0f + expf(-x));
    }

    struct silu_args args = {
        .in_off  = (uint32_t)in_off,
        .out_off = (uint32_t)out_off,
        .d       = D,
    };

    gpuMemcpy(ctx, args_off, &args, sizeof(args), 0);
    printf("[5] args uploaded\n");
    gpuMemcpy(ctx, in_off, h_in, D * 4, 0); // host → GPU

    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return 1;
    }
    printf("[6] kernel done\n");

    gpuMemcpy(ctx, out_off, h_out, D * 4, 1);
    printf("[7] result read back\n");

    int ok = 1;
    for (int i = 0; i < D; i++) {
        if (fabsf(h_out[i] - h_ref[i]) >= 1e-5f) {
            printf("result failed is:%f\n", h_out[i]);
            ok = 0;
        }
    }
    /* 抽样打印 */

    printf("out[0]   = %.10f  ref[0]   = %.10f\n", h_out[0], h_ref[0]);
    printf("out[1]   = %.10f  ref[1]   = %.10f\n", h_out[1], h_ref[1]);
    printf("out[895] = %.10f  ref[895] = %.10f\n", h_out[895], h_ref[895]);

    /* sum check — softmax 输出和应该 = 1 */
    float sum_check = 0;
    for (int i = 0; i < D; i++)
        sum_check += h_out[i];
    printf("sum(out) = %.7f  (expect 1.0)\n", sum_check);

    /* max error */
    float max_err = 0;
    for (int i = 0; i < D; i++) {
        float err = fabsf(h_out[i] - h_ref[i]);
        if (err > max_err)
            max_err = err;
    }
    printf("max_err = %.2e\n", max_err);
    if (ok)
        printf("test pass\n");
    gpuDestroy(ctx);
    free(ctx);

    return ok ? 0 : 1;
}
