#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D 896

typedef struct {
    uint32_t in_off;
    uint32_t out_off;
    uint32_t weight_off;
    uint32_t d;
    float    inv_d;
    float    eps;
} rmsnorm_args;

void rmsnorm_cpu(float *out, const float *in, const float *weight, int d,
                 float eps) {
    float sumsq = 0.0f;
    for (int i = 0; i < d; i++)
        sumsq += in[i] * in[i];
    float rms  = sqrtf(sumsq / d + eps);
    float rrms = 1.0f / rms;
    for (int i = 0; i < d; i++)
        out[i] = in[i] * rrms * weight[i];
}

int main(void) {
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
    FILE *f = fopen("/tmp/rmsnorm.bin", "rb");
    if (!f) {
        fprintf(stderr, "open rmsnorm.bin failed\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t ksize = ftell(f);
    rewind(f);
    uint8_t *kbuf = malloc(ksize);
    fread(kbuf, 1, ksize, f);
    fclose(f);

    /* VRAM 分配 */
    uint64_t kernel_off = gpuMalloc(ctx, ksize);
    printf("kernel_off = 0x%lx  ksize=%zu\n", kernel_off, ksize);

    uint64_t args_off = gpuMalloc(ctx, sizeof(rmsnorm_args));
    printf("args_off   = 0x%lx  size=%zu\n", args_off, sizeof(rmsnorm_args));

    uint64_t in_off = gpuMalloc(ctx, D * 4);
    printf("in_off     = 0x%lx  size=%d\n", in_off, D * 4);

    uint64_t out_off = gpuMalloc(ctx, D * 4);
    printf("out_off    = 0x%lx  size=%d\n", out_off, D * 4);

    uint64_t weight_off = gpuMalloc(ctx, D * 4);
    printf("weight_off = 0x%lx  size=%d\n", weight_off, D * 4);
    printf("[2] alloc OK\n");

    gpuMemcpy(ctx, kernel_off, kbuf, ksize, 0);
    free(kbuf);
    printf("[3] kernel uploaded\n");

    /* 准备数据:输入 + weight */
    float h_in[D], h_weight[D];
    for (int i = 0; i < D; i++) {
        h_in[i]     = 1.0f; /* 全 1,期望 rms ≈ 1.0 */
        h_weight[i] = 2.0f; /* 全 2,期望 output ≈ 2.0 */
    }
    gpuMemcpy(ctx, in_off, h_in, D * 4, 0);
    gpuMemcpy(ctx, weight_off, h_weight, D * 4, 0);
    printf("[4] input + weight uploaded\n");

    rmsnorm_args args = {
        .in_off     = (uint32_t)in_off,
        .out_off    = (uint32_t)out_off,
        .weight_off = (uint32_t)weight_off,
        .d          = D,
        .inv_d      = 1.0f / D,
        .eps        = 1e-6f,
    };
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

    /* 读结果 */
    float h_out[D];
    gpuMemcpy(ctx, out_off, h_out, D * 4, 1);
    printf("[7] result read back\n");

    /* CPU 参考 */
    float h_ref[D];
    rmsnorm_cpu(h_ref, h_in, h_weight, D, 1e-6f);

    /* 比对 */
    int   ok = 1, fail_count = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < D; i++) {
        float diff = fabsf(h_out[i] - h_ref[i]);
        if (diff > max_diff)
            max_diff = diff;
        if (diff >= 1e-3f) {
            if (fail_count < 5) {
                printf("  FAIL i=%d  expected=%.6f  got=%.6f  diff=%.6f\n", i,
                       h_ref[i], h_out[i], diff);
            }
            fail_count++;
            ok = 0;
        }
    }
    printf("[8] %s  max_diff=%.6f  fails=%d/%d\n", ok ? "PASS" : "FAIL",
           max_diff, fail_count, D);

    gpuDestroy(ctx);
    free(ctx);
    return ok ? 0 : 1;
}