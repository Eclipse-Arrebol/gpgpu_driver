#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEQ_LEN 16
#define S_SIZE 1024
#define P_SIZE 1024

struct softmax16_args {
    uint32_t S_off; /* offset 0:整个 S 矩阵起始 */
    uint32_t P_off; /* offset 4:整个 P 矩阵起始 */
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
    FILE *f = fopen("/tmp/softmax16.bin", "rb");
    if (!f) {
        fprintf(stderr, "open softmax16.bin failed\n");
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

    uint64_t args_off = gpuMalloc(ctx, sizeof(struct softmax16_args));
    printf("args_off   = 0x%lx  size=%zu\n", args_off,
           sizeof(struct softmax16_args));

    uint64_t S_off = gpuMalloc(ctx, S_SIZE);
    printf("in_off     = 0x%lx  size=%d\n", S_off, S_SIZE);

    uint64_t P_off = gpuMalloc(ctx, P_SIZE);
    printf("out_off    = 0x%lx  size=%d\n", P_off, P_SIZE);

    gpuMemcpy(ctx, kernel_off, kbuf, ksize, 0);
    free(kbuf);
    printf("[3] kernel uploaded\n");

    float *S = malloc(S_SIZE);
    float *P = malloc(P_SIZE);

    float *P_ref = malloc(P_SIZE);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            S[i * 16 + j] = (i + 1) * 0.5f - j * 0.3f + (i * j) * 0.05f;
        }
    }

    for (int i = 0; i < 16; i++) {
        /* 找 max */
        float m = -INFINITY;
        for (int j = 0; j < 16; j++) {
            float v = S[i * 16 + j];
            if (v > m)
                m = v;
        }
        /* 算 exp 和 sum */
        float sum = 0.0f;
        float e[16];
        for (int j = 0; j < 16; j++) {
            e[j] = expf(S[i * 16 + j] - m);
            sum += e[j];
        }
        /* divide */
        for (int j = 0; j < 16; j++) {
            P_ref[i * 16 + j] = e[j] / sum;
        }
    }

    struct softmax16_args args = {.P_off = (uint32_t)P_off,
                                  .S_off = (uint32_t)S_off};

    gpuMemcpy(ctx, args_off, &args, sizeof(args), 0);
    printf("[5] args uploaded\n");
    gpuMemcpy(ctx, P_off, P, P_SIZE, 0); // host → GPU
    gpuMemcpy(ctx, S_off, S, S_SIZE, 0);

    uint32_t grid[3]  = {16, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return 1;
    }
    printf("[6] kernel done\n");

    gpuMemcpy(ctx, P_off, P, P_SIZE, 1);
    printf("[7] result read back\n");

    /* 检查每行 sum */
    for (int i = 0; i < 16; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 16; j++)
            sum += P[i * 16 + j];
        if (fabsf(sum - 1.0f) > 1e-4f) {
            printf("row %d sum = %f (expect 1.0)\n", i, sum);
        }
    }

    /* 逐元素 */
    float max_err = 0.0f;
    int   max_i = -1, max_j = -1;
    int   fail = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            int   idx  = i * 16 + j;
            float diff = fabsf(P[idx] - P_ref[idx]);
            if (diff > max_err) {
                max_err = diff;
                max_i   = i;
                max_j   = j;
            }
            if (diff >= 1e-5f) {
                if (fail < 10)
                    printf("FAIL P[%d,%d] got=%f ref=%f diff=%.3e\n", i, j,
                           P[idx], P_ref[idx], diff);
                fail++;
            }
        }
    }
    printf("max_err = %.3e at (%d,%d), fail = %d\n", max_err, max_i, max_j,
           fail);

    /* Spot check */
    printf("P[0,0]   = %f (ref %f)\n", P[0], P_ref[0]);
    printf("P[0,15]  = %f (ref %f)\n", P[15], P_ref[15]);
    printf("P[8,5]   = %f (ref %f)\n", P[8 * 16 + 5], P_ref[8 * 16 + 5]);
    printf("P[15,15] = %f (ref %f)\n", P[15 * 16 + 15], P_ref[15 * 16 + 15]);

    if (fail == 0)
        printf("PASS\n");
    else
        printf("FAIL: %d elements\n", fail);
    gpuDestroy(ctx);
    free(ctx);
    return 0;
}
