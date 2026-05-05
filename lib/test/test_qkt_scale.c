#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEQ_LEN 16
#define HEAD_DIM 64
#define I_SIZE 4096
#define S_SIZE 1024

struct qkt_scale_args {
    uint32_t Q_base;     /* offset 0 */
    uint32_t K_base;     /* offset 4 */
    uint32_t S_base;     /* offset 8 */
    uint32_t scale_bits; /* offset 12 */
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
    FILE *f = fopen("/tmp/qkt_scale.bin", "rb");
    if (!f) {
        fprintf(stderr, "open qkt_scale.bin failed\n");
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

    uint64_t args_off = gpuMalloc(ctx, sizeof(struct qkt_scale_args));
    printf("args_off   = 0x%lx  size=%zu\n", args_off,
           sizeof(struct qkt_scale_args));

    uint64_t Q_off = gpuMalloc(ctx, I_SIZE);
    printf("in_off     = 0x%lx  size=%d\n", Q_off, I_SIZE);

    uint64_t K_off = gpuMalloc(ctx, I_SIZE);
    printf("out_off    = 0x%lx  size=%d\n", K_off, I_SIZE);

    uint64_t S_off = gpuMalloc(ctx, S_SIZE);
    printf("out_off    = 0x%lx  size=%d\n", S_off, S_SIZE);
    printf("[2] alloc OK\n");

    gpuMemcpy(ctx, kernel_off, kbuf, ksize, 0);
    free(kbuf);
    printf("[3] kernel uploaded\n");

    float *Q     = malloc(I_SIZE);
    float *K     = malloc(I_SIZE);
    float *S     = malloc(S_SIZE);
    float *S_ref = malloc(S_SIZE);
    float  scale = 1.0f / sqrtf(64.0f); /* = 0.125 */

    for (int i = 0; i < 16; i++) {
        for (int k = 0; k < 64; k++) {
            Q[i * 64 + k] = (i + 1) * 0.01f + k * 0.001f;
        }
    }
    for (int j = 0; j < 16; j++) {
        for (int k = 0; k < 64; k++) {
            K[j * 64 + k] = (j + 1) * 0.02f - k * 0.0005f;
        }
    }
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            float acc = 0.0f;
            for (int k = 0; k < 64; k++) {
                acc += Q[i * 64 + k] * K[j * 64 + k];
            }
            S_ref[i * 16 + j] = acc * scale;
        }
    }

    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, 4);

    struct qkt_scale_args args = {.Q_base     = (uint32_t)Q_off,
                                  .K_base     = (uint32_t)K_off,
                                  .S_base     = (uint32_t)S_off,
                                  .scale_bits = scale_bits};

    gpuMemcpy(ctx, args_off, &args, sizeof(args), 0);
    printf("[5] args uploaded\n");
    gpuMemcpy(ctx, Q_off, Q, I_SIZE, 0); // host → GPU
    gpuMemcpy(ctx, K_off, K, I_SIZE, 0);

    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return 1;
    }
    printf("[6] kernel done\n");

    gpuMemcpy(ctx, S_off, S, S_SIZE, 1);
    printf("[7] result read back\n");

    int ok = 1;
    for (int i = 0; i < 256; i++) {
        if (fabsf(S_ref[i] - S[i]) >= 1e-5f) {
            printf("result failed is:%f\n", S[i]);
            ok = 0;
        }
    }
    /* 抽样打印 */
    printf("S[0,0]   = %f  (ref %f)\n", S[0], S_ref[0]);
    printf("S[5,3]   = %f  (ref %f)\n", S[5 * 16 + 3], S_ref[5 * 16 + 3]);
    printf("S[8,8]   = %f  (ref %f)\n", S[8 * 16 + 8], S_ref[8 * 16 + 8]);
    printf("S[15,15] = %f  (ref %f)\n", S[15 * 16 + 15], S_ref[15 * 16 + 15]);

    if (ok)
        printf("test pass\n");
    gpuDestroy(ctx);
    free(ctx);

    return ok ? 0 : 1;
}
