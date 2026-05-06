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

#define P_SIZE 1024
#define V_SIZE 4096
#define O_SIZE 4096

struct qkt_scale_args {
    uint32_t Q_base;     /* offset 0 */
    uint32_t K_base;     /* offset 4 */
    uint32_t S_base;     /* offset 8 */
    uint32_t scale_bits; /* offset 12 */
};

struct softmax16_args {
    uint32_t S_off; /* offset 0:整个 S 矩阵起始 */
    uint32_t P_off; /* offset 4:整个 P 矩阵起始 */
};

struct pv_args {
    uint32_t P_off;
    uint32_t V_off;
    uint32_t O_off;
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

    FILE *f1 = fopen("/tmp/qkt_scale.bin", "rb");
    if (!f1) {
        fprintf(stderr, "open qkt_scale.bin failed\n");
        return 1;
    }
    fseek(f1, 0, SEEK_END);
    size_t ksize1 = ftell(f1);
    rewind(f1);
    uint8_t *kbuf1 = malloc(ksize1);
    fread(kbuf1, 1, ksize1, f1);
    fclose(f1);

    printf("[2] get qkt_scale kernel OK\n");
    FILE *f2 = fopen("/tmp/softmax16.bin", "rb");
    if (!f2) {
        fprintf(stderr, "open softmax16.bin failed\n");
        return 1;
    }
    fseek(f2, 0, SEEK_END);
    size_t ksize2 = ftell(f2);
    rewind(f2);
    uint8_t *kbuf2 = malloc(ksize2);
    fread(kbuf2, 1, ksize2, f2);
    fclose(f2);
    printf("[2] get softmax16 kernel OK\n");

    FILE *f3 = fopen("/tmp/pv.bin", "rb");
    if (!f3) {
        fprintf(stderr, "open pv.bin failed\n");
        return 1;
    }
    fseek(f3, 0, SEEK_END);
    size_t ksize3 = ftell(f3);
    rewind(f3);
    uint8_t *kbuf3 = malloc(ksize3);
    fread(kbuf3, 1, ksize3, f3);
    fclose(f3);
    printf("[2] get pv kernel OK\n");

    uint64_t kernel_off1 = gpuMalloc(ctx, ksize1);
    printf("kernel_off = 0x%lx  ksize=%zu\n", kernel_off1, ksize1);
    uint64_t kernel_off2 = gpuMalloc(ctx, ksize2);
    printf("kernel_off = 0x%lx  ksize=%zu\n", kernel_off2, ksize2);
    uint64_t kernel_off3 = gpuMalloc(ctx, ksize3);
    printf("kernel_off = 0x%lx  ksize=%zu\n", kernel_off3, ksize3);

    uint64_t args_off1 = gpuMalloc(ctx, sizeof(struct qkt_scale_args));
    printf("args_off   = 0x%lx  size=%zu\n", args_off1,
           sizeof(struct qkt_scale_args));

    uint64_t args_off2 = gpuMalloc(ctx, sizeof(struct softmax16_args));
    printf("args_off   = 0x%lx  size=%zu\n", args_off2,
           sizeof(struct softmax16_args));

    uint64_t args_off3 = gpuMalloc(ctx, sizeof(struct pv_args));
    printf("args_off   = 0x%lx  size=%zu\n", args_off3, sizeof(struct pv_args));

    uint64_t Q_off = gpuMalloc(ctx, I_SIZE);
    printf("in_off     = 0x%lx  size=%d\n", Q_off, I_SIZE);

    uint64_t K_off = gpuMalloc(ctx, I_SIZE);
    printf("out_off    = 0x%lx  size=%d\n", K_off, I_SIZE);

    uint64_t S_off = gpuMalloc(ctx, S_SIZE);
    printf("out_off    = 0x%lx  size=%d\n", S_off, S_SIZE);

    uint64_t P_off = gpuMalloc(ctx, P_SIZE);
    printf("out_off    = 0x%lx  size=%d\n", P_off, P_SIZE);

    uint64_t V_off = gpuMalloc(ctx, V_SIZE);
    printf("out_off    = 0x%lx  size=%d\n", V_off, V_SIZE);

    uint64_t O_off = gpuMalloc(ctx, O_SIZE);
    printf("out_off    = 0x%lx  size=%d\n", O_off, O_SIZE);

    printf("[2] alloc OK\n");

    gpuMemcpy(ctx, kernel_off1, kbuf1, ksize1, 0);
    free(kbuf1);

    gpuMemcpy(ctx, kernel_off2, kbuf2, ksize2, 0);
    free(kbuf2);

    gpuMemcpy(ctx, kernel_off3, kbuf3, ksize3, 0);
    free(kbuf3);
    printf("[3] kernel uploaded\n");

    float *Q = malloc(I_SIZE);
    float *K = malloc(I_SIZE);
    float *S = malloc(S_SIZE);

    float *V     = malloc(V_SIZE);
    float *P     = malloc(P_SIZE);
    float *O     = malloc(O_SIZE);
    float  scale = 1.0f / sqrtf(64.0f); /* = 0.125 */

    /* 输入打破对称性 */
    for (int i = 0; i < 16; i++)
        for (int k = 0; k < 64; k++)
            Q[i * 64 + k] = (i + 1) * 0.01f + k * 0.001f;

    for (int j = 0; j < 16; j++)
        for (int k = 0; k < 64; k++)
            K[j * 64 + k] = (j + 1) * 0.02f - k * 0.0005f;

    for (int j = 0; j < 16; j++)
        for (int k = 0; k < 64; k++)
            V[j * 64 + k] = (j + 1) * 0.03f + k * 0.0008f;

    float S_ref[256], P_ref[256], O_ref[1024];

    /* Step 1: S = Q @ K^T * scale */
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++) {
            float acc = 0.0f;
            for (int k = 0; k < 64; k++)
                acc += Q[i * 64 + k] * K[j * 64 + k];
            S_ref[i * 16 + j] = acc * scale;
        }

    /* Step 2: P = softmax(S) (per-row) */
    for (int i = 0; i < 16; i++) {
        float m = -INFINITY;
        for (int j = 0; j < 16; j++)
            if (S_ref[i * 16 + j] > m)
                m = S_ref[i * 16 + j];
        float sum = 0.0f;
        for (int j = 0; j < 16; j++) {
            P_ref[i * 16 + j] = expf(S_ref[i * 16 + j] - m);
            sum += P_ref[i * 16 + j];
        }
        for (int j = 0; j < 16; j++)
            P_ref[i * 16 + j] /= sum;
    }

    /* Step 3: O = P @ V */
    for (int i = 0; i < 16; i++)
        for (int k = 0; k < 64; k++) {
            float acc = 0.0f;
            for (int j = 0; j < 16; j++)
                acc += P_ref[i * 16 + j] * V[j * 64 + k];
            O_ref[i * 64 + k] = acc;
        }
    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, 4);

    struct qkt_scale_args args1 = {.Q_base     = (uint32_t)Q_off,
                                   .K_base     = (uint32_t)K_off,
                                   .S_base     = (uint32_t)S_off,
                                   .scale_bits = scale_bits};
    struct softmax16_args args2 = {.P_off = (uint32_t)P_off,
                                   .S_off = (uint32_t)S_off};
    struct pv_args        args3 = {.P_off = (uint32_t)P_off,
                                   .V_off = (uint32_t)V_off,
                                   .O_off = (uint32_t)O_off};

    gpuMemcpy(ctx, args_off1, &args1, sizeof(args1), 0);
    gpuMemcpy(ctx, args_off2, &args2, sizeof(args2), 0);
    gpuMemcpy(ctx, args_off3, &args3, sizeof(args3), 0);
    printf("[5] args uploaded\n");

    gpuMemcpy(ctx, Q_off, Q, I_SIZE, 0); // host → GPU
    gpuMemcpy(ctx, K_off, K, I_SIZE, 0);
    gpuMemcpy(ctx, V_off, V, V_SIZE, 0);

    uint32_t grid1[3]  = {1, 1, 1};
    uint32_t block1[3] = {32, 1, 1};
    ret = gpuLaunchKernel(ctx, kernel_off1, args_off1, grid1, block1, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return 1;
    }
    printf("[6] kernel1 done\n");
    uint32_t grid2[3]  = {16, 1, 1};
    uint32_t block2[3] = {32, 1, 1};
    ret = gpuLaunchKernel(ctx, kernel_off2, args_off2, grid2, block2, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return 1;
    }
    printf("[6] kernel2 done\n");

    uint32_t grid3[3]  = {1, 1, 1};
    uint32_t block3[3] = {32, 1, 1};
    ret = gpuLaunchKernel(ctx, kernel_off3, args_off3, grid3, block3, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return 1;
    }
    printf("[6] kernel3 done\n");

    gpuMemcpy(ctx, O_off, O, O_SIZE, 1);
    printf("[7] result read back\n");

    float max_err = 0.0f;
    int   max_i = -1, max_k = -1;
    int   fail = 0;
    for (int i = 0; i < 16; i++)
        for (int k = 0; k < 64; k++) {
            int   idx  = i * 64 + k;
            float diff = fabsf(O_ref[idx] - O[idx]);
            if (diff > max_err) {
                max_err = diff;
                max_i   = i;
                max_k   = k;
            }
            if (diff >= 1e-4f) {
                if (fail < 10)
                    printf("FAIL O[%d,%d] got=%f ref=%f diff=%.3e\n", i, k,
                           O[idx], O_ref[idx], diff);
                fail++;
            }
        }

    printf("max_err = %.3e at (%d,%d), fail = %d\n", max_err, max_i, max_k,
           fail);
    printf("O[0,0]   = %f (ref %f)\n", O[0], O_ref[0]);
    printf("O[0,63]  = %f (ref %f)\n", O[63], O_ref[63]);
    printf("O[8,32]  = %f (ref %f)\n", O[8 * 64 + 32], O_ref[8 * 64 + 32]);
    printf("O[15,63] = %f (ref %f)\n", O[15 * 64 + 63], O_ref[15 * 64 + 63]);

    gpuDestroy(ctx);
    free(ctx);
    return 0;
}