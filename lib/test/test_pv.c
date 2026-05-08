#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define P_SIZE 1024
#define V_SIZE 4096
#define O_SIZE 4096

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

    /* 读 kernel binary */
    FILE *f = fopen("/tmp/pv.bin", "rb");
    if (!f) {
        fprintf(stderr, "open pv.bin failed\n");
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

    uint64_t args_off = gpuMalloc(ctx, sizeof(struct pv_args));
    printf("args_off   = 0x%lx  size=%zu\n", args_off, sizeof(struct pv_args));

    uint64_t P_off = gpuMalloc(ctx, P_SIZE);
    printf("out_off    = 0x%lx  size=%d\n", P_off, P_SIZE);

    uint64_t V_off = gpuMalloc(ctx, V_SIZE);
    printf("out_off    = 0x%lx  size=%d\n", V_off, V_SIZE);

    uint64_t O_off = gpuMalloc(ctx, O_SIZE);
    printf("out_off    = 0x%lx  size=%d\n", O_off, O_SIZE);

    gpuMemcpy(ctx, kernel_off, kbuf, ksize, 0);
    free(kbuf);
    printf("[3] kernel uploaded\n");

    float *V = malloc(V_SIZE);
    float *P = malloc(P_SIZE);
    float *O = malloc(O_SIZE);

    float *O_ref = malloc(O_SIZE);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            P[i * 16 + j] = (i + 1) * 0.03f + j * 0.02f;
        }
    }
    for (int j = 0; j < 16; j++) {
        for (int k = 0; k < 64; k++) {
            V[j * 64 + k] = (j + 1) * 0.05f - k * 0.001f;
        }
    }

    for (int i = 0; i < 16; i++) {
        for (int k = 0; k < 64; k++) {
            float acc = 0.0f;
            for (int j = 0; j < 16; j++) {
                acc += P[i * 16 + j] * V[j * 64 + k];
            }
            O_ref[i * 64 + k] = acc;
        }
    }

    struct pv_args args = {.P_off = (uint32_t)P_off,
                           .V_off = (uint32_t)V_off,
                           .O_off = (uint32_t)O_off};

    gpuMemcpy(ctx, args_off, &args, sizeof(args), 0);
    printf("[5] args uploaded\n");
    gpuMemcpy(ctx, P_off, P, P_SIZE, 0); // host → GPU
    gpuMemcpy(ctx, V_off, V, V_SIZE, 0);

    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return 1;
    }
    printf("[6] kernel done\n");

    gpuMemcpy(ctx, O_off, O, O_SIZE, 1);
    printf("[7] result read back\n");

    /* 检查每行 sum */

    /* 逐元素 */
    float max_err = 0.0f;
    int   max_i = -1, max_j = -1;
    int   fail = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            int   idx  = i * 16 + j;
            float diff = fabsf(O_ref[idx] - O[idx]);
            if (diff > max_err) {
                max_err = diff;
                max_i   = i;
                max_j   = j;
            }
            if (diff >= 1e-5f) {
                if (fail < 10)
                    printf("FAIL P[%d,%d] got=%f ref=%f diff=%.3e\n", i, j,
                           O[idx], O_ref[idx], diff);
                fail++;
            }
        }
    }
    printf("max_err = %.3e at (%d,%d), fail = %d\n", max_err, max_i, max_j,
           fail);

    /* Spot check */

    if (fail == 0)
        printf("PASS\n");
    else
        printf("FAIL: %d elements\n", fail);
    return 0;
}
