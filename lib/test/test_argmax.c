#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define V_NUM 32
#define LOGIC_SIZE V_NUM * 4

struct pv_args {
    uint32_t logits_base;
    uint32_t V;
    uint32_t out_base;
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
    FILE *f = fopen("/tmp/argmax.bin", "rb");
    if (!f) {
        fprintf(stderr, "open argmax.bin failed\n");
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

    uint64_t L_off = gpuMalloc(ctx, LOGIC_SIZE);
    printf("out_off    = 0x%lx  size=%d\n", L_off, LOGIC_SIZE);

    uint64_t O_off = gpuMalloc(ctx, 4);
    printf("out_off    = 0x%lx  size=%d\n", O_off, 4);

    gpuMemcpy(ctx, kernel_off, kbuf, ksize, 0);
    free(kbuf);
    printf("[3] kernel uploaded\n");

    float    *L = malloc(LOGIC_SIZE);
    uint32_t *O = malloc(4);
    *O          = 0xDEADBEEF;

    uint32_t *O_ref = malloc(4);
    O_ref[0]        = 0;

    for (int i = 0; i < V_NUM; i++) {
        uint32_t neg_inf = 0xFF800000;
        memcpy(&L[i], &neg_inf, 4);
    }

    for (int i = 0; i < V_NUM; i++) {
        if (L[i] > L[O_ref[0]]) {
            O_ref[0] = i;
        }
    }

    struct pv_args args = {.logits_base = (uint32_t)L_off,
                           .V           = V_NUM,
                           .out_base    = (uint32_t)O_off};

    gpuMemcpy(ctx, args_off, &args, sizeof(args), 0);
    printf("[5] args uploaded\n");
    gpuMemcpy(ctx, L_off, L, LOGIC_SIZE, 0); // host → GPU

    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return 1;
    }
    printf("[6] kernel done\n");

    gpuMemcpy(ctx, O_off, O, 4, 1);

    printf("[7] result read back\n");

    /* 检查每行 sum */

    printf("device_idx=%u  ref_idx=%u  ", O[0], O_ref[0]);
    if (O[0] == O_ref[0])
        printf("PASS\n");
    else
        printf("FAIL\n");

    /* 逐元素 */
    gpuDestroy(ctx);
    free(ctx);
    return 0;
}
