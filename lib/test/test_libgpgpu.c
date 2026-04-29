#include "../libgpgpu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VRAM_SIZE (64 * 1024 * 1024) // 64MB

int main(void) {
    gpgpu_ctx *ctx = malloc(sizeof(gpgpu_ctx));
    if (!ctx) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }
    int ret;

    printf("step 1: gpuInit\n");
    fflush(stdout);
    ret = gpuInit(ctx, "/dev/gpgpu", VRAM_SIZE);
    if (ret < 0) {
        fprintf(stderr, "FAIL: gpuInit\n");
        return 1;
    }
    printf("PASS: gpuInit\n");
    fflush(stdout);

    printf("step 2: gpuMalloc 32B\n");
    fflush(stdout);
    uint64_t off_32 = gpuMalloc(ctx, 32);
    if (off_32 == (uint64_t)-1) {
        fprintf(stderr, "FAIL: gpuMalloc 32B\n");
        return 1;
    }
    printf("PASS: gpuMalloc 32B → offset=0x%lx\n", off_32);
    fflush(stdout);

    printf("step 3: gpuMalloc 2048B\n");
    fflush(stdout);
    uint64_t off_2048 = gpuMalloc(ctx, 2048);
    if (off_2048 == (uint64_t)-1) {
        fprintf(stderr, "FAIL: gpuMalloc 2048B\n");
        return 1;
    }
    printf("PASS: gpuMalloc 2048B → offset=0x%lx\n", off_2048);
    fflush(stdout);

    printf("step 4: gpuMalloc 8KB\n");
    fflush(stdout);
    uint64_t off_8k = gpuMalloc(ctx, 8192);
    if (off_8k == (uint64_t)-1) {
        fprintf(stderr, "FAIL: gpuMalloc 8KB\n");
        return 1;
    }
    printf("PASS: gpuMalloc 8KB → offset=0x%lx\n", off_8k);
    fflush(stdout);

    printf("step 5: gpuMemcpy H2D\n");
    fflush(stdout);
    char src[64] = "hello gpgpu";
    char dst[64] = {0};
    ret          = gpuMemcpy(ctx, off_8k, src, sizeof(src), GPU_MEMCPY_H2D);
    if (ret < 0) {
        fprintf(stderr, "FAIL: gpuMemcpy H2D\n");
        return 1;
    }
    printf("PASS: gpuMemcpy H2D\n");
    fflush(stdout);

    printf("step 6: gpuMemcpy D2H\n");
    fflush(stdout);
    ret = gpuMemcpy(ctx, off_8k, dst, sizeof(dst), GPU_MEMCPY_D2H);
    if (ret < 0) {
        fprintf(stderr, "FAIL: gpuMemcpy D2H\n");
        return 1;
    }
    printf("PASS: gpuMemcpy D2H data=\"%s\"\n", dst);
    fflush(stdout);

    printf("step 7: gpuFree\n");
    fflush(stdout);
    gpuFree(ctx, off_32);
    gpuFree(ctx, off_2048);
    gpuFree(ctx, off_8k);
    printf("PASS: gpuFree\n");
    fflush(stdout);

    gpuDestroy(ctx);
    printf("PASS: gpuDestroy\n");
    return 0;
}