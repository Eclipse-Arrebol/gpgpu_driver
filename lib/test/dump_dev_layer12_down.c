#include "../include/weight_loader.h"
#include "../libgpgpu.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define VRAM_SIZE (4ULL * 1024 * 1024 * 1024)
#define N 16
#define I 4864
#define D 896

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <weights_dir>\n", argv[0]);
        return 1;
    }

    gpgpu_ctx *ctx = malloc(sizeof(gpgpu_ctx));
    if (gpuInit(ctx, "/dev/gpgpu", VRAM_SIZE) < 0) {
        fprintf(stderr, "gpuInit failed\n");
        return 1;
    }

    weight_loader_t *wl = weight_loader_load(ctx, argv[1]);
    if (!wl) {
        fprintf(stderr, "weight_loader_load failed\n");
        return 1;
    }

    weight_loader_print_layout(wl);

    float buf[N];

    for (int layer = 11; layer <= 12; layer++) {
        uint64_t off = weight_loader_layer(wl, layer, WL_DOWN_PROJ_W);
        printf("\ndevice layer %d down_proj offset: 0x%llx\n",
               layer, (unsigned long long)off);

        if (gpuMemcpy(ctx, off, buf, sizeof(buf), GPU_MEMCPY_D2H) < 0) {
            fprintf(stderr, "D2H failed for layer %d\n", layer);
            return 1;
        }

        printf("device layer %d down_proj first %d fp32:\n", layer, N);
        for (int i = 0; i < N; i++)
            printf("  [%2d] = %.8e\n", i, buf[i]);
    }

    {
        size_t   bytes = (size_t)I * D * sizeof(float);
        uint64_t off   = weight_loader_layer(wl, 12, WL_DOWN_PROJ_W);
        void    *whole = malloc(bytes);
        if (!whole) {
            fprintf(stderr, "malloc %zu failed\n", bytes);
            return 1;
        }
        if (gpuMemcpy(ctx, off, whole, bytes, GPU_MEMCPY_D2H) < 0) {
            fprintf(stderr, "D2H full layer 12 down_proj failed\n");
            free(whole);
            return 1;
        }
        FILE *fp = fopen("/root/layer12_down_after_load.bin", "wb");
        if (!fp) {
            perror("fopen /root/layer12_down_after_load.bin");
            free(whole);
            return 1;
        }
        fwrite(whole, 1, bytes, fp);
        fclose(fp);
        free(whole);
        printf("\nwrote /root/layer12_down_after_load.bin (%zu bytes)\n", bytes);
    }

    weight_loader_destroy(wl);
    gpuDestroy(ctx);
    free(ctx);
    return 0;
}
