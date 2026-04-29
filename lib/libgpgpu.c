#include "libgpgpu.h"
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

// ── 内部辅助：根据 object_size 找到对应 slab_cache ───
static struct slab_cache *get_slab(gpgpu_ctx *ctx, uint8_t marker) {
    switch (marker) {
    case PAGE_ORDER_SLAB_32:
        return &ctx->slab_32;
    case PAGE_ORDER_SLAB_64:
        return &ctx->slab_64;
    case PAGE_ORDER_SLAB_128:
        return &ctx->slab_128;
    case PAGE_ORDER_SLAB_256:
        return &ctx->slab_256;
    case PAGE_ORDER_SLAB_512:
        return &ctx->slab_512;
    case PAGE_ORDER_SLAB_2048:
        return &ctx->slab_2048;
    default:
        return NULL;
    }
}

// ── 内部辅助：根据 size 选择 slab_cache 和对应 marker ─
static struct slab_cache *route_slab(gpgpu_ctx *ctx, size_t size,
                                     uint8_t *marker_out) {
    if (size <= 32) {
        *marker_out = PAGE_ORDER_SLAB_32;
        return &ctx->slab_32;
    }
    if (size <= 64) {
        *marker_out = PAGE_ORDER_SLAB_64;
        return &ctx->slab_64;
    }
    if (size <= 128) {
        *marker_out = PAGE_ORDER_SLAB_128;
        return &ctx->slab_128;
    }
    if (size <= 256) {
        *marker_out = PAGE_ORDER_SLAB_256;
        return &ctx->slab_256;
    }
    if (size <= 512) {
        *marker_out = PAGE_ORDER_SLAB_512;
        return &ctx->slab_512;
    }
    if (size < 4096) {
        *marker_out = PAGE_ORDER_SLAB_2048;
        return &ctx->slab_2048;
    }
    return NULL;
}

// ── gpuInit ───────────────────────────────────────────
int gpuInit(gpgpu_ctx *ctx, const char *dev_path, uint64_t vram_size) {
    ctx->vram_size = vram_size;

    printf("  gpuInit: opening %s\n", dev_path);
    fflush(stdout);
    ctx->fd = open(dev_path, O_RDWR);
    if (ctx->fd < 0) {
        perror("gpuInit: open");
        return -1;
    }

    ctx->vram_mmap = mmap(NULL, vram_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                          ctx->fd, 1 * PAGE_SIZE);
    if (ctx->vram_mmap == MAP_FAILED) {
        perror("gpuInit: mmap");
        close(ctx->fd);
        return -1;
    }

    if (buddy_init(&ctx->buddy, (uint64_t)ctx->vram_mmap, vram_size) < 0) {
        fprintf(stderr, "gpuInit: buddy_init failed\n");
        munmap(ctx->vram_mmap, vram_size);
        close(ctx->fd);
        return -1;
    }

    if (slab_cache_init(&ctx->slab_32, 32, &ctx->buddy) < 0)
        goto err_slab;

    if (slab_cache_init(&ctx->slab_64, 64, &ctx->buddy) < 0)
        goto err_slab;

    if (slab_cache_init(&ctx->slab_128, 128, &ctx->buddy) < 0)
        goto err_slab;

    if (slab_cache_init(&ctx->slab_256, 256, &ctx->buddy) < 0)
        goto err_slab;

    if (slab_cache_init(&ctx->slab_512, 512, &ctx->buddy) < 0)
        goto err_slab;

    if (slab_cache_init(&ctx->slab_2048, 2048, &ctx->buddy) < 0)
        goto err_slab;

    ctx->buddy.page_order[ctx->slab_32.slab_base / PAGE_SIZE] =
        PAGE_ORDER_SLAB_32;
    ctx->buddy.page_order[ctx->slab_64.slab_base / PAGE_SIZE] =
        PAGE_ORDER_SLAB_64;
    ctx->buddy.page_order[ctx->slab_128.slab_base / PAGE_SIZE] =
        PAGE_ORDER_SLAB_128;
    ctx->buddy.page_order[ctx->slab_256.slab_base / PAGE_SIZE] =
        PAGE_ORDER_SLAB_256;
    ctx->buddy.page_order[ctx->slab_512.slab_base / PAGE_SIZE] =
        PAGE_ORDER_SLAB_512;
    ctx->buddy.page_order[ctx->slab_2048.slab_base / PAGE_SIZE] =
        PAGE_ORDER_SLAB_2048;

    return 0;

err_slab:
    fprintf(stderr, "gpuInit: slab_cache_init failed\n");
    munmap(ctx->vram_mmap, vram_size);
    close(ctx->fd);
    return -1;
}

// ── gpuDestroy ────────────────────────────────────────
void gpuDestroy(gpgpu_ctx *ctx) {
    if (ctx->vram_mmap && ctx->vram_mmap != MAP_FAILED)
        munmap(ctx->vram_mmap, ctx->vram_size);
    if (ctx->fd >= 0)
        close(ctx->fd);
}

// ── gpuMalloc ─────────────────────────────────────────
uint64_t gpuMalloc(gpgpu_ctx *ctx, size_t size) {
    uint8_t            marker;
    struct slab_cache *sc = route_slab(ctx, size, &marker);

    if (sc != NULL) {
        // 走 Slab
        uint64_t offset = slab_alloc(sc);
        if (offset == (uint64_t)-1) {
            fprintf(stderr, "gpuMalloc: slab_alloc failed\n");
            return (uint64_t)-1;
        }
        return offset;
    }

    // 走 Buddy（size >= 4KB）
    uint64_t offset = buddy_alloc(&ctx->buddy, size);
    if (offset == (uint64_t)-1) {
        fprintf(stderr, "gpuMalloc: buddy_alloc failed\n");
        return (uint64_t)-1;
    }
    return offset;
}

// ── gpuFree ───────────────────────────────────────────
int gpuFree(gpgpu_ctx *ctx, uint64_t offset) {
    uint32_t page_idx = offset / PAGE_SIZE;
    uint8_t  marker   = ctx->buddy.page_order[page_idx];

    struct slab_cache *sc = get_slab(ctx, marker);
    if (sc != NULL)
        return slab_free(sc, offset);

    return buddy_free(&ctx->buddy, offset);
}

// ── gpuMemcpy ─────────────────────────────────────────
int gpuMemcpy(gpgpu_ctx *ctx, uint64_t dst, const void *src, size_t size,
              int direction) {
    if (direction == GPU_MEMCPY_H2D) {
        memcpy((uint8_t *)ctx->vram_mmap + dst, src, size);
    } else if (direction == GPU_MEMCPY_D2H) {
        memcpy((void *)src, (uint8_t *)ctx->vram_mmap + dst, size);
    } else {
        fprintf(stderr, "gpuMemcpy: unknown direction %d\n", direction);
        return -1;
    }
    return 0;
}

// ── gpuLaunchKernel ───────────────────────────────────
int gpuLaunchKernel(gpgpu_ctx *ctx, uint64_t kernel_addr, uint64_t args,
                    uint32_t grid[3], uint32_t block[3], uint32_t shared_mem) {
    struct gpgpu_dispatch_args dispatch = {
        .kernel_addr     = kernel_addr,
        .kernel_args     = args,
        .grid_dim        = {grid[0], grid[1], grid[2]},
        .block_dim       = {block[0], block[1], block[2]},
        .shared_mem_size = shared_mem,
    };

    if (ioctl(ctx->fd, GPGPU_IOC_DISPATCH, &dispatch) < 0) {
        perror("gpuLaunchKernel: ioctl");
        return -1;
    }
    return 0;
}