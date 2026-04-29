#pragma once
#include "include/buddy.h"
#include "include/slab_cache.h"
#include <stddef.h>
#include <stdint.h>

// ── 传输方向 ──────────────────────────────────────────
#define GPU_MEMCPY_H2D 0
#define GPU_MEMCPY_D2H 1

// ── page_order Slab 标记 ──────────────────────────────
#define PAGE_ORDER_SLAB_32 0xF0
#define PAGE_ORDER_SLAB_64 0xF1
#define PAGE_ORDER_SLAB_128 0xF2
#define PAGE_ORDER_SLAB_256 0xF3
#define PAGE_ORDER_SLAB_512 0xF4
#define PAGE_ORDER_SLAB_2048 0xF5

// ── ioctl 定义（与驱动保持一致）─────────────────────
#include <linux/ioctl.h>

#define GPGPU_IOC_MAGIC 'G'
#define GPGPU_IOC_DISPATCH _IOW(GPGPU_IOC_MAGIC, 0, struct gpgpu_dispatch_args)

struct gpgpu_dispatch_args {
    uint64_t kernel_addr;
    uint64_t kernel_args;
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint32_t shared_mem_size;
};

// ── 全局上下文 ────────────────────────────────────────
typedef struct {
    int      fd;
    void    *vram_mmap;
    uint64_t vram_size;

    struct buddy_allocator buddy;

    struct slab_cache slab_32;
    struct slab_cache slab_64;
    struct slab_cache slab_128;
    struct slab_cache slab_256;
    struct slab_cache slab_512;
    struct slab_cache slab_2048;
} gpgpu_ctx;

// ── API ───────────────────────────────────────────────
int  gpuInit(gpgpu_ctx *ctx, const char *dev_path, uint64_t vram_size);
void gpuDestroy(gpgpu_ctx *ctx);

uint64_t gpuMalloc(gpgpu_ctx *ctx, size_t size);
int      gpuFree(gpgpu_ctx *ctx, uint64_t offset);

int gpuMemcpy(gpgpu_ctx *ctx, uint64_t dst, const void *src, size_t size,
              int direction);

int gpuLaunchKernel(gpgpu_ctx *ctx, uint64_t kernel_addr, uint64_t args,
                    uint32_t grid[3], uint32_t block[3], uint32_t shared_mem);