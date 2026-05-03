#include "../include/buddy.h"
#include "../include/slab_cache.h"
#include "../libgpgpu.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    /*初始化对应结构体*/
    struct buddy_allocator alloc;
    struct slab_cache      cache;
    uint64_t              *ptr = (uint64_t *)malloc(1ULL * 1024 * 1024 * 1024);
    buddy_init(&alloc, (uint64_t)ptr, 1ULL * 1024 * 1024 * 1024);
    slab_cache_init(&cache, 32, &alloc, PAGE_ORDER_SLAB_32);

    /*分配内存*/
    uint64_t slab_offset = slab_alloc(&cache);
    assert(slab_offset != -1);

    slab_free(&cache, slab_offset);
    assert((uint64_t)cache.free_list - alloc.vram_base == slab_offset);
    printf("test slab passed\n");
}