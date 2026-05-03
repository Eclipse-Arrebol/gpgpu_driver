#pragma once
#include "buddy.h"
#include <stdint.h>
#include <stdio.h>

struct slab_node {
    struct slab_node *next;
};

struct slab_page_meta {
    uint64_t page_offset; // 这个 page 在 VRAM 中的 offset
    uint16_t free_count;  // 当前还有几个 slot 在 free_list 里
    uint16_t total_slots; // 这个 page 总共能放几个 slot(冗余,方便判满)
    struct slab_page_meta *next;
};

struct slab_cache {
    size_t                  object_size;
    uint8_t                 marker;
    struct buddy_allocator *buddy;
    struct slab_node       *free_list; // 全局 free_list,横跨所有 page
    struct slab_page_meta  *pages;     // page meta 链表头
};

int      slab_cache_init(struct slab_cache *cache, size_t object_size,
                         struct buddy_allocator *buddy, uint8_t marker);
uint64_t slab_alloc(struct slab_cache *cache);
int      slab_free(struct slab_cache *cache, uint64_t offset);
