#include "../include/slab_cache.h"

#include <stdint.h>
#include <stdlib.h>

int slab_grow(struct slab_cache *cache) {
    uint64_t offset = buddy_alloc(cache->buddy, PAGE_SIZE);
    if (offset == (uint64_t)-1)
        return -1;

    struct slab_page_meta *meta = malloc(sizeof(struct slab_page_meta));
    if (meta == NULL) {
        buddy_free(cache->buddy, offset);
        return -1;
    }

    meta->page_offset = offset;
    meta->total_slots = PAGE_SIZE / cache->object_size;
    meta->free_count  = meta->total_slots;
    meta->next        = cache->pages;
    cache->pages      = meta;

    cache->buddy->page_order[offset / PAGE_SIZE] = cache->marker;

    int      cnt        = meta->total_slots;
    uint64_t page_vaddr = offset + cache->buddy->vram_base;
    for (int i = 0; i < cnt; i++) {
        struct slab_node *node =
            (struct slab_node *)(page_vaddr + i * cache->object_size);
        node->next       = cache->free_list;
        cache->free_list = node;
    }

    return 0;
}

int slab_cache_init(struct slab_cache *cache, size_t object_size,
                    struct buddy_allocator *buddy, uint8_t marker) {
    /*初始化cache*/

    cache->object_size = object_size;
    cache->buddy       = buddy;
    cache->marker      = marker;
    cache->free_list   = NULL;
    cache->pages       = NULL;

    /*初始化空闲链表*/
    if (slab_grow(cache) < 0)
        return -1;

    return 0;

    return 0;
}

uint64_t slab_alloc(struct slab_cache *cache) {
    uint64_t offset;
    if (!cache->free_list) {
        if (slab_grow(cache) == -1) {
            return (uint64_t)-1;
        }
    }

    offset           = (uint64_t)cache->free_list - cache->buddy->vram_base;
    cache->free_list = cache->free_list->next;

    uint64_t               page_off = offset & ~(PAGE_SIZE - 1);
    struct slab_page_meta *meta     = cache->pages;
    while (meta) {
        if (meta->page_offset == page_off)
            break;
        meta = meta->next;
    }
    // meta 一定能找到(否则就是 bug,可以加个 assert)
    meta->free_count -= 1;

    return offset;
}

int slab_free(struct slab_cache *cache, uint64_t offset) {
    /* 1. slot 插回 free_list 头 */
    struct slab_node *node =
        (struct slab_node *)(offset + cache->buddy->vram_base);
    node->next       = cache->free_list;
    cache->free_list = node;

    /* 2. 找到所属 meta + 它的 prev */
    uint64_t               page_off = offset & ~(PAGE_SIZE - 1);
    struct slab_page_meta *prev     = NULL;
    struct slab_page_meta *meta     = cache->pages;
    while (meta) {
        if (meta->page_offset == page_off)
            break;
        prev = meta;
        meta = meta->next;
    }
    meta->free_count += 1;

    /* 3. 整页空 + 还有其他 page → shrink */
    int has_other = 0;
    for (struct slab_page_meta *m = cache->pages; m; m = m->next) {
        if (m != meta) {
            has_other = 1;
            break;
        }
    }

    if (meta->free_count == meta->total_slots && has_other) {
        /* 3a. 从 free_list 摘掉本页所有 slot */
        uint64_t page_lo = meta->page_offset + cache->buddy->vram_base;
        uint64_t page_hi = page_lo + PAGE_SIZE;

        struct slab_node *fprev = NULL;
        struct slab_node *fcurr = cache->free_list;
        while (fcurr) {
            struct slab_node *fnext = fcurr->next;
            if ((uint64_t)fcurr >= page_lo && (uint64_t)fcurr < page_hi) {
                if (fprev)
                    fprev->next = fnext;
                else
                    cache->free_list = fnext;
            } else {
                fprev = fcurr;
            }
            fcurr = fnext;
        }

        /* 3b. 从 cache->pages 摘掉 meta */
        if (prev)
            prev->next = meta->next;
        else
            cache->pages = meta->next;

        /* 3c. 清 page_order,还给 buddy */
        cache->buddy->page_order[meta->page_offset / PAGE_SIZE] = 0;
        buddy_free(cache->buddy, meta->page_offset);

        /* 3d. 释放 meta */
        free(meta);
    }

    return 0;
}