#include "../include/slab_cache.h"
#include <stdint.h>

int slab_cache_init(struct slab_cache *cache, size_t object_size,
                    struct buddy_allocator *buddy) {
    /*初始化cache*/
    uint64_t offset = buddy_alloc(buddy, PAGE_SIZE);
    if (offset == (uint64_t)-1)
        return -1;
    cache->slab_base   = offset;
    cache->slab_size   = PAGE_SIZE;
    cache->object_size = object_size;
    cache->buddy       = buddy;

    /*初始化空闲链表*/
    int cnt = cache->slab_size / cache->object_size;
    cache->free_list =
        (struct slab_node *)(cache->slab_base + buddy->vram_base);
    struct slab_node *temp = cache->free_list;
    for (int i = 1; i < cnt; i++) {
        temp->next =
            (struct slab_node *)(cache->slab_base + i * cache->object_size +
                                 buddy->vram_base);
        temp = temp->next;
    }
    temp->next = NULL;

    return 0;
}

uint64_t slab_alloc(struct slab_cache *cache) {
    uint64_t offset;
    if (!cache->free_list)
        return (uint64_t)-1;
    offset           = (uint64_t)cache->free_list - cache->buddy->vram_base;
    cache->free_list = cache->free_list->next;

    return offset;
}

int slab_free(struct slab_cache *cache, uint64_t offset) {
    struct slab_node *node =
        (struct slab_node *)(offset + cache->buddy->vram_base);
    node->next       = cache->free_list;
    cache->free_list = node;

    return 0;
}