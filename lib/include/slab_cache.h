#pragma once
#include "buddy.h"
#include <stdint.h>
#include <stdio.h>

struct slab_node {
    struct slab_node *next;
};

struct slab_cache {
    size_t                  object_size; // 一块slab的大小
    struct slab_node       *free_list;   // 空闲链表
    uint64_t                slab_base;   // slab的起始地址
    size_t                  slab_size;   // 这块内存总大小
    struct buddy_allocator *buddy;       // 用于扩容
};

int      slab_cache_init(struct slab_cache *cache, size_t object_size,
                         struct buddy_allocator *buddy);
uint64_t slab_alloc(struct slab_cache *cache);
int      slab_free(struct slab_cache *cache, uint64_t offset);
