#pragma once
#include <stdint.h>
#include <stdio.h>

#define BUDDY_MAX_ORDER 19
#define BUDDY_TOTAL_PAGES 262144
#define PAGE_SIZE 4096
struct buddy_node {
    struct buddy_node *prev;
    struct buddy_node *next;
};

struct buddy_allocator{
    struct buddy_node *free_list[BUDDY_MAX_ORDER];
    uint8_t page_order[BUDDY_TOTAL_PAGES];
    uint64_t vram_base;
};

int buddy_init(struct buddy_allocator *alloc,uint64_t vram_base,uint64_t size);

uint64_t buddy_alloc(struct buddy_allocator *alloc,uint64_t alloc_size);

int buddy_free(struct buddy_allocator *alloc,uint64_t offset);