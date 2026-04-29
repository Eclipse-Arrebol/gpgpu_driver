#include "../include/buddy.h"
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/**
 * buddy_init -
 * @param *alloc - [DESCRIPTION]
 * @param vram_base - [DESCRIPTION]
 * @param size - [DESCRIPTION]
 * @return [DESCRIPTION]
 */
int buddy_init(struct buddy_allocator *alloc, uint64_t vram_base,
               uint64_t size) {
    alloc->vram_base = vram_base;
    for (int i = 0; i < BUDDY_MAX_ORDER; i++)
        alloc->free_list[i] = NULL;
    for (int i = 0; i < BUDDY_TOTAL_PAGES; i++)
        alloc->page_order[i] = 0;

    /* 根据实际 size 计算最大 order，逐级挂载 */
    uint64_t remaining = size;
    uint64_t offset    = 0;

    for (int order = BUDDY_MAX_ORDER - 1; order >= 0; order--) {
        uint64_t block_size = (1ULL << order) * PAGE_SIZE;
        if (remaining >= block_size) {
            struct buddy_node *node = (struct buddy_node *)(vram_base + offset);
            node->prev              = NULL;
            node->next              = alloc->free_list[order];
            if (alloc->free_list[order])
                alloc->free_list[order]->prev = node;
            alloc->free_list[order] = node;
            offset += block_size;
            remaining -= block_size;
        }
    }

    return 0;
}

/**
 *
 * @param alloc
 * @param alloc_size
 * @return uint64_t
 */
uint64_t buddy_alloc(struct buddy_allocator *alloc, uint64_t alloc_size) {
    int order;
    int i;
    /* 分配所需的阶数 */
    for (order = 0; order < BUDDY_MAX_ORDER; order++) {
        if ((1ULL << order) * PAGE_SIZE >= alloc_size)
            break;
    }
    if (order == BUDDY_MAX_ORDER)
        return (uint64_t)-1;

    /*寻找空链表*/
    for (i = order; i < BUDDY_MAX_ORDER; i++) {
        if (alloc->free_list[i])
            break;
    }

    if (i == BUDDY_MAX_ORDER) {
        printf("    buddy_alloc: no free block found!\n");
        fflush(stdout);
        return (uint64_t)-1;
    }

    /* 将大的order块分成合适大小的order块 */
    /* 先从 free_list[i] 摘出块 */
    struct buddy_node *left = alloc->free_list[i];
    alloc->free_list[i]     = left->next;
    if (alloc->free_list[i])
        alloc->free_list[i]->prev = NULL;

    /* 逐级分裂，左半块用 left 传递 */
    while (i > order) {
        struct buddy_node *right =
            (struct buddy_node *)((uint64_t)left +
                                  (1ULL << (i - 1)) * PAGE_SIZE);
        right->prev = NULL;
        right->next = alloc->free_list[i - 1];
        if (alloc->free_list[i - 1])
            alloc->free_list[i - 1]->prev = right;
        alloc->free_list[i - 1] = right;
        i--;
    }

    /* left 就是最终分配的块 */
    uint64_t offset               = (uint64_t)left - alloc->vram_base;
    uint64_t page_index           = offset / PAGE_SIZE;
    alloc->page_order[page_index] = order;
    return offset;
}

int buddy_free(struct buddy_allocator *alloc, uint64_t offset) {
    /*寻找buddy*/
    int                order        = alloc->page_order[offset / PAGE_SIZE];
    uint64_t           buddy_offset = offset ^ ((1ULL << order) * PAGE_SIZE);
    uint64_t           buddy_vaddr  = buddy_offset + alloc->vram_base;
    struct buddy_node *temp         = alloc->free_list[order];
    while (temp) {
        if ((uint64_t)temp == buddy_vaddr)
            break;
        temp = temp->next;
    }

    /*没找到buddy自己回到空闲链表*/
    if (!temp) {
        struct buddy_node *node =
            (struct buddy_node *)(offset + alloc->vram_base);
        node->prev              = NULL;
        node->next              = alloc->free_list[order];
        alloc->free_list[order] = node;
        if (node->next)
            node->next->prev = node;
    }

    /* 找到buddy就合并回到order+1的链表 */
    else {
        if (temp->prev)
            temp->prev->next = temp->next;
        else
            alloc->free_list[order] = temp->next;
        if (temp->next)
            temp->next->prev = temp->prev;
        uint64_t merged_offset = MIN(offset, buddy_offset);
        alloc->page_order[merged_offset / PAGE_SIZE] = order + 1;
        buddy_free(alloc, merged_offset);
    }

    return 0;
}
