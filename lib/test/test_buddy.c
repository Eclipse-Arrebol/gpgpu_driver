#include <stdint.h>
#include <stdio.h>
#include "../include/buddy.h"
#include <stdlib.h>
#include <assert.h>

int main()
{
    /*初始化buddy*/
    struct buddy_allocator alloc;
    uint64_t* ptr= (uint64_t*)malloc(1ULL*1024*1024*1024);
    buddy_init(&alloc, (uint64_t)ptr, 1ULL*1024*1024*1024);
    for (int i = 0; i < 18; i++) {
        assert(alloc.free_list[i] == NULL);
    }
    assert(alloc.free_list[18] != NULL);
    printf("test init passed\n");

    /*分配buddy*/
    uint64_t offset = buddy_alloc(&alloc, 1ULL*1024*1024*1024);
    assert(offset != (uint64_t)-1);
    assert(offset<1ULL*1024*1024*1024);
    for (int i = 0; i < BUDDY_MAX_ORDER; i++) {
        assert(alloc.free_list[i] == NULL);
    }
    printf("test alloc passed\n");
    
    
    /*回收buddy*/
    buddy_free(&alloc, offset);
    for (int i = 0; i < 18; i++) {
        assert(alloc.free_list[i] == NULL);
    }
    assert(alloc.free_list[18] != NULL);
    printf("test free passed\n");
    
}