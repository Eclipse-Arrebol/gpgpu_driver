#include "../include/kv_cache.h"
#include "../libgpgpu.h"
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

kv_cache_t *kv_cache_create(gpgpu_ctx *ctx, uint32_t num_layers,
                            uint32_t num_kv_heads, uint32_t max_seq,
                            uint32_t head_dim) {
    kv_cache_t *kv_cache = malloc(sizeof(kv_cache_t));
    if (!kv_cache) {
        return NULL;
    }
    kv_cache->num_layers   = num_layers;
    kv_cache->num_kv_heads = num_kv_heads;
    kv_cache->max_seq      = max_seq;
    kv_cache->head_dim     = head_dim;
    kv_cache->filled       = 0;
    kv_cache->pending      = 0;

    size_t total_bytes =
        (size_t)num_layers * num_kv_heads * max_seq * head_dim * sizeof(float);
    kv_cache->K_dev_off = gpuMalloc(ctx, total_bytes);
    if (kv_cache->K_dev_off == (uint64_t)-1) {
        free(kv_cache);
        return NULL;
    }

    kv_cache->V_dev_off = gpuMalloc(ctx, total_bytes);
    if (kv_cache->V_dev_off == (uint64_t)-1) {
        gpuFree(ctx, kv_cache->K_dev_off);
        free(kv_cache);
        return NULL;
    }

    return kv_cache;
}

int kv_cache_append_layer(gpgpu_ctx *ctx, kv_cache_t *cache, uint32_t layer,
                          const float *K_new, const float *V_new) {
    if (cache->filled >= cache->max_seq)
        return -1; // 满
    if (layer >= cache->num_layers)
        return -2; // layer 越界
    if (layer != cache->pending)
        return -2; // 顺序错(漏/重复)

    uint64_t layer_K_base = kv_cache_K_off(cache, layer); /* 复用 getter! */
    uint64_t layer_V_base = kv_cache_V_off(cache, layer);
    uint64_t head_stride =
        (uint64_t)cache->max_seq * cache->head_dim * sizeof(float);
    uint64_t row_offset =
        (uint64_t)cache->filled * cache->head_dim * sizeof(float);

    for (uint32_t kh = 0; kh < cache->num_kv_heads; kh++) {
        uint64_t     dst_K = layer_K_base + kh * head_stride + row_offset;
        uint64_t     dst_V = layer_V_base + kh * head_stride + row_offset;
        const float *src_K = K_new + kh * cache->head_dim;
        const float *src_V = V_new + kh * cache->head_dim;
        if (gpuMemcpy(ctx, dst_K, (void *)src_K,
                      cache->head_dim * sizeof(float), 0) < 0)
            return -3;
        if (gpuMemcpy(ctx, dst_V, (void *)src_V,
                      cache->head_dim * sizeof(float), 0) < 0)
            return -3;
    }

    cache->pending++;
    return 0;
}

int kv_cache_commit_token(kv_cache_t *cache) {
    if (cache->pending == 0)
        return -1;
    if (cache->pending != cache->num_layers)
        return -2;
    assert(cache->filled < cache->max_seq && "commit overflows max_seq");
    cache->pending = 0;
    cache->filled++;
    return 0;
}

uint64_t kv_cache_K_off(const kv_cache_t *cache, uint32_t layer) {
    assert(layer < cache->num_layers);
    return cache->K_dev_off + (uint64_t)layer * cache->num_kv_heads *
                                  cache->max_seq * cache->head_dim *
                                  sizeof(float);
}
uint64_t kv_cache_V_off(const kv_cache_t *cache, uint32_t layer) {
    assert(layer < cache->num_layers);
    return cache->V_dev_off + (uint64_t)layer * cache->num_kv_heads *
                                  cache->max_seq * cache->head_dim *
                                  sizeof(float);
}

uint32_t kv_cache_cur_pos(const kv_cache_t *cache) {
    assert(cache->filled < cache->max_seq && "kv cache full");
    return cache->filled;
}

void kv_cache_destroy(gpgpu_ctx *ctx, kv_cache_t *cache) {
    if (!cache)
        return;
    gpuFree(ctx, cache->K_dev_off);
    gpuFree(ctx, cache->V_dev_off);
    free(cache);
}