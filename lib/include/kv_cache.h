#pragma once
#include "../libgpgpu.h"
#include <stdint.h>
#include <stdio.h>

typedef struct {
    /* 配置(创建时确定,只读) */
    uint32_t num_layers;
    uint32_t num_kv_heads;
    uint32_t max_seq;
    uint32_t head_dim;

    /* device 端 buffer (gpuMalloc 返回值, 64-bit) */
    uint64_t K_dev_off;
    uint64_t V_dev_off;

    /* 状态 */
    uint32_t filled; /* 已 commit 的 token 数, 范围 [0, max_seq] */
    uint32_t pending; /* 本 token 已 append 的 layer 数, 范围 [0, num_layers] */
} kv_cache_t;

kv_cache_t *kv_cache_create(gpgpu_ctx *ctx, uint32_t num_layers,
                            uint32_t num_kv_heads, uint32_t max_seq,
                            uint32_t head_dim);

/* 把 layer L 的新 K/V tile 写入 device buffer 位置 `filled`。
 * K_new, V_new 形状 [num_kv_heads, head_dim] (host 指针)。
 * 不推进 filled。推进 pending。
 * 返回 0 成功; -1 cache 已满; -2 layer 越界 / 重复 append。
 */
int kv_cache_append_layer(gpgpu_ctx *ctx, kv_cache_t *cache, uint32_t layer,
                          const float *K_new, const float *V_new);

/* 当前 token 所有 layer 都 append 完后调用。
 * filled++,pending 清零。
 * 返回 0 成功;-1 没有 append 任何 layer;-2 pending != num_layers(漏 layer)。
 */
int kv_cache_commit_token(kv_cache_t *cache);

/* getter:给 kernel args 用。
 * K_off / V_off:layer L 的 K/V 张量起始 device offset。
 * cur_pos:当前 token 的 index(= filled,因为还没 commit)。
 *
 * 重要语义:这些 getter 必须在 append 之后、commit 之前调用,
 *          此时 filled 指向当前 token,kernel 用 cur_pos=filled
 * 就能扫到刚写入的数据。
 */
uint64_t kv_cache_K_off(const kv_cache_t *cache, uint32_t layer);
uint64_t kv_cache_V_off(const kv_cache_t *cache, uint32_t layer);
uint32_t kv_cache_cur_pos(const kv_cache_t *cache);

/* 销毁:释放 device buffer。 */
void kv_cache_destroy(gpgpu_ctx *ctx, kv_cache_t *cache);