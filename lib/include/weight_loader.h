/*
 * weight_loader.h — Qwen2-0.5B-Instruct 权重加载器
 *
 * 职责:
 *   1) 从 host 文件系统 (weights/ 目录) 读取所有权重
 *   2) 一次性 gpuMalloc 一整块 region (2.35 GiB),把全部权重 H2D 上去
 *   3) 提供 (layer, tensor) → vram_offset 的查询接口
 *
 * 不做的事:
 *   - 不做 transpose / dtype 转换 (dump 工具已做)
 *   - 不做 alignment 调整 (dump 工具按固定 offset 排列)
 *   - 不做 lazy load (一次性全加载)
 *   - 不预留 scratch 区 (3.7 caller 自己 gpuMalloc)
 *
 * VRAM 布局:
 *   gpuMalloc 给的 region_base 由 buddy 决定,我们不控制. region 内部:
 *     [region_base + 0]                  embedding   (519 MB)
 *     [region_base + 519MB]              lm_head     (519 MB)
 *     [region_base + 1038MB]             final_norm  (3.5 KB)
 *     [region_base + 1038MB+]            layer 0     (57 MB)
 *     ...
 *     [region_base + ...]                layer 23    (57 MB)
 *
 * 所有 VRAM 地址都是 device-side offset (uint64_t), 与 gpuMalloc 返回值同义.
 */
#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include <stddef.h>
#include <stdint.h>

#include "../libgpgpu.h"
#include "weight_layout.h"

/* 不透明类型 */
typedef struct weight_loader weight_loader_t;

/* 按层 tensor 标识 ─ enum 而不是字符串名:错拼是编译期错而不是 runtime fail
 * (原则 21 精神:让 fail 显形) */
typedef enum {
    WL_INPUT_NORM = 0,
    WL_Q_PROJ_W,
    WL_Q_PROJ_B,
    WL_K_PROJ_W,
    WL_K_PROJ_B,
    WL_V_PROJ_W,
    WL_V_PROJ_B,
    WL_O_PROJ_W,
    WL_POST_NORM,
    WL_GATE_PROJ_W,
    WL_UP_PROJ_W,
    WL_DOWN_PROJ_W,
    WL_TENSOR_COUNT,
} wl_tensor_t;

/* ─── 生命周期 ─────────────────────────────────────────────── */

/* 加载 weights_dir 下所有文件到 device VRAM.
 *
 * 失败返回 NULL,fprintf 到 stderr.
 */
weight_loader_t *weight_loader_load(gpgpu_ctx *ctx, const char *weights_dir);

void weight_loader_destroy(weight_loader_t *wl);

/* ─── 查询接口 ─────────────────────────────────────────────── */

/* 按层 tensor 查询 → VRAM offset.
 *   layer ∈ [0, WL_NUM_LAYERS)
 *   name  ∈ [0, WL_TENSOR_COUNT)
 * 范围错 assert fail. */
uint64_t weight_loader_layer(weight_loader_t *wl, int layer, wl_tensor_t name);

/* 全局 tensor */
uint64_t weight_loader_embedding(weight_loader_t *wl);  /* [V, D] row-major */
uint64_t weight_loader_lm_head(weight_loader_t *wl);    /* [D, V] row-major */
uint64_t weight_loader_final_norm(weight_loader_t *wl); /* [D] */

/* ─── 调试 ─────────────────────────────────────────────────── */

/* 读回若干 fp32 到 host buffer (调用 gpuMemcpy D2H).
 * 用于 manifest first8 类校验. 失败返回 -1. */
int weight_loader_dump_floats(weight_loader_t *wl, uint64_t vram_offset,
                              float *out_buf, size_t n_floats);

/* 打印权重区地址簿,3.7 调试常用 */
void weight_loader_print_layout(weight_loader_t *wl);

#endif /* WEIGHT_LOADER_H */