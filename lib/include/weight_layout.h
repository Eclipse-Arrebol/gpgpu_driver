/*
 * weight_layout.h — Qwen2-0.5B-Instruct 权重文件布局
 *
 * 与 dump_weights.py 中的 LAYER_OFFSETS 是同源契约。
 * 任一边改动必须同步对方。建议未来加 test_layout_consistency.py
 * 解析本文件 const,对照 Python 字典,任意 diff 立刻 fail。
 *
 * 文件来源:Python 工具产出的 weights/ 目录,内含
 *   embedding.bin / lm_head.bin / final_norm.bin / layer_NN.bin × 24
 *
 * 所有 *_proj.weight 已 transpose 成 device gemm 直吃的 [K, N] 形态(钉死 7,
 * V24 工作记录)。bf16 → fp32 高 16 位填充语义(钉死 2)。
 *
 * 用户(weight_loader.c)不该手抄这些值,而是 #include 本文件。
 */
#ifndef WEIGHT_LAYOUT_H
#define WEIGHT_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

/* ─── 模型常量 ─────────────────────────────────────────────── */

#define WL_HIDDEN_SIZE 896
#define WL_INTERMEDIATE 4864
#define WL_NUM_LAYERS 24
#define WL_NUM_Q_HEADS 14
#define WL_NUM_KV_HEADS 2
#define WL_HEAD_DIM 64 /* HIDDEN_SIZE / NUM_Q_HEADS */
#define WL_KV_DIM 128  /* NUM_KV_HEADS * HEAD_DIM */
#define WL_VOCAB_SIZE 151936

/* ─── 每层内部固定 offset ──────────────────────────────────── */

/* offset 是 bytes,size 是 fp32 数量 */
#define WL_OFF_INPUT_NORM 0
#define WL_SIZE_INPUT_NORM (WL_HIDDEN_SIZE)

#define WL_OFF_Q_PROJ_W 3584
#define WL_SIZE_Q_PROJ_W (WL_HIDDEN_SIZE * WL_HIDDEN_SIZE)

#define WL_OFF_Q_PROJ_B 3214848
#define WL_SIZE_Q_PROJ_B (WL_HIDDEN_SIZE)

#define WL_OFF_K_PROJ_W 3218432
#define WL_SIZE_K_PROJ_W (WL_HIDDEN_SIZE * WL_KV_DIM)

#define WL_OFF_K_PROJ_B 3677184
#define WL_SIZE_K_PROJ_B (WL_KV_DIM)

#define WL_OFF_V_PROJ_W 3677696
#define WL_SIZE_V_PROJ_W (WL_HIDDEN_SIZE * WL_KV_DIM)

#define WL_OFF_V_PROJ_B 4136448
#define WL_SIZE_V_PROJ_B (WL_KV_DIM)

#define WL_OFF_O_PROJ_W 4136960
#define WL_SIZE_O_PROJ_W (WL_HIDDEN_SIZE * WL_HIDDEN_SIZE)

#define WL_OFF_POST_NORM 7348224
#define WL_SIZE_POST_NORM (WL_HIDDEN_SIZE)

#define WL_OFF_GATE_PROJ_W 7351808
#define WL_SIZE_GATE_PROJ_W (WL_HIDDEN_SIZE * WL_INTERMEDIATE)

#define WL_OFF_UP_PROJ_W 24784384
#define WL_SIZE_UP_PROJ_W (WL_HIDDEN_SIZE * WL_INTERMEDIATE)

#define WL_OFF_DOWN_PROJ_W 42216960
#define WL_SIZE_DOWN_PROJ_W (WL_INTERMEDIATE * WL_HIDDEN_SIZE)

#define WL_LAYER_BYTES 59649536 /* 单层总字节;dump 工具断言一致 */

/* 编译期 sanity check ─ 任一 offset/size 改动若导致总和错位,编译失败 */
_Static_assert(
    WL_OFF_DOWN_PROJ_W + WL_SIZE_DOWN_PROJ_W * 4 == WL_LAYER_BYTES,
    "weight_layout.h: layer offsets sum mismatch with WL_LAYER_BYTES");

_Static_assert(WL_HIDDEN_SIZE / WL_NUM_Q_HEADS == WL_HEAD_DIM,
               "weight_layout.h: HEAD_DIM mismatch");

_Static_assert(WL_NUM_KV_HEADS *WL_HEAD_DIM == WL_KV_DIM,
               "weight_layout.h: KV_DIM mismatch");

/* ─── 全局 tensor sizes(emb / lm_head / final_norm) ─────── */

#define WL_EMBEDDING_BYTES ((size_t)WL_VOCAB_SIZE * WL_HIDDEN_SIZE * 4)
#define WL_LM_HEAD_BYTES ((size_t)WL_HIDDEN_SIZE * WL_VOCAB_SIZE * 4)
#define WL_FINAL_NORM_BYTES ((size_t)WL_HIDDEN_SIZE * 4)

/* 全部权重总字节 */
#define WL_WEIGHTS_TOTAL_BYTES                                                 \
    (WL_EMBEDDING_BYTES + WL_LM_HEAD_BYTES + WL_FINAL_NORM_BYTES +             \
     (size_t)WL_NUM_LAYERS * WL_LAYER_BYTES)
/* = 519MB + 519MB + 3.5KB + 24 * 57MB ≈ 2.35 GiB */

/* ─── 文件名 ───────────────────────────────────────────────── */

#define WL_FILE_EMBEDDING "embedding.bin"
#define WL_FILE_LM_HEAD "lm_head.bin"
#define WL_FILE_FINAL_NORM "final_norm.bin"
#define WL_FILE_LAYER_FMT "layer_%02d.bin"

#endif /* WEIGHT_LAYOUT_H */