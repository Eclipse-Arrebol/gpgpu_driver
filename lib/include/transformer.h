/*
 * transformer.h — Qwen2-0.5B transformer 推理骨架 (M3.7.1)
 *
 * 设计原则:
 *   - workspace = 纯静态资源 + 地址簿 (init 时一次性 gpuMalloc)
 *   - step_ctx = 运行态 (transformer.c 内部 struct, 不暴露给外部;
 *                       外部通过 transformer_step 的独立参数
 *                       cur_pos/dump_step_id 传入)
 *   - DUMP_SCRATCH 宏: #ifdef DUMP_INTERMEDIATES 编译期开关, 生产路径零开销
 *   - prefill = N 次 single-token step (决策 1, 不写 prefill attention kernel)
 *   - 24 层共用一个 layer driver 函数 (决策 7)
 *
 * 隐式约定 (写在 transformer.c 注释中, .h 不重复):
 *   - pv_decode 输出 O [num_q_heads=14, head_dim=64] 写到 SCRATCH_X [1, 896]
 *     row-major [14, 64] ≡ [1, 896], 无需 reshape
 *   - KV cache 走 hybrid 路径: D2H → host append → (KV cache 内部 H2D) (决策 6)
 *     D-37 性能债务: 每层 D2H 2 次 (K+V), 24 层 = 48 次 host-device 往返。
 *     未来可用 D2D 路径 (device 端 kv_cache_append kernel) 消除。
 */
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "kv_cache.h"
#include "weight_loader.h"

/* ─── 错误码 ────────────────────────────────────────────────── */

#define TRANSFORMER_OK           0
#define TRANSFORMER_ERR_INIT    -1  /* workspace 初始化失败 */
#define TRANSFORMER_ERR_ALLOC   -2  /* gpuMalloc 失败 */
#define TRANSFORMER_ERR_KERNEL  -3  /* kernel binary 加载失败 */
#define TRANSFORMER_ERR_LAUNCH  -4  /* gpuLaunchKernel 失败 */
#define TRANSFORMER_ERR_CACHE   -5  /* KV cache append/commit 失败 */
#define TRANSFORMER_ERR_IO      -6  /* D2H / H2D 传输失败 */
#define TRANSFORMER_ERR_SINCOS  -7  /* sincos 表生成失败 */

/* ─── Config ────────────────────────────────────────────────── */

typedef struct {
    uint32_t D;              /* hidden_size = 896 */
    uint32_t intermediate;   /* intermediate_size = 4864 */
    uint32_t num_layers;     /* 24 */
    uint32_t num_q_heads;    /* 14 */
    uint32_t num_kv_heads;   /* 2 */
    uint32_t head_dim;       /* 64 */
    uint32_t vocab_size;     /* 151936 (Qwen2 常量) */
    uint32_t max_seq;        /* KV cache capacity = 128 */
    uint32_t S_max;          /* = 1 for pure decode; reserved for future batch prefill */
    float    rms_norm_eps;   /* 1e-6 */
    float    rope_theta;     /* 1e6 */
} transformer_config_t;

/* 默认 config (Qwen2-0.5B, pure decode, max_seq=128) */
static inline transformer_config_t transformer_default_config(void) {
    transformer_config_t cfg = {
        .D              = 896,
        .intermediate   = 4864,
        .num_layers     = 24,
        .num_q_heads    = 14,
        .num_kv_heads   = 2,
        .head_dim       = 64,
        .vocab_size     = 151936,
        .max_seq        = 128,
        .S_max          = 1,
        .rms_norm_eps   = 1e-6f,
        .rope_theta     = 1000000.0f,
    };
    return cfg;
}

/* ─── Workspace (静态资源 + 地址簿) ─────────────────────────── */

/* 所有 uint64_t 字段都是 device VRAM offset (gpgpu_ctx 内部分配)。
 * 命名风格: SCRATCH_x / RESID_x / LOGITS 全大写无后缀 (历史沿用);
 *           sincos_off / token_id_buf_off / argmax_debug_off 带 _off 后缀;
 *           kbin_xxx / args_xxx 语义前缀区分 kernel binary vs args buffer。
 * 字段名已与 transformer.c 绑定 (~280 处引用), 本轮不重命名。 */
typedef struct {
    /* borrowed 句柄 (caller 管生命周期) */
    gpgpu_ctx       *ctx;
    weight_loader_t *wl;
    kv_cache_t      *kv;

    transformer_config_t cfg;

    /* 9 scratch buffers (device offset) */
    uint64_t RESID_A;        /* [S_max, D]  residual ping-pong */
    uint64_t RESID_B;        /* [S_max, D] */
    uint64_t SCRATCH_X;      /* [S_max, D]  rmsnorm / o_proj / pv 输出共用 */
    uint64_t SCRATCH_Q;      /* [S_max, D]  q_proj 输出 */
    uint64_t SCRATCH_K;      /* [S_max, kv_dim]  k_proj 输出 */
    uint64_t SCRATCH_V;      /* [S_max, kv_dim]  v_proj 输出 */
    uint64_t SCRATCH_SCORES; /* [num_q_heads, max_seq]  attention 分数 */
    uint64_t SCRATCH_FFN_A;  /* [S_max, intermediate]  gate 输出 */
    uint64_t SCRATCH_FFN_B;  /* [S_max, intermediate]  up 输出 */

    /* logits buffer [1, vocab_size] (device offset) */
    uint64_t LOGITS;

    /* sincos 表 (device offset, V24 交错 layout) */
    uint64_t sincos_off;

    /* token_id uint32 buffer (device offset, 输入/输出复用) */
    uint64_t token_id_buf_off;

    /* argmax debug dump buffer (device offset, 32 lanes × 32 bytes) */
    uint64_t argmax_debug_off;

    /* kernel binary offsets (kbin_ 前缀, 与 args_ 区分) */
    uint64_t kbin_rmsnorm;
    uint64_t kbin_gemm;
    uint64_t kbin_broadcast_add;
    uint64_t kbin_silu;
    uint64_t kbin_vmul;
    uint64_t kbin_rope;
    uint64_t kbin_embedding;
    uint64_t kbin_argmax;
    uint64_t kbin_qkt_decode;
    uint64_t kbin_softmax_decode;
    uint64_t kbin_pv_decode;

    /* args buffers (device offset, 每类 kernel 一个) */
    uint64_t args_rmsnorm;
    uint64_t args_gemm;
    uint64_t args_broadcast_add;
    uint64_t args_silu;
    uint64_t args_vmul;
    uint64_t args_rope;
    uint64_t args_embedding;
    uint64_t args_argmax;
    uint64_t args_qkt_decode;
    uint64_t args_softmax_decode;
    uint64_t args_pv_decode;
} transformer_workspace_t;

/* ─── Lifecycle ─────────────────────────────────────────────── */

/*
 * transformer_init:
 *   1. 校验 cfg (assert S_max==1, max_seq==128 与 kv cache 一致)
 *   2. gpuMalloc 9 块 scratch + sincos + token_id_buf
 *   3. 加载所有 kernel binary 到 VRAM
 *   4. gpuMalloc 所有 args buffer
 *   5. 生成 sincos 表 (host 算 + H2D)
 *
 * kernel_bin_dir: 存放 *.bin 的目录路径 (如 "/tmp")
 *                 期望文件: rmsnorm.bin, gemm.bin, broadcast_add.bin,
 *                          silu.bin, vmul.bin, rope.bin,
 *                          embedding_lookup.bin, argmax.bin,
 *                          qkt_decode.bin, softmax_decode.bin, pv_decode.bin
 *
 * 返回: workspace 指针, 或 NULL 失败 (stderr 有诊断)
 */
transformer_workspace_t *transformer_init(
    gpgpu_ctx *ctx, weight_loader_t *wl, kv_cache_t *kv,
    const transformer_config_t *cfg,
    const char *kernel_bin_dir);

void transformer_destroy(transformer_workspace_t *ws);

/* ─── Step ──────────────────────────────────────────────────── */

/*
 * transformer_step: 单 token decode 步
 *
 * 流程: embedding → 24 × decoder_layer → final_norm → lm_head → argmax
 * decoder_layer 内部: rmsnorm → q/k/v_proj → rope → attention → o_proj
 *                     → residual_add → rmsnorm → ffn(gate+up+silu·mul+down)
 *                     → residual_add
 *
 * 输入: input_token_id, cur_pos (KV cache slot), dump_step_id (-1 = 不 dump)
 * 返回: next_token_id (>= 0), 或 TRANSFORMER_ERR_* (< 0)
 */
int transformer_step(transformer_workspace_t *ws,
                     int32_t input_token_id,
                     int32_t cur_pos,
                     int32_t dump_step_id);

/* ─── Dump 框架 ─────────────────────────────────────────────── */

#ifdef DUMP_INTERMEDIATES

/*
 * dump_scratch_impl: 将 device buffer 导出到文件
 *   layer_idx = -1 时, 路径为 dump/step_NN/<name>.bin (非层内 tensor)
 *   layer_idx >= 0 时, 路径为 dump/step_NN/layer_MM_<name>.bin (MM 两位 0 padding)
 *   NN = dump_step_id (两位 0 padding)
 *
 * name 约定与 reference 严格对齐 (ref/step_NN/ 下的文件名):
 *   after_embed,
 *   input_norm, q_proj, k_proj, v_proj,
 *   q_rope, k_rope, attn_out, o_proj, after_attn_res,
 *   post_norm, gate, up, silu_gate, silu_mul, down, after_ffn_res,
 *   final_norm, logits
 *
 * device-only 额外 dump (仅 device 调试, 无 reference 对应):
 *   qkt_scores, softmax_out
 */
void dump_scratch_impl(transformer_workspace_t *ws,
                       int32_t dump_step_id,
                       int32_t layer_idx,
                       const char *name,
                       uint64_t off,
                       size_t nbytes);

#define DUMP_SCRATCH(ws, step_id, layer_idx, name, off, nbytes) \
    dump_scratch_impl((ws), (step_id), (layer_idx), (name), (off), (nbytes))

#else

#define DUMP_SCRATCH(...) ((void)0)

#endif /* DUMP_INTERMEDIATES */

#endif /* TRANSFORMER_H */
