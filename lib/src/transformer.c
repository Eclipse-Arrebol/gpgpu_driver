/*
 * transformer.c — Qwen2-0.5B transformer 推觉骨架 (M3.7.1)
 *
 * 隐式约定:
 *   - pv_decode 输出 O [num_q_heads=14, head_dim=64] 写到 SCRATCH_X [1, 896]
 *     row-major [14, 64] ≡ [1, 896], 无需 reshape
 *   - KV cache 走 hybrid 路径: D2H → host append → (KV cache 内部 H2D) (决策 6)
 *     D-37 性能债务: 每层 D2H 2 次 (K+V), 24 层 = 48 次 host-device 往返。
 *     未来可用 D2D 路径 (device 端 kv_cache_append kernel) 消除。
 *   - o_proj / MLP 无 bias (Qwen2 结构)
 *
 * 9 块 scratch buffer 流转 (decode 路径, S_max=1):
 *
 *   每层:
 *     input_norm:  RESID_A → SCRATCH_X (rmsnorm)
 *     q/k/v_proj:  SCRATCH_X → SCRATCH_Q/K/V (gemm) + bias (broadcast_add)
 *     rope:        SCRATCH_Q, SCRATCH_K in-place
 *     kv append:   SCRATCH_K, SCRATCH_V → D2H → kv_cache_append_layer
 *     attention:   SCRATCH_Q(K=from cache) → SCRATCH_SCORES (qkt_decode)
 *                  → SCRATCH_SCORES in-place (softmax_decode)
 *                  → SCRATCH_SCORES + V_cache → SCRATCH_X (pv_decode)
 *     o_proj:      SCRATCH_X → SCRATCH_Q (gemm, 无 bias)
 *     residual:    RESID_A + SCRATCH_Q → RESID_B (broadcast_add)
 *     post_norm:   RESID_B → SCRATCH_X (rmsnorm)
 *     gate/up:     SCRATCH_X → SCRATCH_FFN_A/B (gemm)
 *     silu+mul:    SCRATCH_FFN_A in-place (silu), then × SCRATCH_FFN_B in-place
 * (vmul) down_proj:   SCRATCH_FFN_A → SCRATCH_X (gemm) residual:    RESID_B +
 * SCRATCH_X → RESID_A (broadcast_add)
 *
 *   层后 (final_norm → lm_head → argmax):
 *     RESID_A → SCRATCH_X (rmsnorm)
 *     SCRATCH_X → LOGITS (gemm)
 *     LOGITS → token_id_buf_off (argmax)
 */

#include "../include/transformer.h"
#include "../include/weight_layout.h"
#include "../libgpgpu.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* =================================================================
 * 块 1 — 内部 args structs + step_ctx_t + helpers 前向声明
 *
 * 每个 struct 的字段顺序严格对应 kernel .S 中 lw aN, offset(a0) 的顺序。
 * 寄存器对齐风险: 字段增删或重排会导致 args buffer 与 kernel 期望不匹配,
 * 表现为 "某个 lane 拿到错误值" 而非 crash, 极难调试。
 * 参见 V27 §6 funct7 错编同源风险。
 * ================================================================= */

/* ─── args structs ──────────────────────────────────────────── */

struct rmsnorm_args {
    uint32_t in_off;     /* lw a1, 0(a0) */
    uint32_t out_off;    /* lw a2, 4(a0) */
    uint32_t weight_off; /* lw a3, 8(a0) */
    uint32_t d; /* lw a4, 12(a0) — 读入但未参与计算, 硬编码 28 */
    float    inv_d; /* flw fa1, 16(a0) */
    float    eps;   /* flw fa2, 20(a0) */
};

struct gemm_args {
    uint32_t a_off; /* lw a1, 0(a0) */
    uint32_t b_off; /* lw a2, 4(a0) */
    uint32_t c_off; /* lw a3, 8(a0) */
    uint32_t M;     /* lw a4, 12(a0) */
    uint32_t K;     /* lw a5, 16(a0) */
    uint32_t N;     /* lw a6, 20(a0) */
};

struct broadcast_add_args {
    uint32_t x_off; /* lw a1, 0(a0) */
    uint32_t y_off; /* lw a2, 4(a0) */
    uint32_t b_off; /* lw a3, 8(a0) */
    uint32_t M;     /* lw a4, 12(a0) */
    uint32_t N;     /* lw a5, 16(a0) */
};

struct silu_args {
    uint32_t in_off;  /* lw a1, 0(a0) */
    uint32_t out_off; /* lw a2, 4(a0) */
    uint32_t D;       /* lw a3, 8(a0) — M3.7.1 新增 runtime D */
};

struct vmul_args {
    uint32_t a_off; /* lw a1, 0(a0) */
    uint32_t b_off; /* lw a2, 4(a0) */
    uint32_t c_off; /* lw a3, 8(a0) */
    uint32_t num;   /* lw a4, 12(a0) — runtime, 单 warp stride-32 循环 */
};

struct rope_args {
    uint32_t x_off;      /* lw a1, 0(a0) */
    uint32_t sincos_off; /* lw a2, 4(a0) */
    uint32_t num_heads;  /* lw a3, 8(a0) */
    uint32_t head_dim;   /* lw a4, 12(a0) */
    uint32_t token_pos;  /* lw a5, 16(a0) */
};

struct embedding_args {
    uint32_t table_base;     /* lw a1, 0(a0) */
    uint32_t token_id_buf;   /* lw a2, 4(a0) */
    uint32_t output_base;    /* lw a3, 8(a0) */
    uint32_t D;              /* lw a4, 12(a0) */
    uint32_t elems_per_lane; /* lw a5, 16(a0) */
};

struct argmax_args {
    uint32_t logits_base; /* lw a1, 0(a0) */
    uint32_t V;           /* lw a2, 4(a0) */
    uint32_t out_base;    /* lw a3, 8(a0) */
    uint32_t debug_base;  /* lw a4, 12(a0) */
};

struct qkt_decode_args {
    uint32_t Q_off;       /* lw a1, 0(a0) */
    uint32_t K_cache_off; /* lw a2, 4(a0) */
    uint32_t scores_off;  /* lw a3, 8(a0) */
    uint32_t cur_pos;     /* lw a4, 12(a0) */
    uint32_t max_seq;     /* lw a5, 16(a0) */
    uint32_t head_dim;    /* lw a6, 20(a0) */
    uint32_t q_per_kv;    /* lw a7, 24(a0) */
    uint32_t scale_bits; /* lw a0, 28(a0) — 复用 a0, 最后 load; float bitcast */
};

struct softmax_decode_args {
    uint32_t scores_off; /* lw a1, 0(a0) */
    uint32_t cur_pos;    /* lw a2, 4(a0) */
    uint32_t max_seq;    /* lw a3, 8(a0) */
};

struct pv_decode_args {
    uint32_t P_off;       /* lw a1, 0(a0) */
    uint32_t V_cache_off; /* lw a2, 4(a0) */
    uint32_t O_off;       /* lw a3, 8(a0) */
    uint32_t cur_pos;     /* lw a4, 12(a0) */
    uint32_t max_seq;     /* lw a5, 16(a0) */
    uint32_t head_dim;    /* lw a6, 20(a0) */
    uint32_t q_per_kv;    /* lw a7, 24(a0) */
};

/* D-40: args buffer 统一 32 字节, 未来超过 32 字节时此处报编译错误 */
_Static_assert(sizeof(struct rmsnorm_args) <= 32, "rmsnorm args too large");
_Static_assert(sizeof(struct gemm_args) <= 32, "gemm args too large");
_Static_assert(sizeof(struct broadcast_add_args) <= 32,
               "broadcast_add args too large");
_Static_assert(sizeof(struct silu_args) <= 32, "silu args too large");
_Static_assert(sizeof(struct vmul_args) <= 32, "vmul args too large");
_Static_assert(sizeof(struct rope_args) <= 32, "rope args too large");
_Static_assert(sizeof(struct embedding_args) <= 32, "embedding args too large");
_Static_assert(sizeof(struct argmax_args) <= 32, "argmax args too large");
_Static_assert(sizeof(struct qkt_decode_args) <= 32,
               "qkt_decode args too large");
_Static_assert(sizeof(struct softmax_decode_args) <= 32,
               "softmax_decode args too large");
_Static_assert(sizeof(struct pv_decode_args) <= 32, "pv_decode args too large");

/* ─── step_ctx_t (内部运行态, 不暴露给外部) ─────────────────── */

typedef struct {
    int32_t cur_pos;
    int32_t dump_step_id; /* -1 = 不 dump */
} step_ctx_t;

/* ─── dispatch helper ──────────────────────────────────────── */

static inline void calc_dispatch_1warp_per_block(uint32_t total,
                                                 uint32_t grid[3],
                                                 uint32_t block[3]) {
    grid[0]  = (total + 31) / 32;
    grid[1]  = 1;
    grid[2]  = 1;
    block[0] = 32;
    block[1] = 1;
    block[2] = 1;
}

/* ─── file-static helpers 前向声明 ────────────────────────── */

static uint64_t load_kernel_bin(gpgpu_ctx *ctx, const char *dir,
                                const char *name);

static float *compute_sincos_table(float rope_theta, int max_pos, int head_dim,
                                   int *out_err);

static int launch_op(gpgpu_ctx *ctx, uint64_t kbin, uint64_t args_off,
                     const void *args, size_t args_size, uint32_t grid[3],
                     uint32_t block[3]);

static int kv_append_from_device(transformer_workspace_t *ws, uint32_t layer,
                                 uint32_t cur_pos, uint64_t K_dev_off,
                                 uint64_t V_dev_off);

static int transformer_decoder_layer(transformer_workspace_t *ws,
                                     uint32_t layer, const step_ctx_t *sctx);

/* per-op helpers */
static int op_rmsnorm(transformer_workspace_t *ws, uint64_t in_off,
                      uint64_t out_off, uint64_t weight_off);
static int op_gemm(transformer_workspace_t *ws, uint64_t a_off, uint64_t b_off,
                   uint64_t c_off, uint32_t M, uint32_t K, uint32_t N);
static int op_broadcast_add(transformer_workspace_t *ws,
                            uint64_t x_off, uint64_t b_off, uint64_t y_off,
                            uint32_t M, uint32_t N);
static int op_silu(transformer_workspace_t *ws, uint64_t in_off,
                   uint64_t out_off, uint32_t D);
static int op_vmul(transformer_workspace_t *ws, uint64_t a_off, uint64_t b_off,
                   uint64_t c_off, uint32_t num);
static int op_rope(transformer_workspace_t *ws, uint64_t x_off,
                   uint32_t num_heads, uint32_t token_pos);
static int op_embedding(transformer_workspace_t *ws, uint64_t table_off,
                        uint64_t out_off);
static int op_argmax(transformer_workspace_t *ws);
static int op_qkt_decode(transformer_workspace_t *ws, uint64_t Q_off,
                         uint64_t K_cache_off, uint64_t scores_off,
                         uint32_t cur_pos, float scale);
static int op_softmax_decode(transformer_workspace_t *ws, uint64_t scores_off,
                             uint32_t cur_pos);
static int op_pv_decode(transformer_workspace_t *ws, uint64_t P_off,
                        uint64_t V_cache_off, uint64_t O_off, uint32_t cur_pos);

/* 块 1 结束, 等 review */

/* =================================================================
 * 块 2 — compute_sincos_table + pos=0 自检
 *
 * D-36 sincos layout:
 *   V24 RoPE (rope.S) 期望交错 layout:
 *     table[pos * head_dim + 2*k + 0] = cos(pos * freq_k)
 *     table[pos * head_dim + 2*k + 1] = sin(pos * freq_k)
 *   其中 k ∈ [0, head_dim/2), freq_k = theta^(-2k/head_dim)
 *
 *   reference 的 rope_cos.bin / rope_sin.bin 是分离 layout,
 *   D-36 自检要求 host 端按 V24 交错 layout 重组后 bit-exact 对照。
 *
 * pos=0 自检 (D-36 硬 gate):
 *   pos=0 时所有角度 = 0 → cos=1.0, sin=0.0。
 *   遍历 head_dim/2 个 pair, 任何一个不满足即 return TRANSFORMER_ERR_SINCOS。
 * ================================================================= */

static float *compute_sincos_table(float rope_theta, int max_pos, int head_dim,
                                   int *out_err) {
    assert(out_err != NULL);
    *out_err = TRANSFORMER_OK;

    assert(head_dim > 0 && head_dim % 2 == 0);
    int    half  = head_dim / 2;
    size_t count = (size_t)max_pos * head_dim;
    float *table = calloc(count, sizeof(float));
    if (!table) {
        *out_err = TRANSFORMER_ERR_ALLOC;
        return NULL;
    }

    float freq[64];
    assert(half <= 64);
    for (int k = 0; k < half; k++)
        freq[k] = 1.0f / powf(rope_theta, (2.0f * (float)k) / (float)head_dim);

    for (int pos = 0; pos < max_pos; pos++) {
        for (int k = 0; k < half; k++) {
            float angle                       = (float)pos * freq[k];
            table[pos * head_dim + k * 2 + 0] = cosf(angle);
            table[pos * head_dim + k * 2 + 1] = sinf(angle);
        }
    }

    /* D-36 pos=0 自检: cos=1.0, sin=0.0 全表 head_dim/2 个 pair */
    for (int k = 0; k < half; k++) {
        float c = table[k * 2 + 0];
        float s = table[k * 2 + 1];
        if (fabsf(c - 1.0f) > 1e-6f || fabsf(s) > 1e-6f) {
            fprintf(stderr,
                    "sincos self-check FAIL: pos=0 k=%d "
                    "cos=%.8e (expect 1) sin=%.8e (expect 0)\n",
                    k, c, s);
            free(table);
            *out_err = TRANSFORMER_ERR_SINCOS;
            return NULL;
        }
    }

    return table;
}

/* 块 2 结束, 等 review */

/* =================================================================
 * 块 3 — load_kernel_bin + transformer_init + transformer_destroy
 *
 * init 路径:
 *   1. 校验 cfg (assert S_max==1, max_seq==128, kv->max_seq 一致)
 *   2. workspace calloc + 所有 offset 字段初始化为 (uint64_t)-1 哨兵
 *   3. gpuMalloc 13 块 buffer (9 scratch + LOGITS + sincos + token_id +
 * argmax_debug)
 *   4. load 11 个 kernel binary 到 VRAM
 *   5. gpuMalloc 11 个 args buffer
 *   6. compute_sincos_table → H2D → free host
 *   7. zero out residual buffers
 *
 * 失败路径: 任何一步失败跳 fail → transformer_destroy(ws)
 * destroy 内部对所有 offset 字段做 != (uint64_t)-1 检查后才 gpuFree,
 * 因为 calloc 已把未走到的字段保持为 -1 哨兵, 不会误 free。
 * ================================================================= */

#define OFF_INVALID ((uint64_t) - 1)

static uint64_t load_kernel_bin(gpgpu_ctx *ctx, const char *dir,
                                const char *name) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "load_kernel_bin: fopen(%s): ", path);
        perror("");
        return OFF_INVALID;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    if (file_size <= 0) {
        fprintf(stderr, "load_kernel_bin: ftell(%s) returned %ld\n", path,
                file_size);
        fclose(f);
        return OFF_INVALID;
    }
    size_t size = (size_t)file_size;
    rewind(f);

    uint8_t *buf = malloc(size);
    if (!buf) {
        fprintf(stderr, "load_kernel_bin: malloc %zu failed\n", size);
        fclose(f);
        return OFF_INVALID;
    }
    size_t nread = fread(buf, 1, size, f);
    fclose(f);
    if (nread != size) {
        fprintf(stderr, "load_kernel_bin: fread short read %zu/%zu for %s\n",
                nread, size, name);
        free(buf);
        return OFF_INVALID;
    }

    uint64_t off = gpuMalloc(ctx, size);
    if (off == OFF_INVALID) {
        fprintf(stderr, "load_kernel_bin: gpuMalloc %zu failed for %s\n", size,
                name);
        free(buf);
        return OFF_INVALID;
    }
    if (gpuMemcpy(ctx, off, buf, size, GPU_MEMCPY_H2D) < 0) {
        fprintf(stderr, "load_kernel_bin: H2D failed for %s\n", name);
        gpuFree(ctx, off);
        free(buf);
        return OFF_INVALID;
    }
    free(buf);
    return off;
}

/* ─── init ────────────────────────────────────────────────── */

transformer_workspace_t *transformer_init(gpgpu_ctx *ctx, weight_loader_t *wl,
                                          kv_cache_t                 *kv,
                                          const transformer_config_t *cfg,
                                          const char *kernel_bin_dir) {
    fprintf(stderr, "[transformer_init] step 1: cfg validate\n");
    if (!ctx || !wl || !kv || !cfg) {
        fprintf(stderr, "transformer_init: null arg\n");
        return NULL;
    }
    if (cfg->S_max != 1) {
        fprintf(stderr, "transformer_init: S_max must be 1, got %u\n",
                cfg->S_max);
        return NULL;
    }
    if (cfg->max_seq != 128) {
        fprintf(stderr, "transformer_init: max_seq must be 128, got %u\n",
                cfg->max_seq);
        return NULL;
    }
    if (cfg->max_seq != kv->max_seq) {
        fprintf(stderr, "transformer_init: cfg.max_seq=%u != kv->max_seq=%u\n",
                cfg->max_seq, kv->max_seq);
        return NULL;
    }

    fprintf(stderr, "[transformer_init] step 2: workspace calloc\n");
    transformer_workspace_t *ws = calloc(1, sizeof(*ws));
    if (!ws)
        return NULL;

    fprintf(stderr, "[transformer_init] step 3: OFF_INVALID init\n");
    /* 所有 offset 字段初始化为 OFF_INVALID 哨兵,
     * 使 transformer_destroy 能安全处理部分初始化的 ws */
    {
        uint64_t *fields[] = {
            &ws->RESID_A,
            &ws->RESID_B,
            &ws->SCRATCH_X,
            &ws->SCRATCH_Q,
            &ws->SCRATCH_K,
            &ws->SCRATCH_V,
            &ws->SCRATCH_SCORES,
            &ws->SCRATCH_FFN_A,
            &ws->SCRATCH_FFN_B,
            &ws->LOGITS,
            &ws->sincos_off,
            &ws->token_id_buf_off,
            &ws->argmax_debug_off,
            &ws->kbin_rmsnorm,
            &ws->kbin_gemm,
            &ws->kbin_broadcast_add,
            &ws->kbin_silu,
            &ws->kbin_vmul,
            &ws->kbin_rope,
            &ws->kbin_embedding,
            &ws->kbin_argmax,
            &ws->kbin_qkt_decode,
            &ws->kbin_softmax_decode,
            &ws->kbin_pv_decode,
            &ws->args_rmsnorm,
            &ws->args_gemm,
            &ws->args_broadcast_add,
            &ws->args_silu,
            &ws->args_vmul,
            &ws->args_rope,
            &ws->args_embedding,
            &ws->args_argmax,
            &ws->args_qkt_decode,
            &ws->args_softmax_decode,
            &ws->args_pv_decode,
        };
        for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++)
            *fields[i] = OFF_INVALID;
    }

    ws->ctx = ctx;
    ws->wl  = wl;
    ws->kv  = kv;
    ws->cfg = *cfg;

    uint32_t D      = cfg->D;
    uint32_t I      = cfg->intermediate;
    uint32_t kv_dim = cfg->num_kv_heads * cfg->head_dim;

    fprintf(stderr, "[transformer_init] step 4: scratch alloc\n");
    /* ── scratch buffers (13 块) ───────────────────────── */
    ws->RESID_A   = gpuMalloc(ctx, (size_t)cfg->S_max * D * 4);
    ws->RESID_B   = gpuMalloc(ctx, (size_t)cfg->S_max * D * 4);
    ws->SCRATCH_X = gpuMalloc(ctx, (size_t)cfg->S_max * D * 4);
    ws->SCRATCH_Q = gpuMalloc(ctx, (size_t)cfg->S_max * D * 4);
    ws->SCRATCH_K = gpuMalloc(ctx, (size_t)cfg->S_max * kv_dim * 4);
    ws->SCRATCH_V = gpuMalloc(ctx, (size_t)cfg->S_max * kv_dim * 4);
    ws->SCRATCH_SCORES =
        gpuMalloc(ctx, (size_t)cfg->num_q_heads * cfg->max_seq * 4);
    ws->SCRATCH_FFN_A = gpuMalloc(ctx, (size_t)cfg->S_max * I * 4);
    ws->SCRATCH_FFN_B = gpuMalloc(ctx, (size_t)cfg->S_max * I * 4);
    ws->LOGITS        = gpuMalloc(ctx, (size_t)cfg->vocab_size * 4);
    ws->sincos_off = gpuMalloc(ctx, (size_t)cfg->max_seq * cfg->head_dim * 4);
    ws->token_id_buf_off = gpuMalloc(ctx, 4);
    ws->argmax_debug_off = gpuMalloc(ctx, 1024);

    {
        uint64_t *bufs[] = {
            &ws->RESID_A,          &ws->RESID_B,       &ws->SCRATCH_X,
            &ws->SCRATCH_Q,        &ws->SCRATCH_K,     &ws->SCRATCH_V,
            &ws->SCRATCH_SCORES,   &ws->SCRATCH_FFN_A, &ws->SCRATCH_FFN_B,
            &ws->LOGITS,           &ws->sincos_off,    &ws->token_id_buf_off,
            &ws->argmax_debug_off,
        };
        for (int i = 0; i < (int)(sizeof(bufs) / sizeof(bufs[0])); i++) {
            if (*bufs[i] == OFF_INVALID) {
                fprintf(stderr, "transformer_init: gpuMalloc failed (buf %d)\n",
                        i);
                goto fail;
            }
        }
    }

    fprintf(stderr, "[transformer_init] step 5: kbin load\n");
    /* ── kernel binaries (11 个) ───────────────────────── */
    ws->kbin_rmsnorm = load_kernel_bin(ctx, kernel_bin_dir, "rmsnorm");
    ws->kbin_gemm    = load_kernel_bin(ctx, kernel_bin_dir, "gemm");
    ws->kbin_broadcast_add =
        load_kernel_bin(ctx, kernel_bin_dir, "broadcast_add");
    ws->kbin_silu = load_kernel_bin(ctx, kernel_bin_dir, "silu");
    ws->kbin_vmul = load_kernel_bin(ctx, kernel_bin_dir, "vmul");
    ws->kbin_rope = load_kernel_bin(ctx, kernel_bin_dir, "rope");
    ws->kbin_embedding =
        load_kernel_bin(ctx, kernel_bin_dir, "embedding_lookup");
    ws->kbin_argmax     = load_kernel_bin(ctx, kernel_bin_dir, "argmax");
    ws->kbin_qkt_decode = load_kernel_bin(ctx, kernel_bin_dir, "qkt_decode");
    ws->kbin_softmax_decode =
        load_kernel_bin(ctx, kernel_bin_dir, "softmax_decode");
    ws->kbin_pv_decode = load_kernel_bin(ctx, kernel_bin_dir, "pv_decode");

    {
        uint64_t *kbins[] = {
            &ws->kbin_rmsnorm,       &ws->kbin_gemm,
            &ws->kbin_broadcast_add, &ws->kbin_silu,
            &ws->kbin_vmul,          &ws->kbin_rope,
            &ws->kbin_embedding,     &ws->kbin_argmax,
            &ws->kbin_qkt_decode,    &ws->kbin_softmax_decode,
            &ws->kbin_pv_decode,
        };
        for (int i = 0; i < (int)(sizeof(kbins) / sizeof(kbins[0])); i++) {
            if (*kbins[i] == OFF_INVALID) {
                fprintf(stderr, "transformer_init: kernel load failed (%s)\n",
                        kernel_bin_dir);
                goto fail;
            }
        }
    }

    fprintf(stderr, "[transformer_init] step 6: args alloc\n");
    /* ── args buffers (11 个) ──────────────────────────── */
    ws->args_rmsnorm        = gpuMalloc(ctx, 32);
    ws->args_gemm           = gpuMalloc(ctx, 32);
    ws->args_broadcast_add  = gpuMalloc(ctx, 32);
    ws->args_silu           = gpuMalloc(ctx, 32);
    ws->args_vmul           = gpuMalloc(ctx, 32);
    ws->args_rope           = gpuMalloc(ctx, 32);
    ws->args_embedding      = gpuMalloc(ctx, 32);
    ws->args_argmax         = gpuMalloc(ctx, 32);
    ws->args_qkt_decode     = gpuMalloc(ctx, 32);
    ws->args_softmax_decode = gpuMalloc(ctx, 32);
    ws->args_pv_decode      = gpuMalloc(ctx, 32);

    {
        uint64_t *aoffs[] = {
            &ws->args_rmsnorm,       &ws->args_gemm,
            &ws->args_broadcast_add, &ws->args_silu,
            &ws->args_vmul,          &ws->args_rope,
            &ws->args_embedding,     &ws->args_argmax,
            &ws->args_qkt_decode,    &ws->args_softmax_decode,
            &ws->args_pv_decode,
        };
        for (int i = 0; i < (int)(sizeof(aoffs) / sizeof(aoffs[0])); i++) {
            if (*aoffs[i] == OFF_INVALID) {
                fprintf(stderr, "transformer_init: args alloc failed (%d)\n",
                        i);
                goto fail;
            }
        }
    }

    fprintf(stderr, "[transformer_init] step 7: sincos H2D\n");
    /* ── sincos 表 (host 算 + H2D) ─────────────────────── */
    {
        int    sincos_err = TRANSFORMER_OK;
        float *sincos = compute_sincos_table(cfg->rope_theta, (int)cfg->max_seq,
                                             (int)cfg->head_dim, &sincos_err);
        if (!sincos) {
            fprintf(stderr, "transformer_init: sincos failed (err=%d)\n",
                    sincos_err);
            goto fail;
        }
        if (gpuMemcpy(ctx, ws->sincos_off, sincos,
                      (size_t)cfg->max_seq * cfg->head_dim * 4,
                      GPU_MEMCPY_H2D) < 0) {
            fprintf(stderr, "transformer_init: sincos H2D failed\n");
            free(sincos);
            goto fail;
        }
#ifdef DUMP_INTERMEDIATES
        /* D-36: dump sincos 表到 dump/sincos.bin (一次性, init 时),
         * Eclipse 离线用 numpy 重组 reference rope_cos.bin / rope_sin.bin
         * 按 V24 交错 layout, bit-exact 对照. */
        {
            if (mkdir("dump", 0755) < 0 && errno != EEXIST) {
                fprintf(stderr, "transformer_init: mkdir(dump) failed: ");
                perror("");
            }
            FILE *df = fopen("dump/sincos.bin", "wb");
            if (df) {
                fwrite(sincos, sizeof(float),
                       (size_t)cfg->max_seq * cfg->head_dim, df);
                fclose(df);
            }
        }
#endif
        free(sincos);
    }

    fprintf(stderr, "[transformer_init] step 8: residual zero out\n");
    /* zero out residual buffers — 功能上不必要 (RESID_A 第一次被 embedding
     * 覆盖, RESID_B 第一次被 attention 覆盖), 但 debug 时若有 off-by-one
     * 漏写, 看到 0 比看到随机值更易定位 */
    {
        size_t resid_bytes = (size_t)cfg->S_max * D * 4;
        void  *zero        = calloc(1, resid_bytes);
        if (zero) {
            gpuMemcpy(ctx, ws->RESID_A, zero, resid_bytes, GPU_MEMCPY_H2D);
            gpuMemcpy(ctx, ws->RESID_B, zero, resid_bytes, GPU_MEMCPY_H2D);
            free(zero);
        }
    }

    fprintf(stderr,
            "[transformer] workspace initialized: "
            "D=%u I=%u layers=%u heads=%u/%u max_seq=%u vocab=%u\n",
            D, I, cfg->num_layers, cfg->num_q_heads, cfg->num_kv_heads,
            cfg->max_seq, cfg->vocab_size);
    fprintf(stderr, "[transformer_init] step 9: done\n");

    return ws;

fail:
    transformer_destroy(ws);
    return NULL;
}

/* ─── destroy ─────────────────────────────────────────────── */

void transformer_destroy(transformer_workspace_t *ws) {
    if (!ws)
        return;

    gpgpu_ctx *ctx = ws->ctx;

    if (ctx) {
        /* scratch + auxiliary (13 块) */
        uint64_t scratch[] = {
            ws->RESID_A,          ws->RESID_B,       ws->SCRATCH_X,
            ws->SCRATCH_Q,        ws->SCRATCH_K,     ws->SCRATCH_V,
            ws->SCRATCH_SCORES,   ws->SCRATCH_FFN_A, ws->SCRATCH_FFN_B,
            ws->LOGITS,           ws->sincos_off,    ws->token_id_buf_off,
            ws->argmax_debug_off,
        };
        for (int i = 0; i < (int)(sizeof(scratch) / sizeof(scratch[0])); i++)
            if (scratch[i] != OFF_INVALID)
                gpuFree(ctx, scratch[i]);

        /* kernel bins (11 个) */
        uint64_t kbins[] = {
            ws->kbin_rmsnorm,        ws->kbin_gemm,      ws->kbin_broadcast_add,
            ws->kbin_silu,           ws->kbin_vmul,      ws->kbin_rope,
            ws->kbin_embedding,      ws->kbin_argmax,    ws->kbin_qkt_decode,
            ws->kbin_softmax_decode, ws->kbin_pv_decode,
        };
        for (int i = 0; i < (int)(sizeof(kbins) / sizeof(kbins[0])); i++)
            if (kbins[i] != OFF_INVALID)
                gpuFree(ctx, kbins[i]);

        /* args buffers (11 个) */
        uint64_t aoffs[] = {
            ws->args_rmsnorm,        ws->args_gemm,      ws->args_broadcast_add,
            ws->args_silu,           ws->args_vmul,      ws->args_rope,
            ws->args_embedding,      ws->args_argmax,    ws->args_qkt_decode,
            ws->args_softmax_decode, ws->args_pv_decode,
        };
        for (int i = 0; i < (int)(sizeof(aoffs) / sizeof(aoffs[0])); i++)
            if (aoffs[i] != OFF_INVALID)
                gpuFree(ctx, aoffs[i]);
    }

    free(ws);
}

/* 块 3 结束, 等 review */

/* =================================================================
 * 块 4 — launch_op + kv_append_from_device helpers
 *
 * launch_op: write args → H2D args → gpuLaunchKernel
 *   所有 op_* helper 都通过它启动 kernel.
 *   不做 stream 同步: gpuLaunchKernel 是 fire-and-forget,
 *   后续 D2H 读会自动 stall (test_silu.c 的模式).
 *
 * kv_append_from_device: D2H + kv_cache_append_layer 二合一
 *   D-37 性能债务: 每层 D2H 2 次 (K+V), 24 层 = 48 次 host-device 往返.
 *   K_dev_off / V_dev_off 是当前 step 的 k_proj / v_proj 输出
 *   (SCRATCH_K / SCRATCH_V), 单 token 切片, 不需要按 cur_pos 切片.
 * ================================================================= */

static int launch_op(gpgpu_ctx *ctx, uint64_t kbin, uint64_t args_off,
                     const void *args, size_t args_size, uint32_t grid[3],
                     uint32_t block[3]) {
    if (gpuMemcpy(ctx, args_off, args, args_size, GPU_MEMCPY_H2D) < 0)
        return TRANSFORMER_ERR_IO;

    if (gpuLaunchKernel(ctx, kbin, args_off, grid, block, 0) < 0)
        return TRANSFORMER_ERR_LAUNCH;

    return TRANSFORMER_OK;
}

static int kv_append_from_device(transformer_workspace_t *ws, uint32_t layer,
                                 uint32_t cur_pos, uint64_t K_dev_off,
                                 uint64_t V_dev_off) {
    assert(cur_pos == kv_cache_cur_pos(ws->kv));

    uint32_t kv_dim = ws->cfg.num_kv_heads * ws->cfg.head_dim;
    size_t   nbytes = (size_t)kv_dim * sizeof(float);

    float *K_host = malloc(nbytes);
    float *V_host = malloc(nbytes);
    if (!K_host || !V_host) {
        free(K_host);
        free(V_host);
        return TRANSFORMER_ERR_ALLOC;
    }

    if (gpuMemcpy(ws->ctx, K_dev_off, K_host, nbytes, GPU_MEMCPY_D2H) < 0) {
        free(K_host);
        free(V_host);
        return TRANSFORMER_ERR_IO;
    }
    if (gpuMemcpy(ws->ctx, V_dev_off, V_host, nbytes, GPU_MEMCPY_D2H) < 0) {
        free(K_host);
        free(V_host);
        return TRANSFORMER_ERR_IO;
    }

    int rc = kv_cache_append_layer(ws->ctx, ws->kv, layer, K_host, V_host);
    free(K_host);
    free(V_host);
    if (rc != 0) {
        fprintf(stderr,
                "kv_append_from_device: append failed rc=%d "
                "(layer=%u, cur_pos=%u)\n",
                rc, layer, cur_pos);
        return TRANSFORMER_ERR_CACHE;
    }

    return TRANSFORMER_OK;
}

/* 块 4 结束, 等 review */

/* =================================================================
 * 块 5a — 11 个 op_* helpers
 *
 * 每个 op_* = 构造 args struct → calc grid/block → 调 launch_op
 *
 * dispatch 模式 (已翻 11 个 .S 确认):
 *   single warp (grid=[1,1,1] block=[32,1,1]):
 *     rmsnorm, vmul, rope, embedding, argmax
 *   multi-block (calc_dispatch_1warp_per_block):
 *     gemm (total=M*N), broadcast_add (total=M*N), silu (total=D)
 *   multi-block by q_head (grid=[num_q_heads,1,1] block=[32,1,1]):
 *     qkt_decode, softmax_decode, pv_decode
 * ================================================================= */

static int op_rmsnorm(transformer_workspace_t *ws, uint64_t in_off,
                      uint64_t out_off, uint64_t weight_off) {
    struct rmsnorm_args args = {
        .in_off     = in_off,
        .out_off    = out_off,
        .weight_off = weight_off,
        .d          = ws->cfg.D,
        .inv_d      = 1.0f / (float)ws->cfg.D,
        .eps        = ws->cfg.rms_norm_eps,
    };
    uint32_t grid[3] = {1, 1, 1}, block[3] = {32, 1, 1};
    return launch_op(ws->ctx, ws->kbin_rmsnorm, ws->args_rmsnorm, &args,
                     sizeof(args), grid, block);
}

static int op_gemm(transformer_workspace_t *ws, uint64_t a_off, uint64_t b_off,
                   uint64_t c_off, uint32_t M, uint32_t K, uint32_t N) {
    struct gemm_args args = {
        .a_off = a_off,
        .b_off = b_off,
        .c_off = c_off,
        .M     = M,
        .K     = K,
        .N     = N,
    };
    uint32_t grid[3], block[3];
    calc_dispatch_1warp_per_block(M * N, grid, block);
    return launch_op(ws->ctx, ws->kbin_gemm, ws->args_gemm, &args, sizeof(args),
                     grid, block);
}

static int op_broadcast_add(transformer_workspace_t *ws,
                            uint64_t x_off, uint64_t b_off, uint64_t y_off,
                            uint32_t M, uint32_t N) {
    struct broadcast_add_args args = {
        .x_off = x_off, .y_off = y_off, .b_off = b_off,
        .M = M, .N = N,
    };
    uint32_t grid[3], block[3];
    calc_dispatch_1warp_per_block(M * N, grid, block);
    return launch_op(ws->ctx, ws->kbin_broadcast_add, ws->args_broadcast_add,
                     &args, sizeof(args), grid, block);
}

static int op_silu(transformer_workspace_t *ws, uint64_t in_off,
                   uint64_t out_off, uint32_t D) {
    struct silu_args args = {
        .in_off  = in_off,
        .out_off = out_off,
        .D       = D,
    };
    uint32_t grid[3], block[3];
    calc_dispatch_1warp_per_block(D, grid, block);
    return launch_op(ws->ctx, ws->kbin_silu, ws->args_silu, &args, sizeof(args),
                     grid, block);
}

static int op_vmul(transformer_workspace_t *ws, uint64_t a_off, uint64_t b_off,
                   uint64_t c_off, uint32_t num) {
    assert(num % 32 == 0 && "vmul.S D-38: num must be multiple of 32");
    struct vmul_args args = {
        .a_off = a_off,
        .b_off = b_off,
        .c_off = c_off,
        .num   = num,
    };
    uint32_t grid[3] = {1, 1, 1}, block[3] = {32, 1, 1};
    return launch_op(ws->ctx, ws->kbin_vmul, ws->args_vmul, &args, sizeof(args),
                     grid, block);
}

static int op_rope(transformer_workspace_t *ws, uint64_t x_off,
                   uint32_t num_heads, uint32_t token_pos) {
    struct rope_args args = {
        .x_off      = x_off,
        .sincos_off = ws->sincos_off,
        .num_heads  = num_heads,
        .head_dim   = ws->cfg.head_dim,
        .token_pos  = token_pos,
    };
    uint32_t grid[3] = {1, 1, 1}, block[3] = {32, 1, 1};
    return launch_op(ws->ctx, ws->kbin_rope, ws->args_rope, &args, sizeof(args),
                     grid, block);
}

static int op_embedding(transformer_workspace_t *ws, uint64_t table_off,
                        uint64_t out_off) {
    struct embedding_args args = {
        .table_base     = table_off,
        .token_id_buf   = ws->token_id_buf_off,
        .output_base    = out_off,
        .D              = ws->cfg.D,
        .elems_per_lane = ws->cfg.D / 32,
    };
    uint32_t grid[3] = {1, 1, 1}, block[3] = {32, 1, 1};
    return launch_op(ws->ctx, ws->kbin_embedding, ws->args_embedding, &args,
                     sizeof(args), grid, block);
}

static int op_argmax(transformer_workspace_t *ws) {
    struct argmax_args args = {
        .logits_base = ws->LOGITS,
        .V           = ws->cfg.vocab_size,
        .out_base    = ws->token_id_buf_off,
        .debug_base  = ws->argmax_debug_off,
    };
    uint32_t grid[3] = {1, 1, 1}, block[3] = {32, 1, 1};
    return launch_op(ws->ctx, ws->kbin_argmax, ws->args_argmax, &args,
                     sizeof(args), grid, block);
}

static int op_qkt_decode(transformer_workspace_t *ws, uint64_t Q_off,
                         uint64_t K_cache_off, uint64_t scores_off,
                         uint32_t cur_pos, float scale) {
    union {
        float    f;
        uint32_t u;
    } sf                        = {.f = scale};
    struct qkt_decode_args args = {
        .Q_off       = Q_off,
        .K_cache_off = K_cache_off,
        .scores_off  = scores_off,
        .cur_pos     = cur_pos,
        .max_seq     = ws->cfg.max_seq,
        .head_dim    = ws->cfg.head_dim,
        .q_per_kv    = ws->cfg.num_q_heads / ws->cfg.num_kv_heads,
        .scale_bits  = sf.u,
    };
    uint32_t grid[3] = {ws->cfg.num_q_heads, 1, 1}, block[3] = {32, 1, 1};
    return launch_op(ws->ctx, ws->kbin_qkt_decode, ws->args_qkt_decode, &args,
                     sizeof(args), grid, block);
}

static int op_softmax_decode(transformer_workspace_t *ws, uint64_t scores_off,
                             uint32_t cur_pos) {
    struct softmax_decode_args args = {
        .scores_off = scores_off,
        .cur_pos    = cur_pos,
        .max_seq    = ws->cfg.max_seq,
    };
    uint32_t grid[3] = {ws->cfg.num_q_heads, 1, 1}, block[3] = {32, 1, 1};
    return launch_op(ws->ctx, ws->kbin_softmax_decode, ws->args_softmax_decode,
                     &args, sizeof(args), grid, block);
}

static int op_pv_decode(transformer_workspace_t *ws, uint64_t P_off,
                        uint64_t V_cache_off, uint64_t O_off,
                        uint32_t cur_pos) {
    struct pv_decode_args args = {
        .P_off       = P_off,
        .V_cache_off = V_cache_off,
        .O_off       = O_off,
        .cur_pos     = cur_pos,
        .max_seq     = ws->cfg.max_seq,
        .head_dim    = ws->cfg.head_dim,
        .q_per_kv    = ws->cfg.num_q_heads / ws->cfg.num_kv_heads,
    };
    uint32_t grid[3] = {ws->cfg.num_q_heads, 1, 1}, block[3] = {32, 1, 1};
    return launch_op(ws->ctx, ws->kbin_pv_decode, ws->args_pv_decode, &args,
                     sizeof(args), grid, block);
}

/* 块 5a 结束, 等 review */

/* =================================================================
 * 块 5b — transformer_decoder_layer (24 层共用的 layer driver)
 *
 * 22 步线性流程 (V27 §2.2 决策 7):
 *   Attention: input_norm → q/k/v_proj + bias → rope → kv_append
 *              → qkt_decode → softmax_decode → pv_decode
 *              → o_proj → residual_add_1
 *   FFN:       post_norm → gate_proj → up_proj → silu → vmul
 *              → down_proj → residual_add_2
 *
 * 入口: RESID_A = 当前层输入
 * 出口: RESID_A = 当前层输出 (下一层输入, ping-pong)
 *
 * buffer 流转图参见 transformer.c 顶部注释.
 * ================================================================= */

static int transformer_decoder_layer(transformer_workspace_t *ws,
                                     uint32_t layer, const step_ctx_t *sctx) {
    weight_loader_t *wl     = ws->wl;
    kv_cache_t      *kv     = ws->kv;
    gpgpu_ctx       *ctx    = ws->ctx;
    uint32_t         D      = ws->cfg.D;
    uint32_t         kv_dim = ws->cfg.num_kv_heads * ws->cfg.head_dim;
    uint32_t         I      = ws->cfg.intermediate;
    int32_t          sid    = sctx->dump_step_id;
    uint32_t         pos    = (uint32_t)sctx->cur_pos;
    int              rc;

    (void)ctx;
#ifndef DUMP_INTERMEDIATES
    (void)sid;
#endif

    /* ── 1. input_norm ───────────────────────────────── */
    rc = op_rmsnorm(ws, ws->RESID_A, ws->SCRATCH_X,
                    weight_loader_layer(wl, layer, WL_INPUT_NORM));
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "input_norm", ws->SCRATCH_X,
                 (size_t)D * 4);

    /* ── 2-3. q_proj ─────────────────────────────────── */
    rc = op_gemm(ws, ws->SCRATCH_X, weight_loader_layer(wl, layer, WL_Q_PROJ_W),
                 ws->SCRATCH_Q, 1, D, D);
    if (rc)
        return rc;
    rc = op_broadcast_add(ws, ws->SCRATCH_Q,
                          weight_loader_layer(wl, layer, WL_Q_PROJ_B),
                          ws->SCRATCH_Q, 1, D);
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "q_proj", ws->SCRATCH_Q,
                 (size_t)D * 4);

    /* ── 4-5. k_proj ─────────────────────────────────── */
    rc = op_gemm(ws, ws->SCRATCH_X, weight_loader_layer(wl, layer, WL_K_PROJ_W),
                 ws->SCRATCH_K, 1, D, kv_dim);
    if (rc)
        return rc;
    rc = op_broadcast_add(ws, ws->SCRATCH_K,
                          weight_loader_layer(wl, layer, WL_K_PROJ_B),
                          ws->SCRATCH_K, 1, kv_dim);
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "k_proj", ws->SCRATCH_K,
                 (size_t)kv_dim * 4);

    /* ── 6-7. v_proj ─────────────────────────────────── */
    rc = op_gemm(ws, ws->SCRATCH_X, weight_loader_layer(wl, layer, WL_V_PROJ_W),
                 ws->SCRATCH_V, 1, D, kv_dim);
    if (rc)
        return rc;
    rc = op_broadcast_add(ws, ws->SCRATCH_V,
                          weight_loader_layer(wl, layer, WL_V_PROJ_B),
                          ws->SCRATCH_V, 1, kv_dim);
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "v_proj", ws->SCRATCH_V,
                 (size_t)kv_dim * 4);

    /* ── 8. rope(Q) ──────────────────────────────────── */
    rc = op_rope(ws, ws->SCRATCH_Q, ws->cfg.num_q_heads, pos);
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "q_rope", ws->SCRATCH_Q,
                 (size_t)D * 4);

    /* ── 9. rope(K) ──────────────────────────────────── */
    rc = op_rope(ws, ws->SCRATCH_K, ws->cfg.num_kv_heads, pos);
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "k_rope", ws->SCRATCH_K,
                 (size_t)kv_dim * 4);

    /* ── 10. kv append (D2H → host → kv_cache) ──────── */
    rc = kv_append_from_device(ws, layer, pos, ws->SCRATCH_K, ws->SCRATCH_V);
    if (rc)
        return rc;

    /* ── 11. qkt_decode ──────────────────────────────── */
    float scale = 1.0f / sqrtf((float)ws->cfg.head_dim);
    rc          = op_qkt_decode(ws, ws->SCRATCH_Q, kv_cache_K_off(kv, layer),
                                ws->SCRATCH_SCORES, pos, scale);
    if (rc)
        return rc;
#ifdef DUMP_INTERMEDIATES
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "qkt_scores", ws->SCRATCH_SCORES,
                 (size_t)ws->cfg.num_q_heads * ws->cfg.max_seq * 4);
#endif

    /* ── 12. softmax_decode ──────────────────────────── */
    rc = op_softmax_decode(ws, ws->SCRATCH_SCORES, pos);
    if (rc)
        return rc;
#ifdef DUMP_INTERMEDIATES
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "softmax_out", ws->SCRATCH_SCORES,
                 (size_t)ws->cfg.num_q_heads * ws->cfg.max_seq * 4);
#endif

    /* ── 13. pv_decode → SCRATCH_X ───────────────────── */
    rc = op_pv_decode(ws, ws->SCRATCH_SCORES, kv_cache_V_off(kv, layer),
                      ws->SCRATCH_X, pos);
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "attn_out", ws->SCRATCH_X,
                 (size_t)D * 4);

    /* ── 14. o_proj → SCRATCH_Q ──────────────────────── */
    rc = op_gemm(ws, ws->SCRATCH_X, weight_loader_layer(wl, layer, WL_O_PROJ_W),
                 ws->SCRATCH_Q, 1, D, D);
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "o_proj", ws->SCRATCH_Q,
                 (size_t)D * 4);

    /* ── 15. residual 1: RESID_A + SCRATCH_Q → RESID_B ─ */
    rc = op_broadcast_add(ws, ws->RESID_A, ws->SCRATCH_Q, ws->RESID_B, 1, D);
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "after_attn_res", ws->RESID_B,
                 (size_t)D * 4);

    /* ── 16. post_norm ───────────────────────────────── */
    rc = op_rmsnorm(ws, ws->RESID_B, ws->SCRATCH_X,
                    weight_loader_layer(wl, layer, WL_POST_NORM));
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "post_norm", ws->SCRATCH_X,
                 (size_t)D * 4);

    /* ── 17. gate_proj → SCRATCH_FFN_A ───────────────── */
    rc = op_gemm(ws, ws->SCRATCH_X,
                 weight_loader_layer(wl, layer, WL_GATE_PROJ_W),
                 ws->SCRATCH_FFN_A, 1, D, I);
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "gate", ws->SCRATCH_FFN_A,
                 (size_t)I * 4);

    /* ── 18. up_proj → SCRATCH_FFN_B ─────────────────── */
    rc =
        op_gemm(ws, ws->SCRATCH_X, weight_loader_layer(wl, layer, WL_UP_PROJ_W),
                ws->SCRATCH_FFN_B, 1, D, I);
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "up", ws->SCRATCH_FFN_B,
                 (size_t)I * 4);

    /* ── 19. silu(FFN_A) in-place ────────────────────── */
    rc = op_silu(ws, ws->SCRATCH_FFN_A, ws->SCRATCH_FFN_A, I);
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "silu_gate", ws->SCRATCH_FFN_A,
                 (size_t)I * 4);

    /* ── 20. vmul(FFN_A, FFN_B) → FFN_A in-place ────── */
    rc =
        op_vmul(ws, ws->SCRATCH_FFN_A, ws->SCRATCH_FFN_B, ws->SCRATCH_FFN_A, I);
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "silu_mul", ws->SCRATCH_FFN_A,
                 (size_t)I * 4);

    /* ── 21. down_proj → SCRATCH_X ───────────────────── */
    {
        uint64_t down_w_off = weight_loader_layer(wl, layer, WL_DOWN_PROJ_W);
        fprintf(stderr,
                "[down @ layer %u] a_off=0x%llx b_off=0x%llx c_off=0x%llx "
                "M=1 K=%u N=%u\n",
                layer,
                (unsigned long long)ws->SCRATCH_FFN_A,
                (unsigned long long)down_w_off,
                (unsigned long long)ws->SCRATCH_X,
                I, D);
        fflush(stderr);
#ifdef DUMP_INTERMEDIATES
        if (layer == 12 && sid >= 0) {
            float *tmp = malloc((size_t)I * 4);
            if (tmp) {
                gpuMemcpy(ws->ctx, ws->SCRATCH_FFN_A, tmp,
                          (size_t)I * 4, GPU_MEMCPY_D2H);
                char path[256];
                snprintf(path, sizeof(path),
                         "dump/step_%02d/layer_12_ffn_a_just_before_down.bin",
                         sid);
                FILE *fp = fopen(path, "wb");
                if (fp) { fwrite(tmp, 4, I, fp); fclose(fp); }
                free(tmp);
            }
            size_t w_bytes = (size_t)I * D * 4;
            float *w_tmp = malloc(w_bytes);
            if (w_tmp) {
                gpuMemcpy(ws->ctx, down_w_off, w_tmp, w_bytes, GPU_MEMCPY_D2H);
                char wpath[256];
                snprintf(wpath, sizeof(wpath),
                         "dump/step_%02d/layer_12_down_w_d2h.bin", sid);
                FILE *wfp = fopen(wpath, "wb");
                if (wfp) { fwrite(w_tmp, 1, w_bytes, wfp); fclose(wfp); }
                free(w_tmp);
            }
        }
#endif
        rc = op_gemm(ws, ws->SCRATCH_FFN_A, down_w_off, ws->SCRATCH_X,
                     1, I, D);
    }
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "down", ws->SCRATCH_X, (size_t)D * 4);

    /* ── 22. residual 2: RESID_B + SCRATCH_X → RESID_A ─ */
    rc = op_broadcast_add(ws, ws->RESID_B, ws->SCRATCH_X, ws->RESID_A, 1, D);
    if (rc)
        return rc;
    DUMP_SCRATCH(ws, sid, (int32_t)layer, "after_ffn_res", ws->RESID_A,
                 (size_t)D * 4);

    return TRANSFORMER_OK;
}

/* 块 5b 结束, 等 review */

/* =================================================================
 * 块 6 — transformer_step (公共入口) + dump_scratch_impl (调试工具)
 *
 * transformer_step: 单 token 推理, 返回 next_token_id (>= 0)
 * dump_scratch_impl: device buffer → 文件, #ifdef DUMP_INTERMEDIATES 编译
 * ================================================================= */

#ifdef DUMP_INTERMEDIATES

void dump_scratch_impl(transformer_workspace_t *ws,
                       int32_t dump_step_id, int32_t layer_idx,
                       const char *name, uint64_t off, size_t nbytes) {
    static int last_step_id = -1;

    if (dump_step_id < 0) return;

    char dir[128];
    snprintf(dir, sizeof(dir), "dump/step_%02d", dump_step_id);

    if (dump_step_id != last_step_id) {
        if (mkdir("dump", 0755) < 0 && errno != EEXIST) {
            fprintf(stderr, "dump_scratch: mkdir(dump) failed\n");
            return;
        }
        if (mkdir(dir, 0755) < 0 && errno != EEXIST) {
            fprintf(stderr, "dump_scratch: mkdir(%s) failed\n", dir);
            return;
        }
        last_step_id = dump_step_id;
    }

    char path[256];
    if (layer_idx < 0)
        snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    else
        snprintf(path, sizeof(path), "%s/layer_%02d_%s.bin",
                 dir, layer_idx, name);

    void *host_buf = malloc(nbytes);
    if (!host_buf) return;

    if (gpuMemcpy(ws->ctx, off, host_buf, nbytes, GPU_MEMCPY_D2H) < 0) {
        free(host_buf);
        return;
    }

    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(host_buf, 1, nbytes, f);
        fclose(f);
    }
    free(host_buf);
}

#endif /* DUMP_INTERMEDIATES */

int transformer_step(transformer_workspace_t *ws,
                     int32_t input_token_id,
                     int32_t cur_pos,
                     int32_t dump_step_id) {
    weight_loader_t *wl = ws->wl;
    uint32_t D         = ws->cfg.D;
    int32_t  sid       = dump_step_id;
    int rc;

    step_ctx_t sctx = { .cur_pos = cur_pos, .dump_step_id = sid };

    /* ── 1. H2D input token id ───────────────────────── */
    uint32_t tok_u32 = (uint32_t)input_token_id;
    if (gpuMemcpy(ws->ctx, ws->token_id_buf_off,
                  &tok_u32, 4, GPU_MEMCPY_H2D) < 0)
        return TRANSFORMER_ERR_IO;

    /* ── 2. embedding ────────────────────────────────── */
    rc = op_embedding(ws, weight_loader_embedding(wl), ws->RESID_A);
    if (rc) return rc;
    DUMP_SCRATCH(ws, sid, -1, "after_embed",
                 ws->RESID_A, (size_t)D * 4);

    /* ── 3. 24 层 decoder ────────────────────────────── */
    for (uint32_t layer = 0; layer < ws->cfg.num_layers; layer++) {
        rc = transformer_decoder_layer(ws, layer, &sctx);
        if (rc) return rc;
    }

    /* ── 4. final norm ───────────────────────────────── */
    rc = op_rmsnorm(ws, ws->RESID_A, ws->SCRATCH_X,
                    weight_loader_final_norm(wl));
    if (rc) return rc;
    DUMP_SCRATCH(ws, sid, -1, "final_norm",
                 ws->SCRATCH_X, (size_t)D * 4);

    /* ── 5. lm_head → LOGITS ─────────────────────────── */
    rc = op_gemm(ws, ws->SCRATCH_X,
                 weight_loader_lm_head(wl),
                 ws->LOGITS, 1, D, ws->cfg.vocab_size);
    if (rc) return rc;
    DUMP_SCRATCH(ws, sid, -1, "logits",
                 ws->LOGITS, (size_t)ws->cfg.vocab_size * 4);

    /* ── 6. argmax → token_id_buf ────────────────────── */
    rc = op_argmax(ws);
    if (rc) return rc;

    /* ── 7. KV cache commit (选项 C: argmax 后, D2H 前) */
    if (kv_cache_commit_token(ws->kv) < 0) {
        fprintf(stderr, "transformer_step: kv commit failed\n");
        return TRANSFORMER_ERR_CACHE;
    }

    /* ── 8. D2H read next token ──────────────────────── */
    uint32_t next_tok = 0;
    if (gpuMemcpy(ws->ctx, ws->token_id_buf_off,
                  &next_tok, 4, GPU_MEMCPY_D2H) < 0)
        return TRANSFORMER_ERR_IO;

    return (int)next_tok;
}

/* 块 6 结束 — M3.7.1 transformer.c 主体完成 */
