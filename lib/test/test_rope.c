/* test_rope.c — Milestone 3.6.5 RoPE 单算子测试 (V24)
 *
 * V24 改动 (相对 V22 版):
 *   #1 ROPE_THETA = 1e6,匹配 Qwen2 (config.json 实证)
 *   #2 Pair 排布改为 GPT-NeoX style: (k, k+head_dim/2),匹配 HF modeling_qwen2
 *   #3 容差改混合:tol = max(1e-5, 2e-5 * |ref|),per-element pass 而非只看
 * max_err #4 加 high token_pos case (pos=4096, pos=65535) #5 helper self-check
 * 加 Qwen2 频率谱验证 #6 MAX_TEST_POS 256 → 65536
 *
 * 设计要点:
 *   - 单 warp 32 thread 派发 (每 lane 一对,head_dim/2 = 32 = lane 数)
 *   - in-place:device 端 x_off 既是输入又是输出
 *   - sincos 表 layout 不变:[pos*head_dim + 2k + 0/1],cos/sin 仍相邻 8 字节
 *   - host reference 用 fmaf 强制单舍入,匹配 device fmadd.s
 *   - baseline (token_pos=0) 走 bit-pattern 比对 (恒等变换);其余走混合容差
 *
 * V24 关键约定 (隐性约定显性化,V23 原则 22):
 *   - ROPE_THETA = 1000000.0f (不是 RoFormer 原论文的 10000)
 *   - Pair: x[k] 与 x[k + head_dim/2] 配对 (NeoX),不是 (2k, 2k+1) (GPT-J)
 *   - sincos_table layout 不变,只改 x 索引
 */

#include "../libgpgpu.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Qwen2 RoPE 常量 (config.json 实证) ──────────────── */

/* Qwen2 全系列 (0.5B / 1.5B / 7B / 72B) config.json 中 rope_theta = 1000000.0
 * 不是 RoFormer 原论文的 10000.0。改这个数会让 high token_pos 下 attention
 * 完全错位 但 low pos 几乎看不出。属于 V23 原则 22 同款隐性约定,显性化记录。 */
static const float ROPE_THETA = 1000000.0f;

/* ─── 测试矩阵 ─────────────────────────────────────── */

typedef struct {
    const char *name;
    uint32_t    num_heads;
    uint32_t    head_dim;
    uint32_t    token_pos;
} TestCase;

static const TestCase test_cases[] = {
    {"baseline_pos0", 1, 64, 0},    /* 恒等变换,bit-perfect 判据 */
    {"single_pos1", 1, 64, 1},      /* 单 head 真实非零旋转 */
    {"two_heads_pos5", 2, 64, 5},   /* 模拟 K (num_kv_heads=2) */
    {"qwen_q_pos100", 14, 64, 100}, /* Qwen Q 真实尺度,低 pos */
    {"qwen_q_pos4096", 14, 64, 4096}, /* V24:中规模 pos,验证 sin/cos 精度 */
    {"qwen_q_pos65535", 14, 64,
     65535}, /* V24:接近表上限,argument reduction 高压 */
};
static const int NUM_TESTS = sizeof(test_cases) / sizeof(test_cases[0]);

#define MAX_TEST_POS                                                           \
    65536 /* V24:覆盖到 Qwen2-0.5B max_pos=131072 一半,16MB 表 */
#define TEST_HEAD_DIM 64

/* 容差 (V23 原则 18 应用) */
static const float ABS_TOL = 1e-5f;
static const float REL_TOL = 2e-5f;

/* ─── kernel args 结构 ────────────────────────────── */

typedef struct {
    uint32_t x_off;
    uint32_t sincos_off;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t token_pos;
} RopeArgs;

/* ─── helpers ────────────────────────────────────── */

/* sincos 预计算
 * 布局不变:table[pos * head_dim + k*2 + 0] = cos, +1 = sin
 * 改动:base 10000 → ROPE_THETA = 1e6 */
static float *compute_sincos_table(int max_pos, int head_dim) {
    assert(head_dim > 0 && head_dim % 2 == 0);
    assert(max_pos > 0);

    int    half  = head_dim / 2;
    size_t bytes = (size_t)max_pos * head_dim * sizeof(float);
    float *table = malloc(bytes);
    if (!table)
        return NULL;

    float freq[half];
    for (int k = 0; k < half; k++) {
        freq[k] = 1.0f / powf(ROPE_THETA, (2.0f * (float)k) / (float)head_dim);
    }

    for (int pos = 0; pos < max_pos; pos++) {
        for (int k = 0; k < half; k++) {
            float angle                       = (float)pos * freq[k];
            table[pos * head_dim + k * 2 + 0] = cosf(angle);
            table[pos * head_dim + k * 2 + 1] = sinf(angle);
        }
    }
    return table;
}

/* host reference RoPE,严格平行 device kernel 的 5 条 fp 操作
 * V24:pair 排布改为 NeoX style (k, k + head_dim/2)
 * in-place 修改 x */
static void rope_reference(float *x, const float *sincos_table, int num_heads,
                           int head_dim, int token_pos) {
    int half = head_dim / 2;

    for (int k = 0; k < half; k++) {
        float cos_k = sincos_table[token_pos * head_dim + k * 2 + 0];
        float sin_k = sincos_table[token_pos * head_dim + k * 2 + 1];

        for (int h = 0; h < num_heads; h++) {
            /* V24 关键改动:NeoX pair = (k, k+half),不是 (2k, 2k+1) */
            int   idx_lo = h * head_dim + k;
            int   idx_hi = h * head_dim + k + half;
            float x_lo   = x[idx_lo];
            float x_hi   = x[idx_hi];

            /* 5 条 fp 操作严格匹配 kernel,顺序/类型/fmaf 全部保持 */
            float t1     = x_lo * cos_k;          // fmul.s
            float t2     = x_hi * sin_k;          // fmul.s
            float t3     = x_hi * cos_k;          // fmul.s
            float new_lo = t1 - t2;               // fsub.s
            float new_hi = fmaf(x_lo, sin_k, t3); // fmadd.s,强制单舍入

            x[idx_lo] = new_lo;
            x[idx_hi] = new_hi;
        }
    }
}

/* 单 block 1D dispatch */
static inline void calc_dispatch_1d(uint32_t total_threads, uint32_t grid[3],
                                    uint32_t block[3]) {
    grid[0]  = 1;
    grid[1]  = 1;
    grid[2]  = 1;
    block[0] = total_threads;
    block[1] = 1;
    block[2] = 1;
}

/* sincos helper 自检(不调 device)
 * V24:加 Qwen2 频率谱验证,锁死 ROPE_THETA=1e6 */
static void sincos_self_check(void) {
    float *t = compute_sincos_table(4, TEST_HEAD_DIM);
    assert(t);

    /* 1. pos=0:严格 cos=1, sin=0 — baseline 恒等变换的根基 */
    for (int k = 0; k < TEST_HEAD_DIM / 2; k++) {
        float c = t[0 * TEST_HEAD_DIM + k * 2 + 0];
        float s = t[0 * TEST_HEAD_DIM + k * 2 + 1];
        assert(c == 1.0f);
        assert(s == 0.0f);
    }

    /* 2. pos=1, k=0:freq=1, angle=1, cos≈0.5403, sin≈0.8415
     * 注:k=0 的 freq 与 ROPE_THETA 无关 (1/theta^0 = 1),这条对 10000 和 1e6
     * 都成立 */
    {
        float c = t[1 * TEST_HEAD_DIM + 0 * 2 + 0];
        float s = t[1 * TEST_HEAD_DIM + 0 * 2 + 1];
        assert(fabsf(c - 0.5403023f) < 1e-5f);
        assert(fabsf(s - 0.8414710f) < 1e-5f);
    }

    /* 3. V24 新增:pos=1, k=1, 验证 ROPE_THETA=1e6
     * freq[1] = 1 / 1e6^(2/64) = 1 / 1e6^0.03125 ≈ 1 / 1.5399 ≈ 0.6494
     * angle = 1 * 0.6494 ≈ 0.6494
     * cos(0.6494) ≈ 0.7967, sin(0.6494) ≈ 0.6044
     *
     * 如果 ROPE_THETA 被错改回 10000:
     *   freq[1] = 1 / 10000^(2/64) = 1 / 10000^0.03125 ≈ 1 / 1.3335 ≈ 0.7499
     *   angle ≈ 0.7499, cos ≈ 0.7316 ≠ 0.7967 — 这条 assert 会爆,立刻显形 */
    {
        float c = t[1 * TEST_HEAD_DIM + 1 * 2 + 0];
        float s = t[1 * TEST_HEAD_DIM + 1 * 2 + 1];
        assert(fabsf(c - 0.7967f) < 1e-3f);
        assert(fabsf(s - 0.6044f) < 1e-3f);
    }

    /* 4. pos=1, k=31:freq 极小,cos 几乎 1, sin 几乎 0
     * V24 备注:1e6 下 freq[31] = 1/1e6^(62/64) ≈ 1/7.5e5 ≈ 1.3e-6 (10000 下是
     * ~1e-4) 都是"几乎 0/1",这条 assert 对两个 theta 都过,纯保守 sanity */
    {
        float c = t[1 * TEST_HEAD_DIM + 31 * 2 + 0];
        float s = t[1 * TEST_HEAD_DIM + 31 * 2 + 1];
        assert(c > 0.9999999f);
        assert(s > 0.0f && s < 1e-3f);
    }

    free(t);
    printf("[0] sincos helper self-check PASS (ROPE_THETA=%.0f)\n", ROPE_THETA);
}

/* ─── run_one_test ───────────────────────────────── */

static int run_one_test(gpgpu_ctx *ctx, uint64_t kernel_off,
                        uint64_t sincos_off, const TestCase *tc) {
    int      ret;
    uint32_t num_heads = tc->num_heads;
    uint32_t head_dim  = tc->head_dim;
    uint32_t token_pos = tc->token_pos;

    printf("\n=== [%s] num_heads=%u head_dim=%u token_pos=%u ===\n", tc->name,
           num_heads, head_dim, token_pos);

    /* step 1: 算 x 大小 */
    size_t elems   = (size_t)num_heads * head_dim;
    size_t x_bytes = elems * sizeof(float);

    /* step 2: 生成测试数据(打破对称用 1e-5 偏移,V18 §5) */
    float *h_x_input = malloc(x_bytes);
    for (size_t i = 0; i < elems; i++) {
        h_x_input[i] = ((int)(i % 7) - 3) * 0.1f + 1e-5f;
    }

    /* step 3: reference 画布 */
    float *h_x_ref = malloc(x_bytes);
    memcpy(h_x_ref, h_x_input, x_bytes);

    /* step 4: device alloc */
    uint64_t x_off = gpuMalloc(ctx, x_bytes);
    if (x_off == (uint64_t)-1) {
        fprintf(stderr, "alloc x failed\n");
        free(h_x_input);
        free(h_x_ref);
        return 0;
    }
    uint64_t args_off = gpuMalloc(ctx, sizeof(RopeArgs));
    if (args_off == (uint64_t)-1) {
        fprintf(stderr, "alloc args failed\n");
        gpuFree(ctx, x_off);
        free(h_x_input);
        free(h_x_ref);
        return 0;
    }

    /* step 5: H2D x */
    gpuMemcpy(ctx, x_off, h_x_input, x_bytes, GPU_MEMCPY_H2D);

    /* step 6: H2D args */
    RopeArgs args = {
        .x_off      = (uint32_t)x_off,
        .sincos_off = (uint32_t)sincos_off,
        .num_heads  = num_heads,
        .head_dim   = head_dim,
        .token_pos  = token_pos,
    };
    gpuMemcpy(ctx, args_off, &args, sizeof(args), GPU_MEMCPY_H2D);
    printf("  args: x_off=0x%lx sincos_off=0x%lx num_heads=%u head_dim=%u "
           "token_pos=%u\n",
           x_off, sincos_off, num_heads, head_dim, token_pos);

    /* step 7: dispatch — 单 warp 32 thread */
    uint32_t total = 32;
    uint32_t grid[3], block[3];
    calc_dispatch_1d(total, grid, block);

    /* step 8: launch */
    ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "launch failed: %d\n", ret);
        gpuFree(ctx, x_off);
        gpuFree(ctx, args_off);
        free(h_x_input);
        free(h_x_ref);
        return 0;
    }
    printf("  kernel done\n");

    /* step 9: D2H */
    float *h_x_after = malloc(x_bytes);
    gpuMemcpy(ctx, x_off, h_x_after, x_bytes, GPU_MEMCPY_D2H);

    /* step 10: host reference */
    extern float *g_h_sincos;
    rope_reference(h_x_ref, g_h_sincos, num_heads, head_dim, token_pos);

    /* step 11: 比对 */
    int ok = 1;

    if (token_pos == 0) {
        /* baseline:bit-pattern 严格相等 (恒等变换,与 pair 排布无关) */
        int bit_diff = 0;
        for (size_t i = 0; i < elems; i++) {
            uint32_t a = *(uint32_t *)&h_x_after[i];
            uint32_t b = *(uint32_t *)&h_x_input[i];
            if (a != b)
                bit_diff++;
        }
        printf("  [baseline] bit-diff: %d / %zu\n", bit_diff, elems);
        ok = (bit_diff == 0);

        for (int i = 0; i < 3 && (size_t)i < elems; i++) {
            printf("    [%d] in_bits=0x%08x  out_bits=0x%08x  in=%g out=%g\n",
                   i, *(uint32_t *)&h_x_input[i], *(uint32_t *)&h_x_after[i],
                   h_x_input[i], h_x_after[i]);
        }
    } else {
        /* V24:混合容差 + per-element pass 计数 */
        int   exact_match = 0;
        int   pass_count  = 0;
        float max_err     = -1.0f;
        int   max_idx     = -1;
        float max_err_ref = 0.0f;

        for (size_t i = 0; i < elems; i++) {
            float got = h_x_after[i];
            float ref = h_x_ref[i];
            if (got == ref)
                exact_match++;

            float err = fabsf(got - ref);
            float tol = fmaxf(ABS_TOL, REL_TOL * fabsf(ref));

            if (err <= tol)
                pass_count++;
            if (err > max_err) {
                max_err     = err;
                max_idx     = (int)i;
                max_err_ref = ref;
            }
        }

        float tol_at_max = fmaxf(ABS_TOL, REL_TOL * fabsf(max_err_ref));
        printf("  exact_match: %d / %zu\n", exact_match, elems);
        printf("  pass_count:  %d / %zu (tol = max(%g, %g*|ref|))\n",
               pass_count, elems, ABS_TOL, REL_TOL);
        printf("  max_err = %g at idx %d (ref=%g, tol_there=%g)\n", max_err,
               max_idx, max_err_ref, tol_at_max);

        if (max_err < 0.0f) {
            printf("  WARNING: validation loop did not execute\n");
            ok = 0;
        } else {
            ok = (pass_count == (int)elems);
        }

        /* 头 3 个 bit pattern */
        for (int i = 0; i < 3 && (size_t)i < elems; i++) {
            printf("    [%d] got_bits=0x%08x  ref_bits=0x%08x  got=%g ref=%g\n",
                   i, *(uint32_t *)&h_x_after[i], *(uint32_t *)&h_x_ref[i],
                   h_x_after[i], h_x_ref[i]);
        }

        /* FAIL 时打印更多样本 */
        if (!ok) {
            printf("  --- first 16 samples ---\n");
            for (size_t i = 0; i < elems && i < 16; i++) {
                float err = fabsf(h_x_after[i] - h_x_ref[i]);
                float tol = fmaxf(ABS_TOL, REL_TOL * fabsf(h_x_ref[i]));
                printf("    [%zu] got=%g ref=%g err=%g tol=%g %s\n", i,
                       h_x_after[i], h_x_ref[i], err, tol,
                       err <= tol ? "ok" : "FAIL");
            }
        }
    }

    /* step 12: 清理 */
    gpuFree(ctx, x_off);
    gpuFree(ctx, args_off);
    free(h_x_input);
    free(h_x_ref);
    free(h_x_after);

    /* step 13 */
    printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

float *g_h_sincos = NULL;

/* ─── main ───────────────────────────────────────── */

int main(void) {
    int ret;

    /* phase 0 */
    sincos_self_check();

    /* phase 1.1: gpuInit */
    gpgpu_ctx *ctx = malloc(sizeof(gpgpu_ctx));
    if (!ctx) {
        fprintf(stderr, "malloc ctx failed\n");
        return 1;
    }

    /* V24:sincos 表 16MB,加上 kernel/args/x 留余量,256MB 仍然够用 */
    ret = gpuInit(ctx, "/dev/gpgpu", 256ULL * 1024 * 1024);
    if (ret < 0) {
        fprintf(stderr, "gpuInit failed: %d\n", ret);
        return 1;
    }
    printf("[1] gpuInit OK\n");

    /* phase 1.2: 读 rope.bin */
    FILE *f = fopen("/tmp/rope.bin", "rb");
    if (!f) {
        perror("fopen /tmp/rope.bin");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t kernel_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    printf("[1.5] rope.bin size = %zu bytes\n", kernel_size);

    uint64_t kernel_off = gpuMalloc(ctx, kernel_size);
    if (kernel_off == (uint64_t)-1) {
        fprintf(stderr, "alloc kernel failed\n");
        return 1;
    }

    uint8_t *kbuf = malloc(kernel_size);
    fread(kbuf, 1, kernel_size, f);
    fclose(f);
    gpuMemcpy(ctx, kernel_off, kbuf, kernel_size, GPU_MEMCPY_H2D);
    free(kbuf);
    printf("[2] kernel uploaded, kernel_off=0x%lx\n", kernel_off);

    /* phase 1.3 + 1.4: 预算 sincos + 上传 */
    g_h_sincos = compute_sincos_table(MAX_TEST_POS, TEST_HEAD_DIM);
    if (!g_h_sincos) {
        fprintf(stderr, "compute_sincos_table failed (16MB)\n");
        return 1;
    }

    size_t sincos_bytes = (size_t)MAX_TEST_POS * TEST_HEAD_DIM * sizeof(float);
    uint64_t sincos_off = gpuMalloc(ctx, sincos_bytes);
    if (sincos_off == (uint64_t)-1) {
        fprintf(stderr, "alloc sincos failed (need %zu bytes)\n", sincos_bytes);
        return 1;
    }
    gpuMemcpy(ctx, sincos_off, g_h_sincos, sincos_bytes, GPU_MEMCPY_H2D);
    printf("[3] sincos uploaded, sincos_off=0x%lx, size=%zu bytes (%.1f MB)\n",
           sincos_off, sincos_bytes, sincos_bytes / 1024.0 / 1024.0);

    /* phase 2 */
    int total_pass = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        int ok = run_one_test(ctx, kernel_off, sincos_off, &test_cases[i]);
        if (!ok) {
            printf("\n*** FAIL at test case [%s], stopping ***\n",
                   test_cases[i].name);
            break;
        }
        total_pass++;
    }

    /* phase 3 */
    free(g_h_sincos);
    gpuDestroy(ctx);
    free(ctx);

    printf("\n%d / %d cases passed\n", total_pass, NUM_TESTS);
    return total_pass == NUM_TESTS ? 0 : 1;
}