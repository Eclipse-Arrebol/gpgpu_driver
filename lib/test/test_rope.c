/* test_rope.c — Milestone 3.5 第二关 RoPE 单算子测试
 *
 * 设计要点:
 *   - 单 warp 32 thread 派发(每 lane 一对,head_dim/2 = 32 = lane 数)
 *   - in-place:device 端 x_off 既是输入又是输出
 *   - host 端 3 份 buffer:input(原始) / ref(reference 算的) / after(D2H 读回)
 *   - baseline (token_pos=0) 走 bit-pattern 比对(恒等变换);其余走容差 + 三指标
 *   - 所有浮点字面量 f 后缀;reference 里 fmaf 强制单舍入,匹配 device fmadd.s
 */

#include "../libgpgpu.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── 测试矩阵:从这里改 ─────────────────────────── */

typedef struct {
    const char *name;
    uint32_t    num_heads;
    uint32_t    head_dim;
    uint32_t    token_pos;
} TestCase;

static const TestCase test_cases[] = {
    {"baseline_pos0", 1, 64, 0},    // 恒等变换,bit-perfect 判据
    {"single_pos1", 1, 64, 1},      // 单 head 真实非零旋转
    {"two_heads_pos5", 2, 64, 5},   // 模拟 K
    {"qwen_q_pos100", 14, 64, 100}, // Qwen Q 真实尺度
};
static const int NUM_TESTS = sizeof(test_cases) / sizeof(test_cases[0]);

#define MAX_TEST_POS 256 // sincos 预算 256 个 pos,所有 case 共享
#define TEST_HEAD_DIM 64 // 当前所有 case 都是 64

/* ─── kernel args 结构(必须和 kernel 端的 lw 偏移一致)──── */

typedef struct {
    uint32_t x_off;
    uint32_t sincos_off;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t token_pos;
} RopeArgs;

/* ─── helpers ────────────────────────────────────── */

/* sincos 预计算
 * 布局:行主序,table[pos * head_dim + k*2 + 0] = cos
 *                  table[pos * head_dim + k*2 + 1] = sin
 */
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
        freq[k] = 1.0f / powf(10000.0f, (2.0f * (float)k) / (float)head_dim);
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
 * in-place 修改 x */
static void rope_reference(float *x, const float *sincos_table, int num_heads,
                           int head_dim, int token_pos) {
    int half = head_dim / 2;

    for (int k = 0; k < half; k++) {
        float cos_k = sincos_table[token_pos * head_dim + k * 2 + 0];
        float sin_k = sincos_table[token_pos * head_dim + k * 2 + 1];

        for (int h = 0; h < num_heads; h++) {
            int   idx_lo = h * head_dim + 2 * k;
            int   idx_hi = h * head_dim + 2 * k + 1;
            float x_lo   = x[idx_lo];
            float x_hi   = x[idx_hi];

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

/* 单 block 1D dispatch,沿用 vmul 模板 */
static inline void calc_dispatch_1d(uint32_t total_threads, uint32_t grid[3],
                                    uint32_t block[3]) {
    grid[0]  = 1;
    grid[1]  = 1;
    grid[2]  = 1;
    block[0] = total_threads;
    block[1] = 1;
    block[2] = 1;
}

/* sincos helper 自检:不调 device,纯 host 行为验证 */
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

    /* 2. pos=1, k=0:freq=1, angle=1,cos≈0.5403, sin≈0.8415 */
    {
        float c = t[1 * TEST_HEAD_DIM + 0 * 2 + 0];
        float s = t[1 * TEST_HEAD_DIM + 0 * 2 + 1];
        assert(fabsf(c - 0.5403023f) < 1e-5f);
        assert(fabsf(s - 0.8414710f) < 1e-5f);
    }

    /* 3. pos=1, k=31:freq 极小,cos 几乎 1, sin 几乎 0 */
    {
        float c = t[1 * TEST_HEAD_DIM + 31 * 2 + 0];
        float s = t[1 * TEST_HEAD_DIM + 31 * 2 + 1];
        assert(c > 0.9999999f);
        assert(s > 0.0f && s < 1e-3f);
    }

    free(t);
    printf("[0] sincos helper self-check PASS\n");
}

/* ─── run_one_test ───────────────────────────────── */

/* 返回:1 = PASS, 0 = FAIL */
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

    /* step 2: 生成测试数据
     * 打破对称的偏移用 1e-5f(V18 §5:0.1 量级数据上 1e-5 起步,1e-7 会被吃) */
    float *h_x_input = malloc(x_bytes);
    for (size_t i = 0; i < elems; i++) {
        h_x_input[i] = ((int)(i % 7) - 3) * 0.1f + 1e-5f;
    }

    /* step 3: reference 画布,从 h_x_input 复制 */
    float *h_x_ref = malloc(x_bytes);
    memcpy(h_x_ref, h_x_input, x_bytes);

    /* step 4: device alloc x_off + args_off */
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
        free(h_x_input);
        free(h_x_ref);
        return 0;
    }

    /* step 5: 上传 x(in-place 输入) */
    gpuMemcpy(ctx, x_off, h_x_input, x_bytes, GPU_MEMCPY_H2D);

    /* step 6: 准备并上传 args */
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

    /* step 7: dispatch — 单 warp 32 thread,每 lane 一对 */
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

    /* step 9: D2H 读回 */
    float *h_x_after = malloc(x_bytes);
    gpuMemcpy(ctx, x_off, h_x_after, x_bytes, GPU_MEMCPY_D2H);

    /* step 10: host reference 在 h_x_ref 上 in-place 修改 */
    /* 注意:sincos 表 device 上是 sincos_off,host 端跑 reference 用的是
     * 我们最初算出来的那张 host 表;但 run_one_test 没拿到 host 表指针。
     * 解法:把 host 表指针通过参数传下来。下面 main 里会传。
     * 这里假设 ctx 里有方法或者我们需要补参数。看 step 11 注释。
     */

    /* —— 为了简洁,把 host sincos 表通过 ctx 之外的全局或额外参数传进来。
     *    本骨架先把它作为 static 全局指针,main 里赋值。详见下方 g_h_sincos。 */
    extern float *g_h_sincos;
    rope_reference(h_x_ref, g_h_sincos, num_heads, head_dim, token_pos);

    /* step 11: 比对 */
    int ok = 1;

    if (token_pos == 0) {
        /* baseline:bit-pattern 严格相等(恒等变换) */
        int bit_diff = 0;
        for (size_t i = 0; i < elems; i++) {
            uint32_t a = *(uint32_t *)&h_x_after[i];
            uint32_t b = *(uint32_t *)&h_x_input[i];
            if (a != b)
                bit_diff++;
        }
        printf("  [baseline] bit-diff: %d / %zu\n", bit_diff, elems);
        ok = (bit_diff == 0);

        /* baseline 头 3 个 bit pattern 打印 */
        for (int i = 0; i < 3 && (size_t)i < elems; i++) {
            printf("    [%d] in_bits=0x%08x  out_bits=0x%08x  in=%g out=%g\n",
                   i, *(uint32_t *)&h_x_input[i], *(uint32_t *)&h_x_after[i],
                   h_x_input[i], h_x_after[i]);
        }
    } else {
        /* 常规:三指标(exact_match + max_err + 头尾样本) */
        int   exact_match = 0;
        float max_err     = -1.0f;
        int   max_idx     = -1;
        for (size_t i = 0; i < elems; i++) {
            if (h_x_after[i] == h_x_ref[i])
                exact_match++;
            float err = fabsf(h_x_after[i] - h_x_ref[i]);
            if (err > max_err) {
                max_err = err;
                max_idx = (int)i;
            }
        }
        printf("  exact_match: %d / %zu\n", exact_match, elems);
        printf("  max_err = %g at idx %d\n", max_err, max_idx);
        if (max_err < 0.0f) {
            printf("  WARNING: validation loop did not execute\n");
        }
        ok = (max_err < 1e-3f);

        /* 头 3 个 bit pattern */
        for (int i = 0; i < 3 && (size_t)i < elems; i++) {
            printf("    [%d] got_bits=0x%08x  ref_bits=0x%08x  got=%g ref=%g\n",
                   i, *(uint32_t *)&h_x_after[i], *(uint32_t *)&h_x_ref[i],
                   h_x_after[i], h_x_ref[i]);
        }

        /* FAIL 时打印更多样本协助 debug */
        if (!ok) {
            printf("  --- first 16 samples ---\n");
            for (size_t i = 0; i < elems && i < 16; i++) {
                printf("    [%zu] got=%g ref=%g err=%g\n", i, h_x_after[i],
                       h_x_ref[i], fabsf(h_x_after[i] - h_x_ref[i]));
            }
        }
    }

    /* step 12: 清理(sincos_off 和 kernel_off 跨 case 复用,不释放) */
    gpuFree(ctx, x_off);
    gpuFree(ctx, args_off);
    free(h_x_input);
    free(h_x_ref);
    free(h_x_after);

    /* step 13: 返回 */
    printf("  %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

/* host sincos 表的全局指针(给 run_one_test 用) */
float *g_h_sincos = NULL;

/* ─── main ───────────────────────────────────────── */

int main(void) {
    int ret;

    /* phase 0: helper 自检(不调 device) */
    sincos_self_check();

    /* phase 1.1: gpuInit */
    gpgpu_ctx *ctx = malloc(sizeof(gpgpu_ctx));
    if (!ctx) {
        fprintf(stderr, "malloc ctx failed\n");
        return 1;
    }

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
        fprintf(stderr, "compute_sincos_table failed\n");
        return 1;
    }

    size_t sincos_bytes = (size_t)MAX_TEST_POS * TEST_HEAD_DIM * sizeof(float);
    uint64_t sincos_off = gpuMalloc(ctx, sincos_bytes);
    if (sincos_off == (uint64_t)-1) {
        fprintf(stderr, "alloc sincos failed\n");
        return 1;
    }
    gpuMemcpy(ctx, sincos_off, g_h_sincos, sincos_bytes, GPU_MEMCPY_H2D);
    printf("[3] sincos uploaded, sincos_off=0x%lx, size=%zu bytes\n",
           sincos_off, sincos_bytes);

    /* phase 2: 逐档跑 */
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

    /* phase 3: 清理 */
    free(g_h_sincos);
    gpuDestroy(ctx);
    free(ctx);

    printf("\n%d / %d cases passed\n", total_pass, NUM_TESTS);
    return total_pass == NUM_TESTS ? 0 : 1;
}