/*
 * test_argmax.c  (V22 重写版,Milestone 3.6.1)
 *
 * 取代旧版退化 reference (全 -INF) 的本质未测问题。
 * 覆盖矩阵 (参照 V22 原则 11/12/15 + test_rope.c 模板):
 *   - baseline (V=32, 无空 lane 无残余)
 *   - empty_lane_small (V=15, 17 个空 lane)
 *   - tail_remainder_33 (V=33, 残余 1, 最大值放在残余位)
 *   - tail_remainder_40 (V=40, 残余 8, 最大值放在尾部)
 *   - multi_round (V=64, 多轮无残余)
 *   - multi_round_tail (V=77, 多轮 + 残余)
 *   - tie_break (V=32, 两个并列最大值, 保留小 idx)
 *   - tie_break_empty_lane (V=15, tie + 空 lane 双重压力)
 *   - large_v (V=1000, 接近真实 LM head 规模的小型版本)
 *
 * 数据 pattern: L[i] = sinf(i * 0.7) * 10 + i * 0.01,每个 i 唯一,
 * 然后把唯一最大值显式放在 max_idx 处 (L[max_idx] = 100.0)。
 * 这样 reference 输出确定为 max_idx,且 max_idx 可控落点。
 *
 * 失败诊断: 打印 device_idx/ref_idx 双方对应的 logits 值,
 * 以及 max_idx 周围的 ±2 邻域,辅助定位 bug 类型 (空 lane / 残余 / tie-break)。
 */

#include "../libgpgpu.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SENTINEL_OUT 0xDEADBEEFu
#define V_MAX 2048 /* 单 case logits 最大长度,large_v=1000 够用 */

struct argmax_args {
    uint32_t logits_base;
    uint32_t V;
    uint32_t out_base;
    uint32_t debug_base; /* per-lane ft2 bit-pattern dump (32 * 4 bytes) */
};

typedef struct {
    const char *name;
    uint32_t    V;
    int      max_idx; /* 唯一最大值放置位置,-1 表示 tie-break case */
    int      tie_idx_a;    /* tie-break: 第一个最大值位置 */
    int      tie_idx_b;    /* tie-break: 第二个最大值位置 */
    uint32_t expected_idx; /* reference 输出 */
} TestCase;

static const TestCase test_cases[] = {
    {"baseline_V32", 32, 17, -1, -1, 17},
    {"empty_lane_V15", 15, 14, -1, -1, 14},
    {"empty_lane_V15_low", 15, 0, -1, -1, 0},
    {"tail_remainder_V33", 33, 32, -1, -1, 32},
    {"tail_remainder_V40", 40, 39, -1, -1, 39},
    {"multi_round_V64", 64, 50, -1, -1, 50},
    {"multi_round_tail_V77", 77, 76, -1, -1, 76},
    {"tie_break_V32", 32, -1, 3, 20, 3},
    {"tie_break_empty_V15", 15, -1, 2, 11, 2},
    {"large_v_V1000", 1000, 777, -1, -1, 777},
    {"large_v_tail", 1000, 999, -1, -1, 999},
};

#define NUM_CASES (sizeof(test_cases) / sizeof(test_cases[0]))

/* ------------------------------------------------------------------ */
/* host reference: 严格平行 argmax 语义,tie-break 保留小 idx          */
/* ------------------------------------------------------------------ */
static uint32_t argmax_reference(const float *L, uint32_t V) {
    uint32_t best_idx = 0;
    float    best_val = L[0];
    for (uint32_t i = 1; i < V; i++) {
        if (L[i] > best_val) {
            best_val = L[i];
            best_idx = i;
        }
        /* 严格 >,相等时不更新 → 自动保留小 idx */
    }
    return best_idx;
}

/* ------------------------------------------------------------------ */
/* 填 logits: 唯一最大值或 tie-break pair                             */
/* ------------------------------------------------------------------ */
static void fill_logits(float *L, const TestCase *tc) {
    /* 基础 pattern: 每个 i 唯一,数值范围 [-10, 10] */
    for (uint32_t i = 0; i < tc->V; i++) {
        L[i] = sinf((float)i * 0.7f) * 10.0f + (float)i * 0.01f;
    }

    if (tc->max_idx >= 0) {
        /* 唯一最大值 case */
        L[tc->max_idx] = 100.0f;
    } else {
        /* tie-break case: 两个并列最大值 */
        assert(tc->tie_idx_a >= 0 && tc->tie_idx_b >= 0);
        assert(tc->tie_idx_a < tc->tie_idx_b); /* 保证 expected 是小的那个 */
        L[tc->tie_idx_a] = 50.0f;
        L[tc->tie_idx_b] = 50.0f;
    }
}

/* ------------------------------------------------------------------ */
/* helper 自检: reference 在受控输入下行为正确                        */
/* ------------------------------------------------------------------ */
static void reference_self_check(void) {
    /* sanity 1: 单调递增,最大值在末位 */
    float L1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    assert(argmax_reference(L1, 8) == 7);

    /* sanity 2: 单调递减,最大值在首位 */
    float L2[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    assert(argmax_reference(L2, 8) == 0);

    /* sanity 3: tie-break 保留小 idx */
    float L3[8] = {1, 5, 3, 5, 2, 5, 1, 1};
    assert(argmax_reference(L3, 8) == 1);

    /* sanity 4: 最大值在中间 */
    float L4[5] = {1, 2, 99, 3, 4};
    assert(argmax_reference(L4, 5) == 2);

    printf("[self-check] reference passed 4 sanity tests\n");
}

/* ------------------------------------------------------------------ */
/* 失败诊断: 打印 device/ref 双方对应值 + max_idx 邻域                */
/* ------------------------------------------------------------------ */
static void diagnose_failure(const TestCase *tc, const float *L,
                             uint32_t device_idx, uint32_t ref_idx) {
    printf("    FAIL DIAGNOSTIC for case '%s' (V=%u):\n", tc->name, tc->V);

    printf("      device_idx = %u", device_idx);
    if (device_idx < tc->V) {
        printf("  L[%u] = %.6f", device_idx, L[device_idx]);
    } else {
        printf("  (OUT OF RANGE,V=%u)", tc->V);
    }
    printf("\n");

    printf("      ref_idx    = %u  L[%u] = %.6f\n", ref_idx, ref_idx,
           L[ref_idx]);

    /* 打印 expected 位置周围 ±2 邻域 */
    int center = (int)tc->expected_idx;
    int lo     = center - 2;
    if (lo < 0)
        lo = 0;
    int hi = center + 2;
    if (hi >= (int)tc->V)
        hi = (int)tc->V - 1;
    printf("      neighborhood around expected:\n");
    for (int i = lo; i <= hi; i++) {
        printf("        L[%d] = %.6f%s\n", i, L[i],
               (i == center) ? "  <-- expected max" : "");
    }

    /* bug 分类提示 */
    if (device_idx == 0 && ref_idx != 0) {
        printf("      hint: device returned 0,可能是空 lane 残留 idx=0 在 "
               "reduce 中获胜\n");
    }
    if (device_idx < (tc->V & ~31u) && ref_idx >= (tc->V & ~31u)) {
        printf("      hint: ref 在尾部残余区 (V & 31 = %u),device 没看到 → "
               "怀疑 srli t3,a1,5 后无残余处理\n",
               tc->V & 31u);
    }
}

/* ------------------------------------------------------------------ */
/* 单 case 执行                                                       */
/* ------------------------------------------------------------------ */
static int run_one_test(gpgpu_ctx *ctx, uint64_t kernel_off, uint64_t args_off,
                        const TestCase *tc) {
    printf("\n--- case: %s (V=%u, expected_idx=%u) ---\n", tc->name, tc->V,
           tc->expected_idx);

    size_t logits_bytes = (size_t)tc->V * 4;

    /* alloc device buffer (每 case 独立,避免脏数据残留) */
    uint64_t L_off = gpuMalloc(ctx, logits_bytes);
    uint64_t O_off = gpuMalloc(ctx, 4);
    uint64_t D_off =
        gpuMalloc(ctx, 32 * 8 * 4); /* 32 lanes * 8 slots * 4 bytes */
    if (L_off == (uint64_t)-1 || O_off == (uint64_t)-1 ||
        D_off == (uint64_t)-1) {
        printf("    gpuMalloc failed\n");
        return 0;
    }

    /* host buffer */
    float   *L     = malloc(logits_bytes);
    uint32_t O_dev = SENTINEL_OUT;
    uint32_t dbg[32 * 8];
    for (int i = 0; i < 32 * 8; i++)
        dbg[i] = 0xDEADBEEFu;
    assert(L);

    /* 填数据 */
    fill_logits(L, tc);

    /* host reference */
    uint32_t ref_idx = argmax_reference(L, tc->V);
    if (ref_idx != tc->expected_idx) {
        printf("    [BUG IN TEST] ref_idx=%u != expected=%u — fill_logits "
               "写错了\n",
               ref_idx, tc->expected_idx);
        free(L);
        return 0;
    }

    /* 写哨兵到 device out / debug (检测 kernel 漏写) */
    gpuMemcpy(ctx, O_off, &O_dev, 4, 0);
    gpuMemcpy(ctx, D_off, dbg, 32 * 8 * 4, 0);

    /* 上传 logits */
    gpuMemcpy(ctx, L_off, L, logits_bytes, 0);

    /* 上传 args */
    struct argmax_args args = {
        .logits_base = (uint32_t)L_off,
        .V           = tc->V,
        .out_base    = (uint32_t)O_off,
        .debug_base  = (uint32_t)D_off,
    };
    gpuMemcpy(ctx, args_off, &args, sizeof(args), 0);

    /* launch */
    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    int      ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        printf("    gpuLaunchKernel failed: %d\n", ret);
        free(L);
        return 0;
    }

    /* 取回结果 */
    gpuMemcpy(ctx, O_off, &O_dev, 4, 1);
    gpuMemcpy(ctx, D_off, dbg, 32 * 8 * 4, 1);

    /* 哨兵检查 */
    if (O_dev == SENTINEL_OUT) {
        printf("    FAIL: output unchanged (sentinel 0xDEADBEEF intact) → "
               "kernel 没写 out_base\n");
        free(L);
        return 0;
    }

    /* 判定 */
    int pass = (O_dev == tc->expected_idx);
    printf("    device_idx=%u  ref_idx=%u  expected=%u  %s\n", O_dev, ref_idx,
           tc->expected_idx, pass ? "PASS" : "FAIL");

    if (!pass) {
        diagnose_failure(tc, L, O_dev, ref_idx);

        /*
         * dump 布局 (kernel 端按下面槽位写,每 lane 占 8 个 uint32):
         *   slot 0: phaseA 之后 ft2 bit pattern (= initial s0 as float)
         *   slot 1: 第 1 步 reduce 后 ft2 (XOR 16)
         *   slot 2: 第 2 步 reduce 后 ft2 (XOR 8)
         *   slot 3: 第 3 步 reduce 后 ft2 (XOR 4)
         *   slot 4: 第 4 步 reduce 后 ft2 (XOR 2)
         *   slot 5: 第 5 步 reduce 后 ft2 (XOR 1)
         *   slot 6: 最终 s0 (integer)
         *   slot 7: reserved
         */
        printf(
            "      DEBUG DUMP (per-lane ft2 across reduce steps, all hex):\n");
        printf("        lane  init     after16  after8   after4   after2   "
               "after1   final_s0\n");
        for (int lane = 0; lane < 32; lane++) {
            uint32_t *row = &dbg[lane * 8];
            printf("        %2d   ", lane);
            for (int slot = 0; slot < 7; slot++) {
                printf(" %08x", row[slot]);
            }
            printf("\n");
        }
    }

    free(L);
    return pass;
}

/* ------------------------------------------------------------------ */
int main(void) {
    int ret;
    printf("=== test_argmax (V22 重写) ===\n");

    /* helper 自检 */
    reference_self_check();

    /* ---------- ctx 初始化 ---------- */
    gpgpu_ctx *ctx = malloc(sizeof(gpgpu_ctx));
    if (!ctx) {
        fprintf(stderr, "malloc ctx failed\n");
        return 1;
    }

    ret = gpuInit(ctx, "/dev/gpgpu", 64ULL * 1024 * 1024);
    if (ret < 0) {
        fprintf(stderr, "gpuInit failed: %d\n", ret);
        return 1;
    }
    printf("[init] gpuInit OK\n");

    /* ---------- 加载 kernel binary ---------- */
    FILE *f = fopen("/tmp/argmax.bin", "rb");
    if (!f) {
        fprintf(stderr, "open argmax.bin failed\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t ksize = ftell(f);
    rewind(f);
    uint8_t *kbuf = malloc(ksize);
    fread(kbuf, 1, ksize, f);
    fclose(f);

    uint64_t kernel_off = gpuMalloc(ctx, ksize);
    uint64_t args_off   = gpuMalloc(ctx, sizeof(struct argmax_args));
    if (kernel_off == (uint64_t)-1 || args_off == (uint64_t)-1) {
        fprintf(stderr, "gpuMalloc kernel/args failed\n");
        return 1;
    }
    gpuMemcpy(ctx, kernel_off, kbuf, ksize, 0);
    free(kbuf);
    printf("[init] kernel uploaded (%zu bytes)\n", ksize);

    /* ---------- 跑所有 case ---------- */
    int total_pass = 0;
    int total      = (int)NUM_CASES;
    for (int i = 0; i < total; i++) {
        if (run_one_test(ctx, kernel_off, args_off, &test_cases[i])) {
            total_pass++;
        }
    }

    /* ---------- summary ---------- */
    printf("\n=== summary: %d / %d cases passed ===\n", total_pass, total);

    /* 关键档独立报告,方便看 bug 模式 */
    printf("\nbreakdown by bug class (按 V22 方法论分类):\n");
    printf("  baseline (V=32):              查看 baseline_V32 是否 PASS\n");
    printf("  空 lane bug (V=15):           查看 empty_lane_V15* 是否 PASS\n");
    printf("  尾部残余 bug (V & 31 != 0):   查看 tail_remainder_V33/V40 是否 "
           "PASS\n");
    printf("  tie-break bug:                查看 tie_break_V32 是否 PASS\n");
    printf(
        "  组合压力 (空 lane + tie):     查看 tie_break_empty_V15 是否 PASS\n");
    printf("  大 V (近真实规模):            查看 large_v_V1000* 是否 PASS\n");

    gpuDestroy(ctx);
    free(ctx);
    return (total_pass == total) ? 0 : 1;
}