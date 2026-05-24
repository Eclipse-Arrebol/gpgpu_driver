/* test_embedding.c — Milestone 3.6.6 Embedding lookup 单算子测试 (V24)
 *
 * V24 改动 (相对原版):
 *   #1 main 主体抽成 run_one_test(V, token_id, name),支持多 case
 *   #2 加 case:V=151936 真规模 (qwen2-0.5B vocab),覆盖边界 token_id
 *   #3 加 tie_word_embeddings 隐性约定注释 (顶部)
 *   #4 (kernel 端) 删末尾退出约定疑问注释
 *   #5 候选原则 21:bit-pattern 编码 = diagnostic 之锚
 *
 * Qwen2-0.5B config (config.json 实证):
 *   vocab_size       = 151936
 *   hidden_size      = 896
 *   tie_word_embeddings = true
 *
 * tie_word_embeddings 隐性约定 (3.7 串联前显性化,V23 原则 22 同款):
 *   embedding table[V=151936][D=896] 与 LM head gemm 权重 W[K=896, N=151936]
 *   在真实推理时共享同一份 VRAM 存储,但索引方式不同:
 *     - embedding kernel: table[token_id * D + d]    (token_id 作外侧索引)
 *     - LM head gemm:    W[d * V + token_id]         (token_id 作内侧索引)
 *   3.7 串联时需决定:alloc 一份共享 + kernel 索引适配,还是 alloc 两份。
 *   当前 3.6.6 单算子测试假设 embedding layout,与 LM head 无关。
 *
 * 测试方法:
 *   - bit-pattern exact match (无容差,V23 原则 18 应用的反面:embedding 没有
 *     数值路径,任何容差都是退化判据)
 *   - table 内容编码 (token_id << 16) | d,fail 时高/低 16 位直接揭示错误形态
 *   - 双哨兵 0xDEADBEEF (H2D) + 0xCAFEBABE (D2H 前重新污染),防止旧值伪造
 *   - 多 case:小规模 sanity → 真规模边界,覆盖 32-bit 算术高位 bit pattern 切换
 */

#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

const uint32_t D = 896; /* Qwen2-0.5B hidden_size */

struct embedding_args {
    uint32_t table_base;
    uint32_t token_id_buf;
    uint32_t output_base;
    uint32_t D;
    uint32_t elems_per_lane;
};

/* ─── case 矩阵 ────────────────────────────────── */

typedef struct {
    const char *name;
    uint32_t    V;
    uint32_t    token_id;
} TestCase;

static const TestCase test_cases[] = {
    /* case 1: 小规模 sanity (原 V22 case 保留),快速烟测 */
    {"small_vocab", 2048, 2047},

    /* case 2: 真规模边界 token_id=V-1
     * row_base = 151935 * 896 * 4 = 544,535,040 字节 ≈ 519 MB
     * 仍在 32-bit signed [0, 2^31-1] 范围,kernel 算术安全
     * 但比 small_vocab 的 ~7MB 高两个数量级,验证大 row_base 路径 */
    {"qwen_vocab_max", 151936, 151935},

    /* case 3: 真规模中间值,避免边界巧合
     * token_id=100000 时 row_base ≈ 342 MB,与 case 2 高位 bit pattern 不同 */
    {"qwen_vocab_100k", 151936, 100000},
};
static const int NUM_TESTS = sizeof(test_cases) / sizeof(test_cases[0]);

/* ─── run_one_test ────────────────────────────── */

/* 返回:1 = PASS, 0 = FAIL
 * 选项 A:每个 case 独立 alloc/上传/free,case 间完全无污染 */
static int run_one_test(gpgpu_ctx *ctx, uint64_t kernel_off,
                        const TestCase *tc) {
    int      ret;
    uint32_t V        = tc->V;
    uint32_t TOKEN_ID = tc->token_id;

    printf("\n=== [%s] V=%u token_id=%u ===\n", tc->name, V, TOKEN_ID);

    /* 算 buffer 大小(size_t 防止 V*D*4 在 uint32 算术下溢出) */
    size_t table_bytes = (size_t)V * (size_t)D * sizeof(float);
    size_t out_bytes   = (size_t)D * sizeof(float);
    printf("  table_bytes = %zu (%.1f MB), out_bytes = %zu\n", table_bytes,
           table_bytes / (1024.0 * 1024.0), out_bytes);

    /* host 端 malloc */
    uint32_t *table_u32 = malloc(table_bytes);
    if (!table_u32) {
        fprintf(stderr, "  malloc table_u32 (%zu bytes) failed\n", table_bytes);
        return 0;
    }
    uint32_t *expected_u32 = malloc(out_bytes);
    uint32_t *O_u32        = malloc(out_bytes);
    if (!expected_u32 || !O_u32) {
        fprintf(stderr, "  malloc expected/O failed\n");
        free(table_u32);
        free(expected_u32);
        free(O_u32);
        return 0;
    }

    /* 准备 table —— bit pattern 编码,完全避开 fp 算术
     * table[t][d] = (t << 16) | d
     * 高 16 位 = token_id,低 16 位 = d,fail 时直接看出错位形态 */
    for (uint32_t t = 0; t < V; t++) {
        for (uint32_t d = 0; d < D; d++) {
            table_u32[(size_t)t * D + d] = (t << 16) | d;
        }
    }

    /* 准备 expected */
    for (uint32_t d = 0; d < D; d++) {
        expected_u32[d] = (TOKEN_ID << 16) | d;
    }

    /* host O 哨兵化 (V15 §4 教训) */
    for (uint32_t d = 0; d < D; d++) {
        O_u32[d] = 0xDEADBEEF;
    }

    /* device alloc */
    uint64_t args_off  = gpuMalloc(ctx, sizeof(struct embedding_args));
    uint64_t Table_off = gpuMalloc(ctx, table_bytes);
    uint64_t Id_off    = gpuMalloc(ctx, 4);
    uint64_t O_off     = gpuMalloc(ctx, out_bytes);

    if (args_off == (uint64_t)-1 || Table_off == (uint64_t)-1 ||
        Id_off == (uint64_t)-1 || O_off == (uint64_t)-1) {
        fprintf(stderr, "  gpuMalloc failed (table=%zu MB)\n",
                table_bytes / (1024 * 1024));
        free(table_u32);
        free(expected_u32);
        free(O_u32);
        return 0;
    }
    printf("  args_off=0x%lx Table_off=0x%lx Id_off=0x%lx O_off=0x%lx\n",
           args_off, Table_off, Id_off, O_off);

    /* args */
    struct embedding_args args = {
        .table_base     = (uint32_t)Table_off,
        .token_id_buf   = (uint32_t)Id_off,
        .output_base    = (uint32_t)O_off,
        .D              = D,
        .elems_per_lane = D / 32,
    };

    /* H2D */
    gpuMemcpy(ctx, args_off, &args, sizeof(args), 0);
    printf("  uploading table (%.1f MB)...\n", table_bytes / (1024.0 * 1024.0));
    gpuMemcpy(ctx, Table_off, table_u32, table_bytes, 0);
    gpuMemcpy(ctx, Id_off, &TOKEN_ID, 4, 0);
    gpuMemcpy(ctx, O_off, O_u32, out_bytes, 0); /* 哨兵 H2D */
    printf("  H2D done\n");

    /* launch */
    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "  gpuLaunchKernel failed: %d\n", ret);
        gpuFree(ctx, args_off);
        gpuFree(ctx, Table_off);
        gpuFree(ctx, Id_off);
        gpuFree(ctx, O_off);
        free(table_u32);
        free(expected_u32);
        free(O_u32);
        return 0;
    }
    printf("  kernel done\n");

    /* 重新污染 host O,防止旧值伪造 */
    for (uint32_t d = 0; d < D; d++) {
        O_u32[d] = 0xCAFEBABE;
    }

    /* D2H */
    gpuMemcpy(ctx, O_off, O_u32, out_bytes, 1);
    printf("  D2H done\n");

    /* 哨兵扫描:device 是否漏写 */
    int sentinel_hit = 0;
    for (uint32_t d = 0; d < D; d++) {
        if (O_u32[d] == 0xDEADBEEF) {
            if (sentinel_hit < 10) {
                printf("  SENTINEL: lane %u (d=%u) did not write\n", d % 32, d);
            }
            sentinel_hit++;
        }
    }
    if (sentinel_hit) {
        printf("  FATAL: %d positions not written by device\n", sentinel_hit);
        gpuFree(ctx, args_off);
        gpuFree(ctx, Table_off);
        gpuFree(ctx, Id_off);
        gpuFree(ctx, O_off);
        free(table_u32);
        free(expected_u32);
        free(O_u32);
        return 0;
    }

    /* bit-pattern exact match */
    int exact_match  = 0;
    int fail         = 0;
    int first_fail_d = -1;
    for (uint32_t d = 0; d < D; d++) {
        if (O_u32[d] == expected_u32[d]) {
            exact_match++;
        } else {
            if (first_fail_d < 0)
                first_fail_d = (int)d;
            if (fail < 10) {
                printf("  FAIL d=%u got=0x%08x expected=0x%08x  "
                       "(lane=%u, k=%u)\n",
                       d, O_u32[d], expected_u32[d], d % 32, d / 32);
            }
            fail++;
        }
    }
    printf("  exact match: %d / %u, fail = %d\n", exact_match, D, fail);

    /* 头 4 + 尾 4 样本 */
    printf("  samples:\n");
    for (uint32_t d = 0; d < 4; d++) {
        printf("    O[%u]: got=0x%08x  expected=0x%08x\n", d, O_u32[d],
               expected_u32[d]);
    }
    printf("    ...\n");
    for (uint32_t d = D - 4; d < D; d++) {
        printf("    O[%u]: got=0x%08x  expected=0x%08x\n", d, O_u32[d],
               expected_u32[d]);
    }

    /* fail 时诊断 (原则 21:bit-pattern 编码 = diagnostic 之锚) */
    if (fail > 0 && first_fail_d >= 0) {
        uint32_t got = O_u32[first_fail_d];
        uint32_t exp = expected_u32[first_fail_d];
        printf("\n  DIAGNOSTIC: first fail at d=%d\n", first_fail_d);
        printf("    got = 0x%08x  -> 高16位=%u (作为 token_id)  "
               "低16位=%u (作为 d)\n",
               got, got >> 16, got & 0xffff);
        printf("    exp = 0x%08x  -> 高16位=%u                    "
               "低16位=%u\n",
               exp, exp >> 16, exp & 0xffff);
        printf("    如果 got 高16位是某个奇怪 token_id,怀疑 mul 算错\n");
        printf("    如果 got 全是 0xDEADBEEF,怀疑 device 漏写"
               "(应已被哨兵扫描捕获)\n");
        printf("    如果 got 是另一个合法 (t,d) bit pattern,"
               "怀疑地址偏移错位\n");
    }

    /* 清理 (选项 A:每 case 独立) */
    gpuFree(ctx, args_off);
    gpuFree(ctx, Table_off);
    gpuFree(ctx, Id_off);
    gpuFree(ctx, O_off);
    free(table_u32);
    free(expected_u32);
    free(O_u32);

    printf("  %s\n", fail == 0 ? "PASS" : "FAIL");
    return fail == 0;
}

/* ─── main ────────────────────────────────────── */

int main(void) {
    int ret;

    /* phase 1: gpuInit
     * 注:case 3 (V=151936) 需要 ~519MB VRAM 给 table,加上其他余量,
     *     1GB 不够,需要 2GB (3.6.4 已扩容) */
    gpgpu_ctx *ctx = malloc(sizeof(gpgpu_ctx));
    if (!ctx) {
        fprintf(stderr, "malloc ctx failed\n");
        return 1;
    }

    ret = gpuInit(ctx, "/dev/gpgpu", 2ULL * 1024 * 1024 * 1024);
    if (ret < 0) {
        fprintf(stderr,
                "gpuInit failed: %d (need 2GB VRAM,"
                " 启动 QEMU 时加 -device gpgpu,vram_size=2147483648)\n",
                ret);
        return 1;
    }
    printf("[1] gpuInit OK (2GB VRAM)\n");

    /* phase 2: 读 kernel binary */
    FILE *f = fopen("/tmp/embedding_lookup.bin", "rb");
    if (!f) {
        fprintf(stderr, "open embedding_lookup.bin failed\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t ksize = ftell(f);
    rewind(f);
    uint8_t *kbuf = malloc(ksize);
    fread(kbuf, 1, ksize, f);
    fclose(f);

    /* kernel_off 跨 case 复用 */
    uint64_t kernel_off = gpuMalloc(ctx, ksize);
    if (kernel_off == (uint64_t)-1) {
        fprintf(stderr, "alloc kernel failed\n");
        free(kbuf);
        return 1;
    }
    gpuMemcpy(ctx, kernel_off, kbuf, ksize, 0);
    free(kbuf);
    printf("[2] kernel uploaded, kernel_off=0x%lx, size=%zu\n", kernel_off,
           ksize);

    /* phase 3: 逐 case 跑 */
    int total_pass = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        int ok = run_one_test(ctx, kernel_off, &test_cases[i]);
        if (!ok) {
            printf("\n*** FAIL at test case [%s], stopping ***\n",
                   test_cases[i].name);
            break;
        }
        total_pass++;
    }

    /* phase 4: 清理 */
    gpuDestroy(ctx);
    free(ctx);

    printf("\n%d / %d cases passed\n", total_pass, NUM_TESTS);
    return total_pass == NUM_TESTS ? 0 : 1;
}