#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

const uint32_t V        = 2048; // ← 从 151936 改
const uint32_t D        = 896;  // 不变
const uint32_t TOKEN_ID = 2047; // ← 从 151935 改

struct embedding_args {
    uint32_t table_base;
    uint32_t token_id_buf;
    uint32_t output_base;
    uint32_t D;
    uint32_t elems_per_lane;
};

int main() {
    int ret;

    gpgpu_ctx *ctx = malloc(sizeof(gpgpu_ctx));
    if (!ctx) {
        fprintf(stderr, "malloc ctx failed\n");
        return 1;
    }

    ret = gpuInit(ctx, "/dev/gpgpu", 1024ULL * 1024 * 1024);
    if (ret < 0) {
        fprintf(stderr, "gpuInit failed: %d\n", ret);
        return 1;
    }
    printf("[1] gpuInit OK\n");

    /* 读 kernel binary */
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

    /* 计算 buffer 大小(用 size_t,防止 V*D*4 在 uint32 算术下溢出) */
    size_t table_bytes =
        (size_t)V * (size_t)D * sizeof(float);    // 544,571,392 ≈ 519 MB
    size_t out_bytes = (size_t)D * sizeof(float); // 3584 字节
    printf("table_bytes = %zu (%.1f MB)\n", table_bytes,
           table_bytes / (1024.0 * 1024.0));
    printf("out_bytes   = %zu\n", out_bytes);

    /* gpuMalloc */
    uint64_t kernel_off = gpuMalloc(ctx, ksize);
    uint64_t args_off   = gpuMalloc(ctx, sizeof(struct embedding_args));
    uint64_t Table_off  = gpuMalloc(ctx, table_bytes);
    uint64_t Id_off     = gpuMalloc(ctx, 4);
    uint64_t O_off      = gpuMalloc(ctx, out_bytes);
    printf("kernel_off=0x%lx  args_off=0x%lx  Table_off=0x%lx  Id_off=0x%lx  "
           "O_off=0x%lx\n",
           kernel_off, args_off, Table_off, Id_off, O_off);

    /* 上传 kernel */
    gpuMemcpy(ctx, kernel_off, kbuf, ksize, 0);
    free(kbuf);
    printf("[3] kernel uploaded\n");

    /* 准备 table —— 用 bit pattern,完全避开 fp 算术 */
    uint32_t *table_u32 = malloc(table_bytes);
    if (!table_u32) {
        fprintf(stderr, "malloc table_u32 (%zu bytes) failed\n", table_bytes);
        return 1;
    }
    for (uint32_t t = 0; t < V; t++) {
        for (uint32_t d = 0; d < D; d++) {
            table_u32[(size_t)t * D + d] = (t << 16) | d;
        }
    }
    printf("[4] table generated\n");

    /* 准备 expected —— 同样 bit pattern */
    uint32_t *expected_u32 = malloc(out_bytes);
    for (uint32_t d = 0; d < D; d++) {
        expected_u32[d] = (TOKEN_ID << 16) | d;
    }

    /* 准备 token_id */
    uint32_t token_id_value = TOKEN_ID;

    /* host O 哨兵化(V15 §4 教训) */
    uint32_t *O_u32 = malloc(out_bytes);
    for (uint32_t d = 0; d < D; d++)
        O_u32[d] = 0xDEADBEEF;

    /* args */
    struct embedding_args args = {
        .table_base     = (uint32_t)Table_off,
        .token_id_buf   = (uint32_t)Id_off,
        .output_base    = (uint32_t)O_off,
        .D              = D,
        .elems_per_lane = D / 32,
    };

    /* 上传所有输入 */
    gpuMemcpy(ctx, args_off, &args, sizeof(args), 0);
    printf("[5a] args uploaded\n");

    /* 上传 table —— 519 MB,可能耗时,打个时间戳 */
    printf("[5b] uploading table (%.1f MB)...\n",
           table_bytes / (1024.0 * 1024.0));
    gpuMemcpy(ctx, Table_off, table_u32, table_bytes, 0);
    printf("[5b] table uploaded\n");

    gpuMemcpy(ctx, Id_off, &token_id_value, 4, 0);
    gpuMemcpy(ctx, O_off, O_u32, out_bytes, 0); /* 哨兵 H2D */
    printf("[5c] token_id and output sentinel uploaded\n");

    /* launch */
    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {32, 1, 1};
    ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "gpuLaunchKernel failed: %d\n", ret);
        return 1;
    }
    printf("[6] kernel done\n");

    /* 重新污染 host O,防止旧值伪造 */
    for (uint32_t d = 0; d < D; d++)
        O_u32[d] = 0xCAFEBABE;

    /* 严格 D*4 字节读回 */
    gpuMemcpy(ctx, O_off, O_u32, out_bytes, 1);
    printf("[7] result read back\n");

    /* 哨兵扫描:device 是否漏写 */
    int sentinel_hit = 0;
    for (uint32_t d = 0; d < D; d++) {
        if (O_u32[d] == 0xDEADBEEF) {
            if (sentinel_hit < 10)
                printf("SENTINEL: lane %u (d=%u) did not write\n", d % 32, d);
            sentinel_hit++;
        }
    }
    if (sentinel_hit) {
        printf("FATAL: %d positions not written by device\n", sentinel_hit);
        free(table_u32);
        free(expected_u32);
        free(O_u32);
        gpuDestroy(ctx);
        free(ctx);
        return 1;
    }

    /* bit-pattern exact match 验证 */
    int exact_match  = 0;
    int fail         = 0;
    int first_fail_d = -1;
    for (uint32_t d = 0; d < D; d++) {
        if (O_u32[d] == expected_u32[d]) {
            exact_match++;
        } else {
            if (first_fail_d < 0)
                first_fail_d = (int)d;
            if (fail < 10)
                printf(
                    "FAIL d=%u got=0x%08x expected=0x%08x  (lane=%u, k=%u)\n",
                    d, O_u32[d], expected_u32[d], d % 32, d / 32);
            fail++;
        }
    }
    printf("exact match: %d / %u, fail = %d\n", exact_match, D, fail);

    /* 打印样本(头 4 + 尾 4) */
    printf("samples:\n");
    for (uint32_t d = 0; d < 4; d++) {
        printf("  O[%u]: got=0x%08x  expected=0x%08x\n", d, O_u32[d],
               expected_u32[d]);
    }
    printf("  ...\n");
    for (uint32_t d = D - 4; d < D; d++) {
        printf("  O[%u]: got=0x%08x  expected=0x%08x\n", d, O_u32[d],
               expected_u32[d]);
    }

    /* 如果失败,打印第一个 fail 位置的诊断信息 */
    if (fail > 0 && first_fail_d >= 0) {
        uint32_t got = O_u32[first_fail_d];
        uint32_t exp = expected_u32[first_fail_d];
        printf("\nDIAGNOSTIC: first fail at d=%d\n", first_fail_d);
        printf("  got = 0x%08x  -> 高16位=%u (作为 token_id)  低16位=%u (作为 "
               "d)\n",
               got, got >> 16, got & 0xffff);
        printf("  exp = 0x%08x  -> 高16位=%u                  低16位=%u\n", exp,
               exp >> 16, exp & 0xffff);
        printf("  如果 got 高16位是某个奇怪 token_id,怀疑 mul 算错\n");
        printf("  如果 got 全是 0xDEADBEEF,怀疑 device "
               "漏写(应已被哨兵扫描捕获)\n");
        printf("  如果 got 是另一个合法 (t,d) bit pattern,怀疑地址偏移错位\n");
    }

    free(table_u32);
    free(expected_u32);
    free(O_u32);
    gpuDestroy(ctx);
    free(ctx);
    return fail ? 1 : 0;
}