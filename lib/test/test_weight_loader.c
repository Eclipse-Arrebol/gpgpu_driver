/*
 * test_weight_loader.c — Milestone 3.6.7 Step 2 验证测试
 *
 * 验证三个独立命题:
 *   1) weight_loader_load 能成功加载所有 28 个文件 (emb + lm_head + final_norm
 * + 24 层) 2) D2H 读回 3 个 tensor 的前 8 个 fp32, 与 manifest first8 bit-exact
 *   3) 加载耗时量化 (债务 20 是否真痛的依据)
 *
 * Bit-exact 期望值来自 host 端 dump_weights.py 输出的 manifest.json:
 *   - stats_embedding.first8
 *   - stats_layer_0.q_proj.weight.first8  (注意:transpose 后的形态)
 *   - stats_final_norm.first8
 *
 * 退出码:0 = 全 PASS,非 0 = FAIL
 *
 * 用法:
 *   ./test_weight_loader <weights_dir>
 *   (例如 ./test_weight_loader /root/weights)
 */

#include "../include/weight_loader.h"
#include "../libgpgpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ─── manifest 期望值 ────────────────────────────────────────
 *
 * 来源:weights/manifest.json (V25 dump 实测).
 * 这些是 bf16 → fp32 高 16 位填充后的 bit-exact 值, 任何差异都说明
 *   - dump 工具改了
 *   - bf16→fp32 cast 路径变了
 *   - 或 weight_loader 读错文件/算错 offset
 */

/* stats_embedding.first8 */
static const float EXPECTED_EMBEDDING[8] = {
    -0.0203857421875f, -0.00921630859375f,  0.011474609375f,
    0.00860595703125f, -0.006439208984375f, -0.022216796875f,
    0.01519775390625f, 0.0111083984375f,
};

/* stats_layer_0.q_proj.weight.first8 (transpose 后形态) */
static const float EXPECTED_Q_PROJ_W_L0[8] = {
    -0.00244140625f, 0.02001953125f,   -0.01513671875f,   -0.0235595703125f,
    0.060791015625f, -0.021728515625f, -0.0125732421875f, 0.004150390625f,
};

/* stats_final_norm.first8 */
static const float EXPECTED_FINAL_NORM[8] = {
    7.5625f, 8.0f, 7.21875f, 7.3125f, 7.46875f, 7.375f, 6.0f, 7.1875f,
};

/* ─── bit-exact 对比工具 ─────────────────────────────────────
 *
 * 不用 fabsf < tol. 这是 dump 出的 bit-exact 路径, 任何 != 都是 bug.
 */
static int compare_bit_exact(const char *tag, const float *got,
                             const float *expected, size_t n) {
    int n_fail = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t got_bits, exp_bits;
        memcpy(&got_bits, &got[i], 4);
        memcpy(&exp_bits, &expected[i], 4);

        if (got_bits == exp_bits) {
            printf("  [%zu] PASS  got=0x%08x (%+ .8e)\n", i, got_bits, got[i]);
        } else {
            printf(
                "  [%zu] FAIL  got=0x%08x (%+ .8e)  expected=0x%08x (%+ .8e)\n",
                i, got_bits, got[i], exp_bits, expected[i]);

            /* 原则 21 同款 diagnostic */
            if (got_bits == 0xDEADBEEFu) {
                printf("       → SENTINEL: D2H 读到 sentinel, host buffer 未被 "
                       "device 写覆盖.\n");
                printf("         检查 gpuMemcpy(D2H) 方向参数, 或 vram offset "
                       "错位.\n");
            } else if (got_bits == 0x00000000u) {
                printf(
                    "       → got=0: VRAM 地址未写入过权重, 或读到全 0 区.\n");
                printf("         检查 weight_loader 是否真完成 H2D, 或 region "
                       "offset 算错.\n");
            } else if ((got_bits & 0xFFFFu) == 0x0000u) {
                printf("       → low 16 bits = 0: bf16→fp32 cast 路径 OK, "
                       "但值不对.\n");
                printf("         可能 offset 算错 (读到了同一文件的别处或别的 "
                       "tensor).\n");
            } else {
                printf("       → low 16 bits ≠ 0: bf16→fp32 cast 路径变了, "
                       "或读到非权重数据.\n");
            }
            n_fail++;
        }
    }
    printf("  %s: %s (%d/%zu fail)\n", tag, (n_fail == 0) ? "PASS" : "FAIL",
           n_fail, n);
    return (n_fail == 0);
}

/* ─── 单 case 验证 ──────────────────────────────────────────── */
static int verify_tensor(weight_loader_t *wl, const char *tag,
                         uint64_t vram_offset, const float *expected) {
    printf("\n--- verifying: %s @ vram=0x%llx ---\n", tag,
           (unsigned long long)vram_offset);

    float got[8];
    /* 灌 sentinel ─ 确保不是"恰好原本是 expected" */
    uint32_t sent_bits = 0xDEADBEEFu;
    for (int i = 0; i < 8; i++)
        memcpy(&got[i], &sent_bits, 4);

    int rc = weight_loader_dump_floats(wl, vram_offset, got, 8);
    if (rc < 0) {
        printf("  FAIL: weight_loader_dump_floats returned %d\n", rc);
        return 0;
    }

    return compare_bit_exact(tag, got, expected, 8);
}

/* ─── main ─────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <weights_dir>\n", argv[0]);
        return 2;
    }
    const char *weights_dir = argv[1];

    printf("=== test_weight_loader (Milestone 3.6.7 Step 2) ===\n");

    /* gpuInit ─ 4GB VRAM, 与 Step 0 一致 */
    gpgpu_ctx *ctx = malloc(sizeof(gpgpu_ctx));
    if (!ctx) {
        fprintf(stderr, "FAIL: malloc ctx\n");
        return 1;
    }

    int rc = gpuInit(ctx, "/dev/gpgpu", 4096ULL * 1024 * 1024);
    if (rc < 0) {
        fprintf(stderr, "FAIL: gpuInit returned %d\n", rc);
        return 1;
    }
    printf("[init] gpuInit OK, VRAM=4GB\n");

    /* 加载权重并计时 */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    weight_loader_t *wl = weight_loader_load(ctx, weights_dir);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    if (!wl) {
        fprintf(stderr, "FAIL: weight_loader_load returned NULL\n");
        gpuDestroy(ctx);
        return 1;
    }
    printf("\n[timing] weight_loader_load took %.2f s (%.1f MB/s)\n", elapsed,
           (double)WL_WEIGHTS_TOTAL_BYTES / (1024.0 * 1024.0) / elapsed);

    /* 三个 tensor 的 bit-exact 验证 */
    int total_pass = 0;

    if (verify_tensor(wl, "embedding[0..7]", weight_loader_embedding(wl),
                      EXPECTED_EMBEDDING))
        total_pass++;

    if (verify_tensor(wl, "layer_0 q_proj.w (transpose) [0..7]",
                      weight_loader_layer(wl, 0, WL_Q_PROJ_W),
                      EXPECTED_Q_PROJ_W_L0))
        total_pass++;

    if (verify_tensor(wl, "final_norm[0..7]", weight_loader_final_norm(wl),
                      EXPECTED_FINAL_NORM))
        total_pass++;

    printf("\n=== summary: %d / 3 tensors verified bit-exact ===\n",
           total_pass);
    printf("\nbreakdown:\n");
    printf("  load time: %.2f s for %.2f GiB = %.1f MB/s\n", elapsed,
           (double)WL_WEIGHTS_TOTAL_BYTES / (1024.0 * 1024.0 * 1024.0),
           (double)WL_WEIGHTS_TOTAL_BYTES / (1024.0 * 1024.0) / elapsed);
    printf("  bit-exact path: 3 tensors @ 8 floats each\n");
    printf(
        "  debt (20): VRAM trap-path performance ─ above MB/s number is the\n");
    printf("    metric. If < 10 MB/s, debt 20 真痛, 考虑改 ram_ptr.\n");

    weight_loader_destroy(wl);
    gpuDestroy(ctx);
    free(ctx);
    return (total_pass == 3) ? 0 : 1;
}