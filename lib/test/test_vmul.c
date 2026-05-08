#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ─── shape:从这里改测试矩阵 ─────────────────────── */

const uint32_t len = 4864;
/* ────────────────────────────────────────────────── */

typedef struct {
    uint32_t a_off; // 输入向量 A 在 VRAM 的偏移
    uint32_t b_off; // 输入向量 B 在 VRAM 的偏移
    uint32_t c_off; // 输出向量 C 在 VRAM 的偏移
    uint32_t len;   // 向量长度(元素个数)
} VmulArgs;

/* 单 block 1D 派发:device model 的 grid 是顺序执行,
 * 多 block 没并行收益,统一塞单 block 最干净 */
static inline void calc_dispatch_1d(uint32_t total_threads, uint32_t grid[3],
                                    uint32_t block[3]) {
    grid[0]  = 1;
    grid[1]  = 1;
    grid[2]  = 1;
    block[0] = total_threads;
    block[1] = 1;
    block[2] = 1;
}

int main() {
    int ret;

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

    /* kernel 大小动态读,避免硬编码踩 V16 的 KERNEL_SIZE 截断坑 */
    FILE *f = fopen("/tmp/vmul.bin", "rb");
    if (!f) {
        perror("fopen vmul.bin");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t kernel_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    printf("[1.5] vmul.bin size = %zu bytes\n", kernel_size);

    /* V16 §4 教训:每个 gpuMalloc 后立刻检查返回值 */
    uint64_t kernel_off = gpuMalloc(ctx, kernel_size);
    if (kernel_off == (uint64_t)-1) {
        fprintf(stderr, "alloc kernel failed\n");
        return 1;
    }

    uint64_t args_off = gpuMalloc(ctx, sizeof(VmulArgs));
    if (args_off == (uint64_t)-1) {
        fprintf(stderr, "alloc args failed\n");
        return 1;
    }

    uint64_t a_off = gpuMalloc(ctx, len * 4);
    if (a_off == (uint64_t)-1) {
        fprintf(stderr, "alloc A failed\n");
        return 1;
    }

    uint64_t b_off = gpuMalloc(ctx, len * 4);
    if (b_off == (uint64_t)-1) {
        fprintf(stderr, "alloc B failed\n");
        return 1;
    }

    uint64_t c_off = gpuMalloc(ctx, len * 4);
    if (c_off == (uint64_t)-1) {
        fprintf(stderr, "alloc C failed\n");
        return 1;
    }

    printf("[2] alloc: kernel=0x%lx args=0x%lx A=0x%lx B=0x%lx C=0x%lx\n",
           kernel_off, args_off, a_off, b_off, c_off);

    /* 上传 kernel */
    uint8_t *kbuf = malloc(kernel_size);
    fread(kbuf, 1, kernel_size, f);
    fclose(f);
    gpuMemcpy(ctx, kernel_off, kbuf, kernel_size, GPU_MEMCPY_H2D);
    free(kbuf);
    printf("[3] kernel uploaded\n");

    /* 准备数据 */
    float *ha = malloc(len * sizeof(float));
    float *hb = malloc(len * sizeof(float));
    float *hc = malloc(len * sizeof(float));

    for (int i = 0; i < len; i++)
        ha[i] = ((i % 7) - 3) * 0.1f; // [-0.3, 0.3]
    for (int i = 0; i < len; i++)
        hb[i] = ((i % 11) - 5) * 0.1f + 1e-7f;

    /* C 哨兵初始化:V11 教训,任何"device 没写"的位置都能被识别 */
    for (int i = 0; i < len; i++)
        hc[i] = -999999.0f;

    gpuMemcpy(ctx, a_off, ha, len * 4, GPU_MEMCPY_H2D);
    gpuMemcpy(ctx, b_off, hb, len * 4, GPU_MEMCPY_H2D);
    gpuMemcpy(ctx, c_off, hc, len * 4, GPU_MEMCPY_H2D);
    printf("[4] input + C sentinel uploaded\n");

    VmulArgs args = {.a_off = (uint32_t)a_off,
                     .b_off = (uint32_t)b_off,
                     .c_off = (uint32_t)c_off,
                     .len   = len};
    gpuMemcpy(ctx, args_off, &args, sizeof(args), GPU_MEMCPY_H2D);
    printf("[5] args uploaded: len=%u\n", args.len);

    /* 派发:M*N 个线程 */
    uint32_t total = 32;
    uint32_t grid[3], block[3];
    calc_dispatch_1d(total, grid, block);
    printf("[6] dispatch: %u threads = %u warps "
           "(grid={%u,%u,%u} block={%u,%u,%u})\n",
           total, (total + 31) / 32, grid[0], grid[1], grid[2], block[0],
           block[1], block[2]);

    ret = gpuLaunchKernel(ctx, kernel_off, args_off, grid, block, 0);
    if (ret < 0) {
        fprintf(stderr, "launch failed: %d\n", ret);
        return 1;
    }
    printf("[7] kernel done\n");

    gpuMemcpy(ctx, c_off, hc, len * 4, GPU_MEMCPY_D2H);
    printf("[8] result read back\n");

    /* 验证 */
    float *ref = malloc(len * sizeof(float));
    for (int i = 0; i < len; i++) {
        ref[i] = ha[i] * hb[i];
    }

    /* V16 §5 教训:max_err + exact_match 双指标,消歧"循环没跑"vs"全对" */
    int   exact_match   = 0;
    int   sentinel_hits = 0;
    float max_err       = -1.0f;
    int   max_idx       = -1;
    for (int i = 0; i < len; i++) {
        if (hc[i] == -999999.0f)
            sentinel_hits++;
        if (hc[i] == ref[i])
            exact_match++;
        float err = fabsf(hc[i] - ref[i]);
        if (err > max_err) {
            max_err = err;
            max_idx = i;
        }
    }
    printf("[9] exact match: %d / %d\n", exact_match, len);
    printf("    sentinel still: %d (should be 0)\n", sentinel_hits);
    printf("    max_err = %g at idx %d\n", max_err, max_idx);
    if (max_err < 0)
        printf("    WARNING: validation loop did not execute\n");

    int ok = (sentinel_hits == 0) && (max_err < 1e-3f);
    if (!ok) {
        for (int i = 0; i < len && i < 16; i++)
            printf("    [%d] got=%g  ref=%g  err=%g\n", i, hc[i], ref[i],
                   fabsf(hc[i] - ref[i]));
    }

    for (int i = 0; i < 3 && i < len; i++) {
        printf("    [%d] got_bits=0x%08x ref_bits=0x%08x  got=%g ref=%g\n", i,
               *(uint32_t *)&hc[i], *(uint32_t *)&ref[i], hc[i], ref[i]);
    }

    free(ha);
    free(hb);
    free(hc);
    free(ref);
    gpuDestroy(ctx);
    free(ctx);
    printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}