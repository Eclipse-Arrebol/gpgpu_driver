/*
 * weight_loader.c — 实现见 weight_loader.h 头注释
 *
 * V25 实战调整 (二次): buddy round-up 把 683MB 请求 round 到 1GiB,4 块方案
 * 总占 3.95 GiB,超出可用 VRAM. 改 8 块,每个 layer-region 只装 4 个 layer
 * (228 MB → round up 到 order=16 即 256 MB),总占 3.5 GiB 留 500 MB 给 scratch.
 *
 * 8 块划分见 NUM_REGIONS 附近常量. 接口完全不变.
 *
 * 错误处理风格:
 *   - 加载阶段失败 → 返回 NULL/-1, fprintf 原因. 不 abort.
 *   - 范围越界 (layer/name) → assert. 那是编程错而非数据错.
 */

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "weight_loader.h"

/* ─── 8 块划分常量 ─────────────────────────────────────────── */

/* V25 实测:buddy round-up 把 683MB → 1GiB,4 块方案总占 3.95 GiB 超出 VRAM.
 * 改 8 块,每块 layer 含 4 个 layer (228 MB),round up 到 order=16 (256 MB).
 * 总占 3.5 GiB,剩 500 MB 给 scratch.
 *
 * region 划分:
 *   region[0] = embedding              (519 MB → 1 GiB block)
 *   region[1] = lm_head                (519 MB → 1 GiB block)
 *   region[2] = final_norm + L 0..3    (228 MB → 256 MB block)
 *   region[3] = L 4..7                 (228 MB → 256 MB block)
 *   region[4] = L 8..11                (228 MB → 256 MB block)
 *   region[5] = L 12..15               (228 MB → 256 MB block)
 *   region[6] = L 16..19               (228 MB → 256 MB block)
 *   region[7] = L 20..23               (228 MB → 256 MB block)
 */
#define LAYERS_PER_REGION 4 /* 每个 layer-region 含 4 层 */
#define NUM_LAYER_REGIONS (WL_NUM_LAYERS / LAYERS_PER_REGION) /* = 6 */
#define NUM_REGIONS                                                            \
    (2 + NUM_LAYER_REGIONS) /* emb + lm_head + 6 layer-regions = 8 */

_Static_assert(
    WL_NUM_LAYERS % LAYERS_PER_REGION == 0,
    "weight_loader: WL_NUM_LAYERS must be divisible by LAYERS_PER_REGION");

/* ─── 内部结构 ─────────────────────────────────────────────── */

struct weight_loader {
    gpgpu_ctx *ctx;

    /* 4 块 region 的 gpuMalloc 返回值. destroy 时各 free 一次. */
    uint64_t region_base[NUM_REGIONS];
    size_t   region_bytes[NUM_REGIONS];

    /* 各全局 tensor 在 VRAM 上的绝对 offset (= region_base + 内部 offset) */
    uint64_t embedding_addr;
    uint64_t lm_head_addr;
    uint64_t final_norm_addr;

    /* 每层 base 地址 */
    uint64_t layer_base[WL_NUM_LAYERS];
};

/* ─── tensor 名 → 每层 offset/size 表 ──────────────────────── */

typedef struct {
    size_t      offset;
    size_t      n_floats;
    const char *name;
} tensor_info_t;

static const tensor_info_t TENSOR_TABLE[WL_TENSOR_COUNT] = {
    [WL_INPUT_NORM]  = {WL_OFF_INPUT_NORM, WL_SIZE_INPUT_NORM, "input_norm"},
    [WL_Q_PROJ_W]    = {WL_OFF_Q_PROJ_W, WL_SIZE_Q_PROJ_W, "q_proj.w"},
    [WL_Q_PROJ_B]    = {WL_OFF_Q_PROJ_B, WL_SIZE_Q_PROJ_B, "q_proj.b"},
    [WL_K_PROJ_W]    = {WL_OFF_K_PROJ_W, WL_SIZE_K_PROJ_W, "k_proj.w"},
    [WL_K_PROJ_B]    = {WL_OFF_K_PROJ_B, WL_SIZE_K_PROJ_B, "k_proj.b"},
    [WL_V_PROJ_W]    = {WL_OFF_V_PROJ_W, WL_SIZE_V_PROJ_W, "v_proj.w"},
    [WL_V_PROJ_B]    = {WL_OFF_V_PROJ_B, WL_SIZE_V_PROJ_B, "v_proj.b"},
    [WL_O_PROJ_W]    = {WL_OFF_O_PROJ_W, WL_SIZE_O_PROJ_W, "o_proj.w"},
    [WL_POST_NORM]   = {WL_OFF_POST_NORM, WL_SIZE_POST_NORM, "post_norm"},
    [WL_GATE_PROJ_W] = {WL_OFF_GATE_PROJ_W, WL_SIZE_GATE_PROJ_W, "gate.w"},
    [WL_UP_PROJ_W]   = {WL_OFF_UP_PROJ_W, WL_SIZE_UP_PROJ_W, "up.w"},
    [WL_DOWN_PROJ_W] = {WL_OFF_DOWN_PROJ_W, WL_SIZE_DOWN_PROJ_W, "down.w"},
};

/* ─── 工具函数 ─────────────────────────────────────────────── */

static int check_file_size(const char *path, size_t expected, const char *tag) {
    struct stat st;
    if (stat(path, &st) < 0) {
        fprintf(stderr, "weight_loader: stat(%s): %s\n", path, strerror(errno));
        return -1;
    }
    if ((size_t)st.st_size != expected) {
        fprintf(stderr,
                "weight_loader: %s size mismatch: got %lld expected %zu (%s)\n",
                tag, (long long)st.st_size, expected, path);
        return -1;
    }
    return 0;
}

/* 读 path 整文件到 host buffer,再一次性 H2D 到 vram_offset. */
static int load_file_to_vram(gpgpu_ctx *ctx, const char *path,
                             uint64_t vram_offset, size_t expected_bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "weight_loader: fopen(%s): %s\n", path,
                strerror(errno));
        return -1;
    }

    void *buf = malloc(expected_bytes);
    if (!buf) {
        fprintf(stderr, "weight_loader: host malloc %zu bytes for %s failed\n",
                expected_bytes, path);
        fclose(f);
        return -1;
    }

    size_t got = fread(buf, 1, expected_bytes, f);
    fclose(f);
    if (got != expected_bytes) {
        fprintf(stderr,
                "weight_loader: short read on %s: got %zu expected %zu\n", path,
                got, expected_bytes);
        free(buf);
        return -1;
    }

    int rc = gpuMemcpy(ctx, vram_offset, buf, expected_bytes, GPU_MEMCPY_H2D);
    free(buf);
    if (rc < 0) {
        fprintf(stderr,
                "weight_loader: gpuMemcpy H2D failed for %s "
                "(vram=0x%llx, size=%zu, rc=%d)\n",
                path, (unsigned long long)vram_offset, expected_bytes, rc);
        return -1;
    }
    return 0;
}

/* ─── 公共 API ─────────────────────────────────────────────── */

weight_loader_t *weight_loader_load(gpgpu_ctx *ctx, const char *weights_dir) {
    if (!ctx || !weights_dir) {
        fprintf(stderr, "weight_loader: NULL ctx or weights_dir\n");
        return NULL;
    }

    /* ─── 1. 算每块 region 的总 bytes ────────────────────────── */

    /* region[0] = embedding, region[1] = lm_head
     * region[2] = final_norm + first LAYERS_PER_REGION layers
     * region[3..7] = next groups of LAYERS_PER_REGION layers
     */
    size_t plan_bytes[NUM_REGIONS];
    plan_bytes[0] = WL_EMBEDDING_BYTES;
    plan_bytes[1] = WL_LM_HEAD_BYTES;
    plan_bytes[2] =
        WL_FINAL_NORM_BYTES + (size_t)LAYERS_PER_REGION * WL_LAYER_BYTES;
    for (int r = 3; r < NUM_REGIONS; r++) {
        plan_bytes[r] = (size_t)LAYERS_PER_REGION * WL_LAYER_BYTES;
    }

    /* ─── 2. 提前 stat 所有文件验大小 ────────────────────────── */

    char path[1024];

    snprintf(path, sizeof(path), "%s/%s", weights_dir, WL_FILE_EMBEDDING);
    if (check_file_size(path, WL_EMBEDDING_BYTES, "embedding") < 0)
        return NULL;

    snprintf(path, sizeof(path), "%s/%s", weights_dir, WL_FILE_LM_HEAD);
    if (check_file_size(path, WL_LM_HEAD_BYTES, "lm_head") < 0)
        return NULL;

    snprintf(path, sizeof(path), "%s/%s", weights_dir, WL_FILE_FINAL_NORM);
    if (check_file_size(path, WL_FINAL_NORM_BYTES, "final_norm") < 0)
        return NULL;

    for (int i = 0; i < WL_NUM_LAYERS; i++) {
        snprintf(path, sizeof(path), "%s/" WL_FILE_LAYER_FMT, weights_dir, i);
        char tag[32];
        snprintf(tag, sizeof(tag), "layer_%02d", i);
        if (check_file_size(path, WL_LAYER_BYTES, tag) < 0)
            return NULL;
    }

    fprintf(stderr,
            "[weight_loader] all %d files validated, will alloc %d regions:\n",
            WL_NUM_LAYERS + 3, NUM_REGIONS);
    for (int r = 0; r < NUM_REGIONS; r++) {
        fprintf(stderr, "  region[%d] = %zu bytes (%.2f MB)\n", r,
                plan_bytes[r], (double)plan_bytes[r] / (1024.0 * 1024.0));
    }

    /* ─── 3. 分配 loader 结构体 ──────────────────────────────── */

    weight_loader_t *wl = calloc(1, sizeof(*wl));
    if (!wl) {
        fprintf(stderr, "weight_loader: host calloc failed\n");
        return NULL;
    }
    wl->ctx = ctx;
    for (int r = 0; r < NUM_REGIONS; r++) {
        wl->region_base[r]  = (uint64_t)-1;
        wl->region_bytes[r] = plan_bytes[r];
    }

    /* ─── 4. 4 次独立 gpuMalloc ─────────────────────────────── */

    for (int r = 0; r < NUM_REGIONS; r++) {
        uint64_t base = gpuMalloc(ctx, plan_bytes[r]);
        if (base == (uint64_t)-1) {
            fprintf(stderr,
                    "weight_loader: gpuMalloc(region[%d], %zu bytes) failed.\n",
                    r, plan_bytes[r]);
            goto fail;
        }
        wl->region_base[r] = base;
        fprintf(stderr, "  region[%d] gpuMalloc -> 0x%llx (%zu bytes)\n", r,
                (unsigned long long)base, plan_bytes[r]);
    }

    /* ─── 5. 在每个 region 内部算 tensor 地址 ──────────────── */

    wl->embedding_addr = wl->region_base[0]; /* region[0] 全是 emb */
    wl->lm_head_addr   = wl->region_base[1]; /* region[1] 全是 lm_head */

    /* region[2]: final_norm 在前, 然后 LAYERS_PER_REGION 个 layer */
    uint64_t cursor     = wl->region_base[2];
    wl->final_norm_addr = cursor;
    cursor += WL_FINAL_NORM_BYTES;
    for (int i = 0; i < LAYERS_PER_REGION; i++) {
        wl->layer_base[i] = cursor;
        cursor += WL_LAYER_BYTES;
    }

    /* region[3..7]: 每块 LAYERS_PER_REGION 个 layer */
    for (int r = 3; r < NUM_REGIONS; r++) {
        cursor = wl->region_base[r];
        int layer_start =
            (r - 2) * LAYERS_PER_REGION; /* r=3 → 4, r=4 → 8, ... */
        for (int i = 0; i < LAYERS_PER_REGION; i++) {
            wl->layer_base[layer_start + i] = cursor;
            cursor += WL_LAYER_BYTES;
        }
    }

    /* ─── 6. H2D 实际加载 ───────────────────────────────────── */

    snprintf(path, sizeof(path), "%s/%s", weights_dir, WL_FILE_EMBEDDING);
    if (load_file_to_vram(ctx, path, wl->embedding_addr, WL_EMBEDDING_BYTES) <
        0)
        goto fail;
    fprintf(stderr, "  embedding   @ 0x%llx (%zu bytes) loaded\n",
            (unsigned long long)wl->embedding_addr, (size_t)WL_EMBEDDING_BYTES);

    snprintf(path, sizeof(path), "%s/%s", weights_dir, WL_FILE_LM_HEAD);
    if (load_file_to_vram(ctx, path, wl->lm_head_addr, WL_LM_HEAD_BYTES) < 0)
        goto fail;
    fprintf(stderr, "  lm_head     @ 0x%llx (%zu bytes) loaded\n",
            (unsigned long long)wl->lm_head_addr, (size_t)WL_LM_HEAD_BYTES);

    snprintf(path, sizeof(path), "%s/%s", weights_dir, WL_FILE_FINAL_NORM);
    if (load_file_to_vram(ctx, path, wl->final_norm_addr, WL_FINAL_NORM_BYTES) <
        0)
        goto fail;
    fprintf(stderr, "  final_norm  @ 0x%llx (%zu bytes) loaded\n",
            (unsigned long long)wl->final_norm_addr,
            (size_t)WL_FINAL_NORM_BYTES);

    for (int i = 0; i < WL_NUM_LAYERS; i++) {
        snprintf(path, sizeof(path), "%s/" WL_FILE_LAYER_FMT, weights_dir, i);
        if (load_file_to_vram(ctx, path, wl->layer_base[i], WL_LAYER_BYTES) < 0)
            goto fail;
        fprintf(stderr, "  layer %2d    @ 0x%llx (%zu bytes) loaded\n", i,
                (unsigned long long)wl->layer_base[i], (size_t)WL_LAYER_BYTES);
    }

    fprintf(stderr, "[weight_loader] all weights loaded.\n");
    return wl;

fail:
    /* 已成功 alloc 的 region 都要 free */
    for (int r = 0; r < NUM_REGIONS; r++) {
        if (wl->region_base[r] != (uint64_t)-1) {
            gpuFree(ctx, wl->region_base[r]);
        }
    }
    free(wl);
    return NULL;
}

void weight_loader_destroy(weight_loader_t *wl) {
    if (!wl)
        return;
    for (int r = 0; r < NUM_REGIONS; r++) {
        if (wl->region_base[r] != (uint64_t)-1) {
            gpuFree(wl->ctx, wl->region_base[r]);
        }
    }
    free(wl);
}

uint64_t weight_loader_layer(weight_loader_t *wl, int layer, wl_tensor_t name) {
    assert(wl != NULL);
    assert(layer >= 0 && layer < WL_NUM_LAYERS);
    assert((int)name >= 0 && name < WL_TENSOR_COUNT);
    return wl->layer_base[layer] + TENSOR_TABLE[name].offset;
}

uint64_t weight_loader_embedding(weight_loader_t *wl) {
    assert(wl);
    return wl->embedding_addr;
}
uint64_t weight_loader_lm_head(weight_loader_t *wl) {
    assert(wl);
    return wl->lm_head_addr;
}
uint64_t weight_loader_final_norm(weight_loader_t *wl) {
    assert(wl);
    return wl->final_norm_addr;
}

int weight_loader_dump_floats(weight_loader_t *wl, uint64_t vram_offset,
                              float *out_buf, size_t n_floats) {
    assert(wl);
    return gpuMemcpy(wl->ctx, vram_offset, out_buf, n_floats * sizeof(float),
                     GPU_MEMCPY_D2H);
}

void weight_loader_print_layout(weight_loader_t *wl) {
    assert(wl);
    fprintf(stderr, "─── weight_loader layout (4 regions) ───\n");
    for (int r = 0; r < NUM_REGIONS; r++) {
        fprintf(stderr, "region[%d] base=0x%llx size=%zu bytes\n", r,
                (unsigned long long)wl->region_base[r], wl->region_bytes[r]);
    }
    fprintf(stderr, "embedding   @ 0x%llx\n",
            (unsigned long long)wl->embedding_addr);
    fprintf(stderr, "lm_head     @ 0x%llx\n",
            (unsigned long long)wl->lm_head_addr);
    fprintf(stderr, "final_norm  @ 0x%llx\n",
            (unsigned long long)wl->final_norm_addr);
    for (int i = 0; i < WL_NUM_LAYERS; i++) {
        fprintf(stderr, "layer %2d    @ 0x%llx\n", i,
                (unsigned long long)wl->layer_base[i]);
    }
}