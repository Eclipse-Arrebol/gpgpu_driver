#include "../include/kv_cache.h"
#include "../libgpgpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks = 0, g_fails = 0;

/* ====== Part A 已有的 g_checks, CHECK, test_state_machine ====== */
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        g_checks++;                                                            \
        if (!(cond)) {                                                         \
            printf("  FAIL: %s  (line %d)\n", msg, __LINE__);                  \
            g_fails++;                                                         \
        } else {                                                               \
            printf("  ok:   %s\n", msg);                                       \
        }                                                                      \
    } while (0)

void test_state_machine(gpgpu_ctx *ctx) {
    printf("\n===== Part A: state machine =====\n");

    kv_cache_t *c = kv_cache_create(ctx, 2, 2, 4, 8);
    CHECK(c != NULL, "create returns non-NULL");
    CHECK(c->filled == 0, "initial filled == 0");
    CHECK(c->pending == 0, "initial pending == 0");

    /* append_layer 需要真实的 K_new/V_new 指针,内容无所谓,
       Part A 不验证 device 数据,但指针不能是野指针 */
    float dummy[2 * 8] = {0}; /* num_kv_heads * head_dim */

    int r = kv_cache_append_layer(ctx, c, 0, dummy, dummy);
    CHECK(r == 0, "append L=0 returns 0");
    CHECK(c->pending == 1, "pending == 1 after append L=0");
    CHECK(c->filled == 0, "filled still 0 after append");

    r = kv_cache_append_layer(ctx, c, 1, dummy, dummy);
    CHECK(r == 0, "append L=1 returns 0");
    CHECK(c->pending == 2, "pending == 2 after append L=1");

    r = kv_cache_commit_token(c);
    CHECK(r == 0, "commit returns 0");
    CHECK(c->filled == 1, "filled == 1 after commit");
    CHECK(c->pending == 0, "pending == 0 after commit");

    /* 第二轮,验证 filled 继续推进 */
    kv_cache_append_layer(ctx, c, 0, dummy, dummy);
    kv_cache_append_layer(ctx, c, 1, dummy, dummy);
    kv_cache_commit_token(c);
    CHECK(c->filled == 2, "filled == 2 after second commit");

    /* 偏移正确性 */
    uint64_t layer_bytes =
        (uint64_t)c->num_kv_heads * c->max_seq * c->head_dim * sizeof(float);
    CHECK(kv_cache_K_off(c, 0) == c->K_dev_off, "K_off(0) == K_dev_off");
    CHECK(kv_cache_K_off(c, 1) - kv_cache_K_off(c, 0) == layer_bytes,
          "K_off layer stride correct");
    CHECK(kv_cache_V_off(c, 1) - kv_cache_V_off(c, 0) == layer_bytes,
          "V_off layer stride correct");

    kv_cache_destroy(ctx, c);

    /* 错误路径 1: commit 前没 append */
    kv_cache_t *e1 = kv_cache_create(ctx, 2, 2, 4, 8);
    CHECK(kv_cache_commit_token(e1) == -1, "commit without append returns -1");
    kv_cache_destroy(ctx, e1);

    /* 错误路径 2: 漏 layer 就 commit */
    kv_cache_t *e2 = kv_cache_create(ctx, 2, 2, 4, 8);
    kv_cache_append_layer(ctx, e2, 0, dummy, dummy);
    CHECK(kv_cache_commit_token(e2) == -2,
          "commit with missing layer returns -2");
    kv_cache_destroy(ctx, e2);

    /* 错误路径 3: layer 顺序错(重复 append L=0) */
    kv_cache_t *e3 = kv_cache_create(ctx, 2, 2, 4, 8);
    kv_cache_append_layer(ctx, e3, 0, dummy, dummy);
    CHECK(kv_cache_append_layer(ctx, e3, 0, dummy, dummy) == -2,
          "duplicate append L=0 returns -2");
    kv_cache_destroy(ctx, e3);

    kv_cache_t *e4 = kv_cache_create(ctx, 2, 2, 4, 8); /* max_seq=4 */
    for (int t = 0; t < 4; t++) {                      /* 填满 4 个 token */
        kv_cache_append_layer(ctx, e4, 0, dummy, dummy);
        kv_cache_append_layer(ctx, e4, 1, dummy, dummy);
        kv_cache_commit_token(e4);
    }
    CHECK(kv_cache_append_layer(ctx, e4, 0, dummy, dummy) == -1,
          "append into full cache returns -1");
    kv_cache_destroy(ctx, e4);
    /* ... 继续 ... */
}

/* ====== Part B ====== */
struct qkt_decode_args {
    uint32_t Q_off;
    uint32_t K_cache_off;
    uint32_t scores_off;
    uint32_t cur_pos;
    uint32_t max_seq;
    uint32_t head_dim;
    uint32_t q_per_kv;
    uint32_t scale_bits;
};

struct softmax_decode_args {
    uint32_t scores_off;
    uint32_t cur_pos;
    uint32_t max_seq;
};

struct pv_decode_args {
    uint32_t P_off;
    uint32_t V_cache_off;
    uint32_t O_off;
    uint32_t cur_pos;
    uint32_t max_seq;
    uint32_t head_dim;
    uint32_t q_per_kv;
};

static void
attention_ref(const float *Q,       /* [num_q_heads, head_dim] */
              const float *K_cache, /* [num_kv_heads, max_seq, head_dim] */
              const float *V_cache, /* [num_kv_heads, max_seq, head_dim] */
              float       *O,       /* [num_q_heads, head_dim] */
              int num_q_heads, int num_kv_heads, int head_dim, int max_seq,
              int S, int q_per_kv, float scale) {
    float *scores = calloc(num_q_heads * S, sizeof(float));

    /* QK^T * scale */
    for (int qh = 0; qh < num_q_heads; qh++) {
        int kh = qh / q_per_kv;
        for (int j = 0; j < S; j++) {
            float acc = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                acc += Q[qh * head_dim + k] *
                       K_cache[kh * max_seq * head_dim + j * head_dim + k];
            }
            scores[qh * S + j] = acc * scale;
        }
    }

    /* softmax per row */
    for (int qh = 0; qh < num_q_heads; qh++) {
        float *row  = scores + qh * S;
        float  maxv = -1e30f;
        for (int j = 0; j < S; j++)
            if (row[j] > maxv)
                maxv = row[j];
        float sum = 0.0f;
        for (int j = 0; j < S; j++) {
            row[j] = expf(row[j] - maxv);
            sum += row[j];
        }
        for (int j = 0; j < S; j++)
            row[j] /= sum;
    }

    /* P * V */
    for (int qh = 0; qh < num_q_heads; qh++) {
        int kh = qh / q_per_kv;
        for (int k = 0; k < head_dim; k++) {
            float acc = 0.0f;
            for (int j = 0; j < S; j++) {
                acc += scores[qh * S + j] *
                       V_cache[kh * max_seq * head_dim + j * head_dim + k];
            }
            O[qh * head_dim + k] = acc;
        }
    }

    free(scores);
}

/* ---- 加载单个 kernel binary ---- */
static uint64_t load_kernel(gpgpu_ctx *g_ctx, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "open %s failed\n", path);
        return 0;
    }
    fseek(f, 0, SEEK_END);
    size_t ksize = ftell(f);
    rewind(f);
    uint8_t *buf = malloc(ksize);
    fread(buf, 1, ksize, f);
    fclose(f);
    uint64_t off = gpuMalloc(g_ctx, ksize);
    gpuMemcpy(g_ctx, off, buf, ksize, 0);
    free(buf);
    printf("[init] loaded %s, size=%zu\n", path, ksize);
    return off;
}

static float K_at(int step, int kh, int k) {
    return (step + 1) * 0.1f + (kh + 1) * 0.01f + k * 0.001f;
}
static float V_at(int step, int kh, int k) {
    return (step + 1) * 0.2f - (kh + 1) * 0.01f + k * 0.002f;
}
static float Q_at(int step, int qh, int k) {
    return (step + 1) * 0.05f + (qh + 1) * 0.03f + k * 0.0005f;
}

static int test_multistep_decode(gpgpu_ctx *ctx, uint64_t qkt_off,
                                 uint64_t softmax_off, uint64_t pv_off) {
    /* === 配置 === */
    const int   num_layers   = 1;
    const int   num_kv_heads = 2;
    const int   num_q_heads  = 4;
    const int   head_dim     = 32;
    const int   max_seq      = 40;
    const int   N            = 40;
    const int   q_per_kv     = num_q_heads / num_kv_heads;
    const float scale        = 1.0f;
    const float tol          = 1e-4f;

    printf("\n===== Part B: multi-step decode =====\n");
    printf("  num_q_heads=%d num_kv_heads=%d head_dim=%d max_seq=%d N=%d\n",
           num_q_heads, num_kv_heads, head_dim, max_seq, N);

    /* === 循环外:创建 kv_cache === */
    kv_cache_t *cache =
        kv_cache_create(ctx, num_layers, num_kv_heads, max_seq, head_dim);
    if (!cache) {
        fprintf(stderr, "kv_cache_create failed\n");
        return -1;
    }

    /* === 循环外:分配 device buffer === */
    uint64_t qkt_args_off = gpuMalloc(ctx, sizeof(struct qkt_decode_args));
    uint64_t softmax_args_off =
        gpuMalloc(ctx, sizeof(struct softmax_decode_args));
    uint64_t pv_args_off = gpuMalloc(ctx, sizeof(struct pv_decode_args));

    size_t Q_size      = (size_t)num_q_heads * head_dim * sizeof(float);
    size_t scores_size = (size_t)num_q_heads * max_seq * sizeof(float);
    size_t O_size      = (size_t)num_q_heads * head_dim * sizeof(float);

    uint64_t Q_off      = gpuMalloc(ctx, Q_size);
    uint64_t scores_off = gpuMalloc(ctx, scores_size);
    uint64_t O_off      = gpuMalloc(ctx, O_size);

    /* === 循环外:分配 host buffer === */
    size_t KV_hist_size =
        (size_t)num_kv_heads * max_seq * head_dim * sizeof(float);
    float *K_hist = calloc(num_kv_heads * max_seq * head_dim, sizeof(float));
    float *V_hist = calloc(num_kv_heads * max_seq * head_dim, sizeof(float));
    float *K_new  = malloc((size_t)num_kv_heads * head_dim * sizeof(float));
    float *V_new  = malloc((size_t)num_kv_heads * head_dim * sizeof(float));
    float *Q_h    = malloc(Q_size);
    float *O_h    = calloc(num_q_heads * head_dim, sizeof(float));
    float *O_ref  = calloc(num_q_heads * head_dim, sizeof(float));

    /* === 主循环 === */
    float overall_max_err  = 0.0f;
    int   overall_max_step = -1;
    int   total_fail       = 0;

    for (int step = 0; step < N; step++) {
        /* 1. 生成 K_new, V_new */
        for (int kh = 0; kh < num_kv_heads; kh++) {
            for (int k = 0; k < head_dim; k++) {
                K_new[kh * head_dim + k] = K_at(step, kh, k);
                V_new[kh * head_dim + k] = V_at(step, kh, k);
            }
        }

        /* 2. 同步写到 host K_hist / V_hist 的 step 位置 */
        for (int kh = 0; kh < num_kv_heads; kh++) {
            for (int k = 0; k < head_dim; k++) {
                size_t idx = (size_t)kh * max_seq * head_dim +
                             (size_t)step * head_dim + k;
                K_hist[idx] = K_new[kh * head_dim + k];
                V_hist[idx] = V_new[kh * head_dim + k];
            }
        }

        /* 3. append 到 device cache */
        int r = kv_cache_append_layer(ctx, cache, 0, K_new, V_new);
        if (r != 0) {
            fprintf(stderr, "append failed at step %d: %d\n", step, r);
            break;
        }

        /* 4. 生成 Q_step,上传 */
        for (int qh = 0; qh < num_q_heads; qh++)
            for (int k = 0; k < head_dim; k++)
                Q_h[qh * head_dim + k] = Q_at(step, qh, k);
        gpuMemcpy(ctx, Q_off, Q_h, Q_size, 0);

        /* 5. 拿 offset 和 cur_pos */
        uint32_t cur_pos = kv_cache_cur_pos(cache);
        uint64_t K_off   = kv_cache_K_off(cache, 0);
        uint64_t V_off   = kv_cache_V_off(cache, 0);

        /* 6/7/8. 上传 args 并 launch */
        uint32_t scale_bits;
        memcpy(&scale_bits, &scale, sizeof(float));

        struct qkt_decode_args qkt_args = {
            .Q_off       = (uint32_t)Q_off,
            .K_cache_off = (uint32_t)K_off,
            .scores_off  = (uint32_t)scores_off,
            .cur_pos     = cur_pos,
            .max_seq     = max_seq,
            .head_dim    = head_dim,
            .q_per_kv    = q_per_kv,
            .scale_bits  = scale_bits,
        };
        gpuMemcpy(ctx, qkt_args_off, &qkt_args, sizeof(qkt_args), 0);

        struct softmax_decode_args sm_args = {
            .scores_off = (uint32_t)scores_off,
            .cur_pos    = cur_pos,
            .max_seq    = max_seq,
        };
        gpuMemcpy(ctx, softmax_args_off, &sm_args, sizeof(sm_args), 0);

        struct pv_decode_args pv_args = {
            .P_off       = (uint32_t)scores_off,
            .V_cache_off = (uint32_t)V_off,
            .O_off       = (uint32_t)O_off,
            .cur_pos     = cur_pos,
            .max_seq     = max_seq,
            .head_dim    = head_dim,
            .q_per_kv    = q_per_kv,
        };
        gpuMemcpy(ctx, pv_args_off, &pv_args, sizeof(pv_args), 0);

        uint32_t grid[3]  = {(uint32_t)num_q_heads, 1, 1};
        uint32_t block[3] = {32, 1, 1};
        float   *zero_scores =
            calloc((size_t)num_q_heads * max_seq, sizeof(float));
        gpuMemcpy(ctx, scores_off, zero_scores, scores_size, 0);
        free(zero_scores);

        if (gpuLaunchKernel(ctx, qkt_off, qkt_args_off, grid, block, 0) < 0)
            break;
        if (gpuLaunchKernel(ctx, softmax_off, softmax_args_off, grid, block,
                            0) < 0)
            break;
        if (gpuLaunchKernel(ctx, pv_off, pv_args_off, grid, block, 0) < 0)
            break;

        /* 9. 下载 O */
        gpuMemcpy(ctx, O_off, O_h, O_size, 1);

        /* 10. host reference */
        attention_ref(Q_h, K_hist, V_hist, O_ref, num_q_heads, num_kv_heads,
                      head_dim, max_seq, /*S=*/step + 1, q_per_kv, scale);

        /* 11. 比对 */
        float step_max_err = 0.0f;
        for (int qh = 0; qh < num_q_heads; qh++) {
            for (int k = 0; k < head_dim; k++) {
                int   idx  = qh * head_dim + k;
                float diff = fabsf(O_h[idx] - O_ref[idx]);
                if (diff > step_max_err)
                    step_max_err = diff;
            }
        }
        if (step_max_err > overall_max_err) {
            overall_max_err  = step_max_err;
            overall_max_step = step;
        }
        if (step_max_err >= tol)
            total_fail++;

        printf("  step %2d: cur_pos=%2u  S=%d  max_err=%.3e%s\n", step, cur_pos,
               step + 1, step_max_err, step_max_err >= tol ? "  FAIL" : "");

        /* 12. commit */
        kv_cache_commit_token(cache);
    }

    printf("\n  overall max_err = %.3e at step %d  (tol=%.1e)\n",
           overall_max_err, overall_max_step, tol);
    printf("  %s\n", total_fail == 0 ? "PASS" : "FAIL");

    /* === 清理 === */
    free(K_hist);
    free(V_hist);
    free(K_new);
    free(V_new);
    free(Q_h);
    free(O_h);
    free(O_ref);
    kv_cache_destroy(ctx, cache);

    return total_fail;
}

int main(void) {
    gpgpu_ctx *ctx = malloc(sizeof(gpgpu_ctx));
    if (gpuInit(ctx, "/dev/gpgpu", 64ULL * 1024 * 1024) < 0)
        return 1;
    printf("[init] gpuInit OK\n");

    /* Part A */
    test_state_machine(ctx);

    /* Part B 需要 kernel */
    uint64_t qkt_off     = load_kernel(ctx, "/tmp/qkt_decode.bin");
    uint64_t softmax_off = load_kernel(ctx, "/tmp/softmax_decode.bin");
    uint64_t pv_off      = load_kernel(ctx, "/tmp/pv_decode.bin");
    int part_b_fail = test_multistep_decode(ctx, qkt_off, softmax_off, pv_off);

    printf("\n===== SUMMARY =====\n");
    printf("Part A checks: %d, fails: %d\n", g_checks, g_fails);
    printf("Part B fails: %d\n", part_b_fail);

    gpuDestroy(ctx);
    free(ctx);
    return (g_fails + part_b_fail) == 0 ? 0 : 1;
}