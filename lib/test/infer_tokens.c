/*
 * infer_tokens.c - toy Qwen2-0.5B token-level greedy inference.
 *
 * This intentionally does not implement tokenizer or sampling. It reads prompt
 * token ids from a text file, runs transformer_step(), and writes generated
 * token ids to a text file.
 *
 * Usage:
 *   ./infer_tokens <weights_dir> <kernel_bin_dir> <input_tokens.txt> \
 *                  <max_new_tokens> <output_tokens.txt> [eos_token_id]
 */

#include "../include/kv_cache.h"
#include "../include/transformer.h"
#include "../include/weight_loader.h"
#include "../libgpgpu.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VRAM_SIZE (4ULL * 1024 * 1024 * 1024)

static int read_tokens(const char *path, int32_t **out_tokens,
                       size_t *out_count) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "infer_tokens: fopen(%s): %s\n", path,
                strerror(errno));
        return -1;
    }

    size_t   cap = 64;
    size_t   n   = 0;
    int32_t *ids = malloc(cap * sizeof(*ids));
    if (!ids) {
        fclose(fp);
        return -1;
    }

    long value;
    while (fscanf(fp, "%ld", &value) == 1) {
        if (value < 0 || value > INT32_MAX) {
            fprintf(stderr, "infer_tokens: invalid token id %ld\n", value);
            free(ids);
            fclose(fp);
            return -1;
        }
        if (n == cap) {
            cap *= 2;
            int32_t *next = realloc(ids, cap * sizeof(*ids));
            if (!next) {
                free(ids);
                fclose(fp);
                return -1;
            }
            ids = next;
        }
        ids[n++] = (int32_t)value;
    }

    if (ferror(fp)) {
        fprintf(stderr, "infer_tokens: read error on %s\n", path);
        free(ids);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    if (n == 0) {
        fprintf(stderr, "infer_tokens: no token ids in %s\n", path);
        free(ids);
        return -1;
    }

    *out_tokens = ids;
    *out_count  = n;
    return 0;
}

static int write_generated(const char *path, const int32_t *tokens,
                           size_t count) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "infer_tokens: fopen(%s): %s\n", path,
                strerror(errno));
        return -1;
    }
    for (size_t i = 0; i < count; i++) {
        fprintf(fp, "%s%d", (i == 0) ? "" : " ", tokens[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
    return 0;
}

static long parse_long_arg(const char *s, const char *name) {
    char *end = NULL;
    errno     = 0;
    long v    = strtol(s, &end, 10);
    if (errno || !end || *end != '\0') {
        fprintf(stderr, "infer_tokens: invalid %s: %s\n", name, s);
        return -1;
    }
    return v;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    if (argc != 6 && argc != 7) {
        fprintf(stderr,
                "Usage: %s <weights_dir> <kernel_bin_dir> <input_tokens.txt> "
                "<max_new_tokens> <output_tokens.txt> [eos_token_id]\n",
                argv[0]);
        return 2;
    }

    const char *weights_dir    = argv[1];
    const char *kernel_bin_dir = argv[2];
    const char *input_path     = argv[3];
    const char *output_path    = argv[5];

    long max_new_long = parse_long_arg(argv[4], "max_new_tokens");
    if (max_new_long <= 0) {
        fprintf(stderr, "infer_tokens: max_new_tokens must be > 0\n");
        return 2;
    }
    size_t max_new = (size_t)max_new_long;

    int32_t eos_token = -1;
    if (argc == 7) {
        long eos_long = parse_long_arg(argv[6], "eos_token_id");
        if (eos_long < 0 || eos_long > INT32_MAX)
            return 2;
        eos_token = (int32_t)eos_long;
    }

    int32_t *prompt       = NULL;
    size_t   prompt_count = 0;
    if (read_tokens(input_path, &prompt, &prompt_count) < 0)
        return 1;

    gpgpu_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        free(prompt);
        return 1;
    }

    int ret = gpuInit(ctx, "/dev/gpgpu", VRAM_SIZE);
    if (ret < 0) {
        fprintf(stderr, "infer_tokens: gpuInit failed: %d\n", ret);
        free(prompt);
        free(ctx);
        return 1;
    }
    printf("[init] gpuInit OK\n");

    transformer_config_t cfg = transformer_default_config();
    if (prompt_count > cfg.max_seq ||
        prompt_count + max_new - 1 > cfg.max_seq) {
        fprintf(stderr,
                "infer_tokens: prompt_count(%zu) + max_new_tokens(%zu) - 1 "
                "exceeds max_seq(%u)\n",
                prompt_count, max_new, cfg.max_seq);
        gpuDestroy(ctx);
        free(prompt);
        free(ctx);
        return 1;
    }
    for (size_t i = 0; i < prompt_count; i++) {
        if (prompt[i] < 0 || (uint32_t)prompt[i] >= cfg.vocab_size) {
            fprintf(stderr,
                    "infer_tokens: token[%zu]=%d outside vocab_size=%u\n", i,
                    prompt[i], cfg.vocab_size);
            gpuDestroy(ctx);
            free(prompt);
            free(ctx);
            return 1;
        }
    }

    weight_loader_t *wl = weight_loader_load(ctx, weights_dir);
    if (!wl) {
        fprintf(stderr, "infer_tokens: weight_loader_load failed\n");
        gpuDestroy(ctx);
        free(prompt);
        free(ctx);
        return 1;
    }
    printf("[init] weight_loader_load OK\n");

    kv_cache_t *kv =
        kv_cache_create(ctx, cfg.num_layers, cfg.num_kv_heads, cfg.max_seq,
                        cfg.head_dim);
    if (!kv) {
        fprintf(stderr, "infer_tokens: kv_cache_create failed\n");
        weight_loader_destroy(wl);
        gpuDestroy(ctx);
        free(prompt);
        free(ctx);
        return 1;
    }
    printf("[init] kv_cache_create OK\n");

    transformer_workspace_t *ws =
        transformer_init(ctx, wl, kv, &cfg, kernel_bin_dir);
    if (!ws) {
        fprintf(stderr, "infer_tokens: transformer_init failed\n");
        kv_cache_destroy(ctx, kv);
        weight_loader_destroy(wl);
        gpuDestroy(ctx);
        free(prompt);
        free(ctx);
        return 1;
    }
    printf("[init] transformer_init OK\n");

    int32_t *generated = calloc(max_new, sizeof(*generated));
    if (!generated) {
        transformer_destroy(ws);
        kv_cache_destroy(ctx, kv);
        weight_loader_destroy(wl);
        gpuDestroy(ctx);
        free(prompt);
        free(ctx);
        return 1;
    }

    printf("[prefill] %zu prompt tokens\n", prompt_count);
    int32_t next = -1;
    for (size_t i = 0; i < prompt_count; i++) {
        next = transformer_step(ws, prompt[i], (int32_t)i, -1);
        if (next < 0) {
            fprintf(stderr, "infer_tokens: prefill step %zu failed: %d\n", i,
                    next);
            ret = 1;
            goto cleanup;
        }
        printf("prefill step %zu: in=%d next=%d\n", i, prompt[i], next);
    }

    size_t gen_count = 0;
    generated[gen_count++] = next;
    printf("generate step 0: token=%d\n", next);

    for (size_t i = 1; i < max_new; i++) {
        if (eos_token >= 0 && next == eos_token)
            break;
        int32_t pos = (int32_t)(prompt_count + i - 1);
        next        = transformer_step(ws, next, pos, -1);
        if (next < 0) {
            fprintf(stderr, "infer_tokens: decode step %zu failed: %d\n", i,
                    next);
            ret = 1;
            goto cleanup;
        }
        generated[gen_count++] = next;
        printf("generate step %zu: token=%d\n", i, next);
    }

    if (write_generated(output_path, generated, gen_count) < 0) {
        ret = 1;
        goto cleanup;
    }
    printf("[done] wrote %zu generated tokens to %s\n", gen_count,
           output_path);
    ret = 0;

cleanup:
    free(generated);
    transformer_destroy(ws);
    kv_cache_destroy(ctx, kv);
    weight_loader_destroy(wl);
    gpuDestroy(ctx);
    free(prompt);
    free(ctx);
    return ret;
}
