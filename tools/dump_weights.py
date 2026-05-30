#!/usr/bin/env python3
"""
dump_weights.py — Qwen2-0.5B-Instruct → device-ready binary dump

产物布局(方案 C 按层打包 + 方案 Y tie 双份):
  weights/
    manifest.json
    embedding.bin                519 MB   [V=151936, D=896] row-major fp32
    lm_head.bin                  519 MB   [D=896, V=151936] row-major fp32 (transpose of embedding)
    final_norm.bin              3584 B    [896] fp32
    layer_00.bin           ~56.9 MB   按下表固定 offset 排
    ...
    layer_23.bin

每层 layer_NN.bin 内部固定 offset(bytes):
       0  input_layernorm.weight         [896]            3584
    3584  self_attn.q_proj.weight        [896, 896]    3211264
 3214848  self_attn.q_proj.bias          [896]            3584
 3218432  self_attn.k_proj.weight        [896, 128]     458752
 3677184  self_attn.k_proj.bias          [128]             512
 3677696  self_attn.v_proj.weight        [896, 128]     458752
 4136448  self_attn.v_proj.bias          [128]             512
 4136960  self_attn.o_proj.weight        [896, 896]    3211264
 7348224  post_attention_layernorm.weight [896]           3584
 7351808  mlp.gate_proj.weight           [896, 4864]  17432576
24784384  mlp.up_proj.weight             [896, 4864]  17432576
42216960  mlp.down_proj.weight           [4864, 896]  17432576
59649536  (TOTAL per-layer bytes)

所有 *_proj.weight 已 transpose 成 device gemm 直吃的 [K, N] 形态。
PyTorch .T 只改 stride; 必须 .contiguous() 真挪数据后再 tobytes(),
否则 dump 出来是原 row-major,3.7 时 gemm 全垃圾(钉死 7)。

bf16 → fp32 用 torch 默认 cast,等价于"高 16 位填充、低 16 位填 0",
与 HF 推理路径内部的 cast 语义 bit-exact(钉死 2)。
"""

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

# torch + transformers 是必需依赖
try:
    import torch
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
except ImportError as e:
    sys.exit(f"missing dep: {e}\n  pip install torch safetensors huggingface_hub")


# ─── 模型常量(Qwen2-0.5B-Instruct 实测,与 HF config.json 对照锁定) ──
MODEL_ID         = "Qwen/Qwen2-0.5B-Instruct"
HIDDEN_SIZE      = 896      # d_model
INTERMEDIATE     = 4864     # FFN 中间宽度
NUM_LAYERS       = 24
NUM_Q_HEADS      = 14
NUM_KV_HEADS     = 2        # GQA
HEAD_DIM         = 64       # hidden / num_q_heads
KV_DIM           = NUM_KV_HEADS * HEAD_DIM   # = 128, K/V proj 输出维度
VOCAB_SIZE       = 151936
RMS_NORM_EPS     = 1e-6
ROPE_THETA       = 1_000_000.0     # 与 V24 锁定一致
TIE_WORD_EMB     = True

# 每层内部固定 offset(必须与上表一致;device 端 weight_loader 用同一份)
LAYER_OFFSETS = {
    "input_layernorm.weight":        (        0, (HIDDEN_SIZE,)),
    "self_attn.q_proj.weight":       (     3584, (HIDDEN_SIZE, HIDDEN_SIZE)),
    "self_attn.q_proj.bias":         (  3214848, (HIDDEN_SIZE,)),
    "self_attn.k_proj.weight":       (  3218432, (HIDDEN_SIZE, KV_DIM)),
    "self_attn.k_proj.bias":         (  3677184, (KV_DIM,)),
    "self_attn.v_proj.weight":       (  3677696, (HIDDEN_SIZE, KV_DIM)),
    "self_attn.v_proj.bias":         (  4136448, (KV_DIM,)),
    "self_attn.o_proj.weight":       (  4136960, (HIDDEN_SIZE, HIDDEN_SIZE)),
    "post_attention_layernorm.weight": (7348224, (HIDDEN_SIZE,)),
    "mlp.gate_proj.weight":          (  7351808, (HIDDEN_SIZE, INTERMEDIATE)),
    "mlp.up_proj.weight":            ( 24784384, (HIDDEN_SIZE, INTERMEDIATE)),
    "mlp.down_proj.weight":          ( 42216960, (INTERMEDIATE, HIDDEN_SIZE)),
}
LAYER_BYTES = 59_649_536    # 必须与最后一项 offset + size 一致

# 哪些 *.weight 需要 transpose 后再 dump(钉死 3)
NEEDS_TRANSPOSE = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
}


# ─── 工具函数 ───────────────────────────────────────────────────

def tensor_to_fp32_bytes(t: torch.Tensor) -> bytes:
    """bf16/fp16/fp32 → fp32 bytes, 已确保 contiguous."""
    if not t.is_contiguous():
        # 钉死 7: transpose 后必须 contiguous,否则 tobytes 拿到原内存
        t = t.contiguous()
    t = t.to(torch.float32).contiguous()
    return t.numpy().tobytes()


def assert_shape(name: str, t: torch.Tensor, expected: tuple):
    if tuple(t.shape) != expected:
        sys.exit(f"FAIL shape mismatch: {name}\n"
                 f"  HF gave {tuple(t.shape)}, expected {expected}\n"
                 f"  ─ Qwen2 config may have changed; check NUM_KV_HEADS, HEAD_DIM, etc.")


def sha256_first_n(data: bytes, n: int = 32) -> str:
    """前 n 字节的 sha256,manifest 里留一份用于 sanity check."""
    return hashlib.sha256(data[:n]).hexdigest()[:16]


def stats(t: torch.Tensor) -> dict:
    """tensor 摘要,manifest 里留以供 HF reference 对照."""
    t32 = t.to(torch.float32)
    return {
        "shape":  list(t.shape),
        "mean":   float(t32.mean()),
        "std":    float(t32.std()),
        "min":    float(t32.min()),
        "max":    float(t32.max()),
        # 前 8 个元素 ─ 与 HF 跑同一个 tensor 对比,bit-exact 应当全等
        "first8": [float(x) for x in t32.flatten()[:8].tolist()],
    }


# ─── 主流程 ─────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weights",
                    help="output directory (default: ./weights)")
    ap.add_argument("--cache-dir", default=None,
                    help="HF cache dir (default: ~/.cache/huggingface)")
    ap.add_argument("--check-only", action="store_true",
                    help="don't dump, just print HF tensor stats")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 拉权重(命中缓存秒回)
    print(f"[1/5] Downloading {MODEL_ID} ...")
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=args.cache_dir,
        allow_patterns=["*.safetensors", "*.json"],
    )
    print(f"  model files at: {model_path}")

    # Step 2: 找 safetensors 文件并打开
    st_files = sorted(Path(model_path).glob("*.safetensors"))
    if not st_files:
        sys.exit("FAIL: no .safetensors found in cache")
    print(f"[2/5] Opening {len(st_files)} safetensors file(s)")

    # 把所有 tensor 名→file 路径建索引
    tensor_index = {}
    for st in st_files:
        with safe_open(st, framework="pt") as f:
            for k in f.keys():
                tensor_index[k] = st
    print(f"  total tensors in checkpoint: {len(tensor_index)}")

    def load_tensor(name: str) -> torch.Tensor:
        if name not in tensor_index:
            sys.exit(f"FAIL: tensor '{name}' not found in checkpoint")
        with safe_open(tensor_index[name], framework="pt") as f:
            return f.get_tensor(name)

    # Step 3: dump embedding + lm_head
    print(f"[3/5] Dumping embedding + lm_head")
    emb = load_tensor("model.embed_tokens.weight")
    assert_shape("embed_tokens.weight", emb, (VOCAB_SIZE, HIDDEN_SIZE))

    if args.check_only:
        print(f"  embed_tokens: {stats(emb)}")
    else:
        # embedding.bin: [V, D] row-major, kernel 直接吃
        with open(out_dir / "embedding.bin", "wb") as f:
            f.write(tensor_to_fp32_bytes(emb))

        # lm_head.bin: 钉死 6 ─ tie 时 HF safetensors 里没有 lm_head.weight,
        # 回退到 embed_tokens.T。无论 tie 与否,我们 device 端都要 [D, V] 形态.
        if "lm_head.weight" in tensor_index:
            lm_head = load_tensor("lm_head.weight")
            assert_shape("lm_head.weight", lm_head, (VOCAB_SIZE, HIDDEN_SIZE))
            assert TIE_WORD_EMB == False, "found lm_head.weight but config says tied"
        else:
            assert TIE_WORD_EMB, "no lm_head.weight but config says NOT tied"
            print(f"  (lm_head tied to embedding, generating transpose)")
            lm_head = emb  # 同一份数据,transpose 后写出

        with open(out_dir / "lm_head.bin", "wb") as f:
            # transpose: [V, D] → [D, V]
            f.write(tensor_to_fp32_bytes(lm_head.T))

    # Step 4: dump 24 层
    print(f"[4/5] Dumping {NUM_LAYERS} layers")
    layer_stats = {}

    for layer_idx in range(NUM_LAYERS):
        prefix = f"model.layers.{layer_idx}."
        buf = bytearray(LAYER_BYTES)  # 整层一次性 alloc,按 offset 填
        per_layer_stats = {}

        for local_name, (offset, expected_shape) in LAYER_OFFSETS.items():
            full_name = prefix + local_name
            t = load_tensor(full_name)

            # HF 原始 shape ─ weight 要在 transpose 前 assert
            if local_name in NEEDS_TRANSPOSE:
                # NEEDS_TRANSPOSE 的 expected_shape 是 transpose 后形态,
                # HF 原 shape 是反过来的
                hf_shape = (expected_shape[1], expected_shape[0])
                assert_shape(full_name, t, hf_shape)
                t = t.T.contiguous()  # 钉死 7: 必须 contiguous
            else:
                assert_shape(full_name, t, expected_shape)

            data = tensor_to_fp32_bytes(t)
            expected_bytes = 4
            for d in expected_shape:
                expected_bytes *= d
            if len(data) != expected_bytes:
                sys.exit(f"FAIL byte count: {full_name} got {len(data)} expected {expected_bytes}")

            # 写入到 layer buffer 对应 offset
            buf[offset:offset + expected_bytes] = data

            # 只在 layer 0 留 stats 减小 manifest 大小
            if layer_idx == 0:
                per_layer_stats[local_name] = stats(t) | {
                    "offset": offset,
                    "bytes":  expected_bytes,
                    "head32_sha": sha256_first_n(data),
                }

        if args.check_only:
            if layer_idx == 0:
                for k, v in per_layer_stats.items():
                    print(f"  layer0.{k}: {v}")
            continue

        with open(out_dir / f"layer_{layer_idx:02d}.bin", "wb") as f:
            f.write(bytes(buf))

        if layer_idx == 0:
            layer_stats["layer_0"] = per_layer_stats
        print(f"  layer {layer_idx:2d} done ({LAYER_BYTES} bytes)")

    # Step 5: dump final norm + manifest
    print(f"[5/5] Dumping final_norm + manifest")
    final_norm = load_tensor("model.norm.weight")
    assert_shape("model.norm.weight", final_norm, (HIDDEN_SIZE,))

    if not args.check_only:
        with open(out_dir / "final_norm.bin", "wb") as f:
            f.write(tensor_to_fp32_bytes(final_norm))

    # manifest ─ device 端 weight_loader 不需要,但调试/对照非常关键
    manifest = {
        "model_id":        MODEL_ID,
        "hidden_size":     HIDDEN_SIZE,
        "intermediate":    INTERMEDIATE,
        "num_layers":      NUM_LAYERS,
        "num_q_heads":     NUM_Q_HEADS,
        "num_kv_heads":    NUM_KV_HEADS,
        "head_dim":        HEAD_DIM,
        "kv_dim":          KV_DIM,
        "vocab_size":      VOCAB_SIZE,
        "rms_norm_eps":    RMS_NORM_EPS,
        "rope_theta":      ROPE_THETA,
        "tie_word_embeddings": TIE_WORD_EMB,
        "dtype":           "fp32",
        "layer_bytes":     LAYER_BYTES,
        "layer_offsets":   {k: {"offset": v[0], "shape": list(v[1])}
                            for k, v in LAYER_OFFSETS.items()},
        "files": {
            "embedding":  {"shape": [VOCAB_SIZE, HIDDEN_SIZE],
                           "bytes": VOCAB_SIZE * HIDDEN_SIZE * 4},
            "lm_head":    {"shape": [HIDDEN_SIZE, VOCAB_SIZE],
                           "bytes": HIDDEN_SIZE * VOCAB_SIZE * 4,
                           "note":  "transpose of embedding (tie_word_embeddings)"},
            "final_norm": {"shape": [HIDDEN_SIZE], "bytes": HIDDEN_SIZE * 4},
        },
        "stats_layer_0": layer_stats.get("layer_0", {}),
        "stats_embedding": stats(emb),
        "stats_final_norm": stats(final_norm),
    }

    if not args.check_only:
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  manifest at: {out_dir / 'manifest.json'}")

    # Summary
    total_bytes = (
        VOCAB_SIZE * HIDDEN_SIZE * 4 * 2  # emb + lm_head
        + HIDDEN_SIZE * 4                  # final_norm
        + NUM_LAYERS * LAYER_BYTES
    )
    print(f"\nDone. Total dump size: {total_bytes:,} bytes "
          f"({total_bytes / 1024 / 1024 / 1024:.2f} GB)")


if __name__ == "__main__":
    main()