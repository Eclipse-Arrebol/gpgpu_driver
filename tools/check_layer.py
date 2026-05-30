#!/usr/bin/env python3
"""
check_layer.py — 相对 ULP 对照 device dump vs reference

策略:对照 step_00(prefill 第 1 个 token),看 layer 0 内部 16 个算子,
按 V28 §3.2 顺序逐个 check,全部跑完显示结果。

跳查:layer 23 末端 + final_norm + logits

判定规则:rel_err = max_err / max(|ref|.max(), 1e-30)
        est_ULP = rel_err * 2**23
        PASS if est_ULP <= threshold (按算子类型区分)
"""
import numpy as np
import sys

REF_DIR  = "ref"
DUMP_DIR = "dump"

# Qwen2-0.5B 常量
D            = 896
KV_DIM       = 128        # 2 * 64
I            = 4864
NUM_Q_HEADS  = 14
NUM_KV_HEADS = 2
HEAD_DIM     = 64
VOCAB        = 151936

# Layer 0 内部 16 个算子按 V28 §3.2 顺序
# (name, dev_numel, ref_layout, note)
SPECS = [
    ("input_norm",      D,                       "seq_first",  "rmsnorm"),
    ("q_proj",          D,                       "seq_first",  "gemm + bias"),
    ("k_proj",          KV_DIM,                  "seq_first",  "gemm + bias"),
    ("v_proj",          KV_DIM,                  "seq_first",  "gemm + bias"),
    ("q_rope",          NUM_Q_HEADS * HEAD_DIM,  "head_seq",   "[num_heads, seq, head_dim] in ref"),
    ("k_rope",          NUM_KV_HEADS * HEAD_DIM, "head_seq",   "[num_heads, seq, head_dim] in ref"),
    ("attn_out",        D,                       "seq_first",  "pv_decode output"),
    ("o_proj",          D,                       "seq_first",  "gemm (no bias)"),
    ("after_attn_res",  D,                       "seq_first",  "broadcast_add"),
    ("post_norm",       D,                       "seq_first",  "rmsnorm"),
    ("gate",            I,                       "seq_first",  "gemm"),
    ("up",              I,                       "seq_first",  "gemm"),
    ("silu_gate",       I,                       "seq_first",  "silu(gate)"),
    ("silu_mul",        I,                       "seq_first",  "silu_gate * up"),
    ("down",            D,                       "seq_first",  "gemm"),
    ("after_ffn_res",   D,                       "seq_first",  "broadcast_add"),
]

# 4 ULP 给 elementwise(broadcast_add, silu, vmul, rope, embedding, argmax)
# 32 ULP 给 reduction 算子(rmsnorm, gemm, softmax, attn, qkt/pv_decode)
ULP_TOL = {
    "input_norm":       32,
    "q_proj":           32,
    "k_proj":           32,
    "v_proj":           32,
    "q_rope":            4,
    "k_rope":            4,
    "attn_out":         32,
    "o_proj":           32,
    "after_attn_res":    4,
    "post_norm":        32,
    "gate":             32,
    "up":               32,
    "silu_gate":         4,
    "silu_mul":          4,
    "down":             32,
    "after_ffn_res":     4,
}

def load_fp32(path):
    return np.fromfile(path, dtype=np.float32)

def get_ref_slice(ref_path, dev_numel, ref_layout, step_idx_in_seq):
    """
    从 ref/step_00/<file> 中切出对应 device dump step_NN 的位置。

    ref_layout:
      - "seq_first" : ref shape [seq, *], 取 ref[step_idx_in_seq, :]
      - "head_seq"  : ref shape [num_heads, seq, head_dim],
                      取 ref[:, step_idx_in_seq, :].reshape(-1)  (与 device [num_heads, head_dim] 对齐)
    """
    ref_raw = load_fp32(ref_path)
    if ref_layout == "seq_first":
        seq = 10
        assert ref_raw.size == seq * dev_numel, \
            f"ref size {ref_raw.size} != seq*dev_numel={seq*dev_numel} for {ref_path}"
        ref_full = ref_raw.reshape(seq, dev_numel)
        return ref_full[step_idx_in_seq]
    elif ref_layout == "head_seq":
        num_heads = dev_numel // HEAD_DIM
        seq = 10
        assert ref_raw.size == num_heads * seq * HEAD_DIM, \
            f"ref size {ref_raw.size} != {num_heads}*{seq}*{HEAD_DIM} for {ref_path}"
        ref_full = ref_raw.reshape(num_heads, seq, HEAD_DIM)
        ref_slice = ref_full[:, step_idx_in_seq, :].reshape(-1)
        return ref_slice
    else:
        raise ValueError(f"unknown ref_layout {ref_layout}")

def check_one(step_idx, layer_idx, name, dev_numel, ref_layout, note, ulp_tol=None):
    ref_path = f"{REF_DIR}/step_00/layer_{layer_idx:02d}_{name}.bin"
    dev_path = f"{DUMP_DIR}/step_{step_idx:02d}/layer_{layer_idx:02d}_{name}.bin"

    ref_slice = get_ref_slice(ref_path, dev_numel, ref_layout, step_idx_in_seq=step_idx)
    dev = load_fp32(dev_path)

    if dev.size != dev_numel:
        print(f"[layer {layer_idx:02d}] {name:<18s} SHAPE MISMATCH "
              f"dev.size={dev.size} != expected {dev_numel}")
        return False

    diff = np.abs(ref_slice - dev)
    max_err = diff.max()
    max_i   = int(np.argmax(diff))

    absmax_ref = np.abs(ref_slice).max()
    denom = max(absmax_ref, 1e-30)
    rel_err = max_err / denom
    est_ulp = rel_err * (2**23)

    threshold = ulp_tol if ulp_tol is not None else ULP_TOL.get(name, 4)
    passed = (est_ulp <= threshold)

    status = "PASS" if passed else "FAIL"
    print(f"[layer {layer_idx:02d}] {name:<18s} "
          f"absmax={absmax_ref:.2e}  max_err={max_err:.2e}  "
          f"rel_err={rel_err:.1e}  ≈{est_ulp:.1f} ULP  "
          f"≤{threshold}  {status}  ({note})")

    if not passed:
        print(f"             ref[:8]    = {ref_slice[:8]}")
        print(f"             dev[:8]    = {dev[:8]}")
        print(f"             ref[max_i] = {ref_slice[max_i]:.6e} (i={max_i})")
        print(f"             dev[max_i] = {dev[max_i]:.6e}")
        print(f"             ref stats: mean={ref_slice.mean():.3e} std={ref_slice.std():.3e} "
              f"absmax={absmax_ref:.3e}")
        print(f"             dev stats: mean={dev.mean():.3e} std={dev.std():.3e} "
              f"absmax={np.abs(dev).max():.3e}")

    return passed

def check_after_embed():
    ref_path = f"{REF_DIR}/step_00/after_embed.bin"
    dev_path = f"{DUMP_DIR}/step_00/after_embed.bin"
    ref_full = load_fp32(ref_path).reshape(10, D)
    dev = load_fp32(dev_path)
    max_err = np.abs(ref_full[0] - dev).max()
    absmax_ref = np.abs(ref_full[0]).max()
    denom = max(absmax_ref, 1e-30)
    rel_err = max_err / denom
    est_ulp = rel_err * (2**23)
    status = "PASS" if max_err == 0 else "FAIL"
    print(f"[non-layer] after_embed  absmax={absmax_ref:.2e}  max_err={max_err:.2e}  "
          f"rel_err={rel_err:.1e}  ≈{est_ulp:.1f} ULP  {status}")
    return max_err == 0

def check_non_layer(step_idx, name, dev_numel):
    """对照 non-layer tensor (after_embed / final_norm / logits)"""
    ref_path = f"{REF_DIR}/step_00/{name}.bin"
    dev_path = f"{DUMP_DIR}/step_{step_idx:02d}/{name}.bin"
    ref_full = load_fp32(ref_path).reshape(10, dev_numel)
    ref_slice = ref_full[step_idx]
    dev = load_fp32(dev_path)

    diff = np.abs(ref_slice - dev)
    max_err = diff.max()
    absmax = np.abs(ref_slice).max()
    rel_err = max_err / max(absmax, 1e-30)
    ulp = rel_err * (2**23)
    max_i = int(np.argmax(diff))

    print(f"[non-layer] {name:18s}  "
          f"absmax={absmax:.3e}  max_err={max_err:.3e}  "
          f"≈{ulp:.1f} ULP")
    print(f"             ref[max_i={max_i}]={ref_slice[max_i]:.6e}  "
          f"dev[max_i]={dev[max_i]:.6e}")
    return ref_slice, dev

def check_logits_topk(step_idx, k=10):
    """专门看 logits 的 top-k 排名"""
    ref_path = f"{REF_DIR}/step_00/logits.bin"
    dev_path = f"{DUMP_DIR}/step_{step_idx:02d}/logits.bin"
    ref_full = load_fp32(ref_path).reshape(10, VOCAB)
    ref = ref_full[step_idx]
    dev = load_fp32(dev_path)

    ref_topk = np.argsort(-ref)[:k]
    dev_topk = np.argsort(-dev)[:k]

    print(f"\n=== logits top-{k} comparison (step_{step_idx:02d}) ===")
    print(f"  ref top-{k}: {ref_topk.tolist()}")
    print(f"  dev top-{k}: {dev_topk.tolist()}")
    print(f"  ref top-{k} values: {ref[ref_topk]}")
    print(f"  dev top-{k} values: {dev[dev_topk]}")
    print(f"  ref argmax = {int(np.argmax(ref))}")
    print(f"  dev argmax = {int(np.argmax(dev))}")
    print(f"  ref logits stats: min={ref.min():.3e} max={ref.max():.3e} "
          f"mean={ref.mean():.3e} std={ref.std():.3e}")
    print(f"  dev logits stats: min={dev.min():.3e} max={dev.max():.3e} "
          f"mean={dev.mean():.3e} std={dev.std():.3e}")
    diff = np.abs(ref - dev)
    print(f"  |ref-dev| max={diff.max():.3e}  mean={diff.mean():.3e}  "
          f"absmax(ref)={np.abs(ref).max():.3e}")

def check_layer_full(step_idx, layer_idx):
    """跑一个 layer 的全部 16 个算子,返回 first_fail_name"""
    print(f"\n--- Layer {layer_idx:02d} full check (step {step_idx:02d}) ---")
    first_fail = None
    for name, dev_numel, ref_layout, note in SPECS:
        passed = check_one(step_idx=step_idx, layer_idx=layer_idx, name=name,
                           dev_numel=dev_numel, ref_layout=ref_layout, note=note)
        if not passed and first_fail is None:
            first_fail = name
    return first_fail

def main():
    print(f"=== Compare device dump vs reference (ULP-based) ===")
    print(f"=== step_00 (prefill token 0, token_id=100, pos=0) ===\n")

    ok = check_after_embed()
    if not ok:
        print("\n=== after_embed FAIL, abort ===")
        sys.exit(1)

    print(f"\n--- Layer 0: {len(SPECS)} operators ---\n")

    all_pass = True
    first_fail = None

    for name, dev_numel, ref_layout, note in SPECS:
        ok = check_one(step_idx=0, layer_idx=0,
                       name=name, dev_numel=dev_numel,
                       ref_layout=ref_layout, note=note)
        if not ok and first_fail is None:
            first_fail = name
            all_pass = False

    print()
    if all_pass:
        print("=== ALL PASS (layer 0, 16 operators) ===")
    else:
        print(f"=== FIRST FAIL at layer_00/{first_fail} ===")

    # Layer 12 detailed (第一个出问题的层)
    check_layer_full(step_idx=0, layer_idx=12)

    # Layer 11 (最后一个 OK 的层) for comparison
    print("\n--- Layer 11 (last known good) full check ---")
    check_layer_full(step_idx=0, layer_idx=11)

    # 扫描所有 24 层的 after_ffn_res 误差演化
    print("\n--- All 24 layers: after_ffn_res evolution ---")
    prev_ulp = 0.0
    for L in range(24):
        name = "after_ffn_res"
        ref_path = f"{REF_DIR}/step_00/layer_{L:02d}_{name}.bin"
        dev_path = f"{DUMP_DIR}/step_00/layer_{L:02d}_{name}.bin"
        ref_full = load_fp32(ref_path).reshape(10, D)
        ref_slice = ref_full[0]
        dev = load_fp32(dev_path)

        diff = np.abs(ref_slice - dev)
        max_err = diff.max()
        absmax = np.abs(ref_slice).max()
        rel_err = max_err / max(absmax, 1e-30)
        ulp = rel_err * (2**23)

        if prev_ulp > 1e-3:
            ratio = ulp / prev_ulp
            ratio_str = f"x{ratio:6.2f}"
        else:
            ratio_str = "  --  "

        flag = ""
        if prev_ulp > 1e-3 and ulp / prev_ulp > 3.0:
            flag = "  <== JUMP"
        if ulp > 1e6:
            flag = "  <== BLOWN UP"

        print(f"  layer {L:02d}: absmax={absmax:.3e}  max_err={max_err:.3e}  "
              f"≈{ulp:8.1f} ULP  ratio={ratio_str}{flag}")
        prev_ulp = ulp

    # 跳查 layer 23 的最后一步
    print("\n--- Layer 23 (last) end-of-block check ---")
    check_one(step_idx=0, layer_idx=23, name="after_ffn_res",
              dev_numel=D, ref_layout="seq_first",
              note="layer 23 末端,进 final_norm 前", ulp_tol=64)

    # Non-layer tensors
    print("\n--- Non-layer tensors at step_00 ---")
    check_non_layer(0, "final_norm", D)

    # logits 单独看
    check_logits_topk(0, k=10)

if __name__ == "__main__":
    main()
