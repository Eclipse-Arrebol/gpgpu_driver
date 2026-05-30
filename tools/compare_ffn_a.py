import numpy as np
import sys
import os

step = sys.argv[1] if len(sys.argv) > 1 else "00"
base = f"dump/step_{step}"

silu_mul_path = os.path.join(base, "layer_12_silu_mul.bin")
just_before_path = os.path.join(base, "layer_12_ffn_a_just_before_down.bin")

if not os.path.exists(silu_mul_path):
    print(f"ERROR: {silu_mul_path} not found"); sys.exit(1)
if not os.path.exists(just_before_path):
    print(f"ERROR: {just_before_path} not found"); sys.exit(1)

silu_mul = np.fromfile(silu_mul_path, dtype=np.float32)
just_before = np.fromfile(just_before_path, dtype=np.float32)

print(f"silu_mul.shape:      {silu_mul.shape}")
print(f"just_before.shape:   {just_before.shape}")

diff = np.abs(silu_mul - just_before)
print(f"max |diff| = {diff.max():.6e}")
print(f"any diff != 0:  {(diff != 0).any()}")

if (diff != 0).any():
    i = int(np.argmax(diff))
    print(f"first nonzero at i={i}: silu_mul={silu_mul[i]} just_before={just_before[i]}")
    mask = diff != 0
    print(f"total nonzero: {mask.sum()} / {len(diff)}")
else:
    print("PASS: bit-exact match")
