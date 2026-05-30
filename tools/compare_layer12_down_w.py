#!/usr/bin/env python3
"""Compare layer 12 down_proj weight: host file vs device VRAM D2H."""
import numpy as np

I = 4864
D = 896
DOWN_OFFSET_IN_LAYER = 42216960  # WL_OFF_DOWN_PROJ_W
DOWN_BASE_ADDR = 0xC2842E00

# Host file layer 12 down_proj weight.
host_w = np.fromfile(
    "weights/layer_12.bin",
    dtype=np.float32,
    count=I * D,
    offset=DOWN_OFFSET_IN_LAYER,
)
print(f"host_w.shape = {host_w.shape}  size = {host_w.size}")

# Device VRAM content dumped immediately before layer 12 down gemm.
dev_w = np.fromfile("dump/step_00/layer_12_down_w_d2h.bin", dtype=np.float32)
print(f"dev_w.shape  = {dev_w.shape}  size = {dev_w.size}")

assert host_w.size == dev_w.size, (
    f"SIZE MISMATCH: host {host_w.size} vs dev {dev_w.size}"
)

diff = np.abs(host_w - dev_w)
print(f"\nmax |diff|  = {diff.max():.6e}")
print(f"mean |diff| = {diff.mean():.6e}")
print(f"any diff != 0:  {(diff != 0).any()}")
print(f"total nonzero count: {(diff != 0).sum()}")

if (diff != 0).any():
    nz = np.where(diff != 0)[0]
    print(f"\nfirst nonzero idx: {nz[0]}")
    print(f"last  nonzero idx: {nz[-1]}")

    chunks = 16
    chunk_size = host_w.size // chunks
    print(
        f"\n--- chunk analysis ({chunks} chunks of {chunk_size} fp32 = "
        f"{chunk_size * 4} bytes each) ---"
    )
    for c in range(chunks):
        lo = c * chunk_size
        hi = (c + 1) * chunk_size if c < chunks - 1 else host_w.size
        d = diff[lo:hi]
        n_nz = int((d != 0).sum())
        max_d = float(d.max())
        addr_lo = DOWN_BASE_ADDR + lo * 4
        addr_hi = DOWN_BASE_ADDR + hi * 4
        marker = "  <== HAS DIFF" if n_nz > 0 else ""
        print(
            f"  chunk {c:2d}: idx [{lo:9d}, {hi:9d})  "
            f"addr [0x{addr_lo:08x}, 0x{addr_hi:08x})  "
            f"nz={n_nz:7d}  max_d={max_d:.3e}{marker}"
        )

    first_idx = int(nz[0])
    row = first_idx // D
    col = first_idx % D
    print(
        f"\nfirst nonzero idx={first_idx}: row={row} (of {I}), "
        f"col={col} (of {D})"
    )
    print(f"  host[{first_idx}] = {host_w[first_idx]:.6e}")
    print(f"  dev [{first_idx}] = {dev_w[first_idx]:.6e}")

    last_idx = int(nz[-1])
    row = last_idx // D
    col = last_idx % D
    print(f"last nonzero idx={last_idx}: row={row}, col={col}")
    print(f"  host[{last_idx}] = {host_w[last_idx]:.6e}")
    print(f"  dev [{last_idx}] = {dev_w[last_idx]:.6e}")
else:
    print("\nBIT-EXACT: device VRAM matches host file completely.")
