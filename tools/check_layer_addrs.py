layer_addrs = {
    0: 0x10000e00,
    1: 0x138e3c00,
    2: 0x171c6a00,
    3: 0x1aaa9800,
    4: 0x20000000,
    5: 0x238e2e00,
    6: 0x271c5c00,
    7: 0x2aaa8a00,
    8: 0x30000000,
    9: 0x338e2e00,
    10: 0x371c5c00,
    11: 0x3aaa8a00,
    12: 0xc0000000,
    13: 0xc38e3c00,
    14: 0xc71c5c00,
    15: 0xcaaa8a00,
    16: 0xd0000000,
    17: 0xd38e2e00,
    18: 0xd71c5c00,
    19: 0xdaaa8a00,
    20: 0xe0000000,
    21: 0xe38e2e00,
    22: 0xe71c5c00,
    23: 0xeaaa8a00,
}

WL_LAYER_BYTES = 59649536  # 0x38e2e00

print(f"WL_LAYER_BYTES = {WL_LAYER_BYTES} = 0x{WL_LAYER_BYTES:x}\n")
for i in range(1, 24):
    diff = layer_addrs[i] - layer_addrs[i-1]
    expected = WL_LAYER_BYTES
    delta = diff - expected
    flag = "" if delta == 0 else f"  <== OFFSET {delta:+d} ({delta:+x})"
    region_boundary = i in [4, 8, 12, 16, 20]
    rb = "  (region boundary)" if region_boundary else ""
    print(f"layer {i-1:2d} -> {i:2d}: diff = 0x{diff:08x} = {diff:11d}  delta from WL_LAYER_BYTES: {delta:+d}{flag}{rb}")
