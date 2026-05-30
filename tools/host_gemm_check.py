import numpy as np

D = 896
I = 4864

DOWN_OFFSET_IN_LAYER = 42216960  # WL_OFF_DOWN_PROJ_W
DOWN_NUMEL = I * D  # 4864 * 896

def host_matmul_down(layer_id, step_id):
    a = np.fromfile(
        f"dump/step_{step_id:02d}/layer_{layer_id:02d}_silu_mul.bin",
        dtype=np.float32)
    assert a.size == I

    weight_path = f"weights/layer_{layer_id:02d}.bin"
    b = np.fromfile(weight_path, dtype=np.float32,
                    count=DOWN_NUMEL, offset=DOWN_OFFSET_IN_LAYER)
    assert b.size == DOWN_NUMEL
    b = b.reshape(I, D)

    c_host = a @ b
    return c_host.astype(np.float32)

for layer_id in [11, 12]:
    print(f"\n=== Layer {layer_id} down host vs device ===")
    c_host = host_matmul_down(layer_id, step_id=0)
    c_dev = np.fromfile(
        f"dump/step_00/layer_{layer_id:02d}_down.bin",
        dtype=np.float32)
    assert c_dev.size == D
    diff = np.abs(c_host - c_dev)
    print(f"  host[:8]    = {c_host[:8]}")
    print(f"  dev[:8]     = {c_dev[:8]}")
    print(f"  max |diff|  = {diff.max():.3e}")
    print(f"  mean |diff| = {diff.mean():.3e}")
    print(f"  argmax diff = {int(np.argmax(diff))}")
    print(f"  c_host stats: mean={c_host.mean():.3e} std={c_host.std():.3e} absmax={np.abs(c_host).max():.3e}")
    print(f"  c_dev  stats: mean={c_dev.mean():.3e}  std={c_dev.std():.3e}  absmax={np.abs(c_dev).max():.3e}")
