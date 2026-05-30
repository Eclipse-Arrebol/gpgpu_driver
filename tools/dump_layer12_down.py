import numpy as np

WL_OFF_DOWN_PROJ_W = 42216960
n = 16

for layer in [11, 12]:
    path = f"weights/layer_{layer:02d}.bin"
    data = np.fromfile(path, dtype=np.float32, count=n, offset=WL_OFF_DOWN_PROJ_W)
    print(f"host layer_{layer:02d}.bin @ offset {WL_OFF_DOWN_PROJ_W}, first {n} fp32:")
    print(data)
    print(f"raw bytes: {data.tobytes().hex()}")
    print()
