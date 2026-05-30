# silu.S — SiLU activation: out[i] = x[i] * sigmoid(x[i])
#
# M3.7.1 改造: 单 warp 硬编码 D=896 → multi-block runtime D
# 沿用 broadcast_add.S 的 multi-block grid pattern:
#   mhartid layout: [31:13] block_id | [12:5] warp_id | [4:0] lane_id
#   linear_id = block_id * 32 + lane_id
#   grid = ceil(D/32), block = 32
#   每 thread 处理 1 个元素
#
# args 布局:
#   0:  in_off  (fp32 [D])
#   4:  out_off (fp32 [D], 可 == in_off 即 in-place)
#   8:  D       (int32, 元素总数)

.macro VEXP_S rd, rs1
    .word (0x7c << 25) | (0 << 20) | (\rs1 << 15) | (0 << 12) | (\rd << 7) | 0x53
.endm

.globl _start
_start:
    lw   a1, 0(a0)        # a1 = in_off
    lw   a2, 4(a0)        # a2 = out_off
    lw   a3, 8(a0)        # a3 = D

    # ===== thread id 计算 (沿用 broadcast_add.S pattern) =====
    csrrs t6, mhartid, x0
    andi  t1, t6, 0x1F          # t1 = lane_id (bit 0~4)
    srli  t6, t6, 13            # t6 = block_id (bit 13~31)
    slli  t6, t6, 5             # block_id * 32
    add   t0, t6, t1            # t0 = linear_id = block_id*32 + lane_id

    # 越界检查 (放在所有计算之前, 原则 22)
    bge   t0, a3, done_oob

    # ===== 计算元素地址 =====
    slli  t1, t0, 2             # t1 = linear_id * 4
    add   t2, a1, t1            # t2 = &in[linear_id]
    add   t3, a2, t1            # t3 = &out[linear_id]

    # ===== 常量准备 =====
    li    t4, 0x3F800000        # 1.0f 的位模式
    fmv.s.x ft4, t4             # ft4 = 1.0f

    # ===== SiLU: x * sigmoid(x) =====
    flw    ft1, 0(t2)           # ft1 = x
    fsgnjn.s ft2, ft1, ft1      # ft2 = -x
    VEXP_S 2, 2                 # ft2 = exp(-x)
    fadd.s  ft2, ft4, ft2       # ft2 = 1 + exp(-x)
    fdiv.s  ft3, ft4, ft2       # ft3 = sigmoid(x) = 1/(1+exp(-x))
    fmul.s  ft0, ft1, ft3       # ft0 = x * sigmoid(x) = SiLU(x)
    fsw    ft0, 0(t3)           # out[linear_id] = SiLU(x)

done_oob:
    ebreak
