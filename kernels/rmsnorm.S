.section .text
.macro VSHFL_XOR rd, rs1, mask
    .word (\mask << 20) | (\rs1 << 15) | (\rd << 7) | 0x0B
.endm
.globl _start
_start:
    lw   a1, 0(a0)        # in_off
    lw   a2, 4(a0)        # out_off
    lw   a3, 8(a0)        # weight_off
    lw   a4, 12(a0)       # d (未直接使用)
    flw  fa1, 16(a0)      # inv_d = 1.0f/d
    flw  fa2, 20(a0)      # eps

    csrrs t0, mhartid, x0
    andi  t0, t0, 0x1F      # lane_id

    li    t3, 112           # 28 * 4
    mul   t1, t0, t3        # t1 = lane_id * 112（字节偏移）

    # === 阶段1：每个 lane 累加自己的 28 个 x² ===
    add   t2, a1, t1        # 输入起始地址
    fmv.s.x ft1, zero       # ft1 = 0.0f（累加器）
    li    t0, 28
sum_loop:
    flw   ft0, 0(t2)
    fmul.s ft0, ft0, ft0
    fadd.s ft1, ft1, ft0
    addi  t2, t2, 4
    addi  t0, t0, -1
    bnez  t0, sum_loop

    # === 阶段2：跨 lane 求和（warp shuffle）===
    fmv.s ft0, ft1           # ft0 = 本地平方和
    VSHFL_XOR 2, 0, 1
    fadd.s ft0, ft0, ft2
    VSHFL_XOR 2, 0, 2
    fadd.s ft0, ft0, ft2
    VSHFL_XOR 2, 0, 4
    fadd.s ft0, ft0, ft2
    VSHFL_XOR 2, 0, 8
    fadd.s ft0, ft0, ft2
    VSHFL_XOR 2, 0, 16
    fadd.s ft0, ft0, ft2    # ft0 = 全局平方和（所有 lane 之和）

    # === 阶段3：计算 rrms = 1 / rms ===
    fmul.s  ft1, ft0, fa1
    fadd.s  ft1, ft1, fa2
    fsqrt.s ft1, ft1
    li      t5, 0x3F800000
    fmv.s.x ft5, t5
    fdiv.s  ft1, ft5, ft1     # ft1 = rrms

    # === 阶段4：归一化并乘以 gamma ===
    add   t2, a1, t1
    add   t3, a3, t1
    add   t4, a2, t1
    li    t0, 28
norm_loop:
    flw    ft2, 0(t2)
    flw    ft3, 0(t3)
    fmul.s ft4, ft2, ft1     # x * rrms
    fmul.s ft4, ft4, ft3     # * gamma
    fsw    ft4, 0(t4)
    addi   t2, t2, 4
    addi   t3, t3, 4
    addi   t4, t4, 4
    addi   t0, t0, -1
    bnez   t0, norm_loop

    ebreak