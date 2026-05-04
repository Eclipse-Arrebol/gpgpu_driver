.macro VEXP_S rd, rs1
    .word (0x7c << 25) | (0 << 20) | (\rs1 << 15) | (0 << 12) | (\rd << 7) | 0x53
.endm

.globl _start
_start:
    lw   a1, 0(a0)        # in_off
    lw   a2, 4(a0)        # out_off
    lw   a3, 8(a0)        # D

    csrrs t0, mhartid, x0
    andi  t0, t0, 0x1F
    li    t3, 112           # 28 * 4
    mul   t1, t0, t3

    li    t0, 28
    add   t2, a1, t1        # 输入指针
    add   t3, a2, t1        # 输出指针

    li    t4, 0x3F800000    # 用 t4 保存常数 1.0 的位模式
    fmv.s.x ft4, t4          # ft4 = 1.0

loop:
    flw    ft1, 0(t2)        # x
    fsgnjn.s ft2, ft1, ft1   # ft2 = -x
    VEXP_S 2, 2          # ft2 = exp(-x)
    fadd.s  ft2, ft4, ft2    # ft2 = 1 + exp(-x)
    fdiv.s  ft3, ft4, ft2    # ft3 = sigmoid(x) 注意用 ft3
    fmul.s  ft0, ft1, ft3    # ft0 = x * sigmoid(x) = SiLU(x)
    fsw     ft0, 0(t3)       # 存储结果

    addi   t2, t2, 4
    addi   t3, t3, 4
    addi   t0, t0, -1
    bnez   t0, loop

    ebreak