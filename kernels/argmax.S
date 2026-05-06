.section .text

.macro REDUCE_STEP
    fmv.x.s a5, ft4
    flt.s t5, ft1, ft3
    bnez  t5, 1f
    feq.s t5, ft1, ft3
    beqz  t5, 2f
    bgeu  a5, s0, 2f
1:  fmv.s ft1, ft3
    fmv.s ft2, ft4
    mv    s0, a5
2:
.endm

.macro VSHFL_XOR rd, rs1, mask
    .word (0x00 << 25) | (\mask << 20) | (\rs1 << 15) | (0 << 12) | (\rd << 7) | 0x0B
.endm

.globl _start
_start:
    lw    a1, 4(a0)
    lw    a2, 8(a0)
    lw    a0, 0(a0)

    csrr  t0, mhartid
    andi  t0, t0, 0x1F

    li    s0, 0
    li    a3, 0xFF800000
    fmv.s.x ft1, a3

    srli  t3, a1, 5

    li    t1, 0
phaseA_loop:
    slli  t2, t1, 5
    add   t2, t2, t0
    slli  t4, t2, 2
    add   t4, t4, a0
    flw   ft0, 0(t4)
    flt.s t5, ft1, ft0
    beqz  t5, phaseA_skip
    fmv.s ft1, ft0
    mv    s0, t2
phaseA_skip:
    addi  t1, t1, 1
    bne   t1, t3, phaseA_loop

    fmv.s.x ft2, s0

    VSHFL_XOR  3, 1, 16
    VSHFL_XOR  4, 2, 16
    REDUCE_STEP

    VSHFL_XOR  3, 1, 8
    VSHFL_XOR  4, 2, 8
    REDUCE_STEP

    VSHFL_XOR  3, 1, 4
    VSHFL_XOR  4, 2, 4
    REDUCE_STEP

    VSHFL_XOR  3, 1, 2
    VSHFL_XOR  4, 2, 2
    REDUCE_STEP

    VSHFL_XOR  3, 1, 1
    VSHFL_XOR  4, 2, 1
    REDUCE_STEP

    bnez  t0, done
    fmv.x.s a4, ft2
    sw    a4, 0(a2)
done:
    ebreak