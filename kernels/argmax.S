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

# 把 ft2 写到 dump[lane * 8 + slot]
# 用 t6 临时,不破坏 t0..t4 (REDUCE 在用)
.macro DUMP_FT2 slot
    slli  t6, t0, 5             # lane * 32 (= lane * 8 * 4)
    addi  t6, t6, (\slot * 4)   # + slot * 4
    add   t6, t6, a6            # + debug_base
    fmv.x.s a7, ft2
    sw    a7, 0(t6)
.endm

.globl _start
_start:
    lw    a1, 4(a0)              # V
    lw    a2, 8(a0)              # out_base
    lw    a6, 12(a0)             # debug_base
    lw    a0, 0(a0)              # logits_base

    csrr  t0, mhartid
    andi  t0, t0, 0x1F

    li    s0, 0x7F7FFFFF
    li    a3, 0xFF800000
    fmv.s.x ft1, a3

    addi  t3, a1, 31
    srli  t3, t3, 5

    li    t1, 0
phaseA_loop:
    slli  t2, t1, 5
    add   t2, t2, t0
    bgeu  t2, a1, phaseA_skip
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

    DUMP_FT2 0                   # slot 0: phaseA 之后

    VSHFL_XOR  3, 1, 16
    VSHFL_XOR  4, 2, 16
    REDUCE_STEP
    DUMP_FT2 1                   # slot 1: after XOR 16

    VSHFL_XOR  3, 1, 8
    VSHFL_XOR  4, 2, 8
    REDUCE_STEP
    DUMP_FT2 2                   # slot 2: after XOR 8

    VSHFL_XOR  3, 1, 4
    VSHFL_XOR  4, 2, 4
    REDUCE_STEP
    DUMP_FT2 3                   # slot 3: after XOR 4

    VSHFL_XOR  3, 1, 2
    VSHFL_XOR  4, 2, 2
    REDUCE_STEP
    DUMP_FT2 4                   # slot 4: after XOR 2

    VSHFL_XOR  3, 1, 1
    VSHFL_XOR  4, 2, 1
    REDUCE_STEP
    DUMP_FT2 5                   # slot 5: after XOR 1

    # slot 6: 最终 s0
    slli  t6, t0, 5
    addi  t6, t6, 24             # 6 * 4
    add   t6, t6, a6
    sw    s0, 0(t6)

    bnez  t0, done
    fmv.x.s a4, ft2
    sw    a4, 0(a2)
done:
    ebreak