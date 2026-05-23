.section .text

.macro VEXP_S rd, rs1
    .word (0x7c << 25) | (0 << 20) | (\rs1 << 15) | (0 << 12) | (\rd << 7) | 0x53
.endm

.macro VSHFL_XOR rd, rs1, mask
    .word (\mask << 20) | (\rs1 << 15) | (\rd << 7) | 0x0B
.endm

.globl _start
_start:
    lw    a1, 0(a0)          # a1 = scores_off
    lw    a2, 4(a0)          # a2 = cur_pos
    lw    a3, 8(a0)          # a3 = max_seq

    csrr  t0, mhartid
    andi  s3, t0, 0x1F       # s3 = lane_id
    srli  t1, t0, 13         # t1 = qh

    addi  s2, a2, 1          # s2 = S = cur_pos + 1

    # scores_row_base = scores_off + qh * max_seq * 4
    mul   t0, t1, a3
    slli  t0, t0, 2
    add   s1, a1, t0         # s1 = scores_row_base

    # iter_count = ceil((S - lane_id) / 32), 0 if lane_id >= S
    sub   s4, s2, s3
    addi  s4, s4, 31
    srli  s4, s4, 5
    #beqz  s4, done

    # ====== Pass 1: local_max ======
    li    t0, 0xff800000
    fmv.s.x ft0, t0          # ft0 = -INF

    mv    s5, s4
    slli  t1, s3, 2
    add   t1, s1, t1

p1_loop:
    beqz  s5, p1_done        # 0 次循环时跳过 loop 体
    flw   ft1, 0(t1)
    fmax.s ft0, ft0, ft1
    addi  t1, t1, 128
    addi  s5, s5, -1
    j     p1_loop 
p1_done:
    VSHFL_XOR 3, 0, 1
    fmax.s ft0, ft0, ft3
    VSHFL_XOR 3, 0, 2
    fmax.s ft0, ft0, ft3
    VSHFL_XOR 3, 0, 4
    fmax.s ft0, ft0, ft3
    VSHFL_XOR 3, 0, 8
    fmax.s ft0, ft0, ft3
    VSHFL_XOR 3, 0, 16
    fmax.s ft0, ft0, ft3    # ft0 = warp_max

    fmv.s ft5, ft0

    # ====== Pass 2: exp(x - warp_max), in-place, accumulate local_sum ======
    fmv.s.x ft6, x0         # ft6 = 0.0

    mv    s5, s4
    slli  t1, s3, 2
    add   t1, s1, t1

p2_loop:
    beqz s5, p2_done
    flw   ft0, 0(t1)
    fsub.s ft0, ft0, ft5
    VEXP_S 0, 0
    fsw   ft0, 0(t1)
    fadd.s ft6, ft6, ft0
    addi  t1, t1, 128
    addi  s5, s5, -1
    j     p2_loop 
p2_done:
    VSHFL_XOR 3, 6, 1
    fadd.s ft6, ft6, ft3
    VSHFL_XOR 3, 6, 2
    fadd.s ft6, ft6, ft3
    VSHFL_XOR 3, 6, 4
    fadd.s ft6, ft6, ft3
    VSHFL_XOR 3, 6, 8
    fadd.s ft6, ft6, ft3
    VSHFL_XOR 3, 6, 16
    fadd.s ft6, ft6, ft3    # ft6 = warp_sum

    # ====== Pass 3: divide by warp_sum, in-place ======
    mv    s5, s4
    beqz  s5, done  
    slli  t1, s3, 2
    add   t1, s1, t1

p3_loop:
    flw   ft0, 0(t1)
    fdiv.s ft0, ft0, ft6
    fsw   ft0, 0(t1)
    addi  t1, t1, 128
    addi  s5, s5, -1
    bnez  s5, p3_loop

done:
    ebreak
