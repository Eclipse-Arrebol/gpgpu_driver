# pv_decode.S
# 输入: a0: 参数结构体指针
# 输出: 无，直接写回 O 数组

.section .text
.globl _start
_start:
    lw    a1, 0(a0)          # a1 = P_off
    lw    a2, 4(a0)          # a2 = V_cache_off
    lw    a3, 8(a0)          # a3 = O_off
    lw    a4, 12(a0)         # a4 = cur_pos
    lw    a5, 16(a0)         # a5 = max_seq
    lw    a6, 20(a0)         # a6 = head_dim
    lw    a7, 24(a0)         # a7 = q_per_kv

    csrr  t1, mhartid
    andi  s8, t1, 0x1F       # s8 = lane_id
    srli  s7, t1, 13         # s7 = block_id (qh)

    div   s9, s7, a7         # s9 = kh = qh / q_per_kv
    addi  s10, a4, 1         # s10 = S = cur_pos + 1
    li    t0, 32
    div   s11, a6, t0        # s11 = n_k = head_dim / 32

    slli  s3, a6, 2          # s3 = head_dim * 4

    # P_row_base = P_off + qh * max_seq * 4
    mul   t0, s7, a5
    slli  t0, t0, 2
    add   s0, a1, t0         # s0 = P_row_base

    # V_kh_base = V_cache_off + kh * max_seq * head_dim * 4
    mul   t0, s9, a5
    mul   t0, t0, a6
    slli  t0, t0, 2
    add   s1, a2, t0         # s1 = V_kh_base

    # O_row_base = O_off + qh * head_dim * 4
    mul   t0, s7, a6
    slli  t0, t0, 2
    add   s2, a3, t0         # s2 = O_row_base

    mv    t1, zero
    beqz  s11, done

k_loop:
    slli  t2, t1, 5
    add   t2, s8, t2         # t2 = k
    bge   t2, a6, end_k

    fmv.s.x ft0, x0

    mv    t3, zero
    slli  t4, t2, 2
    add   t5, s1, t4
    mv    t6, s0

j_loop:
    beq   t3, s10, j_loop_end
    flw   ft1, 0(t6)
    flw   ft2, 0(t5)
    fmul.s ft3, ft1, ft2
    fadd.s ft0, ft0, ft3
    addi  t6, t6, 4
    add   t5, t5, s3
    addi  t3, t3, 1
    j     j_loop

j_loop_end:
    add   t0, s2, t4
    fsw   ft0, 0(t0)

end_k:
    addi  t1, t1, 1
    blt   t1, s11, k_loop

done:
    ebreak
