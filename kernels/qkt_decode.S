.section .text
.globl _start
_start:

    lw a1,0(a0)   # Q_off  (Q 是 [num_q_heads, head_dim])
    lw a2,4(a0)   # K_cache_off
    lw a3,8(a0)   # scores_off
    lw a4,12(a0)   # cur_pos
    lw a5,16(a0)   # max_seq
    lw a6,20(a0)   # head_dim
    lw a7,24(a0)   # q_per_kv
    lw a0,28(a0)   # scale_bits

    addi s1,a4,1  # cur_pos + 1
    csrr t1, mhartid
    andi  t0, t1, 0x1F  # lane_id
    srli  t1, t1, 13    # block_id

    div t3,t1,a7        # kh

    mul t4,t1,a6
    slli t4,t4,2
    add t4,t4,a1  # Q的行起始地址

    mul t5,t1,a5
    slli t5,t5,2
    add t5,t5,a3  # score 的行起始地址

    mul t6,t3,a6
    mul t6,t6,a5
    slli t6,t6,2
    add t6,t6,a2  # K_cache的行起始地址

    fmv.s.x ft5,a0       # scale → 浮点寄存器（只算一次）
loop:
    bge t0,s1,done
    mul s2,a6,t0
    slli s2,s2,2
    add s2,s2,t6  # k_subtable + j * head_dim * 4

    li s3,0
    fmv.s.x ft1,zero  # acc
loop_sum:
    bge s3,a6,dot_done
    slli s4,s3,2   # k*4
    add s5,s4,s2  # K_cache[kh, j, k]
    add s6,s4,t4  # Q[qh, k]

    flw ft2,0(s5)
    flw ft3,0(s6)
    fmadd.s ft1,ft2,ft3,ft1  # acc += Q[k] * K[k]
    
    addi s3,s3,1
    j loop_sum
dot_done:
    fmul.s ft1,ft1,ft5       # acc *= scale
    slli s7,t0,2
    add s7,t5,s7
    fsw ft1,0(s7)
    
    addi t0,t0,32
    j loop
done:
    ebreak





