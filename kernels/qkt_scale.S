.section .text

.macro VSHFL_XOR rd, rs1, mask
    .insn r 0x0B, 0, 0, \rd, \rs1, x\mask
.endm

.globl _start
_start:
    lw   a1, 0(a0)        # Q_off
    lw   a2, 4(a0)        # K_off
    lw   a3, 8(a0)        # S_off
    lw   a4, 12(a0)       # scale
    fmv.s.x  fs0, a4

    csrrs t0, mhartid, x0
    andi  t0, t0, 0x1F

    srli t1,t0,1          # t1作为i Q的行
    andi t2,t0,0x1
    slli t2,t2,3          #t2 作为j k的行

    li t3,256
    mul t3,t3,t1
    add t3, a1, t3;     # Q的基础base

    li t4,32
    mul t4,t4,t0
    add t4,a3,t4        #S的基础base

    li    t0, 0
1:
    # 计算 K_row_ptr = K_base + (j_base + local_k) * 256
    add   t5, t2, t0          # j = j_base + local_k
    slli  t5, t5, 8           # j * 256（256 = 2^8）
    add   t5, a2, t5          # K_row_ptr

    # 点积 acc = 0
    fcvt.s.w ft2, x0          # ft2 = 0.0
    li    t6, 0               # k = 0
    mv    a5, t3              # 用 a5 当 Q 的游标
    mv    a6, t5              # 用 a6 当 K 的游标
2:
    flw   ft0, 0(a5)
    flw   ft1, 0(a6)
    fmadd.s ft2, ft0, ft1, ft2
    addi  a5, a5, 4
    addi  a6, a6, 4
    addi  t6, t6, 1
    li    a7, 64
    bne   t6, a7, 2b          # k != 64 继续

    # acc *= scale
    fmul.s ft2, ft2, fs0

    # 存到 S_lane_ptr[local_k * 4]
    slli  a5, t0, 2           # local_k * 4
    add   a5, t4, a5          # S_lane_ptr + local_k * 4
    fsw   ft2, 0(a5)

    addi  t0, t0, 1
    li    a6, 8
    bne   t0, a6, 1b          # local_k != 8 继续
    ebreak
    
    







    


    


