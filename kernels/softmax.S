.section .text
.macro VEXP_S rd, rs1
    .word (0x7c << 25) | (0 << 20) | (\rs1 << 15) | (0 << 12) | (\rd << 7) | 0x53
.endm

.macro VSHFL_XOR rd, rs1, mask
    .word (\mask << 20) | (\rs1 << 15) | (\rd << 7) | 0x0B
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

    li t0,28
    add t2, a1, t1   # t2 = a1 + lane_id*112
    li      t4, 0xFF800000
    fmv.s.x ft0, t4
loop:
    flw ft1,0(t2)
    fmax.s ft0, ft0, ft1
    addi t2,t2,4
    addi t0,t0,-1
    bnez t0,loop

    fmv.s ft0, ft0    
    VSHFL_XOR 2, 0, 1           # ft2 = lane^1 的 ft0
    fmax.s ft0, ft0, ft2        # ft0 = max(本 lane, 邻居)
    VSHFL_XOR 2, 0, 2
    fmax.s ft0, ft0, ft2
    VSHFL_XOR 2, 0, 4
    fmax.s ft0, ft0, ft2
    VSHFL_XOR 2, 0, 8
    fmax.s ft0, ft0, ft2
    VSHFL_XOR 2, 0, 16
    fmax.s ft0, ft0, ft2        # 现在 ft0 包含了所有 32 个 lane 的最大值

    li t0,28
    add t2, a1, t1 
    fmv.s.x ft1, zero
loop_sum:
    flw   ft2, 0(t2)
    fsub.s ft2, ft2, ft0
    VEXP_S 2, 2  
    fadd.s ft1, ft1, ft2
    addi  t2, t2, 4
    addi  t0, t0, -1
    bnez  t0, loop_sum    #ft2作为部分和


    fmv.s ft1, ft1  

    VSHFL_XOR 2, 1, 1
    fadd.s ft1, ft1, ft2
    VSHFL_XOR 2, 1, 2
    fadd.s ft1, ft1, ft2
    VSHFL_XOR 2, 1, 4
    fadd.s ft1, ft1, ft2
    VSHFL_XOR 2, 1, 8
    fadd.s ft1, ft1, ft2
    VSHFL_XOR 2, 1, 16
    fadd.s ft1, ft1, ft2   # 拿到全部和

    li t0,28
    add t2, a1, t1 
    add t3, a2, t1
loop_result:
    flw   ft2, 0(t2)
    fsub.s ft2, ft2, ft0
    VEXP_S 2, 2
    fdiv.s ft3,ft2,ft1
    fsw ft3,0(t3)
    addi  t2, t2, 4
    addi  t3, t3, 4
    addi  t0, t0, -1
    bnez  t0, loop_result


    ebreak





