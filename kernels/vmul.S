.section .text
.globl _start
_start:
    lw      a1, 0(a0)              # a_off base
    lw      a2, 4(a0)              # b_off base
    lw      a3, 8(a0)              # c_off base
    lw      a4, 12(a0)             # num

    srli a5,a4,5

    csrr    t0, mhartid
    andi    t0, t0, 0x1f           # s0 = lane id

    li t1,0

loop:
    bge t1, a5, done # if t1 >a5t1 then target
    slli t2,t1,5
    add t2,t2,t0
    slli t2,t2,2

    add t3,t2,a1  # a的游标
    add t4,t2,a2  # b的游标
    add t5,t2,a3  # c的游标

    flw ft1,0(t3)
    flw ft2,0(t4)
    fmul.s ft3,ft1,ft2
    fsw ft3,0(t5)

    addi t1,t1,1
    j loop

done:
    ebreak