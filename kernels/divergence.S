.section .text
.globl _start
_start:
    lw a1,0(a0)
    lw a2,4(a0)

    csrrs t0, mhartid, x0
    andi  t0, t0, 0x1F

    slli  t1, t0, 2 

    andi  t4,t0,1
    bnez  t4,odd_case

    add   t2, a1, t1
    li t3,0xAAAA
    sw t3,0(t2)
    j after

odd_case:
    add   t2, a2, t1
    li t3,0xBBBB
    sw t3,0(t2)


after:
    ebreak

