.section .text
.globl _start
_start:
    lw a1,0(a0)
    lw a2,4(a0)
    lw a3,8(a0)

    csrrs t0, mhartid, x0
    andi  t0, t0, 0x1F

    slli  t1, t0, 2 

    add   t2, a1, t1
    flw   ft1,0(t2)

    add   t3, a2, t1
    flw   ft2,0(t3)

    add   t4, a3, t1
    flw   ft3,0(t4)

    fmadd.s ft4,ft1,ft2,ft3

    fsw    ft4, 0(t4)

    ebreak