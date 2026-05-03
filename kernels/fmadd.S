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

    # DEBUG: 把 a3 (期望是 c_off=0x2e80) 写到 a_off+0(=0x2f80)
    sw     a3, 0(a1)
    # DEBUG: 把 t4 (期望是 c_off + lane*4) 写到 a_off+4
    sw     t4, 4(a1)
    # DEBUG: 把 a0 (kernel 启动时的 args 指针) 写到 a_off+8
    sw     a0, 8(a1)

    ebreak