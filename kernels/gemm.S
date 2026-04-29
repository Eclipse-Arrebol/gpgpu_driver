.section .text
.globl _start
_start:

    lw   a1,  0(a0)      # a1 = a_off
    lw   a2,  4(a0)      # a2 = b_off
    lw   a3,  8(a0)      # a3 = c_off
    lw   a4,  12(a0)      # K
    lw   a5,  16(a0)      # blockDim.x

    csrrs t0, mhartid, x0
    srli  t6, t0, 5        # t6 = warp_id（bit[12:5]）
    andi  t6, t6, 0xFF     # 取 warp_id
    andi  t0, t0, 0x1F     # t0 = lane_id

    # 完整 linear_id = warp_id * 32 + lane_id
    slli  t6, t6, 5
    add   t0, t6, t0       # t0 = 完整 linear thread id

    div t1,t0,a5
    rem t2,t0,a5

    mul  t3, t1, a4      # t3 = row * K
    slli t3, t3, 2       # t3 = row * K * 4（字节偏移）
    add  t3, a1, t3      # t3 = a_off + row * K * 4

    li t0,0
    fmv.w.x ft3, x0
loop:
    bge  t0, a4, done

    slli t5, t0, 2
    add t4,t3,t5

    mul t5,t0,a5
    add t5,t5,t2
    slli t5,t5,2
    add t5,t5,a2


    flw ft0,0(t4)
    flw ft1,0(t5)

    fmadd.s ft3, ft0, ft1, ft3
    

    addi t0, t0, 1      # k++
    j    loop

done:
    mul  t5, t1, a5      # row * N
    add  t5, t5, t2      # row * N + col
    slli t5, t5, 2       # * 4
    add  t5, t5, a3      # + c_off

    fsw ft3, 0(t5)


    ebreak 


