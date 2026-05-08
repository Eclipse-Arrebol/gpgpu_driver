.section .text
.globl _start
_start:
    lw   a1,  0(a0)       # a1 = a_off
    lw   a2,  4(a0)       # a2 = b_off
    lw   a3,  8(a0)       # a3 = c_off
    lw   a4, 12(a0)       # a4 = M
    lw   a5, 16(a0)       # a5 = K
    lw   a6, 20(a0)       # a6 = N

    csrrs t0, mhartid, x0
    srli  t6, t0, 5       # warp_id (bit[12:5])
    andi  t6, t6, 0xFF
    andi  t0, t0, 0x1F    # lane_id
    slli  t6, t6, 5
    add   t0, t6, t0      # t0 = linear thread id

    # 越界检查:linear_id >= M*N 的线程不参与
    mul   t5, a4, a6      # t5 = M * N
    bge   t0, t5, done_oob

    # row = linear_id / N, col = linear_id % N
    div   t1, t0, a6      # t1 = row
    rem   t2, t0, a6      # t2 = col

    # A 行起点: a_off + row * K * 4
    mul   t3, t1, a5      # t3 = row * K
    slli  t3, t3, 2       # t3 = row * K * 4
    add   t3, a1, t3      # t3 = &A[row][0]

    li    t0, 0           # k = 0
    fmv.w.x ft3, x0       # acc = 0.0
loop:
    bge   t0, a5, done    # k >= K -> 退出
    slli  t5, t0, 2
    add   t4, t3, t5      # &A[row][k]

    mul   t5, t0, a6      # k * N
    add   t5, t5, t2      # k * N + col
    slli  t5, t5, 2
    add   t5, t5, a2      # &B[k][col]

    flw   ft0, 0(t4)
    flw   ft1, 0(t5)
    fmadd.s ft3, ft0, ft1, ft3

    addi  t0, t0, 1
    j     loop
done:
    mul   t5, t1, a6      # row * N
    add   t5, t5, t2      # row * N + col
    slli  t5, t5, 2
    add   t5, t5, a3      # &C[row][col]
    fsw   ft3, 0(t5)
done_oob:
    ebreak