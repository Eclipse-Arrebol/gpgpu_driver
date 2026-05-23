.section .text
.globl _start
_start:
    lw   a1,  0(a0)       # a1 = a_off
    lw   a2,  4(a0)       # a2 = b_off
    lw   a3,  8(a0)       # a3 = c_off
    lw   a4, 12(a0)       # a4 = M
    lw   a5, 16(a0)       # a5 = K
    lw   a6, 20(a0)       # a6 = N

    # ===== thread id 计算 (multi-block, 每 block 1 warp = 32 thread) =====
    # mhartid 布局: [31:13] block_id | [12:5] warp_id | [4:0] lane_id
    # 约定每 block 只有 1 warp,所以 warp_id 永远是 0,直接跳过 5~12 位
    csrrs t6, mhartid, x0
    andi  t1, t6, 0x1F          # t1 = lane_id (bit 0~4)
    srli  t6, t6, 13             # t6 = block_id (bit 13~31)
    slli  t6, t6, 5              # block_id * 32
    add   t0, t6, t1             # t0 = linear thread id = block_id*32 + lane_id

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