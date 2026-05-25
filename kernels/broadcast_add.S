.section .text
.globl _start
_start:
    # broadcast_add: y[m*N + n] = x[m*N + n] + b[n],for m in [0, M), n in [0, N)
    # 支持 in-place: y_off == x_off
    #
    # args 布局(沿用 gemm.S 风格):
    #   0:  x_off (int32)
    #   4:  y_off (int32)
    #   8:  b_off (int32)
    #  12:  M     (int32)
    #  16:  N     (int32)
    #
    # 每个 thread 处理 1 个输出元素;grid = ceil(M*N / 32) 个 block,每 block 1 warp

    lw   a1,  0(a0)             # a1 = x_off
    lw   a2,  4(a0)             # a2 = y_off
    lw   a3,  8(a0)             # a3 = b_off
    lw   a4, 12(a0)             # a4 = M
    lw   a5, 16(a0)             # a5 = N

    # ===== thread id 计算(沿用 gemm.S L10-14 模式)=====
    # mhartid 布局: [31:13] block_id | [12:5] warp_id | [4:0] lane_id
    # 约定每 block 1 warp,warp_id 段跳过
    csrrs t6, mhartid, x0
    andi  t1, t6, 0x1F          # t1 = lane_id (bit 0~4)
    srli  t6, t6, 13            # t6 = block_id
    slli  t6, t6, 5             # block_id * 32
    add   t0, t6, t1            # t0 = linear thread id

    # 越界检查
    mul   t5, a4, a5            # t5 = M * N
    bge   t0, t5, done_oob

    # row = linear_id / N, col = linear_id % N
    div   t1, t0, a5            # t1 = row (= m)
    rem   t2, t0, a5            # t2 = col (= n)

    # ===== 计算 &x[m*N + n] =====
    # 等价于 x_off + linear_id * 4
    slli  t3, t0, 2             # t3 = linear_id * 4
    add   t4, a1, t3            # t4 = &x[m*N + n]

    # ===== 计算 &b[n] =====
    slli  t5, t2, 2             # t5 = n * 4
    add   t5, a3, t5            # t5 = &b[n]

    # ===== load, add, store =====
    flw    ft0, 0(t4)           # ft0 = x[m*N + n]
    flw    ft1, 0(t5)           # ft1 = b[n]
    fadd.s ft2, ft0, ft1        # ft2 = x + b

    # ===== 计算 &y[m*N + n] 并写回 =====
    add   t6, a2, t3            # t6 = &y[m*N + n]
    fsw   ft2, 0(t6)

done_oob:
    ebreak