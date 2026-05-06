.section .text

.globl _start
_start:
    lw    a1, 0(a0)             # P_base
    lw    a2, 4(a0)             # V_base
    lw    a3, 8(a0)             # O_base

    csrrs t0, mhartid, x0
    andi  t0, t0, 0x1F          # lane_id (block=1, 不读 block_id)

    srli  t1, t0, 1             # i = lane_id >> 1
    andi  t2, t0, 0x1
    slli  t2, t2, 5             # k_base = (lane_id & 1) << 5  (0 或 32)

    # P_row_ptr = P_base + i * 64
    slli  t3, t1, 6             # i * 64
    add   t3, a1, t3

    # O_lane_ptr = O_base + lane_id * 128
    slli  t5, t0, 7             # lane_id * 128
    add   t5, a3, t5

    # V_col_base_for_lane = V_base + k_base * 4   (j=0 行的 k_base 列)
    slli  t4, t2, 2             # k_base * 4
    add   t4, a2, t4            # V 起点(j=0, k=k_base)

    li    t6, 0                 # local_k counter

outer:
    # V_col_ptr = V_col_base + local_k * 4
    slli  a7, t6, 2
    add   a6, t4, a7            # V 游标(本次 local_k)
    
    mv    a5, t3                # P 游标 = P_row_ptr

    # acc = 0
    fcvt.s.w ft2, x0
    
    li    a7, 0                 # j counter

inner:
    flw   ft0, 0(a5)
    flw   ft1, 0(a6)
    fmadd.s ft2, ft0, ft1, ft2
    addi  a5, a5, 4             # P 步进
    addi  a6, a6, 256           # V 步进(下一行 V)
    addi  a7, a7, 1
    li    s1, 16
    bne   a7, s1, inner

    # store O[i, k_base + local_k]
    slli  a7, t6, 2
    add   a7, t5, a7            # O 写地址
    fsw   ft2, 0(a7)

    addi  t6, t6, 1
    li    s1, 32
    bne   t6, s1, outer

    ebreak