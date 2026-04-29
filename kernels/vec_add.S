.section .text
.globl _start
_start:
    # a0 = args 结构体的 VRAM offset

    # 读取参数
    lw   a1,  0(a0)      # a1 = a_off
    lw   a2,  4(a0)      # a2 = b_off
    lw   a3,  8(a0)      # a3 = c_off

    # 取 tid = mhartid & 0x1F
    csrrs t0, mhartid, x0
    andi  t0, t0, 0x1F   # t0 = lane id (0~31)

    # 计算 tid * 4（每个 int32 占 4 字节）
    slli  t1, t0, 2      # t1 = tid * 4

    # 计算 A[tid] 的 VRAM 地址，读取
    add   t2, a1, t1     # t2 = a_off + tid*4
    lw    t3, 0(t2)      # t3 = A[tid]

    # 计算 B[tid] 的 VRAM 地址，读取
    add   t4, a2, t1     # t4 = b_off + tid*4
    lw    t5, 0(t4)      # t5 = B[tid]

    # 相加
    add   t6, t3, t5     # t6 = A[tid] + B[tid]

    # 写入 C[tid]
    add   t2, a3, t1     # t2 = c_off + tid*4
    sw    t6, 0(t2)      # C[tid] = t6

    ebreak
