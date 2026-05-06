.section .text


.macro VSHFL_XOR rd, rs1, mask
    .word (0x00 << 25) | (\mask << 20) | (\rs1 << 15) | (0 << 12) | (\rd << 7) | 0x0B
.endm

.macro VEXP_S rd, rs1
    .word (0x7c << 25) | (0 << 20) | (\rs1 << 15) | (0 << 12) | (\rd << 7) | 0x53
.endm

.globl _start
_start:
    lw    a1, 0(a0)              # S_off
    lw    a2, 4(a0)              # P_off

    csrrs t0, mhartid, x0
    srli  t3, t0, 13             # block_id
    andi  t0, t0, 0x1F           # lane_id

    slli  t4, t3, 6              # block_id * 64
    add   t1, a1, t4             # S_row_base
    add   t2, a2, t4             # P_row_base

    li    t6, 16                 # virt threshold

    li    t5, 0xFF800000
    fmv.s.x ft1, t5              # ft1 = -INF (备用)

    bge   t0, t6, A_virt
    slli  t4, t0, 2
    add   t4, t1, t4
    flw   ft0, 0(t4) 
    j     A_reduce
A_virt:
    fmv.s ft0, ft1
A_reduce:
    VSHFL_XOR 3, 0, 1           # ft2 = lane^1 的 ft0
    fmax.s ft0, ft0, ft3        # ft0 = max(本 lane, 邻居)
    VSHFL_XOR 3, 0, 2
    fmax.s ft0, ft0, ft3
    VSHFL_XOR 3, 0, 4
    fmax.s ft0, ft0, ft3
    VSHFL_XOR 3, 0, 8
    fmax.s ft0, ft0, ft3
    VSHFL_XOR 3, 0, 16
    fmax.s ft0, ft0, ft3 

    fmv.s ft3, ft0  

    bge   t0, t6, B_virt
    flw   ft0, 0(t4)             # 重读 x (t4 还是上面的地址)
    fsub.s  ft0, ft0, ft3
    VEXP_S  0, 0
    j     B_reduce
B_virt:
    fcvt.s.w ft0, x0             # 0.0 
B_reduce:
    # 保存 e_local 到某处(下一阶段还要用)
    fmv.s ft2, ft0               # ft2 = e_local

    VSHFL_XOR 4, 0, 1
    fadd.s ft0, ft0, ft4
    VSHFL_XOR 4, 0, 2
    fadd.s ft0, ft0, ft4
    VSHFL_XOR 4, 0, 4
    fadd.s ft0, ft0, ft4
    VSHFL_XOR 4, 0, 8
    fadd.s ft0, ft0, ft4
    VSHFL_XOR 4, 0, 16
    fadd.s ft0, ft0, ft4  

    fmv.s ft4, ft0               # 保存 global_sum
    bge   t0, t6, done
    fdiv.s  ft0, ft2, ft4        # e_local / global_sum
    slli  t4, t0, 2
    add   t4, t2, t4
    fsw   ft0, 0(t4)
done:
    ebreak
