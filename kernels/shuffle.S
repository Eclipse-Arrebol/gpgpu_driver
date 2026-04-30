.section .text

.macro VSHFL_XOR rd, rs1, mask
    .insn r 0x0B, 0, 0, \rd, \rs1, x\mask
.endm

.globl _start
_start:
    # 1. 取参数
    lw a1, 0(a0)        # in_off
    lw a2, 4(a0)        # out_off
    
    # 2. 取 lane_id
    csrrs t0, mhartid, x0
    andi  t0, t0, 0x1F
    
    # 3. 计算自己的输入地址：a1 + lane_id * 4
    slli  t1, t0, 2
    add   t2, a1, t1
    flw   ft0, 0(t2)    # 加载自己的 in[lane_id] 到 ft0


    VSHFL_XOR ft1, ft0, 1
    fadd.s    ft0, ft0, ft1

    VSHFL_XOR ft1, ft0, 2
    fadd.s    ft0, ft0, ft1

    VSHFL_XOR ft1, ft0, 4
    fadd.s    ft0, ft0, ft1

    VSHFL_XOR ft1, ft0, 8
    fadd.s    ft0, ft0, ft1

    VSHFL_XOR ft1, ft0, 16
    fadd.s    ft0, ft0, ft1


    # 4. 五步 xor 归约
    # 提示 1：ft0 累加结果，ft1 临时存伙伴值
    # 提示 2：5 步 mask = 1, 2, 4, 8, 16
    # 提示 3：vshfl.xor 用 .insn r 0x0B, 0, 0, ft1, ft0, <mask>
    # 你来填...
    
    # 5. 写回：a2 + lane_id * 4
    add   t3, a2, t1
    fsw   ft0, 0(t3)
    
    ebreak