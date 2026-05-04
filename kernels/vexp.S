.section .text

# vexp.s rd, rs1   -- funct7=0x60, rs2=0, funct3=0, opcode=0x53
# rd, rs1 传 f-寄存器编号 (0-31)
.macro VEXP_S rd, rs1
    .word (0x7c << 25) | (0 << 20) | (\rs1 << 15) | (0 << 12) | (\rd << 7) | 0x53
.endm

.globl _start
_start:
    flw f1, 0(a0)        # args.x
    lw  a1, 4(a0)        # args.out_off
    li t0, 0x40490fdb
    fmv.s.x f2, t0
    VEXP_S 2, 1          # f2 = vexp.s(f1)
    fsw f2, 0(a1)        # vram[out_off] = f2
    ebreak