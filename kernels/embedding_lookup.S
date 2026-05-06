# ============================================================================
# embedding lookup kernel
# ============================================================================
# Args (a0 -> args struct):
#   args[0] = table_base       (fp32 [V][D] in VRAM, row-major)
#   args[1] = token_id_buf     (uint32 in VRAM, value t in [0, V))
#   args[2] = output_base      (fp32 [D] in VRAM)
#   args[3] = D                (uint32, elements per row)
#   args[4] = elems_per_lane   (uint32, = D/32)
#
# Geometry:
#   32 lanes, lane i handles d in {i, i+32, i+64, ...} (stride-32 interleave)
#
# Register map:
#   t0 = table_base
#   t1 = token_id_buf addr
#   t2 = output_base
#   t3 = D                       (unused after row-base computed; kept for clarity)
#   t4 = elems_per_lane
#   t5 = token_id (loaded from device buffer)
#   t6 = row_base = &table[token_id * D]
#   s0 = lane id (0..31)
#   s1 = k (loop counter, 0..elems_per_lane)
#   s2 = d (= k*32 + lane)
#   s3 = src addr scratch
#   s4 = dst addr scratch
#   s5 = data scratch (bit-pattern copy)
# ============================================================================

    .text
    .globl _start
_start:
    # --- Phase 0: load args ---
    lw      t0, 0(a0)              # table_base
    lw      t1, 4(a0)              # token_id_buf
    lw      t2, 8(a0)              # output_base
    lw      t3, 12(a0)             # D
    lw      t4, 16(a0)             # elems_per_lane

    # --- Phase 1: get lane id ---
    csrr    s0, mhartid
    andi    s0, s0, 0x1f           # s0 = lane id

    # --- Phase 2: load token_id (warp-uniform) ---
    lw      t5, 0(t1)              # t5 = token_id

    # --- Phase 3: row_base = table_base + token_id * D * 4 ---
    mul     t6, t5, t3             # t6 = token_id * D
    slli    t6, t6, 2              # t6 = token_id * D * 4
    add     t6, t0, t6             # t6 = &table[token_id * D]

    # --- Phase 4: per-lane copy loop ---
    li      s1, 0                  # k = 0
loop:
    slli    s2, s1, 5              # s2 = k * 32
    add     s2, s2, s0             # s2 = d = k*32 + lane

    slli    s3, s2, 2              # s3 = d * 4
    add     s4, t2, s3             # s4 = &output[d]
    add     s3, t6, s3             # s3 = &table[token_id*D + d]

    lw      s5, 0(s3)              # bit-pattern load
    sw      s5, 0(s4)              # bit-pattern store

    addi    s1, s1, 1
    blt     s1, t4, loop

    # --- Done ---
    # (按你们 kernel 退出约定,通常是 ebreak 或某个 sentinel;
    #  请按现有 kernel 的退出方式收尾,我不确定你们用的是 ebreak 还是 j .)
    ebreak