/* kernels/rope.S — RoPE kernel (V24)
 *
 * V24 改动:
 *   Pair 排布改为 GPT-NeoX style:lane k 处理 (x[k], x[k+half])
 *   - s3 = k*4 (不是 2*k*4)
 *   - x_hi 在 +128(t1) (不是 +4(t1)),即 head[k+32] 字节偏移
 *
 * 派发:单 warp 32 thread,每 lane 处理 head 内的一对 (k, k+32)
 *      lane_id ∈ [0, 32),恰好 = head_dim/2 = 32 对索引 k
 *
 * sincos 表 layout 不变:[pos*head_dim + 2k + 0/1],cos/sin 相邻
 *
 * args 布局 (RopeArgs):
 *   0:  x_off, 4: sincos_off, 8: num_heads, 12: head_dim, 16: token_pos
 *
 * 算法 (V24, NeoX):
 *   k = lane_id
 *   cos_k = sincos[token_pos*head_dim + 2*k + 0]
 *   sin_k = sincos[token_pos*head_dim + 2*k + 1]
 *   for h in 0..num_heads:
 *       base   = x_off + h*head_dim*4 + k*4
 *       x_lo   = *(base + 0)      ; head[k]
 *       x_hi   = *(base + 128)    ; head[k + 32], 即 head[k + half] 字节偏移
 *       t1     = x_lo * cos_k     fmul.s
 *       t2     = x_hi * sin_k     fmul.s
 *       t3     = x_hi * cos_k     fmul.s
 *       new_lo = t1 - t2          fsub.s
 *       new_hi = x_lo*sin_k + t3  fmadd.s
 *       *(base + 0)   = new_lo
 *       *(base + 128) = new_hi
 */

.section .text
.globl _start
_start:
    /* ===== Phase 1: 加载 args ===== */
    lw      a1, 0(a0)              # a1 = x_off
    lw      a2, 4(a0)              # a2 = sincos_off
    lw      a3, 8(a0)              # a3 = num_heads
    lw      a4, 12(a0)             # a4 = head_dim
    lw      a5, 16(a0)             # a5 = token_pos

    /* ===== Phase 2: 算 lane_id ===== */
    csrr    s0, mhartid
    andi    s0, s0, 0x1f           # s0 = lane_id = k ∈ [0, 32)

    /* ===== Phase 3: 预算常量 ===== */
    /* head_dim_bytes = head_dim * 4 */
    slli    s2, a4, 2              # s2 = head_dim * 4

    /* V24:k_byte_off_in_head = k * 4 (NeoX,不是 2*k*4) */
    slli    s3, s0, 2              # s3 = k * 4

    /* ===== Phase 4: 查 sincos 表 (循环外一次) =====
     * sincos_addr = sincos_off + token_pos * head_dim * 4 + k * 8
     * 注意:sincos 表索引仍是 2k,所以这里需要 k*8 偏移
     * 用 s3<<1 即 k*8 */
    mul     t0, a5, s2             # t0 = token_pos * head_dim * 4
    slli    t2, s3, 1              # t2 = k * 8 (即 2*k*4)
    add     t0, t0, t2             # t0 += 2*k*4 (sincos 索引)
    add     t0, t0, a2             # t0 = sincos addr
    flw     fa0, 0(t0)             # fa0 = cos_k
    flw     fa1, 4(t0)             # fa1 = sin_k

    /* ===== Phase 5: head 循环 (V24 NeoX) ===== */
    li      s1, 0                  # h = 0
head_loop:
    /* base = x_off + h * head_dim * 4 + k * 4 */
    mul     t1, s1, s2             # t1 = h * head_dim * 4
    add     t1, t1, s3             # t1 += k * 4 (NeoX: 不是 2*k*4)
    add     t1, t1, a1             # t1 = x_off + ...

    /* V24:x_lo 在 +0,x_hi 在 +128 (即 +half*4,head_dim=64 → half=32 → 128B)
     * 注:128 = 32*4 在 RV imm 12-bit signed [-2048, 2047] 范围内,安全 */
    flw     fa2, 0(t1)             # fa2 = x_lo = head[k]
    flw     fa3, 128(t1)           # fa3 = x_hi = head[k + 32]

    /* 5 条 fp 算术 */
    fmul.s  ft0, fa2, fa0          # ft0 = x_lo * cos_k     (t1)
    fmul.s  ft1, fa3, fa1          # ft1 = x_hi * sin_k     (t2)
    fmul.s  ft2, fa3, fa0          # ft2 = x_hi * cos_k     (t3)
    fsub.s  fa4, ft0, ft1          # fa4 = new_lo = t1 - t2
    fmadd.s fa5, fa2, fa1, ft2     # fa5 = new_hi = x_lo*sin_k + t3

    /* V24:写回 in-place,x_hi 在 +128 */
    fsw     fa4, 0(t1)             # head[k]      = new_lo
    fsw     fa5, 128(t1)           # head[k + 32] = new_hi

    /* 循环条件 */
    addi    s1, s1, 1              # h++
    blt     s1, a3, head_loop      # if h < num_heads, continue

    /* ===== Phase 6: 结束 ===== */
    ebreak