/* kernels/rope.S — RoPE kernel
 *
 * 派发:单 warp 32 thread,每 lane 处理 head 内的一对 (2k, 2k+1)
 *      lane_id ∈ [0, 32),恰好 = head_dim/2 = 32 对索引 k
 *
 * args 布局(对应 RopeArgs):
 *   0:  uint32_t x_off
 *   4:  uint32_t sincos_off
 *   8:  uint32_t num_heads
 *  12:  uint32_t head_dim
 *  16:  uint32_t token_pos
 *
 * 算法:
 *   k = lane_id
 *   cos_k = sincos[token_pos*head_dim + 2*k + 0]   (查一次,所有 head 共享)
 *   sin_k = sincos[token_pos*head_dim + 2*k + 1]
 *   for h in 0..num_heads:
 *       base = x_off + h*head_dim*4 + 2*k*4
 *       x_lo = *(base + 0)
 *       x_hi = *(base + 4)
 *       t1 = x_lo * cos_k       fmul.s
 *       t2 = x_hi * sin_k       fmul.s
 *       t3 = x_hi * cos_k       fmul.s
 *       new_lo = t1 - t2        fsub.s
 *       new_hi = x_lo*sin_k+t3  fmadd.s  (a*b + c)
 *       *(base + 0) = new_lo
 *       *(base + 4) = new_hi
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
    csrr    s0, mhartid            # s0 = mhartid
    andi    s0, s0, 0x1f           # s0 = lane_id = k ∈ [0, 32)

    /* ===== Phase 3: 预算常量 ===== */
    /* head_dim_bytes = head_dim * 4 */
    slli    s2, a4, 2              # s2 = head_dim * 4

    /* k_byte_off_in_head = 2 * k * 4 = k * 8 */
    slli    s3, s0, 3              # s3 = 2*k*4 = k << 3

    /* ===== Phase 4: 查 sincos 表(循环外只查一次) ===== */
    /* sincos_addr = sincos_off + token_pos * head_dim * 4 + k * 8
     *
     * 注:V18 §2 草图里 sincos 行步进 = head_dim/2 * 2 * 4 = head_dim * 4,
     *    刚好等于 head_dim_bytes,直接复用 s2
     */
    mul     t0, a5, s2             # t0 = token_pos * head_dim * 4
    add     t0, t0, s3             # t0 += 2*k*4
    add     t0, t0, a2             # t0 = sincos_off + ...
    flw     fa0, 0(t0)             # fa0 = cos_k
    flw     fa1, 4(t0)             # fa1 = sin_k

    /* ===== Phase 5: head 循环 ===== */
    li      s1, 0                  # h = 0
head_loop:
    /* base = x_off + h * head_dim * 4 + 2*k*4 */
    mul     t1, s1, s2             # t1 = h * head_dim * 4
    add     t1, t1, s3             # t1 += 2*k*4
    add     t1, t1, a1             # t1 = x_off + ...

    /* 加载 (x_lo, x_hi) */
    flw     fa2, 0(t1)             # fa2 = x_lo
    flw     fa3, 4(t1)             # fa3 = x_hi

    /* 5 条 fp 算术(严格匹配 host reference) */
    fmul.s  ft0, fa2, fa0          # ft0 = x_lo * cos_k     (t1)
    fmul.s  ft1, fa3, fa1          # ft1 = x_hi * sin_k     (t2)
    fmul.s  ft2, fa3, fa0          # ft2 = x_hi * cos_k     (t3)
    fsub.s  fa4, ft0, ft1          # fa4 = new_lo = t1 - t2
    fmadd.s fa5, fa2, fa1, ft2     # fa5 = new_hi = x_lo*sin_k + t3

    /* 写回 in-place */
    fsw     fa4, 0(t1)
    fsw     fa5, 4(t1)

    /* 循环条件 */
    addi    s1, s1, 1              # h++
    blt     s1, a3, head_loop      # if h < num_heads, continue

    /* ===== Phase 6: 结束 ===== */
    ebreak