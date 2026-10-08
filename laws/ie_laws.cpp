// ie_laws.cpp -- the laws of the verified core (core/ie_core.h).
// Each law_<name> returns bool: "for all values of the parameters, it is
// true". cpp-lean translates each law to a Lean proposition; proofs/ has one
// theorem per law. The same functions run as C++ tests (tests/laws_test.cpp).

#include "../core/ie_core.h"

// ---------------------------------------------------------------------
// 1. Index and memory safety
// ---------------------------------------------------------------------

// GEMV grid. LAW grid_cover: each row r < rows is computed by the wave
// r & 7 of workgroup r >> 3, and that workgroup is in the grid.
constexpr auto law_grid_cover(u32 rows, u32 r) -> bool {
  return !(rows < 0x40000u && r < rows) ||
         ((r >> 3u) < ie_gemv_ngroups(rows) && (r & 7u) < ie_rows_per_wg() && ie_gemv_row(r >> 3u, r & 7u) == r);
}

// LAW grid_unique: two different (workgroup, wave) pairs compute different rows.
constexpr auto law_grid_unique(u32 g, u32 v, u32 h, u32 w) -> bool {
  return !(g < 0x8000000u && h < 0x8000000u && v < 8u && w < 8u && ie_gemv_row(g, v) == ie_gemv_row(h, w)) ||
         (g == h && v == w);
}

// Lanes. LAW lane_cover: block b of a row is done by lane b & 31 in step
// b >> 5, and lane_blk gives b back.
constexpr auto law_lane_cover(u32 b) -> bool {
  return ie_blk_lane(b) < ie_wave() && ie_lane_blk(ie_blk_lane(b), ie_blk_step(b)) == b;
}

// LAW lane_unique: two different (lane, step) pairs do different blocks.
constexpr auto law_lane_unique(u32 l, u32 t, u32 m, u32 s) -> bool {
  return !(l < 32u && m < 32u && t < 0x4000000u && s < 0x4000000u && ie_lane_blk(l, t) == ie_lane_blk(m, s)) ||
         (l == m && t == s);
}

// LAW q4_addr: every address of the GPU layout and of the GGUF source is
// inside its array: words < rows*nb*4, scales < rows*nb, source bytes
// < rows*nb*18.
constexpr auto law_q4_addr(u32 rows, u32 nb, u32 r, u32 b, u32 w, u32 k) -> bool {
  return !(ie_sizes_ok(rows, nb) && r < rows && b < nb && w < 4u && k < 16u) ||
         (ie_q4_dst_word(r, b, w, nb) < rows * nb * 4u && ie_q4_dst_scale(r, b, nb) < rows * nb &&
          ie_q4_src_qbyte(r, b, k, nb) < rows * nb * 18u && ie_q4_src_blk(r, b, nb) + 1u < rows * nb * 18u);
}

// LAW q4_inverse: the index map of the repack is one-to-one (its inverse
// gives (r, b, w) back).
constexpr auto law_q4_inverse(u32 rows, u32 nb, u32 r, u32 b, u32 w) -> bool {
  return !(ie_sizes_ok(rows, nb) && r < rows && b < nb && w < 4u) ||
         (ie_q4_word_row(ie_q4_dst_word(r, b, w, nb), nb) == r && ie_q4_word_blk(ie_q4_dst_word(r, b, w, nb), nb) == b &&
          ie_q4_word_w(ie_q4_dst_word(r, b, w, nb)) == w);
}

// LAW q4_onto: every word i of the GPU array is the image of one (r, b, w):
// the repack writes every word.
constexpr auto law_q4_onto(u32 rows, u32 nb, u32 i) -> bool {
  return !(ie_sizes_ok(rows, nb) && 0u < nb && i < rows * nb * 4u) ||
         (ie_q4_word_row(i, nb) < rows && ie_q4_word_blk(i, nb) < nb &&
          ie_q4_dst_word(ie_q4_word_row(i, nb), ie_q4_word_blk(i, nb), ie_q4_word_w(i), nb) == i);
}

// LAW act_addr: the Q8 activation words of block b (8 words) are inside an
// array of nb * 8 words.
constexpr auto law_act_addr(u32 nb, u32 b, u32 v) -> bool {
  return !(nb < 0x800u && b < nb && v < 8u) || b * 8u + v < nb * 8u;
}

// ---------------------------------------------------------------------
// 2. The static memory plan
// ---------------------------------------------------------------------

// LAW plan_fits: every byte of every buffer of an accepted plan is inside
// the arena.
constexpr auto law_plan_fits(Plan p, u32 i, u32 x) -> bool {
  return !(ie_plan_check(p) && i < p.n && ie_in_buf(p, i, x)) || x < p.cap;
}

// LAW plan_aligned: every buffer of an accepted plan starts at a multiple of 256.
constexpr auto law_plan_aligned(Plan p, u32 i) -> bool {
  return !(ie_plan_check(p) && i < p.n) || (ie_off(p, i) & 255u) == 0u;
}

// LAW plan_no_clobber: in step t, two different live buffers of an accepted
// plan share no byte x.
constexpr auto law_plan_no_clobber(Plan p, u32 i, u32 j, u32 t, u32 x) -> bool {
  return !(ie_plan_check(p) && i < p.n && j < p.n && i != j && ie_live(p, i, t) && ie_live(p, j, t) &&
           ie_in_buf(p, i, x)) ||
         !ie_in_buf(p, j, x);
}

// ---------------------------------------------------------------------
// 3. Reduction order
// ---------------------------------------------------------------------

// LAW strided_sum: the 32 lane partial sums (lane l: blocks l, l + 32, ...)
// add up to the plain sum over the nb blocks, for every array x.
constexpr auto law_strided_sum(Mem x, u32 nb) -> bool {
  return !(nb < 0x40000000u) || ie_strided_sum(x, nb) == ie_seq_sum(nb, x, 0u, nb);
}

// LAW wave_sum: the tree reduction of 32 lane values (lane 0 after the
// shuffles by 16, 8, 4, 2, 1) is their plain sum, for every array x.
constexpr auto law_wave_sum(Mem x) -> bool { return ie_wave_sum(x) == ie_seq_sum(32u, x, 0u, 32u); }

// ---------------------------------------------------------------------
// 4. Quantization
// ---------------------------------------------------------------------

// LAW q4_nibble: the kernel's nibble j of a repacked word (packed from the
// GGUF nibble bytes q0..q3 of that word) is the GGUF nibble j.
constexpr auto law_q4_nibble(u32 q0, u32 q1, u32 q2, u32 q3, u32 j) -> bool {
  return !(j < 32u) ||
         ie_q4_nib(ie_pack4(q0, q1, q2, q3), j) ==
             ie_q4_spec_nib((j & 3u) == 0u ? q0 : (j & 3u) == 1u ? q1 : (j & 3u) == 2u ? q2 : q3, j);
}

// LAW q4_word_of: weight j of a block is in word (j & 15) >> 2, which holds
// the GGUF bytes 4w .. 4w + 3: so the byte of weight j is (j & 15), as GGUF says.
constexpr auto law_q4_word_of(u32 j) -> bool {
  return !(j < 32u) || (ie_q4_word_of(j) < 4u && 4u * ie_q4_word_of(j) + (j & 3u) == (j & 15u));
}

// LAW q4q8_word: two dot4 instructions, minus 8 times the activation sums,
// give the exact sum of (q - 8) * a over the 8 weights of a word.
constexpr auto law_q4q8_word(u32 wq, u32 alo, u32 ahi) -> bool {
  return ie_q4q8_word(0u, wq, alo, ahi) - 8u * (ie_sum_s8x4(alo) + ie_sum_s8x4(ahi)) == ie_q4q8_word_spec(wq, alo, ahi);
}

// LAW q4q8_block: the kernel's block dot product, with the activation sum
// that the quantizer stores, is the spec: the sum over j < 32 of
// (weight j - 8) * activation j.
constexpr auto law_q4q8_block(Mem qw, u32 qb, Mem aw, u32 ab) -> bool {
  return ie_q4q8_block(qw, qb, aw, ab, ie_q8_sum(aw, ab)) == ie_q4q8_spec(32u, qw, qb, aw, ab, 0u);
}

// ---------------------------------------------------------------------
// 5. The Q8_0 path (output matrix)
// ---------------------------------------------------------------------

// LAW q8_addr: every address of the Q8_0 GPU layout and of the GGUF source
// is inside its array: words < rows*nb*8, scales < rows*nb, source bytes
// < rows*nb*34.
constexpr auto law_q8_addr(u32 rows, u32 nb, u32 r, u32 b, u32 w, u32 k) -> bool {
  return !(ie_q8_sizes_ok(rows, nb) && r < rows && b < nb && w < 8u && k < 32u) ||
         (ie_q8_dst_word(r, b, w, nb) < rows * nb * 8u && ie_q8_dst_scale(r, b, nb) < rows * nb &&
          ie_q8_src_qbyte(r, b, k, nb) < rows * nb * 34u && ie_q8_src_blk(r, b, nb) + 1u < rows * nb * 34u);
}

// LAW q8_inverse: the index map of the Q8_0 repack is one-to-one.
constexpr auto law_q8_inverse(u32 rows, u32 nb, u32 r, u32 b, u32 w) -> bool {
  return !(ie_q8_sizes_ok(rows, nb) && r < rows && b < nb && w < 8u) ||
         (ie_q8_word_row(ie_q8_dst_word(r, b, w, nb), nb) == r && ie_q8_word_blk(ie_q8_dst_word(r, b, w, nb), nb) == b &&
          ie_q8_word_w(ie_q8_dst_word(r, b, w, nb)) == w);
}

// LAW q8_onto: every word i of the Q8_0 GPU array is the image of one (r, b, w).
constexpr auto law_q8_onto(u32 rows, u32 nb, u32 i) -> bool {
  return !(ie_q8_sizes_ok(rows, nb) && 0u < nb && i < rows * nb * 8u) ||
         (ie_q8_word_row(i, nb) < rows && ie_q8_word_blk(i, nb) < nb &&
          ie_q8_dst_word(ie_q8_word_row(i, nb), ie_q8_word_blk(i, nb), ie_q8_word_w(i), nb) == i);
}

// LAW q8_byte: element j of a repacked word (packed from the GGUF bytes
// q0..q3 of that word) is the GGUF byte j & 3.
constexpr auto law_q8_byte(u32 q0, u32 q1, u32 q2, u32 q3, u32 j) -> bool {
  return ie_byte(ie_pack4(q0, q1, q2, q3), j & 3u) ==
         (((j & 3u) == 0u ? q0 : (j & 3u) == 1u ? q1 : (j & 3u) == 2u ? q2 : q3) & 255u);
}

// LAW q8q8_block: the kernel's block dot product (8 dot4 instructions) is
// the spec: the sum over j < 32 of weight j * activation j.
constexpr auto law_q8q8_block(Mem qw, u32 qb, Mem aw, u32 ab) -> bool {
  return ie_q8q8_block(qw, qb, aw, ab) == ie_q8q8_spec(32u, qw, qb, aw, ab, 0u);
}

// ---------------------------------------------------------------------
// 6. The Q4_K path (MAT_Q4K)
// ---------------------------------------------------------------------

// LAW q4k_addr: every address of the MAT_Q4K layout and of the GGUF Q4_K
// source is inside its array: nibble words < rows*nb*4, (scale, min) words
// < rows*nb, the (d, dmin) pair < ie_q4k_qs_n, the source header bytes and
// nibble bytes < rows*nb*18 (the file size of the tensor).
constexpr auto law_q4k_addr(u32 rows, u32 nb, u32 r, u32 b, u32 w, u32 j) -> bool {
  return !(ie_q4k_sizes_ok(rows, nb) && r < rows && b < nb && w < 4u && j < 32u) ||
         (ie_q4_dst_word(r, b, w, nb) < rows * nb * 4u && ie_q4k_sm(r, b, nb) < rows * nb &&
          ie_q4k_dd(rows, r, b, nb) + 1u < ie_q4k_qs_n(rows, nb) && ie_q4k_src_blk(r, b, nb) + 15u < rows * nb * 18u &&
          ie_q4k_src_qbyte(r, b, j, nb) < rows * nb * 18u);
}

// LAW q4k_inverse: the word map of MAT_Q4K (ie_q4_dst_word) is one-to-one
// for the Q4_K sizes.
constexpr auto law_q4k_inverse(u32 rows, u32 nb, u32 r, u32 b, u32 w) -> bool {
  return !(ie_q4k_sizes_ok(rows, nb) && r < rows && b < nb && w < 4u) ||
         (ie_q4_word_row(ie_q4_dst_word(r, b, w, nb), nb) == r && ie_q4_word_blk(ie_q4_dst_word(r, b, w, nb), nb) == b &&
          ie_q4_word_w(ie_q4_dst_word(r, b, w, nb)) == w);
}

// LAW q4k_onto: every nibble word i of MAT_Q4K is the image of one (r, b, w).
constexpr auto law_q4k_onto(u32 rows, u32 nb, u32 i) -> bool {
  return !(ie_q4k_sizes_ok(rows, nb) && 0u < nb && i < rows * nb * 4u) ||
         (ie_q4_word_row(i, nb) < rows && ie_q4_word_blk(i, nb) < nb &&
          ie_q4_dst_word(ie_q4_word_row(i, nb), ie_q4_word_blk(i, nb), ie_q4_word_w(i), nb) == i);
}

// LAW q4k_nibble: nibble j of a repacked word (its 4 bytes made by
// ie_q4k_pack from the source bytes l0..l3 = j' and h0..h3 = j' + 16) is the
// GGUF nibble of sub-block b in source byte j (j < 16: l_(j&3), else h_(j&3)).
constexpr auto law_q4k_nibble(u32 l0, u32 l1, u32 l2, u32 l3, u32 h0, u32 h1, u32 h2, u32 h3, u32 b, u32 j) -> bool {
  return !(j < 32u) ||
         ie_q4_nib(ie_pack4(ie_q4k_pack(l0, h0, b), ie_q4k_pack(l1, h1, b), ie_q4k_pack(l2, h2, b), ie_q4k_pack(l3, h3, b)), j) ==
             ie_q4k_src_nib(j < 16u ? ((j & 3u) == 0u ? l0 : (j & 3u) == 1u ? l1 : (j & 3u) == 2u ? l2 : l3)
                                    : ((j & 3u) == 0u ? h0 : (j & 3u) == 1u ? h1 : (j & 3u) == 2u ? h2 : h3),
                            b);
}

// LAW q4k_dot: the kernels' sub-block dot product is the spec: the sum over
// j < 32 of nibble j * activation j (exact, mod 2^32; |sum| < 2^16).
constexpr auto law_q4k_dot(Mem qw, u32 qb, Mem aw, u32 ab) -> bool {
  return ie_q4k_dot(qw, qb, aw, ab) == ie_q4k_spec(32u, qw, qb, aw, ab, 0u);
}

// ---------------------------------------------------------------------
// 7. The Q6_K path (MAT_Q6K)
// ---------------------------------------------------------------------

// LAW q6k_addr: every address of the MAT_Q6K layout and of the GGUF Q6_K
// source is inside its array (rows*nb*6 words; ie_q6k_qs_n u16; the source
// tensor has rows*(nb/8)*210 bytes).
constexpr auto law_q6k_addr(u32 rows, u32 nb, u32 r, u32 b, u32 w, u32 h, u32 j) -> bool {
  return !(ie_q4k_sizes_ok(rows, nb) && r < rows && b < nb && w < 4u && h < 2u && j < 32u) ||
         (ie_q4_dst_word(r, b, w, nb) < rows * nb * 4u && ie_q6k_hw(rows, r, b, h, nb) < ie_q6k_qw_n(rows, nb) &&
          rows * nb * 4u <= ie_q6k_hw(rows, r, b, h, nb) && ie_q6k_sc(r, b, nb) < rows * nb &&
          ie_q6k_d(rows, r, b, nb) < ie_q6k_qs_n(rows, nb) && ie_q6k_src_ql(r, b, j, nb) < rows * (nb / 8u) * 210u &&
          ie_q6k_src_qh(r, b, j, nb) < rows * (nb / 8u) * 210u && ie_q6k_src_sc(r, b, h, nb) < rows * (nb / 8u) * 210u &&
          ie_q6k_src_d(r, b, nb) + 1u < rows * (nb / 8u) * 210u);
}

// LAW q6k_unpack: q of weight j of a sub-block, read from the packed words
// (low nibbles as Q4_0: byte e of nibble word gets q_e and q_(e+16); high
// bits by ie_q6k_hpack of the 16 q of the half) is the q it was packed from.
constexpr auto law_q6k_unpack(u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1, u32 b2, u32 b3, u32 c0, u32 c1, u32 c2,
                              u32 c3, u32 d0, u32 d1, u32 d2, u32 d3, u32 e) -> bool {
  // the 16 q of one half, weights 4k + i: a (k = 0), b, c, d (k = 3); the
  // nibble word k holds weights 4k + i (low nibbles); e < 16
  return !(e < 16u) ||
         ie_q6k_q(ie_pack4(e / 4u == 0u ? a0 & 15u : e / 4u == 1u ? b0 & 15u : e / 4u == 2u ? c0 & 15u : d0 & 15u,
                           e / 4u == 0u ? a1 & 15u : e / 4u == 1u ? b1 & 15u : e / 4u == 2u ? c1 & 15u : d1 & 15u,
                           e / 4u == 0u ? a2 & 15u : e / 4u == 1u ? b2 & 15u : e / 4u == 2u ? c2 & 15u : d2 & 15u,
                           e / 4u == 0u ? a3 & 15u : e / 4u == 1u ? b3 & 15u : e / 4u == 2u ? c3 & 15u : d3 & 15u),
                  ie_q6k_hpack(a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3, d0, d1, d2, d3), e) ==
             ((e / 4u == 0u ? (e % 4u == 0u ? a0 : e % 4u == 1u ? a1 : e % 4u == 2u ? a2 : a3)
               : e / 4u == 1u ? (e % 4u == 0u ? b0 : e % 4u == 1u ? b1 : e % 4u == 2u ? b2 : b3)
               : e / 4u == 2u ? (e % 4u == 0u ? c0 : e % 4u == 1u ? c1 : e % 4u == 2u ? c2 : c3)
                              : (e % 4u == 0u ? d0 : e % 4u == 1u ? d1 : e % 4u == 2u ? d2 : d3)) & 63u);
}

// LAW q6k_dot: the kernels' half dot product (4 dot4 of the q bytes minus
// 32 times the half sum of the activations) is the spec: the sum over the
// 16 weights of half h of (q - 32) a.
constexpr auto law_q6k_dot(Mem qw, u32 qb, Mem hw, u32 hb, Mem aw, u32 ab, u32 h) -> bool {
  return !(h < 2u) ||
         ie_q6k_dot(qw, qb, hw, hb, aw, ab, h, ie_q8_hsum(aw, ab, h)) == ie_q6k_spec(16u, qw, qb, hw, hb, aw, ab, 16u * h, 16u * h + 16u);
}

// LAW q6k_dots: the signed form (the q - 32 bytes, 4 signed dot4) is the
// same spec.
constexpr auto law_q6k_dots(Mem qw, u32 qb, Mem hw, u32 hb, Mem aw, u32 ab, u32 h) -> bool {
  return !(h < 2u) || ie_q6k_dots(qw, qb, hw, hb, aw, ab, h) == ie_q6k_spec(16u, qw, qb, hw, hb, aw, ab, 16u * h, 16u * h + 16u);
}

// LAW q4k_dot_u: with the nibbles split once (u = lo, hi of each word),
// the dot product is ie_q4k_dot.
constexpr auto law_q4k_dot_u(Mem qw, u32 qb, Mem u, u32 ub, Mem aw, u32 ab) -> bool {
  return !(IE_LOAD(u, ub) == ie_q4k_lo(IE_LOAD(qw, qb)) && IE_LOAD(u, ub + 1u) == ie_q4k_hi(IE_LOAD(qw, qb)) &&
           IE_LOAD(u, ub + 2u) == ie_q4k_lo(IE_LOAD(qw, qb + 1u)) && IE_LOAD(u, ub + 3u) == ie_q4k_hi(IE_LOAD(qw, qb + 1u)) &&
           IE_LOAD(u, ub + 4u) == ie_q4k_lo(IE_LOAD(qw, qb + 2u)) && IE_LOAD(u, ub + 5u) == ie_q4k_hi(IE_LOAD(qw, qb + 2u)) &&
           IE_LOAD(u, ub + 6u) == ie_q4k_lo(IE_LOAD(qw, qb + 3u)) && IE_LOAD(u, ub + 7u) == ie_q4k_hi(IE_LOAD(qw, qb + 3u))) ||
         ie_q4k_dot_u(u, ub, aw, ab) == ie_q4k_dot(qw, qb, aw, ab);
}

// LAW q6k_dots_s: with the q - 32 words made once, the half dot product is
// ie_q6k_dots.
constexpr auto law_q6k_dots_s(Mem qw, u32 qb, Mem hw, u32 hb, Mem sw, u32 sb, Mem aw, u32 ab, u32 h) -> bool {
  return !(IE_LOAD(sw, sb) == ie_q6k_sw(IE_LOAD(qw, qb), IE_LOAD(hw, hb), h, 0u) &&
           IE_LOAD(sw, sb + 1u) == ie_q6k_sw(IE_LOAD(qw, qb + 1u), IE_LOAD(hw, hb), h, 1u) &&
           IE_LOAD(sw, sb + 2u) == ie_q6k_sw(IE_LOAD(qw, qb + 2u), IE_LOAD(hw, hb), h, 2u) &&
           IE_LOAD(sw, sb + 3u) == ie_q6k_sw(IE_LOAD(qw, qb + 3u), IE_LOAD(hw, hb), h, 3u)) ||
         ie_q6k_dots_s(sw, sb, aw, ab, h) == ie_q6k_dots(qw, qb, hw, hb, aw, ab, h);
}
