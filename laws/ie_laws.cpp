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
