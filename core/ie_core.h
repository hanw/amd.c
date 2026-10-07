/* ie_core.h -- the verified core of the inference engine.
 *
 * One source, two languages:
 *  - C (C11): the host engine and the GPU kernels (clang, amdgcn) include it.
 *  - C++17, the functional subset of cpp-lean: EDG (cpfe-lean) translates it
 *    to Lean 4, and proofs/ proves the laws of laws/ie_laws.cpp about it.
 *
 * Rules (the cpp-lean subset): only u32, bool, structs of these, Mem (a
 * read-only u32 array), ?: and recursion f(n - 1, ...) with fuel n. No loops,
 * no pointers, no signed int, no float. A recursive function is for the
 * host (checkers and specs); the GPU kernels call only non-recursive ones.
 *
 * What the laws say (laws/ie_laws.cpp, proved in Lean for all inputs):
 *  1. Index and memory safety: the GEMV grid gives each row to exactly one
 *     wave; the strided split of the blocks of a row over the 32 lanes gives
 *     each block to exactly one (lane, step); every address is inside its
 *     array; the Q4 repack index map is a bijection.
 *  2. The static memory plan: in an accepted plan no two live buffers share
 *     a byte, and every buffer is inside the arena and aligned.
 *  3. Reduction order: the strided per-lane sums, and the wave tree
 *     reduction, give the plain sum (integers mod 2^32).
 *  4. Quantization: the nibble unpack of a repacked word gives the GGUF
 *     Q4_0 nibble, and the dot4 formula with the "- 8 * sum(a)" correction
 *     gives the exact integer dot product of (q - 8) and a.
 */
#ifndef IE_CORE_H
#define IE_CORE_H

#ifdef __cplusplus
/* EDG / cpp-lean: lean::Mem and lean::load have a fixed meaning in Lean. */
#include "lean_mem.h"
#define IE_FN(T, name, params) constexpr auto name params -> T
#define IE_MK(S) S
#define IE_LOAD(m, i) lean::load(m, i)
using Mem = lean::Mem;
#else
#include <stdbool.h>
typedef struct { const unsigned int *p; } Mem;
#define IE_FN(T, name, params) static inline T name params
#define IE_MK(S) (S)
#define IE_LOAD(m, i) ((m).p[(i)])
#endif

typedef unsigned int u32;

/* ======================================================================
 * 0. Constants of the gfx1201 kernels
 * ==================================================================== */

/* RDNA wave32: 32 lanes per wave. */
IE_FN(u32, ie_wave, (void)) { return 32u; }
/* GEMV: 8 waves per workgroup (256 threads), one row per wave. */
IE_FN(u32, ie_rows_per_wg, (void)) { return 8u; }
/* Arena alignment: 256 bytes. */
IE_FN(u32, ie_align_mask, (void)) { return 255u; }

/* The sizes that the kernels support: rows < 2^18 (a vocabulary of 151936
 * fits) and nb < 2^9 blocks of 32 per row (K <= 16352). Then
 * rows * nb * 18 < 2^32: no address wraps. */
IE_FN(bool, ie_sizes_ok, (u32 rows, u32 nb)) { return rows < 0x40000u && nb < 0x200u; }

/* ======================================================================
 * 1. Q4_0 layout: GGUF source and the GPU (repacked) layout
 * ==================================================================== */

/* GGUF: block b of row r starts at byte (r * nb + b) * 18. Bytes 0..1: the
 * f16 scale. Byte 2 + k (k < 16): nibble k (low 4 bits) and nibble k + 16
 * (high 4 bits). */
IE_FN(u32, ie_q4_src_blk, (u32 r, u32 b, u32 nb)) { return (r * nb + b) * 18u; }
IE_FN(u32, ie_q4_src_qbyte, (u32 r, u32 b, u32 k, u32 nb)) { return (r * nb + b) * 18u + 2u + k; }

/* GPU layout: the nibbles of all blocks as u32 words, 4 words per block, row
 * major: word w of block b of row r is at (r * nb + b) * 4 + w. The scales
 * are a separate f16 array (index r * nb + b). Lane l of a wave reads the 16
 * bytes of block l + 32 t, so 32 lanes read 512 consecutive bytes. */
IE_FN(u32, ie_q4_dst_word, (u32 r, u32 b, u32 w, u32 nb)) { return (r * nb + b) * 4u + w; }
IE_FN(u32, ie_q4_dst_scale, (u32 r, u32 b, u32 nb)) { return r * nb + b; }

/* The inverse of ie_q4_dst_word. */
IE_FN(u32, ie_q4_word_row, (u32 i, u32 nb)) { return (i / 4u) / nb; }
IE_FN(u32, ie_q4_word_blk, (u32 i, u32 nb)) { return (i / 4u) % nb; }
IE_FN(u32, ie_q4_word_w, (u32 i)) { return i % 4u; }

/* Byte e (e < 4) of x. */
IE_FN(u32, ie_byte, (u32 x, u32 e)) { return (x >> (8u * e)) & 255u; }

/* Four bytes into one little-endian word. The repack of word w of a block
 * packs its source nibble bytes 4w .. 4w + 3. */
IE_FN(u32, ie_pack4, (u32 b0, u32 b1, u32 b2, u32 b3)) {
  return (b0 & 255u) | ((b1 & 255u) << 8u) | ((b2 & 255u) << 16u) | ((b3 & 255u) << 24u);
}

/* Weight j (j < 32) of a block in the GPU layout: word (j & 15) >> 2, byte
 * j & 3, the high nibble if j >= 16. This is what the kernel computes. */
IE_FN(u32, ie_q4_word_of, (u32 j)) { return (j & 15u) >> 2u; }
IE_FN(u32, ie_q4_nib, (u32 word, u32 j)) {
  return (word >> (8u * (j & 3u) + 4u * ((j >> 4u) & 1u))) & 15u;
}

/* The GGUF meaning of nibble j, from the nibble byte q_(j & 15) of the
 * block: the low 4 bits if j < 16, else the high 4 bits. */
IE_FN(u32, ie_q4_spec_nib, (u32 qbyte, u32 j)) {
  return j < 16u ? (qbyte & 15u) : ((qbyte & 255u) >> 4u);
}

/* ======================================================================
 * 2. Integer dot products (Q4 weights x Q8 activations)
 * ==================================================================== */

/* Sign extension of the int8 in the low byte (two's complement in u32). */
IE_FN(u32, ie_sext8, (u32 v)) { return (v & 128u) != 0u ? ((v & 255u) | 0xFFFFFF00u) : (v & 255u); }

/* The model of v_dot4_i32_iu8 with A unsigned and B signed
 * (__builtin_amdgcn_sudot4(false, a, true, b, acc, false)):
 * acc + sum over e < 4 of byte(a, e) * sext8(byte(b, e)), mod 2^32.
 * The GPU build uses the instruction itself; the proofs and the CPU build
 * use the model. TRUST: the instruction computes the model (AMD RDNA4 ISA). */
#if defined(__AMDGCN__) && !defined(__cplusplus)
IE_FN(u32, ie_dot4_us, (u32 acc, u32 a, u32 b)) {
  return (u32)__builtin_amdgcn_sudot4(false, (int)a, true, (int)b, (int)acc, false);
}
#else
IE_FN(u32, ie_dot4_us, (u32 acc, u32 a, u32 b)) {
  return acc + ie_byte(a, 0u) * ie_sext8(ie_byte(b, 0u)) + ie_byte(a, 1u) * ie_sext8(ie_byte(b, 1u)) +
         ie_byte(a, 2u) * ie_sext8(ie_byte(b, 2u)) + ie_byte(a, 3u) * ie_sext8(ie_byte(b, 3u));
}
#endif

/* The sum of the 4 signed bytes of a. */
IE_FN(u32, ie_sum_s8x4, (u32 a)) {
  return ie_sext8(ie_byte(a, 0u)) + ie_sext8(ie_byte(a, 1u)) + ie_sext8(ie_byte(a, 2u)) + ie_sext8(ie_byte(a, 3u));
}

/* One Q4 word wq (weights 4w..4w+3 in the low nibbles, 16+4w..16+4w+3 in
 * the high nibbles) against the Q8 words alo (activations 4w..4w+3) and ahi
 * (activations 16+4w..16+4w+3): two dot4 instructions, no "- 8" yet. */
IE_FN(u32, ie_q4q8_word, (u32 acc, u32 wq, u32 alo, u32 ahi)) {
  return ie_dot4_us(ie_dot4_us(acc, wq & 0x0F0F0F0Fu, alo), (wq >> 4u) & 0x0F0F0F0Fu, ahi);
}

/* The same word, by the definition: sum of (q - 8) * a over its 8 weights. */
IE_FN(u32, ie_q4q8_word_spec, (u32 wq, u32 alo, u32 ahi)) {
  return (ie_q4_nib(wq, 0u) - 8u) * ie_sext8(ie_byte(alo, 0u)) + (ie_q4_nib(wq, 1u) - 8u) * ie_sext8(ie_byte(alo, 1u)) +
         (ie_q4_nib(wq, 2u) - 8u) * ie_sext8(ie_byte(alo, 2u)) + (ie_q4_nib(wq, 3u) - 8u) * ie_sext8(ie_byte(alo, 3u)) +
         (ie_q4_nib(wq, 16u) - 8u) * ie_sext8(ie_byte(ahi, 0u)) +
         (ie_q4_nib(wq, 17u) - 8u) * ie_sext8(ie_byte(ahi, 1u)) +
         (ie_q4_nib(wq, 18u) - 8u) * ie_sext8(ie_byte(ahi, 2u)) + (ie_q4_nib(wq, 19u) - 8u) * ie_sext8(ie_byte(ahi, 3u));
}

/* The sum of the 32 signed activations of the Q8 block at word ab of aw.
 * The quantize kernel stores it next to the block (asum). */
IE_FN(u32, ie_q8_sum, (Mem aw, u32 ab)) {
  return ie_sum_s8x4(IE_LOAD(aw, ab)) + ie_sum_s8x4(IE_LOAD(aw, ab + 1u)) + ie_sum_s8x4(IE_LOAD(aw, ab + 2u)) +
         ie_sum_s8x4(IE_LOAD(aw, ab + 3u)) + ie_sum_s8x4(IE_LOAD(aw, ab + 4u)) + ie_sum_s8x4(IE_LOAD(aw, ab + 5u)) +
         ie_sum_s8x4(IE_LOAD(aw, ab + 6u)) + ie_sum_s8x4(IE_LOAD(aw, ab + 7u));
}

/* The integer dot product of one Q4 block (4 words at qb of qw) and one Q8
 * block (8 words at ab of aw, with the sum asum), as the kernel computes it:
 * 8 dot4 instructions, then - 8 * asum. */
IE_FN(u32, ie_q4q8_block, (Mem qw, u32 qb, Mem aw, u32 ab, u32 asum)) {
  return ie_q4q8_word(ie_q4q8_word(ie_q4q8_word(ie_q4q8_word(0u, IE_LOAD(qw, qb), IE_LOAD(aw, ab), IE_LOAD(aw, ab + 4u)),
                                                IE_LOAD(qw, qb + 1u), IE_LOAD(aw, ab + 1u), IE_LOAD(aw, ab + 5u)),
                                   IE_LOAD(qw, qb + 2u), IE_LOAD(aw, ab + 2u), IE_LOAD(aw, ab + 6u)),
                      IE_LOAD(qw, qb + 3u), IE_LOAD(aw, ab + 3u), IE_LOAD(aw, ab + 7u)) -
         8u * asum;
}

/* The spec of the block dot product: the sum over j < 32 of
 * (weight j - 8) * activation j, element by element. Weight j is nibble j of
 * word (j & 15) >> 2; activation j is signed byte j & 3 of word j >> 2.
 * n is the fuel (32). */
IE_FN(u32, ie_q4q8_term, (Mem qw, u32 qb, Mem aw, u32 ab, u32 j)) {
  return (ie_q4_nib(IE_LOAD(qw, qb + ie_q4_word_of(j)), j) - 8u) * ie_sext8(ie_byte(IE_LOAD(aw, ab + (j >> 2u)), j & 3u));
}
IE_FN(u32, ie_q4q8_spec, (u32 n, Mem qw, u32 qb, Mem aw, u32 ab, u32 j)) {
  return n == 0u ? 0u : j < 32u ? ie_q4q8_term(qw, qb, aw, ab, j) + ie_q4q8_spec(n - 1u, qw, qb, aw, ab, j + 1u) : 0u;
}

/* ======================================================================
 * 2b. Q8_0 weights x Q8 activations (the output matrix of many GGUF files)
 * ==================================================================== */

/* The sizes of the Q8_0 path: rows < 2^18, nb < 2^11 blocks per row
 * (K <= 65504) and rows * nb < 2^26 blocks in all. The product cannot wrap
 * (it is < 2^29), and rows * nb * 34 < 2^32: no address wraps. */
IE_FN(bool, ie_q8_sizes_ok, (u32 rows, u32 nb)) { return rows < 0x40000u && nb < 0x800u && rows * nb < 0x4000000u; }

/* GGUF Q8_0: block b of row r starts at byte (r * nb + b) * 34. Bytes 0..1:
 * the f16 scale. Byte 2 + k (k < 32): the int8 weight k. */
IE_FN(u32, ie_q8_src_blk, (u32 r, u32 b, u32 nb)) { return (r * nb + b) * 34u; }
IE_FN(u32, ie_q8_src_qbyte, (u32 r, u32 b, u32 k, u32 nb)) { return (r * nb + b) * 34u + 2u + k; }

/* GPU layout: 8 u32 words per block, row major: word w of block b of row r
 * is at (r * nb + b) * 8 + w and holds the weights 4w .. 4w + 3 (ie_pack4
 * of the source bytes). The f16 scales are a separate array (r * nb + b). */
IE_FN(u32, ie_q8_dst_word, (u32 r, u32 b, u32 w, u32 nb)) { return (r * nb + b) * 8u + w; }
IE_FN(u32, ie_q8_dst_scale, (u32 r, u32 b, u32 nb)) { return r * nb + b; }

/* The inverse of ie_q8_dst_word. */
IE_FN(u32, ie_q8_word_row, (u32 i, u32 nb)) { return (i / 8u) / nb; }
IE_FN(u32, ie_q8_word_blk, (u32 i, u32 nb)) { return (i / 8u) % nb; }
IE_FN(u32, ie_q8_word_w, (u32 i)) { return i % 8u; }

/* The model of v_dot4_i32_iu8 with A and B signed
 * (__builtin_amdgcn_sudot4(true, a, true, b, acc, false)):
 * acc + sum over e < 4 of sext8(byte(a, e)) * sext8(byte(b, e)), mod 2^32.
 * TRUST: as for ie_dot4_us, the instruction computes the model. */
#if defined(__AMDGCN__) && !defined(__cplusplus)
IE_FN(u32, ie_dot4_ss, (u32 acc, u32 a, u32 b)) {
  return (u32)__builtin_amdgcn_sudot4(true, (int)a, true, (int)b, (int)acc, false);
}
#else
IE_FN(u32, ie_dot4_ss, (u32 acc, u32 a, u32 b)) {
  return acc + ie_sext8(ie_byte(a, 0u)) * ie_sext8(ie_byte(b, 0u)) + ie_sext8(ie_byte(a, 1u)) * ie_sext8(ie_byte(b, 1u)) +
         ie_sext8(ie_byte(a, 2u)) * ie_sext8(ie_byte(b, 2u)) + ie_sext8(ie_byte(a, 3u)) * ie_sext8(ie_byte(b, 3u));
}
#endif

/* The integer dot product of one Q8_0 block (8 words at qb of qw) and one Q8
 * block (8 words at ab of aw), as the kernel computes it: 8 dot4. */
IE_FN(u32, ie_q8q8_block, (Mem qw, u32 qb, Mem aw, u32 ab)) {
  return ie_dot4_ss(
      ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(0u, IE_LOAD(qw, qb), IE_LOAD(aw, ab)),
                                                                        IE_LOAD(qw, qb + 1u), IE_LOAD(aw, ab + 1u)),
                                                             IE_LOAD(qw, qb + 2u), IE_LOAD(aw, ab + 2u)),
                                                  IE_LOAD(qw, qb + 3u), IE_LOAD(aw, ab + 3u)),
                                       IE_LOAD(qw, qb + 4u), IE_LOAD(aw, ab + 4u)),
                            IE_LOAD(qw, qb + 5u), IE_LOAD(aw, ab + 5u)),
                 IE_LOAD(qw, qb + 6u), IE_LOAD(aw, ab + 6u)),
      IE_LOAD(qw, qb + 7u), IE_LOAD(aw, ab + 7u));
}

/* The spec: the sum over j < 32 of weight j * activation j, where element j
 * of a block is the signed byte j & 3 of word j >> 2. n is the fuel (32). */
IE_FN(u32, ie_q8q8_term, (Mem qw, u32 qb, Mem aw, u32 ab, u32 j)) {
  return ie_sext8(ie_byte(IE_LOAD(qw, qb + (j >> 2u)), j & 3u)) * ie_sext8(ie_byte(IE_LOAD(aw, ab + (j >> 2u)), j & 3u));
}
IE_FN(u32, ie_q8q8_spec, (u32 n, Mem qw, u32 qb, Mem aw, u32 ab, u32 j)) {
  return n == 0u ? 0u : j < 32u ? ie_q8q8_term(qw, qb, aw, ab, j) + ie_q8q8_spec(n - 1u, qw, qb, aw, ab, j + 1u) : 0u;
}

/* ======================================================================
 * 3. Work split: GEMV grid, lanes, reductions
 * ==================================================================== */

/* GEMV grid: workgroup g, wave v (v < 8) computes row g * 8 + v. */
IE_FN(u32, ie_gemv_ngroups, (u32 rows)) { return (rows + 7u) >> 3u; }
IE_FN(u32, ie_gemv_row, (u32 g, u32 v)) { return g * 8u + v; }

/* In a wave, lane l (l < 32) does blocks l, l + 32, l + 64, ... of its row. */
IE_FN(u32, ie_lane_blk, (u32 l, u32 t)) { return l + 32u * t; }
IE_FN(u32, ie_blk_lane, (u32 b)) { return b & 31u; }
IE_FN(u32, ie_blk_step, (u32 b)) { return b >> 5u; }

/* x[lo] + x[lo + 1] + ... + x[hi - 1] (mod 2^32). n is the fuel. */
IE_FN(u32, ie_seq_sum, (u32 n, Mem x, u32 lo, u32 hi)) {
  return n == 0u ? 0u : lo < hi ? IE_LOAD(x, lo) + ie_seq_sum(n - 1u, x, lo + 1u, hi) : 0u;
}

/* The partial sum of one lane: x[b] + x[b + 32] + ... while < nb. */
IE_FN(u32, ie_lane_sum, (u32 n, Mem x, u32 b, u32 nb)) {
  return n == 0u ? 0u : b < nb ? IE_LOAD(x, b) + ie_lane_sum(n - 1u, x, b + 32u, nb) : 0u;
}
/* The partial sums of lanes l, l + 1, ..., 31, added. */
IE_FN(u32, ie_lanes_sum, (u32 n, Mem x, u32 l, u32 nb)) {
  return n == 0u ? 0u : l < 32u ? ie_lane_sum(nb, x, l, nb) + ie_lanes_sum(n - 1u, x, l + 1u, nb) : 0u;
}
IE_FN(u32, ie_strided_sum, (Mem x, u32 nb)) { return ie_lanes_sum(32u, x, 0u, nb); }

/* The wave tree reduction (shuffle down by 16, 8, 4, 2, 1): the value of
 * lane i after m steps. Lane 0 after 5 steps holds the wave sum. */
IE_FN(u32, ie_tree, (u32 m, Mem x, u32 i)) {
  return m == 0u ? IE_LOAD(x, i) : ie_tree(m - 1u, x, i) + ie_tree(m - 1u, x, i + (32u >> m));
}
IE_FN(u32, ie_wave_sum, (Mem x)) { return ie_tree(5u, x, 0u); }

/* ======================================================================
 * 4. The static memory plan of the activation arena
 * ==================================================================== */

/* Buffer i (i < n) has the bytes [off_i, off_i + size_i) of the arena of
 * cap bytes, and holds live data in the steps first_i .. last_i of one
 * forward pass. The planner (any algorithm) makes the plan; the engine runs
 * ie_plan_check on it before it uses it. */
typedef struct Plan {
  Mem off;
  Mem size;
  Mem first;
  Mem last;
  u32 n;
  u32 cap;
} Plan;

IE_FN(u32, ie_off, (Plan p, u32 i)) { return IE_LOAD(p.off, i); }
IE_FN(u32, ie_size, (Plan p, u32 i)) { return IE_LOAD(p.size, i); }
IE_FN(u32, ie_first, (Plan p, u32 i)) { return IE_LOAD(p.first, i); }
IE_FN(u32, ie_last, (Plan p, u32 i)) { return IE_LOAD(p.last, i); }

/* Buffer i is inside the arena, aligned, and has a lifetime. */
IE_FN(bool, ie_buf_ok, (Plan p, u32 i)) {
  return ie_off(p, i) <= p.cap && ie_size(p, i) <= p.cap - ie_off(p, i) &&
         (ie_off(p, i) & ie_align_mask()) == 0u && ie_first(p, i) <= ie_last(p, i);
}
/* Buffers i and j: if their lifetimes overlap, their bytes do not. */
IE_FN(bool, ie_buf_sep, (Plan p, u32 i, u32 j)) {
  return !(ie_first(p, i) <= ie_last(p, j) && ie_first(p, j) <= ie_last(p, i)) ||
         ie_off(p, i) + ie_size(p, i) <= ie_off(p, j) || ie_off(p, j) + ie_size(p, j) <= ie_off(p, i);
}
/* Buffer i against the buffers j, j + 1, ..., n - 1. */
IE_FN(bool, ie_pairs_ok, (u32 k, Plan p, u32 i, u32 j)) {
  return k == 0u ? true : j < p.n ? ie_buf_sep(p, i, j) && ie_pairs_ok(k - 1u, p, i, j + 1u) : true;
}
/* The buffers i, i + 1, ..., n - 1. */
IE_FN(bool, ie_bufs_ok, (u32 k, Plan p, u32 i)) {
  return k == 0u ? true : i < p.n ? ie_buf_ok(p, i) && ie_pairs_ok(p.n, p, i, i + 1u) && ie_bufs_ok(k - 1u, p, i + 1u) : true;
}
IE_FN(bool, ie_plan_check, (Plan p)) { return ie_bufs_ok(p.n, p, 0u); }

/* Byte x is in buffer i. Buffer i is live in step t. */
IE_FN(bool, ie_in_buf, (Plan p, u32 i, u32 x)) { return ie_off(p, i) <= x && x < ie_off(p, i) + ie_size(p, i); }
IE_FN(bool, ie_live, (Plan p, u32 i, u32 t)) { return ie_first(p, i) <= t && t <= ie_last(p, i); }

#endif
