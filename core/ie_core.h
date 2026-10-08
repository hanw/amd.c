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
 * 2c. Q4_K weights (MAT_Q4K) x Q8 activations
 * ==================================================================== */

/* GGUF Q4_K: super-blocks of 256 weights (8 sub-blocks of 32), 144 bytes:
 * f16 d, f16 dmin, 12 bytes of 6-bit scales and mins, 128 nibble bytes.
 * Sub-blocks 2k and 2k + 1 share the bytes 32k .. 32k + 31: the low nibbles
 * are sub-block 2k, the high nibbles sub-block 2k + 1. Weight =
 * d * sc * q - dmin * mn (q = 0..15).
 * MAT_Q4K (GPU): the nibble words of sub-block b of row r as a Q4_0 block
 * (ie_q4_dst_word; byte j: weights j and j + 16 in the low and high nibble);
 * qs (u16): (sc | mn << 8) of sub-block b at ie_q4k_sm, then the (f16 d,
 * f16 dmin) of each super-block at ie_q4k_dd; ie_q4k_qs_n entries in all.
 * The sizes: those of Q8_0, and whole super-blocks (nb % 8 == 0). Then
 * rows * nb * 18 < 2^31: no address wraps. */
IE_FN(bool, ie_q4k_sizes_ok, (u32 rows, u32 nb)) {
  return rows < 0x40000u && nb < 0x800u && rows * nb < 0x4000000u && nb % 8u == 0u;
}
IE_FN(u32, ie_q4k_sm, (u32 r, u32 b, u32 nb)) { return r * nb + b; }
IE_FN(u32, ie_q4k_dd, (u32 rows, u32 r, u32 b, u32 nb)) { return rows * nb + (r * (nb / 8u) + b / 8u) * 2u; }
IE_FN(u32, ie_q4k_qs_n, (u32 rows, u32 nb)) { return rows * nb + rows * (nb / 8u) * 2u; }

/* GGUF Q4_K source: the super-block of sub-block b of row r starts at byte
 * (r * nb + 8 (b / 8)) * 18 (144 = 8 * 18); byte j (j < 32) of the nibble
 * bytes of sub-block b is at + 16 + 32 * ((b % 8) / 2) + j. */
IE_FN(u32, ie_q4k_src_blk, (u32 r, u32 b, u32 nb)) { return (r * nb + (b / 8u) * 8u) * 18u; }
IE_FN(u32, ie_q4k_src_qbyte, (u32 r, u32 b, u32 j, u32 nb)) {
  return ie_q4k_src_blk(r, b, nb) + 16u + 32u * ((b % 8u) / 2u) + j;
}
/* The nibble of sub-block b in a source byte, and the repacked byte j of
 * the Q4_0-order block from the source bytes j (lo) and j + 16 (hi). */
IE_FN(u32, ie_q4k_src_nib, (u32 qbyte, u32 b)) { return b % 2u == 1u ? (qbyte & 255u) >> 4u : qbyte & 15u; }
IE_FN(u32, ie_q4k_pack, (u32 lo, u32 hi, u32 b)) { return ie_q4k_src_nib(lo, b) | (ie_q4k_src_nib(hi, b) << 4u); }

/* The integer dot product of one MAT_Q4K sub-block (4 words at qb of qw,
 * nibbles q = 0..15) and one Q8 block: the sum over j < 32 of q_j * a_j,
 * as the kernels compute it (8 dot4, no offset). The kernels then add
 * da ((d sc) dot - (dmin mn) asum) in float. */
IE_FN(u32, ie_q4k_dot, (Mem qw, u32 qb, Mem aw, u32 ab)) {
  return ie_q4q8_word(ie_q4q8_word(ie_q4q8_word(ie_q4q8_word(0u, IE_LOAD(qw, qb), IE_LOAD(aw, ab), IE_LOAD(aw, ab + 4u)),
                                                IE_LOAD(qw, qb + 1u), IE_LOAD(aw, ab + 1u), IE_LOAD(aw, ab + 5u)),
                                   IE_LOAD(qw, qb + 2u), IE_LOAD(aw, ab + 2u), IE_LOAD(aw, ab + 6u)),
                      IE_LOAD(qw, qb + 3u), IE_LOAD(aw, ab + 3u), IE_LOAD(aw, ab + 7u));
}
/* ie_q4k_dot with the nibbles already split (u: for k < 4, word 2k = the
 * low nibbles of word k, 2k + 1 = the high nibbles), as the multi-token
 * kernels do once per weight block and use for every token. */
IE_FN(u32, ie_q4k_lo, (u32 wq)) { return wq & 0x0F0F0F0Fu; }
IE_FN(u32, ie_q4k_hi, (u32 wq)) { return (wq >> 4u) & 0x0F0F0F0Fu; }
IE_FN(u32, ie_q4k_dot_u, (Mem u, u32 ub, Mem aw, u32 ab)) {
  return ie_dot4_us(ie_dot4_us(ie_dot4_us(ie_dot4_us(ie_dot4_us(ie_dot4_us(ie_dot4_us(ie_dot4_us(0u, IE_LOAD(u, ub), IE_LOAD(aw, ab)),
      IE_LOAD(u, ub + 1u), IE_LOAD(aw, ab + 4u)), IE_LOAD(u, ub + 2u), IE_LOAD(aw, ab + 1u)), IE_LOAD(u, ub + 3u), IE_LOAD(aw, ab + 5u)),
      IE_LOAD(u, ub + 4u), IE_LOAD(aw, ab + 2u)), IE_LOAD(u, ub + 5u), IE_LOAD(aw, ab + 6u)),
      IE_LOAD(u, ub + 6u), IE_LOAD(aw, ab + 3u)), IE_LOAD(u, ub + 7u), IE_LOAD(aw, ab + 7u));
}
/* The spec: the sum over j < 32 of nibble j * activation j. n is the fuel. */
IE_FN(u32, ie_q4k_term, (Mem qw, u32 qb, Mem aw, u32 ab, u32 j)) {
  return ie_q4_nib(IE_LOAD(qw, qb + ie_q4_word_of(j)), j) * ie_sext8(ie_byte(IE_LOAD(aw, ab + (j >> 2u)), j & 3u));
}
IE_FN(u32, ie_q4k_spec, (u32 n, Mem qw, u32 qb, Mem aw, u32 ab, u32 j)) {
  return n == 0u ? 0u : j < 32u ? ie_q4k_term(qw, qb, aw, ab, j) + ie_q4k_spec(n - 1u, qw, qb, aw, ab, j + 1u) : 0u;
}

/* ======================================================================
 * 2d. Q6_K weights (MAT_Q6K) x Q8 activations
 * ==================================================================== */

/* GGUF Q6_K: super-blocks of 256 weights, 210 bytes: ql[128] (low 4 bits),
 * qh[64] (high 2 bits), int8 scales[16] (one per 16 weights), f16 d.
 * Weight e of a super-block (e = 128 n + 32 m + l): low bits = nibble
 * (m < 2: low, else high) of ql[64 n + 32 (m & 1) + l], high bits =
 * (qh[32 n + l] >> 2m) & 3; value = d * sc[e / 16] * (q - 32), q = 0..63.
 * MAT_Q6K (GPU), sub-block b (32 weights) of row r:
 *  - qw words 0 .. rows*nb*4: the low 4 bits as a Q4_0 block (ie_q4_dst_word);
 *  - qw words rows*nb*4 + ie_q6k_hw: 2 words of high bits, word h for the
 *    weights 16h .. 16h + 15; weight 16h + 4k + i at bits 8i + 2k (so word k
 *    of the activations lines up after a shift by 2k);
 *  - qs (u16): (sc0 | sc1 << 8) (int8 scales of the two halves) at
 *    ie_q6k_sc, then f16 d of each super-block at ie_q6k_d.
 * Sizes: as Q4_K (ie_q4k_sizes_ok). */
IE_FN(u32, ie_q6k_hw, (u32 rows, u32 r, u32 b, u32 h, u32 nb)) { return rows * nb * 4u + (r * nb + b) * 2u + h; }
IE_FN(u32, ie_q6k_qw_n, (u32 rows, u32 nb)) { return rows * nb * 6u; }
IE_FN(u32, ie_q6k_sc, (u32 r, u32 b, u32 nb)) { return r * nb + b; }
IE_FN(u32, ie_q6k_d, (u32 rows, u32 r, u32 b, u32 nb)) { return rows * nb + r * (nb / 8u) + b / 8u; }
IE_FN(u32, ie_q6k_qs_n, (u32 rows, u32 nb)) { return rows * nb + rows * (nb / 8u); }

/* GGUF Q6_K source bytes of weight j (j < 32) of sub-block b of row r. */
IE_FN(u32, ie_q6k_src_blk, (u32 r, u32 b, u32 nb)) { return (r * (nb / 8u) + b / 8u) * 210u; }
IE_FN(u32, ie_q6k_src_ql, (u32 r, u32 b, u32 j, u32 nb)) {
  return ie_q6k_src_blk(r, b, nb) + 64u * ((b % 8u) / 4u) + 32u * ((b % 8u) % 2u) + j;
}
IE_FN(u32, ie_q6k_src_qh, (u32 r, u32 b, u32 j, u32 nb)) { return ie_q6k_src_blk(r, b, nb) + 128u + 32u * ((b % 8u) / 4u) + j; }
IE_FN(u32, ie_q6k_src_sc, (u32 r, u32 b, u32 h, u32 nb)) { return ie_q6k_src_blk(r, b, nb) + 192u + 2u * (b % 8u) + h; }
IE_FN(u32, ie_q6k_src_d, (u32 r, u32 b, u32 nb)) { return ie_q6k_src_blk(r, b, nb) + 208u; }
/* The 6-bit q of a weight of sub-block b from its ql and qh bytes. */
IE_FN(u32, ie_q6k_src_q, (u32 ql, u32 qh, u32 b)) {
  return (((b % 8u) % 4u < 2u ? ql & 15u : (ql & 255u) >> 4u) | ((((qh & 255u) >> (2u * ((b % 8u) % 4u))) & 3u) << 4u));
}

/* The high-bits word of 16 weights (q0 .. q15: weight 4k + i at bits 8i + 2k). */
IE_FN(u32, ie_q6k_h2, (u32 q, u32 e)) { return ((q >> 4u) & 3u) << (8u * (e % 4u) + 2u * (e / 4u)); }
IE_FN(u32, ie_q6k_hpack, (u32 q0, u32 q1, u32 q2, u32 q3, u32 q4, u32 q5, u32 q6, u32 q7, u32 q8, u32 q9, u32 q10,
                          u32 q11, u32 q12, u32 q13, u32 q14, u32 q15)) {
  return ie_q6k_h2(q0, 0u) | ie_q6k_h2(q1, 1u) | ie_q6k_h2(q2, 2u) | ie_q6k_h2(q3, 3u) | ie_q6k_h2(q4, 4u) |
         ie_q6k_h2(q5, 5u) | ie_q6k_h2(q6, 6u) | ie_q6k_h2(q7, 7u) | ie_q6k_h2(q8, 8u) | ie_q6k_h2(q9, 9u) |
         ie_q6k_h2(q10, 10u) | ie_q6k_h2(q11, 11u) | ie_q6k_h2(q12, 12u) | ie_q6k_h2(q13, 13u) | ie_q6k_h2(q14, 14u) |
         ie_q6k_h2(q15, 15u);
}
/* q of weight j (j < 32) from the nibble words (Q4_0 order) and the
 * high-bits word of its half. */
IE_FN(u32, ie_q6k_q, (u32 nibword, u32 hword, u32 j)) {
  return ie_q4_nib(nibword, j) | (((hword >> (8u * (j % 4u) + 2u * ((j % 16u) / 4u))) & 3u) << 4u);
}
/* The 4 values of q (bytes) of word k (k < 4) of half h: weights
 * 16h + 4k .. 16h + 4k + 3, from nibble word k and the high-bits word h. */
IE_FN(u32, ie_q6k_vw, (u32 nibword, u32 hword, u32 h, u32 k)) {
  return (h == 0u ? nibword & 0x0F0F0F0Fu : (nibword >> 4u) & 0x0F0F0F0Fu) | (((hword >> (2u * k)) & 0x03030303u) << 4u);
}
/* The sum of the 16 activations of half h of the Q8 block at word ab: 4
 * dot4 with the bytes 1, 1, 1, 1 (0x01010101). */
IE_FN(u32, ie_q8_hsum, (Mem aw, u32 ab, u32 h)) {
  return ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(0u, IE_LOAD(aw, ab + 4u * h), 0x01010101u), IE_LOAD(aw, ab + 4u * h + 1u), 0x01010101u),
                               IE_LOAD(aw, ab + 4u * h + 2u), 0x01010101u),
                    IE_LOAD(aw, ab + 4u * h + 3u), 0x01010101u);
}
/* The integer dot product of half h of a MAT_Q6K sub-block (nibble words at
 * qb of qw, high-bits word at hb of hw) and the Q8 block at ab:
 * sum over the 16 weights of (q - 32) a, as the kernels compute it: 4 dot4
 * of the q bytes, minus 32 times the half sum hs (ie_q8_hsum). */
IE_FN(u32, ie_q6k_dot, (Mem qw, u32 qb, Mem hw, u32 hb, Mem aw, u32 ab, u32 h, u32 hs)) {
  return ie_dot4_us(ie_dot4_us(ie_dot4_us(ie_dot4_us(0u, ie_q6k_vw(IE_LOAD(qw, qb), IE_LOAD(hw, hb), h, 0u), IE_LOAD(aw, ab + 4u * h)),
                                          ie_q6k_vw(IE_LOAD(qw, qb + 1u), IE_LOAD(hw, hb), h, 1u), IE_LOAD(aw, ab + 4u * h + 1u)),
                               ie_q6k_vw(IE_LOAD(qw, qb + 2u), IE_LOAD(hw, hb), h, 2u), IE_LOAD(aw, ab + 4u * h + 2u)),
                    ie_q6k_vw(IE_LOAD(qw, qb + 3u), IE_LOAD(hw, hb), h, 3u), IE_LOAD(aw, ab + 4u * h + 3u)) -
         32u * hs;
}
/* The same 4 bytes as signed q - 32 (int8): (v | 0x80) - 0x20 has no borrow
 * between bytes (each byte >= 0x80), and ^ 0x80 gives v - 32 mod 256. */
IE_FN(u32, ie_q6k_sw, (u32 nibword, u32 hword, u32 h, u32 k)) {
  return ((ie_q6k_vw(nibword, hword, h, k) | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
}
/* Half h by 4 signed dot4 of the q - 32 bytes (no activation sum needed). */
IE_FN(u32, ie_q6k_dots, (Mem qw, u32 qb, Mem hw, u32 hb, Mem aw, u32 ab, u32 h)) {
  return ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(0u, ie_q6k_sw(IE_LOAD(qw, qb), IE_LOAD(hw, hb), h, 0u), IE_LOAD(aw, ab + 4u * h)),
                                          ie_q6k_sw(IE_LOAD(qw, qb + 1u), IE_LOAD(hw, hb), h, 1u), IE_LOAD(aw, ab + 4u * h + 1u)),
                               ie_q6k_sw(IE_LOAD(qw, qb + 2u), IE_LOAD(hw, hb), h, 2u), IE_LOAD(aw, ab + 4u * h + 2u)),
                    ie_q6k_sw(IE_LOAD(qw, qb + 3u), IE_LOAD(hw, hb), h, 3u), IE_LOAD(aw, ab + 4u * h + 3u));
}
/* ie_q6k_dots with the q - 32 words already made (s: word k = ie_q6k_sw of
 * nibble word k and the high-bits word h), as the multi-token kernels do. */
IE_FN(u32, ie_q6k_dots_s, (Mem sw, u32 sb, Mem aw, u32 ab, u32 h)) {
  return ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(ie_dot4_ss(0u, IE_LOAD(sw, sb), IE_LOAD(aw, ab + 4u * h)), IE_LOAD(sw, sb + 1u),
                                          IE_LOAD(aw, ab + 4u * h + 1u)),
                               IE_LOAD(sw, sb + 2u), IE_LOAD(aw, ab + 4u * h + 2u)),
                    IE_LOAD(sw, sb + 3u), IE_LOAD(aw, ab + 4u * h + 3u));
}
/* The spec: the sum over the weights j = 16h .. 16h + 15 of (q_j - 32) a_j. */
IE_FN(u32, ie_q6k_term, (Mem qw, u32 qb, Mem hw, u32 hb, Mem aw, u32 ab, u32 j)) {
  return (ie_q6k_q(IE_LOAD(qw, qb + ie_q4_word_of(j)), IE_LOAD(hw, hb), j) - 32u) *
         ie_sext8(ie_byte(IE_LOAD(aw, ab + (j >> 2u)), j & 3u));
}
IE_FN(u32, ie_q6k_spec, (u32 n, Mem qw, u32 qb, Mem hw, u32 hb, Mem aw, u32 ab, u32 j, u32 hi)) {
  return n == 0u ? 0u : j < hi ? ie_q6k_term(qw, qb, hw, hb, aw, ab, j) + ie_q6k_spec(n - 1u, qw, qb, hw, hb, aw, ab, j + 1u, hi) : 0u;
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
