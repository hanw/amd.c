/* ie_ext.h -- integer helpers of the weight kinds MAT_Q5K and MAT_IQ4XS
 * (Flint: unsloth UD-Q4_K_XL). Shared by the CPU backend (src/cpu.c) and the
 * GPU kernels, like core/ie_core.h, but NOT part of the verified core: these
 * functions are tested (tests/kernel_test.c, ie_ext_test), not proved.
 *
 * MAT_Q5K (GGUF Q5_K, weights q = 0..31 in sub-blocks of 32):
 *   qw: the low 4 bits as MAT_Q4K (ie_q4_dst_word: byte j of a sub-block
 *       holds weights j and j + 16), then one word per sub-block at
 *       ie_q5k_hw: bit j = bit 4 of weight j; ie_q5k_qw_n words in all.
 *   qs: as MAT_Q4K (ie_q4k_sm, ie_q4k_dd). Weight = d sc q - dmin mn.
 * MAT_IQ4XS (GGUF IQ4_XS, weights kv[q], q = 0..15, kv = kvalues_iq4nl):
 *   qw: the 4-bit indices as MAT_Q4 (the GGUF order is already the Q4_0
 *       order); qs: ie_q4k_sm holds the 6-bit scale ls (0..63) of the
 *       sub-block, ie_q4k_dd the f16 d of the super-block (and 0).
 *       Weight = d (ls - 32) kv[q]. */
#ifndef IE_EXT_H
#define IE_EXT_H

#include "ie_core.h"

static inline u32 ie_q5k_hw(u32 rows, u32 r, u32 b, u32 nb) { return rows * nb * 4u + r * nb + b; }
static inline u32 ie_q5k_qw_n(u32 rows, u32 nb) { return rows * nb * 5u; }

/* The 4 bits of x (bits 0..3) as 4 bytes 0 or 1 (byte e = bit e): the
 * four shifted copies x, x << 7, x << 14, x << 21 do not overlap. */
static inline u32 ie_spread4(u32 x) { return ((x & 15u) * 0x00204081u) & 0x01010101u; }

/* The integer dot product of one MAT_Q5K sub-block (4 nibble words at qb,
 * high-bit word h) and one Q8 block (8 words at ab): sum over j < 32 of
 * (nibble j + 16 bit j of h) * a_j; 8 dot4, as ie_q4k_dot. */
static inline u32 ie_q5k_word(u32 acc, u32 wq, u32 h, u32 w, u32 alo, u32 ahi) {
  const u32 lo = (wq & 0x0F0F0F0Fu) | (ie_spread4(h >> (4u * w)) << 4u);
  const u32 hi = ((wq >> 4u) & 0x0F0F0F0Fu) | (ie_spread4(h >> (16u + 4u * w)) << 4u);
  return ie_dot4_us(ie_dot4_us(acc, lo, alo), hi, ahi);
}
static inline u32 ie_q5k_dot(Mem qw, u32 qb, u32 h, Mem aw, u32 ab) {
  u32 acc = 0u;
  for (u32 w = 0; w < 4u; w++) acc = ie_q5k_word(acc, IE_LOAD(qw, qb + w), h, w, IE_LOAD(aw, ab + w), IE_LOAD(aw, ab + 4u + w));
  return acc;
}

/* kvalues_iq4nl of ggml as 4 words of 4 int8 (entry e: byte e % 4 of word e / 4). */
#define IE_KV0 0xBFAD9881u /* -127, -104, -83, -65 */
#define IE_KV1 0xF6EADDCFu /* -49, -35, -22, -10 */
#define IE_KV2 0x26190D01u /* 1, 13, 25, 38 */
#define IE_KV3 0x71594535u /* 53, 69, 89, 113 */

/* The 4 indices in the low nibbles of the bytes of n4 (0x0N0N0N0N) mapped
 * to their 4 int8 table values. GPU: two byte permutes and a byte select. */
#if defined(__AMDGCN__)
static inline u32 ie_iq4_map(u32 n4) {
  /* v_perm_b32(a, b, sel): byte i = byte sel_i of the 8 bytes {a: 7..4, b: 3..0} */
  const u32 sel = n4 & 0x07070707u;
  const u32 lo = __builtin_amdgcn_perm(IE_KV1, IE_KV0, sel), hi = __builtin_amdgcn_perm(IE_KV3, IE_KV2, sel);
  const u32 m = ((n4 >> 3u) & 0x01010101u) * 255u;
  return (hi & m) | (lo & ~m);
}
#else
static inline u32 ie_kv_byte(u32 e) {
  const u32 t[4] = {IE_KV0, IE_KV1, IE_KV2, IE_KV3};
  return (t[(e & 15u) >> 2u] >> (8u * (e & 3u))) & 255u;
}
static inline u32 ie_iq4_map(u32 n4) {
  return ie_kv_byte(n4) | (ie_kv_byte(n4 >> 8u) << 8u) | (ie_kv_byte(n4 >> 16u) << 16u) | (ie_kv_byte(n4 >> 24u) << 24u);
}
#endif

/* The integer dot product of one MAT_IQ4XS sub-block (4 words at qb) and
 * one Q8 block: sum over j < 32 of kv[q_j] * a_j; 8 signed dot4. */
static inline u32 ie_iq4_word(u32 acc, u32 wq, u32 alo, u32 ahi) {
  return ie_dot4_ss(ie_dot4_ss(acc, ie_iq4_map(wq & 0x0F0F0F0Fu), alo), ie_iq4_map((wq >> 4u) & 0x0F0F0F0Fu), ahi);
}
static inline u32 ie_iq4_dot(Mem qw, u32 qb, Mem aw, u32 ab) {
  u32 acc = 0u;
  for (u32 w = 0; w < 4u; w++) acc = ie_iq4_word(acc, IE_LOAD(qw, qb + w), IE_LOAD(aw, ab + w), IE_LOAD(aw, ab + 4u + w));
  return acc;
}

#endif
