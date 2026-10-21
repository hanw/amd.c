/* model.c -- load Llama / Qwen2 from GGUF, repack Q4_0 into the verified GPU
 * layout, build the op list of one decode step. */
#include "model.h"

#include <math.h>
#include <string.h>

#include "../core/ie_core.h"
#include "../core/ie_ext.h"
#include "common.h"
#include "iq3s_grid.h"

/* ---------------------------------------------------------------- loading */

static const gguf_tensor *need_tensor(gguf_file *g, const char *name) {
  const gguf_tensor *t = gguf_tensor_find(g, name);
  if (!t) ie_die("missing tensor %s", name);
  if (!t->data) ie_die("tensor %s: unsupported type %u", name, t->type);
  return t;
}

/* The 6-bit scale and min of sub-block j (< 8) of a Q4_K / Q5_K block (as
 * ggml's get_scale_min_k4). */
static void q4k_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
  if (j < 4) {
    *sc = q[j] & 63, *m = q[j + 4] & 63;
  } else {
    *sc = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

/* One 256-weight block of IQ4_XS (136 bytes) or IQ3_S (110 bytes) to float
 * (as ggml's dequantize_row_iq4_xs / _iq3_s). */
static void dequant_iq_block(uint32_t type, const uint8_t *blk, float *y) {
  uint16_t h;
  memcpy(&h, blk, 2);
  const float d = ie_f16_to_f32(h);
  if (type == GGML_IQ4_XS) {
    uint16_t sh;
    memcpy(&sh, blk + 2, 2);
    const uint8_t *sl = blk + 4, *qs = blk + 8;
    for (int ib = 0; ib < 8; ib++) {
      const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
      const float dl = d * (float)(ls - 32);
      for (int j = 0; j < 16; j++) {
        y[32 * ib + j] = dl * (float)(int8_t)ie_kv_byte(qs[16 * ib + j] & 15u);
        y[32 * ib + j + 16] = dl * (float)(int8_t)ie_kv_byte(qs[16 * ib + j] >> 4);
      }
    }
    return;
  }
  /* IQ3_S: d, qs[64], qh[8], signs[32], scales[4] */
  const uint8_t *qs = blk + 2, *qh = blk + 66, *signs = blk + 74, *scales = blk + 106;
  for (int ib32 = 0; ib32 < 8; ib32++) {
    const float db = d * (float)(1 + 2 * ((scales[ib32 / 2] >> (4 * (ib32 % 2))) & 0xF));
    for (int l = 0; l < 4; l++) {
      const uint8_t *g1 = (const uint8_t *)(iq3s_grid + (qs[2 * l] | ((qh[ib32] << (8 - 2 * l)) & 256)));
      const uint8_t *g2 = (const uint8_t *)(iq3s_grid + (qs[2 * l + 1] | ((qh[ib32] << (7 - 2 * l)) & 256)));
      for (int j = 0; j < 4; j++) {
        y[j] = db * (float)g1[j] * ((signs[l] >> j) & 1 ? -1.0f : 1.0f);
        y[j + 4] = db * (float)g2[j] * ((signs[l] >> (j + 4)) & 1 ? -1.0f : 1.0f);
      }
      y += 8;
    }
    qs += 8, signs += 4;
  }
}

/* Dequantize n elements of tensor data (F32 / F16 / BF16 / Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K / IQ4_XS / IQ3_S) to float. */
static void dequant_raw(uint32_t type, const uint8_t *p, const char *name, uint64_t first, uint64_t n, float *out) {
  switch (type) {
    case GGML_F32: memcpy(out, p + first * 4, n * 4); break;
    case GGML_F16:
      for (uint64_t i = 0; i < n; i++) {
        uint16_t h;
        memcpy(&h, p + (first + i) * 2, 2);
        out[i] = ie_f16_to_f32(h);
      }
      break;
    case GGML_BF16: /* the high 16 bits of an f32: exact */
      for (uint64_t i = 0; i < n; i++) {
        uint16_t h;
        memcpy(&h, p + (first + i) * 2, 2);
        const uint32_t u = (uint32_t)h << 16;
        memcpy(&out[i], &u, 4);
      }
      break;
    case GGML_Q8_0:
      for (uint64_t i = 0; i < n; i++) {
        uint64_t e = first + i;
        const uint8_t *blk = p + e / 32 * 34;
        uint16_t h;
        memcpy(&h, blk, 2);
        out[i] = ie_f16_to_f32(h) * (float)(int8_t)blk[2 + e % 32];
      }
      break;
    case GGML_Q4_0:
      for (uint64_t i = 0; i < n; i++) {
        uint64_t e = first + i;
        const uint8_t *blk = p + e / 32 * 18;
        uint16_t h;
        memcpy(&h, blk, 2);
        uint32_t j = (uint32_t)(e % 32), q = j < 16 ? (blk[2 + j] & 15u) : (blk[2 + j - 16] >> 4);
        out[i] = ie_f16_to_f32(h) * (float)((int)q - 8);
      }
      break;
    case GGML_Q6_K: {
      /* blocks of 256: ql[128], qh[64], int8 scales[16], f16 d (as ggml's
       * dequantize_row_q6_K) */
      const uint64_t b0 = first / 256, b1 = (first + n + 255) / 256;
      float tmp[256];
      for (uint64_t bk = b0; bk < b1; bk++) {
        const uint8_t *blk = p + bk * 210, *ql = blk, *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        uint16_t h;
        memcpy(&h, blk + 208, 2);
        const float d = ie_f16_to_f32(h);
        for (int half = 0; half < 2; half++) {
          const uint8_t *L = ql + 64 * half, *Hh = qh + 32 * half;
          const int8_t *S = sc + 8 * half;
          float *y = tmp + 128 * half;
          for (int l = 0; l < 32; l++) {
            const int is = l / 16;
            const int q1 = (int)((L[l] & 0xF) | (((Hh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((L[l + 32] & 0xF) | (((Hh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((L[l] >> 4) | (((Hh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((L[l + 32] >> 4) | (((Hh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * S[is] * q1;
            y[l + 32] = d * S[is + 2] * q2;
            y[l + 64] = d * S[is + 4] * q3;
            y[l + 96] = d * S[is + 6] * q4;
          }
        }
        for (int j = 0; j < 256; j++) {
          const uint64_t e = bk * 256 + j;
          if (e >= first && e < first + n) out[e - first] = tmp[j];
        }
      }
      break;
    }
    case GGML_Q4_K:
    case GGML_Q5_K: {
      const int ttype = (int)type;
      /* blocks of 256 (as ggml's dequantize_row_q4_K / _q5_K): f16 d, f16
       * dmin, 12 bytes of 6-bit scales and mins, (Q5_K: qh[32]), qs[128] */
      const int five = ttype == GGML_Q5_K;
      const uint64_t bs = five ? 176 : 144, b0 = first / 256, b1 = (first + n + 255) / 256;
      float tmp[256];
      for (uint64_t bk = b0; bk < b1; bk++) {
        const uint8_t *blk = p + bk * bs, *sc = blk + 4, *qh = blk + 16, *q = blk + (five ? 48 : 16);
        uint16_t h0, h1;
        memcpy(&h0, blk, 2), memcpy(&h1, blk + 2, 2);
        const float d = ie_f16_to_f32(h0), dmin = ie_f16_to_f32(h1);
        for (int j = 0, is = 0; j < 256; j += 64, is += 2, q += 32) {
          uint8_t s0, m0, s1, m1;
          q4k_scale_min(is, sc, &s0, &m0), q4k_scale_min(is + 1, sc, &s1, &m1);
          const float d1 = d * s0, mm1 = dmin * m0, d2 = d * s1, mm2 = dmin * m1;
          for (int l = 0; l < 32; l++) {
            const int h1b = five ? ((qh[l] >> (j / 32)) & 1) << 4 : 0, h2b = five ? ((qh[l] >> (j / 32 + 1)) & 1) << 4 : 0;
            tmp[j + l] = d1 * (float)((q[l] & 0xF) + h1b) - mm1;
            tmp[j + 32 + l] = d2 * (float)((q[l] >> 4) + h2b) - mm2;
          }
        }
        for (int j = 0; j < 256; j++) {
          const uint64_t e = bk * 256 + j;
          if (e >= first && e < first + n) out[e - first] = tmp[j];
        }
      }
      break;
    }
    case GGML_IQ4_XS:
    case GGML_IQ3_S: {
      const uint64_t bs = type == GGML_IQ4_XS ? 136 : 110, b0 = first / 256, b1 = (first + n + 255) / 256;
      float tmp[256];
      for (uint64_t bk = b0; bk < b1; bk++) {
        dequant_iq_block(type, p + bk * bs, tmp);
        for (int j = 0; j < 256; j++) {
          const uint64_t e = bk * 256 + j;
          if (e >= first && e < first + n) out[e - first] = tmp[j];
        }
      }
      break;
    }
    default: ie_die("tensor %s: type %s is not supported here", name, ggml_type_name(type));
  }
}
static void dequant(const gguf_tensor *t, uint64_t first, uint64_t n, float *out) { dequant_raw(t->type, t->data, t->name, first, n, out); }

static void load_vec(vec *v, gguf_file *g, const char *name, uint32_t n, int optional) {
  const gguf_tensor *t = gguf_tensor_find(g, name);
  memset(v, 0, sizeof *v);
  if (!t && optional) return;
  t = need_tensor(g, name);
  if (t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] != n) ie_die("tensor %s: expected %u elements", name, n);
  v->n = n;
  v->f = ie_alloc((size_t)n * 4);
  dequant(t, 0, n, v->f);
}

/* Repack a GGUF Q4_0 matrix into the verified GPU layout. The index maps
 * are those of core/ie_core.h (ie_q4_src_*, ie_q4_dst_*, ie_pack4); the
 * laws q4_addr / q4_inverse / q4_onto / q4_nibble prove that this writes
 * every word once, reads inside the source, and keeps each nibble. */
void repack_q4(mat *w, const uint8_t *src) {
  uint32_t rows = w->rows, nb = w->nb;
  if (!ie_sizes_ok(rows, nb)) ie_die("Q4_0 matrix %ux%u: outside ie_sizes_ok (rows < 2^18, blocks < 512)", rows, w->cols);
  w->qw = ie_alloc((size_t)rows * nb * 16);
  w->qs = ie_alloc((size_t)rows * nb * 2);
  for (u32 r = 0; r < rows; r++)
    for (u32 b = 0; b < nb; b++) {
      u32 s = ie_q4_src_blk(r, b, nb);
      w->qs[ie_q4_dst_scale(r, b, nb)] = (uint16_t)(src[s] | (src[s + 1] << 8));
      for (u32 k = 0; k < 4; k++)
        w->qw[ie_q4_dst_word(r, b, k, nb)] =
            ie_pack4(src[ie_q4_src_qbyte(r, b, 4 * k + 0, nb)], src[ie_q4_src_qbyte(r, b, 4 * k + 1, nb)],
                     src[ie_q4_src_qbyte(r, b, 4 * k + 2, nb)], src[ie_q4_src_qbyte(r, b, 4 * k + 3, nb)]);
    }
}

/* Repack a GGUF Q8_0 matrix into the verified GPU layout (ie_q8_src_*,
 * ie_q8_dst_*, ie_pack4); the laws q8_addr / q8_inverse / q8_onto / q8_byte
 * prove that this writes every word once, reads inside the source, and keeps
 * each int8 weight. */
void repack_q8(mat *w, const uint8_t *src) {
  uint32_t rows = w->rows, nb = w->nb;
  if (!ie_q8_sizes_ok(rows, nb)) ie_die("Q8_0 matrix %ux%u: outside ie_q8_sizes_ok (rows < 2^18, blocks < 2048, rows * blocks < 2^26)", rows, w->cols);
  w->qw = ie_alloc((size_t)rows * nb * 32);
  w->qs = ie_alloc((size_t)rows * nb * 2);
  for (u32 r = 0; r < rows; r++)
    for (u32 b = 0; b < nb; b++) {
      u32 s = ie_q8_src_blk(r, b, nb);
      w->qs[ie_q8_dst_scale(r, b, nb)] = (uint16_t)(src[s] | (src[s + 1] << 8));
      for (u32 k = 0; k < 8; k++)
        w->qw[ie_q8_dst_word(r, b, k, nb)] =
            ie_pack4(src[ie_q8_src_qbyte(r, b, 4 * k + 0, nb)], src[ie_q8_src_qbyte(r, b, 4 * k + 1, nb)],
                     src[ie_q8_src_qbyte(r, b, 4 * k + 2, nb)], src[ie_q8_src_qbyte(r, b, 4 * k + 3, nb)]);
    }
}

/* Repack a GGUF Q4_K matrix into MAT_Q4K (model.h). Sub-block 2k of a
 * 64-weight group holds the low nibbles of its 32 bytes, sub-block 2k + 1
 * the high nibbles; each sub-block is stored as a Q4_0 block (byte j: low
 * nibble = weight j, high nibble = weight j + 16), so the word maps and the
 * nibble order are those of MAT_Q4. */
static void repack_q4k(mat *w, const uint8_t *src) {
  const uint32_t rows = w->rows, nb = w->nb, ns = nb / 8;
  w->qw = ie_alloc((size_t)rows * nb * 16);
  w->qs = ie_alloc((size_t)ie_q4k_qs_n(rows, nb) * 2);
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t sb = 0; sb < ns; sb++) {
      const uint8_t *blk = src + ie_q4k_src_blk(r, sb * 8u, nb); /* the super-block header: d, dmin, scales */
      memcpy(&w->qs[ie_q4k_dd(rows, r, sb * 8u, nb)], blk, 2);
      memcpy(&w->qs[ie_q4k_dd(rows, r, sb * 8u, nb) + 1u], blk + 2, 2);
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t b = sb * 8 + k;
        uint8_t sc, mn;
        q4k_scale_min((int)k, blk + 4, &sc, &mn);
        w->qs[ie_q4k_sm(r, b, nb)] = (uint16_t)(sc | (mn << 8));
        /* byte j of the Q4_0-order block: source bytes j and j + 16 (law q4k_nibble, q4k_addr) */
        uint32_t by[16];
        for (uint32_t j = 0; j < 16; j++)
          by[j] = ie_q4k_pack(src[ie_q4k_src_qbyte(r, b, j, nb)], src[ie_q4k_src_qbyte(r, b, j + 16u, nb)], b);
        for (uint32_t wd = 0; wd < 4; wd++) w->qw[ie_q4_dst_word(r, b, wd, nb)] = ie_pack4(by[4 * wd], by[4 * wd + 1], by[4 * wd + 2], by[4 * wd + 3]);
      }
    }
}

/* Repack a GGUF Q6_K matrix into MAT_Q6K (model.h; index maps and the
 * packing: core/ie_core.h, laws q6k_addr, q6k_unpack). */
static void repack_q6k(mat *w, const uint8_t *src) {
  const uint32_t rows = w->rows, nb = w->nb;
  w->qw = ie_alloc((size_t)ie_q6k_qw_n(rows, nb) * 4);
  w->qs = ie_alloc((size_t)ie_q6k_qs_n(rows, nb) * 2);
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t b = 0; b < nb; b++) {
      uint32_t q[32];
      for (uint32_t j = 0; j < 32; j++) q[j] = ie_q6k_src_q(src[ie_q6k_src_ql(r, b, j, nb)], src[ie_q6k_src_qh(r, b, j, nb)], b);
      for (uint32_t wd = 0; wd < 4; wd++) {
        uint32_t by[4];
        for (uint32_t i = 0; i < 4; i++) by[i] = (q[4 * wd + i] & 15u) | ((q[4 * wd + i + 16] & 15u) << 4);
        w->qw[ie_q4_dst_word(r, b, wd, nb)] = ie_pack4(by[0], by[1], by[2], by[3]);
      }
      for (uint32_t h = 0; h < 2; h++) {
        const uint32_t *p = q + 16 * h;
        w->qw[ie_q6k_hw(rows, r, b, h, nb)] =
            ie_q6k_hpack(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
      }
      w->qs[ie_q6k_sc(r, b, nb)] = (uint16_t)(src[ie_q6k_src_sc(r, b, 0, nb)] | (src[ie_q6k_src_sc(r, b, 1, nb)] << 8));
      if (b % 8 == 0) memcpy(&w->qs[ie_q6k_d(rows, r, b, nb)], src + ie_q6k_src_d(r, b, nb), 2);
    }
}

/* Repack a GGUF Q5_K matrix into MAT_Q5K (core/ie_ext.h): the low nibbles
 * and the scales as MAT_Q4K, bit 4 of each weight in one word per sub-block. */
static void repack_q5k(mat *w, const uint8_t *src) {
  const uint32_t rows = w->rows, nb = w->nb, ns = nb / 8;
  w->qw = ie_alloc((size_t)ie_q5k_qw_n(rows, nb) * 4);
  w->qs = ie_alloc((size_t)ie_q4k_qs_n(rows, nb) * 2);
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t sb = 0; sb < ns; sb++) {
      const uint8_t *blk = src + ((size_t)r * ns + sb) * 176, *qh = blk + 16;
      memcpy(&w->qs[ie_q4k_dd(rows, r, sb * 8u, nb)], blk, 2);
      memcpy(&w->qs[ie_q4k_dd(rows, r, sb * 8u, nb) + 1u], blk + 2, 2);
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t b = sb * 8 + k;
        const uint8_t *q = blk + 48 + 32 * (k / 2);
        uint8_t sc, mn;
        q4k_scale_min((int)k, blk + 4, &sc, &mn);
        w->qs[ie_q4k_sm(r, b, nb)] = (uint16_t)(sc | (mn << 8));
        uint32_t v[32], hw = 0;
        for (uint32_t l = 0; l < 32; l++) {
          v[l] = k % 2 ? (uint32_t)(q[l] >> 4) : q[l] & 15u;
          hw |= (uint32_t)((qh[l] >> k) & 1u) << l;
        }
        for (uint32_t wd = 0; wd < 4; wd++)
          w->qw[ie_q4_dst_word(r, b, wd, nb)] = ie_pack4(v[4 * wd] | v[4 * wd + 16] << 4, v[4 * wd + 1] | v[4 * wd + 17] << 4,
                                                         v[4 * wd + 2] | v[4 * wd + 18] << 4, v[4 * wd + 3] | v[4 * wd + 19] << 4);
        w->qw[ie_q5k_hw(rows, r, b, nb)] = hw;
      }
    }
}

/* Repack a GGUF IQ4_XS matrix into MAT_IQ4XS (core/ie_ext.h). */
static void repack_iq4xs(mat *w, const uint8_t *src) {
  const uint32_t rows = w->rows, nb = w->nb, ns = nb / 8;
  w->qw = ie_alloc((size_t)rows * nb * 16);
  w->qs = ie_alloc((size_t)ie_q4k_qs_n(rows, nb) * 2);
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t sb = 0; sb < ns; sb++) {
      const uint8_t *blk = src + ((size_t)r * ns + sb) * 136, *sl = blk + 4;
      uint16_t sh;
      memcpy(&sh, blk + 2, 2);
      memcpy(&w->qs[ie_q4k_dd(rows, r, sb * 8u, nb)], blk, 2);
      w->qs[ie_q4k_dd(rows, r, sb * 8u, nb) + 1u] = 0;
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t b = sb * 8 + k;
        w->qs[ie_q4k_sm(r, b, nb)] = (uint16_t)(((sl[k / 2] >> (4 * (k % 2))) & 0xF) | (((sh >> (2 * k)) & 3) << 4));
        const uint8_t *q = blk + 8 + 16 * k;
        for (uint32_t wd = 0; wd < 4; wd++) w->qw[ie_q4_dst_word(r, b, wd, nb)] = ie_pack4(q[4 * wd], q[4 * wd + 1], q[4 * wd + 2], q[4 * wd + 3]);
      }
    }
}

uint64_t mat_qw_words(int kind, uint32_t rows, uint32_t cols) {
  const uint64_t nb = cols / 32;
  switch (kind) {
    case MAT_Q4: case MAT_Q4K: case MAT_IQ4XS: return rows * nb * 4;
    case MAT_Q8: return rows * nb * 8;
    case MAT_Q5K: return rows * nb * 5;
    case MAT_Q6K: return rows * nb * 6;
    default: return 0;
  }
}
uint64_t mat_qs_n(int kind, uint32_t rows, uint32_t cols) {
  const uint64_t nb = cols / 32;
  switch (kind) {
    case MAT_Q4: case MAT_Q8: return rows * nb;
    case MAT_Q4K: case MAT_Q5K: case MAT_IQ4XS: return rows * nb + rows * (nb / 8) * 2;
    case MAT_Q6K: return rows * nb + rows * (nb / 8);
    default: return 0;
  }
}
mat mat_expert(const mat *w, uint32_t e) {
  mat v = *w;
  v.n_exp = 1;
  if (w->kind == MAT_F32) {
    v.f = w->f + (size_t)e * w->rows * w->cols;
  } else {
    v.qw = w->qw + (size_t)e * mat_qw_words(w->kind, w->rows, w->cols);
    v.qs = w->qs + (size_t)e * mat_qs_n(w->kind, w->rows, w->cols);
  }
  return v;
}

/* GGUF Q8_0 bytes of n floats (n % 32 == 0), as ggml's quantize_row_q8_0:
 * d = amax / 127 (stored as f16), q = round(x / d). */
static void quant_q8_0_bytes(const float *x, uint64_t n, uint8_t *out) {
  for (uint64_t b = 0; b < n / 32; b++) {
    float amax = 0.0f;
    for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[32 * b + j]));
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    const uint16_t h = ie_f32_to_f16(d);
    memcpy(out + 34 * b, &h, 2);
    for (int j = 0; j < 32; j++) out[34 * b + 2 + j] = (uint8_t)(int8_t)roundf(x[32 * b + j] * id);
  }
}

/* qwen35: Q4_0 matrices of the main layers are loaded as Q8_0 (the Q8_0
 * kernels have the multi-token paths). The MTP layer keeps Q4_0 (MAT_Q4,
 * half the bytes): it runs one token at a time (drafts, and the last token
 * of a verify), and its results only choose the drafts, not the output. */
static int q4_as_q8;
/* Q5_K as MAT_Q5K (1), or requantized to Q8_0 at load (0: the models before
 * flint keep their measured path; IE_Q5K_NATIVE=0/1 overrides) */
static int native_q5k;

/* Make matrix w of rows x cols from the GGUF bytes src of type ttype. A
 * Q8_0 matrix keeps its integer form (MAT_Q8) when q8_ok, else it becomes
 * f32; emb: the token embedding (no K or IQ kinds: OP_EMBED reads Q8, Q4 or f32). */
static void load_mat_src(mat *w, uint32_t ttype, const uint8_t *data, const char *name, uint32_t rows, uint32_t cols,
                         uint64_t *bytes, int q8_ok, int emb) {
  memset(w, 0, sizeof *w);
  w->rows = rows;
  w->cols = cols;
  const int kq = cols % 256 == 0 && ie_q4k_sizes_ok(rows, cols / 32) && !emb;
  if (ttype == GGML_Q4_0 && ie_sizes_ok(rows, cols / 32) && !q4_as_q8) {
    w->kind = MAT_Q4;
    w->nb = cols / 32;
    repack_q4(w, data);
    *bytes += (uint64_t)rows * w->nb * 18;
  } else if (ttype == GGML_Q4_0 && q8_ok && ie_q8_sizes_ok(rows, cols / 32)) {
    /* Q4_0 past the Q4 size limit (K > 16352), or q4_as_q8 (qwen35: the
     * Q8_0 kernels have the multi-token paths): the same weights as Q8_0,
     * exactly (int8 q - 8, the same scale) */
    w->kind = MAT_Q8;
    w->nb = cols / 32;
    const uint8_t *src = data;
    uint8_t *q8 = ie_alloc((size_t)rows * w->nb * 34);
    for (size_t k = 0; k < (size_t)rows * w->nb; k++) {
      memcpy(q8 + 34 * k, src + 18 * k, 2);
      for (int j = 0; j < 16; j++) {
        q8[34 * k + 2 + j] = (uint8_t)(int8_t)((src[18 * k + 2 + j] & 15) - 8);
        q8[34 * k + 18 + j] = (uint8_t)(int8_t)((src[18 * k + 2 + j] >> 4) - 8);
      }
    }
    repack_q8(w, q8);
    free(q8);
    *bytes += (uint64_t)rows * w->nb * 34;
  } else if (ttype == GGML_Q4_K && kq) {
    w->kind = MAT_Q4K;
    w->nb = cols / 32;
    repack_q4k(w, data);
    *bytes += (uint64_t)rows * w->nb * 18;
  } else if (ttype == GGML_Q5_K && kq && native_q5k) {
    w->kind = MAT_Q5K;
    w->nb = cols / 32;
    repack_q5k(w, data);
    *bytes += (uint64_t)rows * w->nb * 22;
  } else if (ttype == GGML_IQ4_XS && kq) {
    w->kind = MAT_IQ4XS;
    w->nb = cols / 32;
    repack_iq4xs(w, data);
    *bytes += (uint64_t)rows * w->nb * 17;
  } else if (ttype == GGML_Q6_K && kq && !getenv("IE_Q6K_AS_Q8")) {
    w->kind = MAT_Q6K;
    w->nb = cols / 32;
    repack_q6k(w, data);
    *bytes += (uint64_t)rows * w->nb * 210 / 8;
  } else if ((ttype == GGML_Q6_K || ttype == GGML_Q5_K || ttype == GGML_Q4_K || ttype == GGML_IQ3_S || ttype == GGML_IQ4_XS) && q8_ok &&
             ie_q8_sizes_ok(rows, cols / 32)) {
    /* Q6_K, Q5_K (not native), IQ3_S (and a K or IQ embedding): no integer
     * kernel; requantize to Q8_0 at load (an approximation: Q8_0 has one
     * scale per 32 weights, Q6_K one per 16) */
    w->kind = MAT_Q8;
    w->nb = cols / 32;
    float *row = ie_alloc((size_t)cols * 4);
    uint8_t *q8 = ie_alloc((size_t)rows * w->nb * 34);
    for (uint32_t r = 0; r < rows; r++) {
      dequant_raw(ttype, data, name, (uint64_t)r * cols, cols, row);
      quant_q8_0_bytes(row, cols, q8 + (size_t)r * w->nb * 34);
    }
    repack_q8(w, q8);
    free(row), free(q8);
    *bytes += (uint64_t)rows * w->nb * 34;
  } else if (ttype == GGML_Q8_0 && q8_ok && ie_q8_sizes_ok(rows, cols / 32)) {
    w->kind = MAT_Q8;
    w->nb = cols / 32;
    repack_q8(w, data);
    *bytes += (uint64_t)rows * w->nb * 34;
  } else {
    w->kind = MAT_F32;
    w->f = ie_alloc((size_t)rows * cols * 4);
    dequant_raw(ttype, data, name, 0, (uint64_t)rows * cols, w->f);
    *bytes += (uint64_t)rows * cols * 4;
  }
}

void model_mat_from_gguf(mat *w, uint32_t ttype, const uint8_t *src, uint32_t rows, uint32_t cols) {
  uint64_t b = 0;
  const int n5 = native_q5k;
  native_q5k = 1;
  load_mat_src(w, ttype, src, "test", rows, cols, &b, 1, 0);
  native_q5k = n5;
}

/* Load matrix `name` of rows x cols (GGUF ne = [cols, rows]). */
static void load_mat(mat *w, gguf_file *g, const char *name, uint32_t rows, uint32_t cols, uint64_t *bytes, int q8_ok) {
  const gguf_tensor *t = need_tensor(g, name);
  if (t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != 1 || t->ne[3] != 1)
    ie_die("tensor %s: shape [%llu, %llu], expected [%u, %u]", name, (unsigned long long)t->ne[0],
           (unsigned long long)t->ne[1], cols, rows);
  load_mat_src(w, t->type, t->data, name, rows, cols, bytes, q8_ok, strcmp(name, "token_embd.weight") == 0);
}

/* Load the expert tensors names[0 .. n) (GGUF ne = [cols, rows[i], n_exp])
 * as one expert matrix: expert e = [part 0 of e; part 1 of e; ...] (rows:
 * the sum). Returns 0 (and loads nothing) if the parts' types differ.
 * bytes: += the bytes of one expert. */
static int load_experts(mat *w, gguf_file *g, const char *const *names, const uint32_t *rows, int n, uint32_t n_exp, uint32_t cols,
                        uint64_t *bytes) {
  const gguf_tensor *t[4];
  uint32_t R = 0;
  for (int i = 0; i < n; i++) {
    t[i] = need_tensor(g, names[i]);
    if (t[i]->ne[0] != cols || t[i]->ne[1] != rows[i] || t[i]->ne[2] != n_exp || t[i]->ne[3] != 1)
      ie_die("tensor %s: shape [%llu, %llu, %llu], expected [%u, %u, %u]", names[i], (unsigned long long)t[i]->ne[0],
             (unsigned long long)t[i]->ne[1], (unsigned long long)t[i]->ne[2], cols, rows[i], n_exp);
    if (t[i]->type != t[0]->type) return 0;
    R += rows[i];
  }
  const uint64_t rb = ggml_nbytes(t[0]->type, cols);
  uint8_t *src = n > 1 ? ie_alloc((size_t)R * rb) : NULL;
  memset(w, 0, sizeof *w);
  uint64_t qwn = 0, qsn = 0;
  for (uint32_t e = 0; e < n_exp; e++) {
    const uint8_t *se = (const uint8_t *)t[0]->data + (size_t)e * rows[0] * rb;
    if (n > 1) {
      size_t o = 0;
      for (int i = 0; i < n; i++) {
        memcpy(src + o, (const uint8_t *)t[i]->data + (size_t)e * rows[i] * rb, (size_t)rows[i] * rb);
        o += (size_t)rows[i] * rb;
      }
      se = src;
    }
    mat x;
    uint64_t xb = 0;
    load_mat_src(&x, t[0]->type, se, names[0], R, cols, &xb, 1, 0);
    if (e == 0) {
      *w = x;
      w->n_exp = n_exp;
      *bytes += xb;
      if (x.kind == MAT_F32) {
        w->f = ie_alloc((size_t)n_exp * R * cols * 4);
      } else {
        qwn = mat_qw_words(x.kind, R, cols), qsn = mat_qs_n(x.kind, R, cols);
        w->qw = ie_alloc((size_t)n_exp * qwn * 4);
        w->qs = ie_alloc((size_t)n_exp * qsn * 2);
      }
    }
    if (x.kind != w->kind) ie_die("tensor %s: experts of different kinds", names[0]);
    if (x.kind == MAT_F32) {
      memcpy(w->f + (size_t)e * R * cols, x.f, (size_t)R * cols * 4);
    } else {
      memcpy(w->qw + (size_t)e * qwn, x.qw, qwn * 4);
      memcpy(w->qs + (size_t)e * qsn, x.qs, qsn * 2);
    }
    free(x.qw), free(x.qs), free(x.f);
  }
  free(src);
  return 1;
}

/* Concatenate the rows of n matrices of one kind and width into dst. The
 * GPU layouts are row major (ie_q4_dst_word / ie_q8_dst_word are
 * (r * nb + b) * W + w), so row r of part i becomes row off_i + r and the
 * index maps stay those of the verified core. The parts are freed. Returns
 * 0 (and changes nothing) if the kinds or widths differ or the sizes are
 * outside the verified limits. */
static int concat_mats(mat *dst, mat *const *src, int n) {
  uint32_t rows = 0;
  for (int i = 0; i < n; i++) {
    if (src[i]->kind != src[0]->kind || src[i]->cols != src[0]->cols) return 0;
    rows += src[i]->rows;
  }
  const int kind = src[0]->kind;
  const uint32_t cols = src[0]->cols, nb = src[0]->nb;
  if (kind == MAT_Q4 && !ie_sizes_ok(rows, nb)) return 0;
  if (kind == MAT_Q6K) { /* two sections in qw and in qs: concatenate each */
    if (!ie_q4k_sizes_ok(rows, nb)) return 0;
    memset(dst, 0, sizeof *dst);
    dst->kind = kind, dst->rows = rows, dst->cols = src[0]->cols, dst->nb = nb;
    dst->qw = ie_alloc((size_t)ie_q6k_qw_n(rows, nb) * 4);
    dst->qs = ie_alloc((size_t)ie_q6k_qs_n(rows, nb) * 2);
    size_t r0 = 0;
    for (int i = 0; i < n; i++) {
      mat *w = src[i];
      memcpy(dst->qw + ie_q4_dst_word((uint32_t)r0, 0, 0, nb), w->qw, (size_t)w->rows * nb * 16);
      memcpy(dst->qw + ie_q6k_hw(rows, (uint32_t)r0, 0, 0, nb), w->qw + ie_q6k_hw(w->rows, 0, 0, 0, nb), (size_t)w->rows * nb * 8);
      memcpy(dst->qs + ie_q6k_sc((uint32_t)r0, 0, nb), w->qs, (size_t)w->rows * nb * 2);
      memcpy(dst->qs + ie_q6k_d(rows, (uint32_t)r0, 0, nb), w->qs + ie_q6k_d(w->rows, 0, 0, nb), (size_t)w->rows * (nb / 8) * 2);
      r0 += w->rows;
      free(w->qw), free(w->qs);
      w->qw = NULL, w->qs = NULL;
    }
    return 1;
  }
  if (kind == MAT_Q4K || kind == MAT_IQ4XS || kind == MAT_Q5K) { /* two sections in qs (Q5K: also in qw): concatenate each */
    if (!ie_q4k_sizes_ok(rows, nb)) return 0;
    memset(dst, 0, sizeof *dst);
    dst->kind = kind, dst->rows = rows, dst->cols = src[0]->cols, dst->nb = nb;
    dst->qw = ie_alloc((size_t)mat_qw_words(kind, rows, src[0]->cols) * 4);
    dst->qs = ie_alloc((size_t)ie_q4k_qs_n(rows, nb) * 2);
    size_t r0 = 0;
    for (int i = 0; i < n; i++) {
      mat *w = src[i];
      memcpy(dst->qw + r0 * nb * 4, w->qw, (size_t)w->rows * nb * 16);
      if (kind == MAT_Q5K) memcpy(dst->qw + ie_q5k_hw(rows, (uint32_t)r0, 0, nb), w->qw + ie_q5k_hw(w->rows, 0, 0, nb), (size_t)w->rows * nb * 4);
      memcpy(dst->qs + ie_q4k_sm((uint32_t)r0, 0, nb), w->qs, (size_t)w->rows * nb * 2);
      memcpy(dst->qs + ie_q4k_dd(rows, (uint32_t)r0, 0, nb), w->qs + ie_q4k_dd(w->rows, 0, 0, nb), (size_t)w->rows * (nb / 8) * 4);
      r0 += w->rows;
      free(w->qw), free(w->qs);
      w->qw = NULL, w->qs = NULL;
    }
    return 1;
  }
  if (kind != MAT_Q4 && kind != MAT_Q8 && kind != MAT_F32) return 0; /* a layout this function does not know */
  if (kind == MAT_Q8 && !ie_q8_sizes_ok(rows, nb)) return 0;
  memset(dst, 0, sizeof *dst);
  dst->kind = kind, dst->rows = rows, dst->cols = cols, dst->nb = nb;
  const size_t words = kind == MAT_Q4 ? 4 : 8;
  if (kind == MAT_F32) {
    dst->f = ie_alloc((size_t)rows * cols * 4);
  } else {
    dst->qw = ie_alloc((size_t)rows * nb * words * 4);
    dst->qs = ie_alloc((size_t)rows * nb * 2);
  }
  size_t r0 = 0;
  for (int i = 0; i < n; i++) {
    mat *w = src[i];
    if (kind == MAT_F32) {
      memcpy(dst->f + r0 * cols, w->f, (size_t)w->rows * cols * 4);
    } else {
      memcpy(dst->qw + r0 * nb * words, w->qw, (size_t)w->rows * nb * words * 4);
      memcpy(dst->qs + r0 * nb, w->qs, (size_t)w->rows * nb * 2);
    }
    r0 += w->rows;
    free(w->qw), free(w->qs), free(w->f);
    w->qw = NULL, w->qs = NULL, w->f = NULL;
  }
  return 1;
}

/* The bias of [q; k; v]: the parts, or zeros for a missing part. n = 0 if
 * no part has a bias. */
static void concat_bias(vec *dst, const vec *const *src, const uint32_t *len, int n) {
  int any = 0;
  uint32_t tot = 0;
  for (int i = 0; i < n; i++) any |= src[i]->n != 0, tot += len[i];
  memset(dst, 0, sizeof *dst);
  if (!any) return;
  dst->n = tot;
  dst->f = calloc(tot, 4);
  if (!dst->f) ie_die("out of memory");
  for (uint32_t i = 0, o = 0; i < (uint32_t)n; o += len[i], i++)
    if (src[i]->n) memcpy(dst->f + o, src[i]->f, len[i] * 4);
}

void (*ie_mat_sink)(mat *w);

/* Give the finished matrices of a layer to the sink (merged parts have no
 * host arrays left and are skipped by the sink). */
static void sink_layer(layer *L) {
  if (!ie_mat_sink) return;
  mat *all[] = {&L->wq, &L->wk, &L->wv, &L->wqkv, &L->wo, &L->wgate, &L->wup, &L->wdown, &L->wgu,
                &L->wlqkv, &L->wz, &L->wbeta, &L->walpha, &L->win, &L->wr, &L->egu, &L->eg, &L->eu, &L->ed};
  for (unsigned i = 0; i < sizeof all / sizeof all[0]; i++) ie_mat_sink(all[i]);
}

static uint32_t opt_u32(gguf_file *g, const char *arch, const char *key, uint32_t def) {
  char k[256];
  int64_t v;
  snprintf(k, sizeof k, "%s.%s", arch, key);
  if (!gguf_get_int(g, k, &v)) return def;
  if (v < 0 || v > 0x7FFFFFFF) ie_die("bad metadata %s", k);
  return (uint32_t)v;
}

static uint32_t need_u32(gguf_file *g, const char *arch, const char *key) {
  char k[256];
  int64_t v;
  snprintf(k, sizeof k, "%s.%s", arch, key);
  if (!gguf_get_int(g, k, &v) || v <= 0 || v > 0x7FFFFFFF) ie_die("missing or bad metadata %s", k);
  return (uint32_t)v;
}

/* A flint layer: attention (QK norm, sinks; window win, RoPE unless
 * norope), then a dense FFN or (moe) the router, the experts and the shared
 * expert. */
static void flint_layer(model *m, layer *L, gguf_file *g, uint32_t i, uint32_t win, int norope, int moe) {
  const uint32_t qd = m->n_head * m->hd, kvd = m->n_kv * m->hd;
  char n[128], n2[128];
#define NAME(s) (snprintf(n, sizeof n, "blk.%u." s, i), n)
  L->swin = win, L->norope = norope, L->moe = moe;
  L->kvi = m->n_kvl++;
  load_vec(&L->attn_norm, g, NAME("attn_norm.weight"), m->dim, 0);
  load_vec(&L->ffn_norm, g, NAME("ffn_norm.weight"), m->dim, 0);
  load_vec(&L->q_norm, g, NAME("attn_q_norm.weight"), m->hd, 0);
  load_vec(&L->k_norm, g, NAME("attn_k_norm.weight"), m->hd, 0);
  load_vec(&L->sinks, g, NAME("attn_sinks.weight"), m->n_head, 1);
  load_mat(&L->wq, g, NAME("attn_q.weight"), qd, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wk, g, NAME("attn_k.weight"), kvd, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wv, g, NAME("attn_v.weight"), kvd, m->dim, &m->weight_bytes, 1);
  load_mat(&L->wo, g, NAME("attn_output.weight"), m->dim, qd, &m->weight_bytes, 1);
  mat *qkv[3] = {&L->wq, &L->wk, &L->wv};
  concat_mats(&L->wqkv, qkv, 3);
  if (!moe) {
    load_mat(&L->wgate, g, NAME("ffn_gate.weight"), m->ffn, m->dim, &m->weight_bytes, 1);
    load_mat(&L->wup, g, NAME("ffn_up.weight"), m->ffn, m->dim, &m->weight_bytes, 1);
    load_mat(&L->wdown, g, NAME("ffn_down.weight"), m->dim, m->ffn, &m->weight_bytes, 1);
  } else {
    load_mat(&L->wr, g, NAME("ffn_gate_inp.weight"), m->n_exp, m->dim, &m->weight_bytes, 0);
    load_vec(&L->ebias, g, NAME("exp_probs_b.bias"), m->n_exp, 1);
    snprintf(n2, sizeof n2, "blk.%u.ffn_up_exps.weight", i);
    const char *gu[2] = {NAME("ffn_gate_exps.weight"), n2};
    const uint32_t r2[2] = {m->eff, m->eff};
    uint64_t eb = 0; /* the bytes of one expert: n_used of them per token */
    if (!load_experts(&L->egu, g, gu, r2, 2, m->n_exp, m->dim, &eb)) {
      load_experts(&L->eg, g, gu, r2, 1, m->n_exp, m->dim, &eb);
      load_experts(&L->eu, g, gu + 1, r2, 1, m->n_exp, m->dim, &eb);
    }
    const char *dn[1] = {NAME("ffn_down_exps.weight")};
    const uint32_t rd[1] = {m->dim};
    load_experts(&L->ed, g, dn, rd, 1, m->n_exp, m->eff, &eb);
    m->weight_bytes += eb * m->n_used;
    if (m->sff) {
      load_mat(&L->wgate, g, NAME("ffn_gate_shexp.weight"), m->sff, m->dim, &m->weight_bytes, 1);
      load_mat(&L->wup, g, NAME("ffn_up_shexp.weight"), m->sff, m->dim, &m->weight_bytes, 1);
      load_mat(&L->wdown, g, NAME("ffn_down_shexp.weight"), m->dim, m->sff, &m->weight_bytes, 1);
    }
  }
#undef NAME
  if (L->wgate.rows) {
    mat *gu2[2] = {&L->wgate, &L->wup};
    concat_mats(&L->wgu, gu2, 2);
  }
  sink_layer(L);
}

void model_load(model *m, gguf_file *g) {
  memset(m, 0, sizeof *m);
  char *arch = gguf_get_str(g, "general.architecture");
  if (!arch) ie_die("missing general.architecture");
  if (strcmp(arch, "llama") == 0) m->arch = ARCH_LLAMA, m->rope = ROPE_NORM;
  else if (strcmp(arch, "qwen2") == 0) m->arch = ARCH_QWEN2, m->rope = ROPE_NEOX;
  else if (strcmp(arch, "qwen35") == 0) m->arch = ARCH_QWEN35, m->rope = ROPE_NEOX;
  else if (strcmp(arch, "flint") == 0) m->arch = ARCH_FLINT, m->rope = ROPE_NEOX;
  else ie_die("architecture %s is not supported (llama, qwen2, qwen35, flint)", arch);
  const int q35 = m->arch == ARCH_QWEN35, fl = m->arch == ARCH_FLINT;
  q4_as_q8 = q35;
  native_q5k = fl;
  if (getenv("IE_Q5K_NATIVE")) native_q5k = getenv("IE_Q5K_NATIVE")[0] == '1';

  m->n_layer = need_u32(g, arch, "block_count");
  /* qwen35: the last nextn_predict_layers blocks are multi-token prediction
   * layers, not used for decoding one token at a time */
  if (q35) m->n_layer -= opt_u32(g, arch, "nextn_predict_layers", 0);
  m->dim = need_u32(g, arch, "embedding_length");
  m->ffn = need_u32(g, arch, "feed_forward_length");
  m->n_head = need_u32(g, arch, "attention.head_count");
  m->n_kv = need_u32(g, arch, "attention.head_count_kv");
  m->ctx_train = need_u32(g, arch, "context_length");
  char k[256];
  double d;
  snprintf(k, sizeof k, "%s.attention.layer_norm_rms_epsilon", arch);
  m->eps = gguf_get_float(g, k, &d) ? (float)d : 1e-5f;
  snprintf(k, sizeof k, "%s.rope.freq_base", arch);
  m->rope_base = gguf_get_float(g, k, &d) ? (float)d : 10000.0f;
  if (m->n_head % m->n_kv) ie_die("bad head counts");
  uint32_t interval = 1;
  if (q35) {
    m->hd = need_u32(g, arch, "attention.key_length");
    if (opt_u32(g, arch, "attention.value_length", m->hd) != m->hd) ie_die("key and value head dims differ");
    m->n_rot = opt_u32(g, arch, "rope.dimension_count", m->hd);
    m->d_conv = need_u32(g, arch, "ssm.conv_kernel");
    m->sd = need_u32(g, arch, "ssm.state_size");
    m->n_kh = need_u32(g, arch, "ssm.group_count");
    m->n_vh = need_u32(g, arch, "ssm.time_step_rank");
    interval = need_u32(g, arch, "full_attention_interval");
    if (need_u32(g, arch, "ssm.inner_size") != m->n_vh * m->sd) ie_die("ssm.inner_size != time_step_rank * state_size");
    /* the kernels (ie_gdn) are written for these sizes */
    if (m->d_conv != 4 || m->sd != 128 || m->n_vh % m->n_kh) ie_die("linear attention: unsupported sizes");
    m->conv_dim = 2 * m->n_kh * m->sd + m->n_vh * m->sd;
  } else if (fl) {
    m->hd = need_u32(g, arch, "attention.key_length");
    if (opt_u32(g, arch, "attention.value_length", m->hd) != m->hd) ie_die("key and value head dims differ");
    m->n_rot = m->hd;
    m->n_exp = need_u32(g, arch, "expert_count");
    m->n_used = need_u32(g, arch, "expert_used_count");
    m->eff = need_u32(g, arch, "expert_feed_forward_length");
    m->sff = opt_u32(g, arch, "expert_shared_feed_forward_length", 0);
    snprintf(k, sizeof k, "%s.expert_weights_scale", arch);
    m->escale = gguf_get_float(g, k, &d) ? (float)d : 1.0f;
    if (opt_u32(g, arch, "expert_gating_func", 2) != 2) ie_die("flint: only sigmoid routing (expert_gating_func 2)");
    snprintf(k, sizeof k, "%s.expert_weights_norm", arch);
    int64_t wn = 1;
    if (gguf_get_int(g, k, &wn) && !wn) ie_die("flint: unnormalized routing weights are not supported");
    if (m->n_used == 0 || m->n_used > 16 || m->n_used > m->n_exp || m->eff % 256 || m->sff % 32) ie_die("flint: unsupported MoE sizes");
  } else {
    if (m->dim % m->n_head) ie_die("bad head counts");
    m->hd = m->dim / m->n_head;
    m->n_rot = m->hd;
  }
  if (m->hd % 2 || m->n_rot % 2 || m->n_rot > m->hd) ie_die("bad head dim");
  if (m->dim % 32 || m->ffn % 32) ie_die("dims must be multiples of 32");

  const gguf_tensor *te = need_tensor(g, "token_embd.weight");
  m->vocab = (uint32_t)te->ne[1];
  uint64_t wb = 0;
  /* qwen35, flint: a Q8_0 embedding stays Q8_0 (1.3 GB for the 27B model; as f32
   * it would be 5 GB) */
  load_mat(&m->tok, g, "token_embd.weight", m->vocab, m->dim, &wb, q35 || fl);
  if (ie_mat_sink) ie_mat_sink(&m->tok);
  if (gguf_tensor_find(g, "output.weight")) {
    load_mat(&m->out, g, "output.weight", m->vocab, m->dim, &m->weight_bytes, 1);
    if (ie_mat_sink) ie_mat_sink(&m->out);
  } else {
    m->out = m->tok;
    m->tied = 1;
    m->weight_bytes += wb; /* the output GEMV reads the whole matrix */
  }
  load_vec(&m->out_norm, g, "output_norm.weight", m->dim, 0);

  uint32_t qd = m->n_head * m->hd, kvd = m->n_kv * m->hd;
  m->l = calloc(m->n_layer + 1, sizeof(layer)); /* + the MTP layer */
  if (!m->l) ie_die("out of memory");
  /* flint: per layer sliding window and RoPE (bool arrays), dense layers first */
  const gguf_kv *swp = NULL, *rpp = NULL;
  uint32_t sw = 0, n_dense = 0;
  if (fl) {
    snprintf(k, sizeof k, "%s.attention.sliding_window_pattern", arch);
    swp = gguf_find(g, k);
    snprintf(k, sizeof k, "%s.attention.rope_pattern", arch);
    rpp = gguf_find(g, k);
    if (!swp || !rpp || swp->type != GGUF_ARR || rpp->type != GGUF_ARR || swp->arr_type != GGUF_BOOL || rpp->arr_type != GGUF_BOOL ||
        swp->arr_n != m->n_layer || rpp->arr_n != m->n_layer)
      ie_die("flint: missing or bad attention.sliding_window_pattern / rope_pattern");
    sw = need_u32(g, arch, "attention.sliding_window");
    n_dense = opt_u32(g, arch, "leading_dense_block_count", 0);
  }
  for (uint32_t i = 0; i < m->n_layer; i++) {
    layer *L = &m->l[i];
    char n[128];
#define NAME(s) (snprintf(n, sizeof n, "blk.%u." s, i), n)
    if (fl) {
      flint_layer(m, L, g, i, swp->arr_data[i] ? sw : 0, !rpp->arr_data[i], i >= n_dense);
      continue;
    }
    load_vec(&L->attn_norm, g, NAME("attn_norm.weight"), m->dim, 0);
    load_vec(&L->ffn_norm, g, q35 ? NAME("post_attention_norm.weight") : NAME("ffn_norm.weight"), m->dim, 0);
    load_mat(&L->wgate, g, NAME("ffn_gate.weight"), m->ffn, m->dim, &m->weight_bytes, 1);
    load_mat(&L->wup, g, NAME("ffn_up.weight"), m->ffn, m->dim, &m->weight_bytes, 1);
    load_mat(&L->wdown, g, NAME("ffn_down.weight"), m->dim, m->ffn, &m->weight_bytes, 1);
    mat *gu[2] = {&L->wgate, &L->wup};
    concat_mats(&L->wgu, gu, 2);
    if (q35 && (i + 1) % interval != 0) { /* linear attention (Gated DeltaNet) */
      const uint32_t inner = m->n_vh * m->sd;
      L->rec = 1;
      L->sti = m->n_rec++;
      load_mat(&L->wlqkv, g, NAME("attn_qkv.weight"), m->conv_dim, m->dim, &m->weight_bytes, 1);
      load_mat(&L->wz, g, NAME("attn_gate.weight"), inner, m->dim, &m->weight_bytes, 1);
      load_mat(&L->wbeta, g, NAME("ssm_beta.weight"), m->n_vh, m->dim, &m->weight_bytes, 1);
      load_mat(&L->walpha, g, NAME("ssm_alpha.weight"), m->n_vh, m->dim, &m->weight_bytes, 1);
      mat *in4[4] = {&L->wlqkv, &L->wz, &L->wbeta, &L->walpha};
      if (!concat_mats(&L->win, in4, 4)) { /* kinds differ (Q4_K_M: F32 beta, alpha): at least [beta; alpha] in one matrix */
        mat ba, *in2[2] = {&L->wbeta, &L->walpha};
        if (concat_mats(&ba, in2, 2)) {
          L->wbeta = ba; /* walpha: rows 0 (merged into wbeta) */
          L->walpha.rows = 0;
        }
      }
      load_vec(&L->conv, g, NAME("ssm_conv1d.weight"), m->conv_dim * m->d_conv, 0);
      load_vec(&L->dt_bias, g, NAME("ssm_dt.bias"), m->n_vh, 0);
      load_vec(&L->ssm_a, g, NAME("ssm_a"), m->n_vh, 0);
      load_vec(&L->ssm_norm, g, NAME("ssm_norm.weight"), m->sd, 0);
      load_mat(&L->wo, g, NAME("ssm_out.weight"), m->dim, inner, &m->weight_bytes, 1);
      sink_layer(L);
      continue;
    }
    L->kvi = m->n_kvl++;
    if (q35) {
      load_vec(&L->q_norm, g, NAME("attn_q_norm.weight"), m->hd, 0);
      load_vec(&L->k_norm, g, NAME("attn_k_norm.weight"), m->hd, 0);
    }
    load_mat(&L->wq, g, NAME("attn_q.weight"), q35 ? 2 * qd : qd, m->dim, &m->weight_bytes, 1);
    load_mat(&L->wk, g, NAME("attn_k.weight"), kvd, m->dim, &m->weight_bytes, 1);
    load_mat(&L->wv, g, NAME("attn_v.weight"), kvd, m->dim, &m->weight_bytes, 1);
    load_mat(&L->wo, g, NAME("attn_output.weight"), m->dim, qd, &m->weight_bytes, 1);
    load_vec(&L->bq, g, NAME("attn_q.bias"), qd, 1);
    load_vec(&L->bk, g, NAME("attn_k.bias"), kvd, 1);
    load_vec(&L->bv, g, NAME("attn_v.bias"), kvd, 1);
    mat *qkv[3] = {&L->wq, &L->wk, &L->wv};
    if (concat_mats(&L->wqkv, qkv, 3)) {
      const vec *b3[3] = {&L->bq, &L->bk, &L->bv};
      const uint32_t n3[3] = {L->wq.rows, kvd, kvd};
      concat_bias(&L->bqkv, b3, n3, 3);
    }
    sink_layer(L);
#undef NAME
  }

  /* tokenizer */
  char *tm = gguf_get_str(g, "tokenizer.ggml.model");
  m->tok_gpt2 = tm && strcmp(tm, "gpt2") == 0;
  free(tm);
  const gguf_kv *tk = gguf_find(g, "tokenizer.ggml.tokens");
  if (tk && tk->type == GGUF_ARR && tk->arr_type == GGUF_STR) {
    m->n_tokens = (uint32_t)tk->arr_n;
    m->tokens = calloc(m->n_tokens ? m->n_tokens : 1, sizeof(char *));
    for (uint32_t i = 0; i < m->n_tokens; i++) {
      gguf_str s = tk->arr_str[i];
      m->tokens[i] = malloc(s.n + 1);
      memcpy(m->tokens[i], s.p, s.n);
      m->tokens[i][s.n] = 0;
    }
  }
  free(arch);
}

static void free_mat(mat *w) {
  free(w->qw);
  free(w->qs);
  free(w->f);
}

int model_load_mtp(model *m, gguf_file *g) {
  if (m->arch != ARCH_QWEN35) return 0;
  int il = -1;
  for (uint64_t i = 0; i < g->n_tensors; i++) {
    unsigned n;
    char tail[64];
    if (sscanf(g->t[i].name, "blk.%u.%63s", &n, tail) == 2 && !strcmp(tail, "nextn.eh_proj.weight")) il = (int)n;
  }
  if (il < 0) return 0;
  const uint32_t qd = m->n_head * m->hd, kvd = m->n_kv * m->hd;
  layer *L = &m->l[m->n_layer];
  memset(L, 0, sizeof *L);
  char n[128];
#define NAME(s) (snprintf(n, sizeof n, "blk.%d." s, il), n)
  load_vec(&m->enorm, g, NAME("nextn.enorm.weight"), m->dim, 0);
  load_vec(&m->hnorm, g, NAME("nextn.hnorm.weight"), m->dim, 0);
  load_vec(&m->head_norm, g, NAME("nextn.shared_head_norm.weight"), m->dim, 1);
  if (!m->head_norm.n) load_vec(&m->head_norm, g, "output_norm.weight", m->dim, 0);
  uint64_t wb = 0; /* the MTP weights are not in weight_bytes (the bytes of one model step) */
  const int q4q8 = q4_as_q8;
  q4_as_q8 = getenv("IE_MTP_Q8") != NULL; /* IE_MTP_Q8=1: Q4_0 as Q8_0 here too (as before) */
  load_mat(&m->eh, g, NAME("nextn.eh_proj.weight"), m->dim, 2 * m->dim, &wb, 1);
  if (ie_mat_sink) ie_mat_sink(&m->eh);
  load_vec(&L->attn_norm, g, NAME("attn_norm.weight"), m->dim, 0);
  load_vec(&L->ffn_norm, g, NAME("post_attention_norm.weight"), m->dim, 0);
  load_vec(&L->q_norm, g, NAME("attn_q_norm.weight"), m->hd, 0);
  load_vec(&L->k_norm, g, NAME("attn_k_norm.weight"), m->hd, 0);
  load_mat(&L->wq, g, NAME("attn_q.weight"), 2 * qd, m->dim, &wb, 1);
  load_mat(&L->wk, g, NAME("attn_k.weight"), kvd, m->dim, &wb, 1);
  load_mat(&L->wv, g, NAME("attn_v.weight"), kvd, m->dim, &wb, 1);
  load_mat(&L->wo, g, NAME("attn_output.weight"), m->dim, qd, &wb, 1);
  load_mat(&L->wgate, g, NAME("ffn_gate.weight"), m->ffn, m->dim, &wb, 1);
  load_mat(&L->wup, g, NAME("ffn_up.weight"), m->ffn, m->dim, &wb, 1);
  load_mat(&L->wdown, g, NAME("ffn_down.weight"), m->dim, m->ffn, &wb, 1);
#undef NAME
  q4_as_q8 = q4q8;
  mat *qkv[3] = {&L->wq, &L->wk, &L->wv};
  concat_mats(&L->wqkv, qkv, 3);
  mat *gu[2] = {&L->wgate, &L->wup};
  concat_mats(&L->wgu, gu, 2);
  sink_layer(L);
  L->kvi = m->n_kvl++;
  m->has_mtp = 1;
  fprintf(stderr, "mtp: head blk.%d loaded (%.2f MB)\n", il, wb / 1e6);
  return 1;
}

int model_load_dflash(model *m, gguf_file *g) {
  char *arch = gguf_get_str(g, "general.architecture");
  const int ok = arch && !strcmp(arch, "dflash");
  free(arch);
  if (!ok) return 0;
  if (m->arch != ARCH_QWEN35) ie_die("dflash: the target must be a qwen35 model");
  dflash *d = calloc(1, sizeof *d);
  if (!d) ie_die("out of memory");
  int64_t v;
  double f;
#define GI(k, dst) (gguf_get_int(g, "dflash." k, &v) ? (void)((dst) = (uint32_t)v) : ie_die("dflash: no dflash." k))
  GI("block_count", d->n_layer), GI("embedding_length", d->dim), GI("feed_forward_length", d->ffn);
  GI("attention.head_count", d->n_head), GI("attention.head_count_kv", d->n_kv), GI("attention.key_length", d->hd);
  GI("block_size", d->block), GI("attention.sliding_window", d->window), GI("selector_top_k", d->top_k);
  GI("selector_rank", d->rank), GI("conv_kernel_size", d->kernel), GI("conv_group_size", d->group);
#undef GI
  d->eps = gguf_get_float(g, "dflash.attention.layer_norm_rms_epsilon", &f) ? (float)f : 1e-6f;
  d->rope_base = gguf_get_float(g, "dflash.rope.freq_base", &f) ? (float)f : 1e7f;
  d->mask = 248070u; /* the mask token of Qwen3.8 (config.json dflash_config.mask_token_id) */
  if (gguf_get_int(g, "dflash.mask_token_id", &v)) d->mask = (uint32_t)v;
  const gguf_kv *tl = gguf_find(g, "dflash.target_layers");
  if (!tl || tl->type != GGUF_ARR || tl->arr_n == 0 || tl->arr_n > 8) ie_die("dflash: no dflash.target_layers");
  d->n_tl = (uint32_t)tl->arr_n;
  for (uint32_t i = 0; i < d->n_tl; i++) { /* the GGUF ids count the embedding as 0: layer id - 1 */
    int32_t x;
    memcpy(&x, tl->arr_data + 4 * (size_t)i, 4);
    if (x < 1 || (uint32_t)x > m->n_layer) ie_die("dflash: target layer %d", x);
    d->tl[i] = (uint32_t)x - 1u;
  }
  if (d->dim != m->dim || d->kernel != 2 || d->hd % 32 || d->hd > 256 || d->dim % d->group || d->top_k > 64 || d->rank % 32 ||
      d->block > 16)
    ie_die("dflash: unsupported shape");
  const uint32_t qd = d->n_head * d->hd, kvd = d->n_kv * d->hd, cp = 2u * d->kernel * (d->dim / d->group);
  uint64_t wb = 0;
  load_vec(&d->enc_norm, g, "enc.output_norm.weight", d->dim, 0);
  load_vec(&d->out_norm, g, "output_norm.weight", d->dim, 0);
  load_mat(&d->fc, g, "fc.weight", d->dim, d->n_tl * d->dim, &wb, 1);
  load_mat(&d->sel_h, g, "selector_hidden.weight", d->rank, d->dim, &wb, 1);
  if (ie_mat_sink) ie_mat_sink(&d->fc), ie_mat_sink(&d->sel_h);
  d->sel_prev = need_tensor(g, "selector_predecessor.weight");
  d->sel_next = need_tensor(g, "selector_successor.weight");
  d->l = calloc(d->n_layer, sizeof *d->l);
  char n[128];
  for (uint32_t il = 0; il < d->n_layer; il++) {
    df_layer *L = &d->l[il];
#define NAME(s) (snprintf(n, sizeof n, "blk.%u." s, il), n)
    load_vec(&L->attn_norm, g, NAME("attn_norm.weight"), d->dim, 0);
    load_vec(&L->ffn_norm, g, NAME("ffn_norm.weight"), d->dim, 0);
    load_vec(&L->q_norm, g, NAME("attn_q_norm.weight"), d->hd, 0);
    load_vec(&L->k_norm, g, NAME("attn_k_norm.weight"), d->hd, 0);
    load_vec(&L->aconv_base, g, NAME("attn_conv_base"), 2 * d->kernel * d->dim, 0);
    load_vec(&L->fconv_base, g, NAME("ffn_conv_base"), 2 * d->kernel * d->dim, 0);
    load_mat(&L->wq, g, NAME("attn_q.weight"), qd, d->dim, &wb, 1);
    load_mat(&L->wk, g, NAME("attn_k.weight"), kvd, d->dim, &wb, 1);
    load_mat(&L->wv, g, NAME("attn_v.weight"), kvd, d->dim, &wb, 1);
    load_mat(&L->wo, g, NAME("attn_output.weight"), d->dim, qd, &wb, 1);
    load_mat(&L->wgate, g, NAME("ffn_gate.weight"), d->ffn, d->dim, &wb, 1);
    load_mat(&L->wup, g, NAME("ffn_up.weight"), d->ffn, d->dim, &wb, 1);
    load_mat(&L->wdown, g, NAME("ffn_down.weight"), d->dim, d->ffn, &wb, 1);
    load_mat(&L->aconv, g, NAME("attn_conv_proj.weight"), cp, d->dim, &wb, 1);
    load_mat(&L->fconv, g, NAME("ffn_conv_proj.weight"), cp, d->dim, &wb, 1);
#undef NAME
    mat *gu[2] = {&L->wgate, &L->wup};
    concat_mats(&L->wgu, gu, 2);
    if (ie_mat_sink) {
      mat *all[] = {&L->wq, &L->wk, &L->wv, &L->wo, &L->wgate, &L->wup, &L->wdown, &L->wgu, &L->aconv, &L->fconv};
      for (unsigned i = 0; i < sizeof all / sizeof all[0]; i++) ie_mat_sink(all[i]);
    }
  }
  m->df = d;
  fprintf(stderr, "dflash: %u layers, block %u, window %u, target layers", d->n_layer, d->block, d->window);
  for (uint32_t i = 0; i < d->n_tl; i++) fprintf(stderr, " %u", d->tl[i]);
  fprintf(stderr, " (%.2f MB)\n", wb / 1e6);
  return 1;
}

void model_df_code(const model *m, int which, uint32_t tok, float *out) {
  const gguf_tensor *t = which ? m->df->sel_next : m->df->sel_prev;
  dequant(t, (uint64_t)tok * m->df->rank, m->df->rank, out);
}

/* GGUF Q4_0 bytes of n floats, as ggml's quantize_row_q4_0_ref: d =
 * (the value of largest magnitude) / -8, q = min(15, (int8)(x / d + 8.5)). */
static void quant_q4_0_bytes(const float *x, uint64_t n, uint8_t *out) {
  for (uint64_t b = 0; b < n / 32; b++) {
    float amax = 0.0f, mx = 0.0f;
    for (int j = 0; j < 32; j++) {
      const float v = x[32 * b + j];
      if (fabsf(v) > amax) amax = fabsf(v), mx = v;
    }
    const float d = mx / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    const uint16_t h = ie_f32_to_f16(d);
    memcpy(out + 18 * b, &h, 2);
    for (int j = 0; j < 16; j++) {
      int q0 = (int)(int8_t)(x[32 * b + j] * id + 8.5f), q1 = (int)(int8_t)(x[32 * b + 16 + j] * id + 8.5f);
      q0 = q0 > 15 ? 15 : q0, q1 = q1 > 15 ? 15 : q1;
      out[18 * b + 2 + j] = (uint8_t)(q0 | (q1 << 4));
    }
  }
}

void model_make_draft_head(model *m, gguf_file *g) {
  const gguf_tensor *t = gguf_tensor_find(g, "output.weight");
  if (!t) t = need_tensor(g, "token_embd.weight"); /* tied */
  mat *w = &m->out_draft;
  memset(w, 0, sizeof *w);
  w->kind = MAT_Q4, w->rows = m->vocab, w->cols = m->dim, w->nb = m->dim / 32;
  /* IE_DRAFT_VOCAB=N (default 131072; 0: all tokens): only the tokens 0 .. N - 1 (BPE ids: roughly the most
   * frequent first) and the special tokens at the end (token type not 1,
   * the run up to the last token) can be drafted. The verify step is not
   * changed, so the output is the same; only fewer drafts may be accepted. */
  m->dv_n = m->dv_s = 0;
  const char *dv = getenv("IE_DRAFT_VOCAB");
  const uint32_t N = dv ? (uint32_t)atoi(dv) : 131072u; /* 27B, 7 drafts: the same acceptance as all tokens, +4 % */
  const gguf_kv *tt = gguf_find(g, "tokenizer.ggml.token_type");
  if (N && N < m->vocab && tt && tt->type == GGUF_ARR && tt->arr_type == GGUF_I32 && tt->arr_n <= m->vocab) {
    uint32_t S = (uint32_t)tt->arr_n;
    for (int32_t ty; S > N; S--) { /* the file data may be unaligned */
      memcpy(&ty, tt->arr_data + (size_t)(S - 1) * 4, 4);
      if (ty == 1) break;
    }
    m->dv_n = N, m->dv_s = S;
    w->rows = N + (uint32_t)tt->arr_n - S;
    fprintf(stderr, "mtp: draft vocabulary: tokens 0 .. %u and %u .. %u\n", N - 1, S, (uint32_t)tt->arr_n - 1);
  }
  if (!ie_sizes_ok(w->rows, w->nb)) { memset(w, 0, sizeof *w); return; }
  float *row = ie_alloc((size_t)m->dim * 4);
  uint8_t *q4 = ie_alloc((size_t)w->rows * w->nb * 18);
  for (uint32_t r = 0; r < w->rows; r++) {
    dequant(t, (uint64_t)model_draft_tok(m, r) * m->dim, m->dim, row);
    quant_q4_0_bytes(row, m->dim, q4 + (size_t)r * w->nb * 18);
  }
  repack_q4(w, q4);
  free(row), free(q4);
  if (ie_mat_sink) ie_mat_sink(w);
  fprintf(stderr, "mtp: draft head Q4_0 (%.2f MB)\n", (double)w->rows * w->nb * 18 / 1e6);
}

void model_free(model *m) {
  free_mat(&m->out_draft);
  if (m->has_mtp) {
    m->n_layer++; /* free the MTP layer with the others */
    free_mat(&m->eh), free(m->enorm.f), free(m->hnorm.f), free(m->head_norm.f);
  }
  free_mat(&m->tok);
  if (!m->tied) free_mat(&m->out);
  free(m->out_norm.f);
  for (uint32_t i = 0; i < m->n_layer; i++) {
    layer *L = &m->l[i];
    free(L->attn_norm.f), free(L->ffn_norm.f), free(L->bq.f), free(L->bk.f), free(L->bv.f), free(L->bqkv.f);
    free_mat(&L->wqkv), free_mat(&L->wgu);
    free_mat(&L->wq), free_mat(&L->wk), free_mat(&L->wv), free_mat(&L->wo);
    free_mat(&L->wgate), free_mat(&L->wup), free_mat(&L->wdown);
    free_mat(&L->wlqkv), free_mat(&L->wz), free_mat(&L->wbeta), free_mat(&L->walpha), free_mat(&L->win);
    free(L->q_norm.f), free(L->k_norm.f), free(L->conv.f), free(L->dt_bias.f), free(L->ssm_a.f), free(L->ssm_norm.f);
    free(L->sinks.f), free(L->ebias.f);
    free_mat(&L->wr), free_mat(&L->egu), free_mat(&L->eg), free_mat(&L->eu), free_mat(&L->ed);
  }
  free(m->l);
  for (uint32_t i = 0; i < m->n_tokens; i++) free(m->tokens[i]);
  free(m->tokens);
}

/* ------------------------------------------------------------ detokenize */

/* GPT2 byte-level: the inverse of bytes_to_unicode(). */
static int gpt2_byte(uint32_t cp) {
  if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) return (int)cp;
  /* the other 68 bytes map to 256, 257, ... in byte order */
  uint32_t n = 0;
  for (uint32_t b = 0; b < 256; b++) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) continue;
    if (cp == 256 + n) return (int)b;
    n++;
  }
  return -1;
}

static uint32_t utf8_next(const unsigned char **s) {
  const unsigned char *p = *s;
  uint32_t c = *p++;
  int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
  if (extra) c &= 0x3Fu >> extra;
  while (extra-- > 0 && (*p & 0xC0) == 0x80) c = (c << 6) | (*p++ & 0x3Fu);
  *s = p;
  return c;
}

char *model_detok(const model *m, uint32_t t, uint32_t *len) {
  const char *s = t < m->n_tokens ? m->tokens[t] : "";
  size_t n = strlen(s);
  char *o = malloc(n + 4);
  uint32_t k = 0;
  if (m->tok_gpt2) {
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
      int b = gpt2_byte(utf8_next(&p));
      if (b >= 0) o[k++] = (char)b;
    }
  } else {
    unsigned hx;
    if (n == 6 && sscanf(s, "<0x%2X>", &hx) == 1) {
      o[k++] = (char)hx;
    } else {
      for (size_t i = 0; i < n; i++) {
        if ((unsigned char)s[i] == 0xE2 && (unsigned char)s[i + 1] == 0x96 && (unsigned char)s[i + 2] == 0x81) {
          o[k++] = ' '; /* U+2581 */
          i += 2;
        } else {
          o[k++] = s[i];
        }
      }
    }
  }
  o[k] = 0;
  *len = k;
  return o;
}

/* --------------------------------------------------------------- the graph */

typedef struct {
  graph *g;
  uint32_t cap_ops;
  int layer; /* the layer of the ops emitted now (-1: outside the layers) */
} gb;

static int new_buf(graph *g, const char *name, uint32_t bytes) {
  if (g->n_bufs >= MAX_BUFS) ie_die("too many buffers");
  buf *b = &g->bufs[g->n_bufs];
  b->name = name;
  b->stride = (bytes + 255u) & ~255u; /* each token's part 256-aligned */
  b->size = b->stride * g->T;
  b->first = UINT32_MAX;
  b->last = 0;
  return (int)g->n_bufs++;
}

/* A buffer that the tokens of a run share (stride 0): only for an op that
 * is launched once per token, one token after the other (attention scores). */
static int new_buf_shared(graph *g, const char *name, uint32_t bytes) {
  int id = new_buf(g, name, bytes);
  buf *b = &g->bufs[id];
  b->size = b->stride, b->stride = 0;
  return id;
}

static void touch(graph *g, int id, uint32_t t) {
  if (id < 0) return;
  buf *b = &g->bufs[id];
  if (t < b->first) b->first = t;
  if (t > b->last) b->last = t;
}

static op *emit(gb *B, int kind, int a, int b, int c) {
  graph *g = B->g;
  if (g->n_ops == B->cap_ops) {
    B->cap_ops = B->cap_ops ? 2 * B->cap_ops : 256;
    g->ops = realloc(g->ops, B->cap_ops * sizeof(op));
    if (!g->ops) ie_die("out of memory");
  }
  op *o = &g->ops[g->n_ops];
  memset(o, 0, sizeof *o);
  o->kind = kind, o->a = a, o->b = b, o->c = c, o->layer = B->layer, o->qo = -1, o->res = -1, o->nout = -1, o->nq = -1;
  o->gt = -1;
  touch(g, a, g->n_ops), touch(g, b, g->n_ops), touch(g, c, g->n_ops);
  g->n_ops++;
  return o;
}

/* The last emitted op also uses buffer id (as an input or output). */
static void use_last(gb *B, int id) {
  if (id >= 0) touch(B->g, id, B->g->n_ops - 1);
}

/* y = W x (+ bias) (+ res), where xq is the Q8 copy of x (or -1) and xf the
 * f32 x. bias may be NULL or empty; res is a buffer id or -1. The result goes
 * to buffer `into` at byte offset `off`, or (into < 0) to a new buffer. */
static int matvec_into(gb *B, const mat *w, int xf, int xq, int into, uint32_t off, const char *name, const vec *bias,
                       int res) {
  int y = into >= 0 ? into : new_buf(B->g, name, w->rows * 4);
  const int quant_in = w->kind != MAT_F32;
  op *o = emit(B, w->kind == MAT_Q4 ? OP_GEMV_Q4 : w->kind == MAT_Q8 ? OP_GEMV_Q8 : w->kind == MAT_Q4K ? OP_GEMV_Q4K
                   : w->kind == MAT_Q6K ? OP_GEMV_Q6K : w->kind == MAT_Q5K ? OP_GEMV_Q5K : w->kind == MAT_IQ4XS ? OP_GEMV_IQ4XS
                   : OP_GEMV_F32,
               quant_in ? xq : xf, y, -1);
  o->w = w;
  o->n = w->cols;
  if (bias && bias->n) o->v = bias;
  o->res = res;
  o->boff = into >= 0 ? off : 0;
  use_last(B, res);
  return y;
}
static int matvec(gb *B, const mat *w, int xf, int xq, const char *name, const vec *bias, int res) {
  return matvec_into(B, w, xf, xq, -1, 0, name, bias, res);
}

static int fuse_quant(gb *B, uint32_t n, const char *name);

/* The last op (a GEMV) also writes out = rmsnorm(its result) * nv and, if
 * quantized, its Q8 copy (a new buffer, returned in *q). */
static int fuse_norm(gb *B, const vec *nv, uint32_t n, const char *name, int quantized, const char *qname, int *q) {
  graph *g = B->g;
  /* By default the norm is a separate RMSNORM op: measured on the R9700, the
   * fused form (the last workgroup of the GEMV does the norm) is slower
   * (Qwen2.5-1.5B: 2.94 vs 2.81 ms/token). IE_NORM_FUSE=1 selects it. */
  const char *nf = getenv("IE_NORM_FUSE");
  const int k = g->ops[g->n_ops - 1].kind;
  const int gemv = k == OP_GEMV_Q4 || k == OP_GEMV_Q4K || k == OP_GEMV_Q6K || k == OP_GEMV_F32 || k == OP_GEMV_Q8;
  if (!(nf && nf[0] == '1') || !gemv) {
    const op *p = &g->ops[g->n_ops - 1];
    int x = p->b, out = new_buf(g, name, n * 4);
    op *o = emit(B, OP_RMSNORM, x, out, -1);
    o->n = n, o->v = nv;
    *q = quantized ? fuse_quant(B, n, qname) : -1;
    return out;
  }
  op *o = &g->ops[g->n_ops - 1];
  int out = new_buf(g, name, n * 4);
  o->nv = nv;
  o->nout = out;
  use_last(B, out);
  *q = -1;
  if (quantized) {
    *q = new_buf(g, qname, (uint32_t)q8_bytes(n / 32));
    g->ops[g->n_ops - 1].nq = *q;
    use_last(B, *q);
  }
  return out;
}

/* A Q8 copy of the f32 result of the last op, written by that op (qo). */
static int fuse_quant(gb *B, uint32_t n, const char *name) {
  int q = new_buf(B->g, name, (uint32_t)q8_bytes(n / 32));
  B->g->ops[B->g->n_ops - 1].qo = q;
  use_last(B, q);
  return q;
}

static int quant(gb *B, int x, uint32_t n, const char *name) {
  int q = new_buf(B->g, name, (uint32_t)q8_bytes(n / 32));
  emit(B, OP_QUANT, x, q, -1)->n = n;
  return q;
}

static int any_q4(const mat *a, const mat *b, const mat *c) {
  return a->kind != MAT_F32 || (b && b->kind != MAT_F32) || (c && c->kind != MAT_F32);
}

/* Does the first GEMV of layer L read a Q8 copy of its input? */
static int layer_in_q8(const layer *L) {
  if (L->rec) return any_q4(&L->wlqkv, &L->wz, &L->wbeta) || L->walpha.kind != MAT_F32;
  return any_q4(&L->wq, &L->wk, &L->wv);
}

/* A full attention block: x + Wo attention(rmsnorm'd input xn / its Q8
 * copy xq). Returns the new x. */
static int attn_block(gb *B, const model *m, const layer *L, int x, int xn, int xq, uint32_t n_ctx) {
  graph *g = B->g;
  const uint32_t qd = m->n_head * m->hd, kvd = m->n_kv * m->hd;
  const int q35 = m->arch == ARCH_QWEN35;
  op *o;
    /* q, k, v in one buffer: [q | k | v] (qwen35: q is [q | gate] per head) */
    const uint32_t qrows = L->wq.rows;
    int qkv;
    if (L->wqkv.rows) {
      qkv = matvec(B, &L->wqkv, xn, xq, "qkv", &L->bqkv, -1);
    } else {
      qkv = new_buf(g, "qkv", (qrows + 2 * kvd) * 4);
      matvec_into(B, &L->wq, xn, xq, qkv, 0, "q", &L->bq, -1);
      matvec_into(B, &L->wk, xn, xq, qkv, qrows * 4, "k", &L->bk, -1);
      matvec_into(B, &L->wv, xn, xq, qkv, (qrows + kvd) * 4, "v", &L->bv, -1);
    }
    int qsrc = qkv;
    const int fl = m->arch == ARCH_FLINT;
    if (q35 || fl) { /* per head QK RMSNorm, RoPE (not for a flint norope layer); qwen35: q heads are [q | gate] */
      qsrc = new_buf(g, "q", qd * 4);
      o = emit(B, OP_QKN_ROPE_KV, qkv, qsrc, -1), o->n = qd, o->nh = m->n_head;
      o->v = &L->q_norm, o->nv = &L->k_norm, o->qstr = q35 ? 2 * m->hd : m->hd, o->norope = L->norope;
    } else {
      o = emit(B, OP_ROPE_KV, qkv, qkv, qkv), o->n = qd, o->nh = m->n_head;
      o->boff = qd * 4, o->coff = (qd + kvd) * 4;
    }
    int att = new_buf(g, "attn", qd * 4);
    const uint32_t sc_cpu = m->n_head * n_ctx, sc_gpu = m->n_head * ie_att_max_split(n_ctx) * (m->hd + 2);
    int sc = new_buf_shared(g, "scores", (sc_cpu > sc_gpu ? sc_cpu : sc_gpu) * 4);
    o = emit(B, OP_ATTN, qsrc, att, sc), o->n = qd;
    o->win = L->swin, o->sk = L->sinks.n ? &L->sinks : NULL;
    if (q35) { /* out *= sigmoid(gate): the gate of head h is at h * 2 hd + hd */
      o->gt = qkv, o->gtoff = m->hd * 4, o->gstride = 2 * m->hd;
      use_last(B, qkv);
    }
    int aq = -1;
    if (L->wo.kind != MAT_F32) aq = m->hd % 32 == 0 && m->hd <= 256 ? fuse_quant(B, qd, "attn_q8") : quant(B, att, qd, "attn_q8");
    return matvec(B, &L->wo, att, aq, "x", NULL, x); /* x = x + Wo att */
}

/* The FFN block: x + Wdown (silu(Wgate h) * Wup h), h = rmsnorm(x) * ffn_norm. */
static int ffn_block(gb *B, const model *m, const layer *L, int x) {
  graph *g = B->g;
  op *o;
  int hq;
  int hn = fuse_norm(B, &L->ffn_norm, m->dim, "ffn_in", any_q4(&L->wgate, &L->wup, NULL), "ffn_in_q8", &hq);
  /* gate and up in one buffer: [gate | up] */
  int gu;
  if (L->wgu.rows) {
    gu = matvec(B, &L->wgu, hn, hq, "gate_up", NULL, -1);
  } else {
    gu = new_buf(g, "gate_up", 2 * m->ffn * 4);
    matvec_into(B, &L->wgate, hn, hq, gu, 0, "gate", NULL, -1);
    matvec_into(B, &L->wup, hn, hq, gu, m->ffn * 4, "up", NULL, -1);
  }
  o = emit(B, OP_SWIGLU, gu, gu, -1); /* gate = silu(gate) * up */
  o->n = m->ffn, o->boff = m->ffn * 4;
  int gq = L->wdown.kind != MAT_F32 ? fuse_quant(B, m->ffn, "ffn_mid_q8") : -1;
  return matvec(B, &L->wdown, gu, gq, "x", NULL, x); /* x = x + Wdown gate */
}

/* flint: the MoE block. x + shared(h) + sum over the n_used chosen experts
 * e of weight_e down_e(silu(gate_e h) * up_e h), h = rmsnorm(x) * ffn_norm. */
static int moe_block(gb *B, const model *m, const layer *L, int x) {
  graph *g = B->g;
  const uint32_t K = m->n_used, eff = m->eff, dim = m->dim;
  op *o;
  int hq;
  int hn = fuse_norm(B, &L->ffn_norm, dim, "ffn_in", 1, "ffn_in_q8", &hq);
  /* the router: logits (f32 weights), then the top K experts and their weights */
  int rl = matvec(B, &L->wr, hn, hq, "router", NULL, -1);
  int rt = new_buf(g, "route", 2 * K * 4);
  o = emit(B, OP_MOE_ROUTE, rl, rt, -1), o->n = m->n_exp, o->nh = K, o->v = L->ebias.n ? &L->ebias : NULL;
  /* the shared expert, with the residual: xs = x + down_s(silu(gate_s h) up_s h) */
  int xs = x;
  if (m->sff) {
    int gu;
    if (L->wgu.rows) {
      gu = matvec(B, &L->wgu, hn, hq, "sh_gate_up", NULL, -1);
    } else {
      gu = new_buf(g, "sh_gate_up", 2 * m->sff * 4);
      matvec_into(B, &L->wgate, hn, hq, gu, 0, "sh_gate", NULL, -1);
      matvec_into(B, &L->wup, hn, hq, gu, m->sff * 4, "sh_up", NULL, -1);
    }
    o = emit(B, OP_SWIGLU, gu, gu, -1), o->n = m->sff, o->boff = m->sff * 4;
    int gq = L->wdown.kind != MAT_F32 ? fuse_quant(B, m->sff, "sh_mid_q8") : -1;
    xs = matvec(B, &L->wdown, gu, gq, "xs", NULL, x);
  }
  /* the experts: gate and up of each chosen expert, [gate | up] per slot */
  const uint32_t gstr = 2 * eff * 4;
  int ye = new_buf(g, "exp_gu", K * gstr);
  const mat *parts[2] = {L->egu.rows ? &L->egu : &L->eg, L->egu.rows ? NULL : &L->eu};
  for (int i = 0; i < 2 && parts[i]; i++) {
    o = emit(B, OP_GEMV_EXP, parts[i]->kind == MAT_F32 ? hn : hq, ye, rt);
    o->w = parts[i], o->n = dim, o->nh = K, o->astr = 0, o->bstr = gstr, o->boff = i * eff * 4;
  }
  const uint32_t qstr = (uint32_t)q8_bytes(eff / 32);
  int mq = new_buf(g, "exp_mid_q8", K * qstr);
  o = emit(B, OP_MOE_SWIGLU, ye, -1, -1), o->n = eff, o->nh = K, o->astr = gstr, o->qo = mq;
  use_last(B, mq);
  int yd = new_buf(g, "exp_down", K * dim * 4);
  o = emit(B, OP_GEMV_EXP, mq, yd, rt), o->w = &L->ed, o->n = eff, o->nh = K, o->astr = qstr, o->bstr = dim * 4;
  if (L->ed.kind == MAT_F32) ie_die("flint: f32 down experts are not supported");
  int xo = new_buf(g, "x", dim * 4);
  o = emit(B, OP_MOE_SUM, yd, xo, rt), o->n = dim, o->nh = K, o->res = xs;
  use_last(B, xs);
  return xo;
}

/* The DFlash ops (see model.h): the context section, then the block. */
static void df_build(gb *B, const model *m) {
  graph *g = B->g;
  const dflash *d = m->df;
  const uint32_t dim = d->dim, qd = d->n_head * d->hd, kvd = d->n_kv * d->hd;
  op *o;
  B->layer = -1;
  g->i_dfc = g->n_ops;
  int fq = quant(B, g->df_feat, d->n_tl * dim, "df_feat_q8");
  int ch = matvec(B, &d->fc, g->df_feat, fq, "df_ctx", NULL, -1);
  int cn = new_buf(g, "df_ctxn", dim * 4);
  o = emit(B, OP_RMSNORM, ch, cn, -1), o->n = dim, o->v = &d->enc_norm;
  int cq = fuse_quant(B, dim, "df_ctxn_q8");
  for (uint32_t il = 0; il < d->n_layer; il++) {
    const df_layer *L = &d->l[il];
    int kv = new_buf(g, "df_kv", 2 * kvd * 4);
    matvec_into(B, &L->wk, cn, cq, kv, 0, "df_k", NULL, -1);
    matvec_into(B, &L->wv, cn, cq, kv, kvd * 4, "df_v", NULL, -1);
    use_last(B, cn), use_last(B, cq);
    o = emit(B, OP_DF_QKV, kv, -1, -1), o->n = kvd, o->nh = 0, o->nv = &L->k_norm, o->layer = (int)il;
  }
  g->i_dfb = g->n_ops;
  int h = new_buf(g, "df_h", dim * 4);
  emit(B, OP_EMBED, h, -1, -1)->n = dim;
  for (uint32_t il = 0; il < d->n_layer; il++) {
    const df_layer *L = &d->l[il];
    /* attention */
    int xn = new_buf(g, "df_xn", dim * 4);
    o = emit(B, OP_RMSNORM, h, xn, -1), o->n = dim, o->v = &L->attn_norm;
    int xq = fuse_quant(B, dim, "df_xn_q8");
    int dyn = matvec(B, &L->aconv, xn, xq, "df_dyn", NULL, -1);
    int xc = new_buf(g, "df_xc", dim * 4);
    o = emit(B, OP_DF_CONV, xn, xc, dyn), o->n = dim, o->nh = 0, o->v = &L->aconv_base;
    int xcq = quant(B, xc, dim, "df_xc_q8");
    int qkv = new_buf(g, "df_qkv", (qd + 2 * kvd) * 4);
    matvec_into(B, &L->wq, xc, xcq, qkv, 0, "df_q", NULL, -1);
    matvec_into(B, &L->wk, xc, xcq, qkv, qd * 4, "df_k", NULL, -1);
    matvec_into(B, &L->wv, xc, xcq, qkv, (qd + kvd) * 4, "df_v", NULL, -1);
    use_last(B, xc), use_last(B, xcq);
    o = emit(B, OP_DF_QKV, qkv, -1, -1), o->n = qd, o->nh = d->n_head, o->v = &L->q_norm, o->nv = &L->k_norm, o->layer = (int)il;
    int att = new_buf(g, "df_att", qd * 4);
    o = emit(B, OP_DF_ATTN, qkv, att, -1), o->n = qd, o->nh = d->n_head, o->layer = (int)il;
    int aq = quant(B, att, qd, "df_att_q8");
    int ob = matvec(B, &L->wo, att, aq, "df_o", NULL, -1);
    int h2 = new_buf(g, "df_h", dim * 4);
    o = emit(B, OP_DF_CONV, ob, h2, dyn), o->n = dim, o->nh = 1, o->v = &L->aconv_base, o->res = h;
    use_last(B, h);
    h = h2;
    /* FFN */
    xn = new_buf(g, "df_xn", dim * 4);
    o = emit(B, OP_RMSNORM, h, xn, -1), o->n = dim, o->v = &L->ffn_norm;
    xq = fuse_quant(B, dim, "df_xn_q8");
    dyn = matvec(B, &L->fconv, xn, xq, "df_dyn", NULL, -1);
    xc = new_buf(g, "df_xc", dim * 4);
    o = emit(B, OP_DF_CONV, xn, xc, dyn), o->n = dim, o->nh = 0, o->v = &L->fconv_base;
    xcq = quant(B, xc, dim, "df_xc_q8");
    int gu;
    if (L->wgu.rows) {
      gu = matvec(B, &L->wgu, xc, xcq, "df_gu", NULL, -1);
    } else {
      gu = new_buf(g, "df_gu", 2 * d->ffn * 4);
      matvec_into(B, &L->wgate, xc, xcq, gu, 0, "df_gate", NULL, -1);
      matvec_into(B, &L->wup, xc, xcq, gu, d->ffn * 4, "df_up", NULL, -1);
      use_last(B, xc), use_last(B, xcq);
    }
    o = emit(B, OP_SWIGLU, gu, gu, -1), o->n = d->ffn, o->boff = d->ffn * 4;
    int gq = fuse_quant(B, d->ffn, "df_mid_q8");
    int dn = matvec(B, &L->wdown, gu, gq, "df_down", NULL, -1);
    h2 = new_buf(g, "df_h", dim * 4);
    o = emit(B, OP_DF_CONV, dn, h2, dyn), o->n = dim, o->nh = 1, o->v = &L->fconv_base, o->res = h;
    use_last(B, h);
    h = h2;
  }
  int hn = new_buf(g, "df_out", dim * 4);
  o = emit(B, OP_RMSNORM, h, hn, -1), o->n = dim, o->v = &d->out_norm;
  int hq = fuse_quant(B, dim, "df_out_q8");
  /* the target's output matrix (all tokens; one launch for the block) */
  g->df_logits = matvec(B, &m->out, hn, hq, "df_logits", NULL, -1);
  g->df_sel = matvec(B, &d->sel_h, hn, hq, "df_sel", NULL, -1);
  use_last(B, hn), use_last(B, hq);
}

void graph_build(graph *g, const model *m, uint32_t n_ctx, uint32_t T, int mtp) {
  memset(g, 0, sizeof *g);
  g->bufs = calloc(MAX_BUFS, sizeof(buf));
  g->n_ctx = n_ctx;
  g->T = T ? T : 1;
  g->h_out = g->mtp_h = g->mtp_g = g->mtp_logits = g->mtp_argmax = -1;
  g->df_feat = g->df_logits = g->df_sel = -1;
  gb B = {g, 0, -1};
  uint32_t dim = m->dim;
  if (mtp && !m->has_mtp) ie_die("the model has no MTP head");
  const int df = mtp && m->df;
  if (df) g->df_feat = new_buf(g, "df_feat", m->df->n_tl * dim * 4);

  int x = new_buf(g, "x", dim * 4);
  emit(&B, OP_EMBED, x, -1, -1)->n = dim;
  /* the norm of layer 0; the later norms are fused into the residual GEMVs */
  B.layer = 0;
  int xn = new_buf(g, "attn_in", dim * 4);
  op *o = emit(&B, OP_RMSNORM, x, xn, -1);
  o->n = dim, o->v = &m->l[0].attn_norm;
  int xq = layer_in_q8(&m->l[0]) ? fuse_quant(&B, dim, "attn_in_q8") : -1;
  for (uint32_t l = 0; l < m->n_layer; l++) {
    const layer *L = &m->l[l];
    B.layer = (int)l;
    if (L->rec) {
      /* linear attention: [q k v (conv_dim) | z | beta | alpha] in one buffer */
      const uint32_t inner = m->n_vh * m->sd, cd = m->conv_dim;
      int lin;
      if (L->win.rows) {
        lin = matvec(&B, &L->win, xn, xq, "lin", NULL, -1);
      } else {
        lin = new_buf(g, "lin", (cd + inner + 2 * m->n_vh) * 4);
        matvec_into(&B, &L->wlqkv, xn, xq, lin, 0, "lqkv", NULL, -1);
        matvec_into(&B, &L->wz, xn, xq, lin, cd * 4, "z", NULL, -1);
        matvec_into(&B, &L->wbeta, xn, xq, lin, (cd + inner) * 4, "beta", NULL, -1);
        if (L->walpha.rows) matvec_into(&B, &L->walpha, xn, xq, lin, (cd + inner + m->n_vh) * 4, "alpha", NULL, -1);
      }
      int go = new_buf(g, "gdn", inner * 4);
      o = emit(&B, OP_GDN, lin, go, -1), o->n = inner;
      int gq = L->wo.kind != MAT_F32 ? fuse_quant(&B, inner, "gdn_q8") : -1;
      x = matvec(&B, &L->wo, go, gq, "x", NULL, x); /* x = x + Wout gdn */
    } else {
    x = attn_block(&B, m, L, x, xn, xq, n_ctx);
    }
    x = L->moe ? moe_block(&B, m, L, x) : ffn_block(&B, m, L, x);
    if (l + 1 < m->n_layer) {
      const layer *N = &m->l[l + 1];
      xn = fuse_norm(&B, &N->attn_norm, dim, "attn_in", layer_in_q8(N), "attn_in_q8", &xq);
    } else {
      xn = fuse_norm(&B, &m->out_norm, dim, "out_in", m->out.kind != MAT_F32, "out_in_q8", &xq);
    }
    if (df) /* DFlash: the outputs of the target layers, side by side */
      for (uint32_t j = 0; j < m->df->n_tl; j++)
        if (m->df->tl[j] == l) {
          o = emit(&B, OP_COPY, x, g->df_feat, -1);
          o->n = dim, o->boff = j * dim * 4;
        }
  }
  B.layer = -1;
  g->i_head = g->n_ops; /* the output matrix and the argmax: ops i_head .. n_main - 1 */
  g->logits = matvec(&B, &m->out, xn, xq, "logits", NULL, -1);
  g->argmax = new_buf(g, "argmax", 8); /* [index, probability (float bits)] */
  emit(&B, OP_ARGMAX, g->logits, g->argmax, -1)->n = m->vocab;
  g->h_out = xn;
  g->n_main = g->n_ops;

  if (mtp) { /* the MTP head (see model.h) */
    const layer *L = &m->l[m->n_layer];
    B.layer = (int)m->n_layer;
    g->mtp_h = new_buf(g, "mtp_h", dim * 4);
    int e = new_buf(g, "mtp_e", dim * 4);
    emit(&B, OP_EMBED, e, -1, -1)->n = dim;
    int cat = new_buf(g, "mtp_cat", 2 * dim * 4); /* [enorm(e) | hnorm(h)] */
    o = emit(&B, OP_RMSNORM, e, cat, -1), o->n = dim, o->v = &m->enorm;
    o = emit(&B, OP_RMSNORM, g->mtp_h, cat, -1), o->n = dim, o->v = &m->hnorm, o->boff = dim * 4;
    int catq = m->eh.kind != MAT_F32 ? quant(&B, cat, 2 * dim, "mtp_cat_q8") : -1;
    int x2 = matvec(&B, &m->eh, cat, catq, "mtp_x", NULL, -1);
    int xn2 = new_buf(g, "mtp_attn_in", dim * 4);
    o = emit(&B, OP_RMSNORM, x2, xn2, -1), o->n = dim, o->v = &L->attn_norm;
    int xq2 = layer_in_q8(L) ? fuse_quant(&B, dim, "mtp_attn_in_q8") : -1;
    x2 = attn_block(&B, m, L, x2, xn2, xq2, n_ctx);
    x2 = ffn_block(&B, m, L, x2);
    g->mtp_g = new_buf(g, "mtp_g", dim * 4);
    o = emit(&B, OP_RMSNORM, x2, g->mtp_g, -1), o->n = dim, o->v = &m->head_norm;
    int gq2 = m->out.kind != MAT_F32 ? fuse_quant(&B, dim, "mtp_g_q8") : -1;
    B.layer = -1;
    g->i_mtp_head = g->n_ops;
    g->mtp_logits = matvec(&B, m->out_draft.rows ? &m->out_draft : &m->out, g->mtp_g, gq2, "mtp_logits", NULL, -1);
    g->mtp_argmax = new_buf(g, "mtp_argmax", 8);
    emit(&B, OP_ARGMAX, g->mtp_logits, g->mtp_argmax, -1)->n = m->out_draft.rows ? m->out_draft.rows : m->vocab;
  }
  g->n_mtp = g->i_dfc = g->i_dfb = g->n_ops;
  if (df) df_build(&B, m);
  /* The host reads (or copies) these buffers between runs: they live
   * through the whole op list. */
  const int keep[] = {g->logits, g->argmax, g->h_out, g->mtp_h, g->mtp_g, g->mtp_logits, g->mtp_argmax, g->df_feat, g->df_logits,
                      g->df_sel};
  for (unsigned k = 0; k < sizeof keep / sizeof keep[0]; k++) touch(g, keep[k], 0), touch(g, keep[k], g->n_ops);

  /* bytes of weights read per token (GEMV + one embedding row) */
  uint64_t wb = 0;
  for (uint32_t i = 0; i < g->n_main; i++) {
    const op *p = &g->ops[i];
    if (p->kind == OP_GEMV_Q4) wb += (uint64_t)p->w->rows * p->w->nb * 18;
    if (p->kind == OP_GEMV_Q4K) wb += (uint64_t)p->w->rows * p->w->nb * 18;
    if (p->kind == OP_GEMV_Q6K) wb += (uint64_t)p->w->rows * p->w->nb * 210 / 8;
    if (p->kind == OP_GEMV_F32) wb += (uint64_t)p->w->rows * p->w->cols * 4;
    if (p->kind == OP_GEMV_Q8) wb += (uint64_t)p->w->rows * p->w->nb * 34;
    if (p->kind == OP_GEMV_Q5K) wb += (uint64_t)p->w->rows * p->w->nb * 22;
    if (p->kind == OP_GEMV_IQ4XS) wb += (uint64_t)p->w->rows * p->w->nb * 17;
    if (p->kind == OP_GEMV_EXP) {
      const int k = p->w->kind;
      const uint64_t per = k == MAT_F32 ? (uint64_t)p->w->rows * p->w->cols * 4
                           : (uint64_t)p->w->rows * p->w->nb * (k == MAT_Q8 ? 34 : k == MAT_Q5K ? 22 : k == MAT_IQ4XS ? 17 : k == MAT_Q6K ? 26 : 18);
      wb += per * p->nh;
    }
  }
  g->weight_bytes_per_token = wb;

  /* the KV cache of each attention layer: n_ctx positions, or a ring for a
   * sliding window layer */
  const uint32_t nkv = m->n_kvl;
  g->kv_len = ie_alloc((size_t)(nkv ? nkv : 1) * 4);
  g->kv_off = ie_alloc((size_t)(nkv ? nkv : 1) * 8);
  for (uint32_t l = 0; l < m->n_layer + (m->has_mtp ? 1u : 0u); l++) {
    const layer *L = &m->l[l];
    if (L->rec) continue;
    uint32_t len = n_ctx;
    if (L->swin) {
      uint32_t r = 1;
      while (r < L->swin + (g->T > IE_KV_SLACK ? g->T : IE_KV_SLACK)) r *= 2; /* room for a run of T tokens */
      if (r < len) len = r;
    }
    g->kv_len[L->kvi] = len;
  }
  for (uint32_t i = 0; i < nkv; i++) g->kv_off[i] = g->kv_total, g->kv_total += g->kv_len[i];

  /* RoPE tables: theta_i = pos * base^(-2i/n_rot), i < n_rot / 2 */
  uint32_t h2 = m->n_rot / 2;
  g->rope_cos = ie_alloc((size_t)n_ctx * h2 * 4);
  g->rope_sin = ie_alloc((size_t)n_ctx * h2 * 4);
  for (uint32_t p = 0; p < n_ctx; p++)
    for (uint32_t i = 0; i < h2; i++) {
      double th = (double)p * pow((double)m->rope_base, -2.0 * i / (double)m->n_rot);
      g->rope_cos[p * h2 + i] = (float)cos(th);
      g->rope_sin[p * h2 + i] = (float)sin(th);
    }
}

void graph_free(graph *g) {
  free(g->ops);
  free(g->bufs);
  free(g->rope_cos);
  free(g->rope_sin);
  free(g->kv_len);
  free(g->kv_off);
  memset(g, 0, sizeof *g);
}
