/* ie_kernels.c -- the GPU kernels for gfx1201 (RDNA4, wave32), in plain C.
 *
 * Build (see Makefile):
 *   clang -x c -std=c11 --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -nogpulib -O3
 *   ld.lld -shared -> build/ie_kernels.hsaco
 *
 * Index, layout, integer dot product, grid, lane split and the order of
 * the wave reduction come from the verified core (core/ie_core.h). In this
 * build ie_dot4_us is the v_dot4_i32_iu8 instruction.
 *
 * Every kernel runs with 256 threads per workgroup (8 waves of 32). The
 * flat work group size 256 is written into the IR by the Makefile (clang 18
 * does not accept amdgpu_flat_work_group_size in C). */
#include "../core/ie_core.h"

#define G __attribute__((address_space(1))) /* global memory */
#define LDS __attribute__((address_space(3), loader_uninitialized))
#define KERNEL __attribute__((amdgpu_kernel, visibility("protected"))) void
#define NT 256u

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned char u8;
typedef signed char i8;
typedef _Float16 f16;

static inline u32 tid(void) { return __builtin_amdgcn_workitem_id_x(); }
static inline u32 wgid(void) { return __builtin_amdgcn_workgroup_id_x(); }
static inline u32 lane(void) { return tid() & 31u; }
/* Token index of a multi-token launch (grid dimension y; 0 for one token).
 * The kernels that take byte strides move their pointers by y strides. */
static inline u32 tokid(void) { return __builtin_amdgcn_workgroup_id_y(); }
#define TOK(p, s) ((p) ? (__typeof__(p))((G u8 *)(p) + (unsigned long)tokid() * (s)) : (p))
#define TOKC(p, s) ((__typeof__(p))((const G u8 *)(p) + (unsigned long)tokid() * (s)))

/* Workgroup barrier with the memory ordering of LDS and global memory. */
static inline void barrier(void) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

/* Value of lane (this lane + off) of the wave: ds_bpermute (byte address). */
static inline float shfl_xor_f(float v, u32 m) {
  return __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((int)((lane() ^ m) * 4u), __builtin_bit_cast(int, v)));
}

/* The wave reduction of ie_tree: step m (1..5) adds the value of lane
 * i + (32 >> m), i.e. shuffle down by 16, 8, 4, 2, 1. Lane 0 holds the sum
 * (law wave_sum). Other lanes hold partial values. */
/* The same reduction with DPP (register moves, no LDS). Step 1: lane i of
 * row 0 gets lane i + 16 (v_permlanex16 with the identity selects). Steps
 * 2-5: lane i gets lane i + 8, 4, 2, 1 inside its row of 16 (DPP row_shl;
 * a source past the row reads 0). Lane 0 adds exactly the same values in
 * the same order as with the shuffles, so its result is the same. */
static inline int xrow_i(int v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false, false); }
static inline int shl_i(int v, int n) {
  switch (n) { /* the DPP control must be a constant: row_shl:n = 0x100 + n */
    case 8: return __builtin_amdgcn_update_dpp(0, v, 0x108, 0xF, 0xF, true);
    case 4: return __builtin_amdgcn_update_dpp(0, v, 0x104, 0xF, 0xF, true);
    case 2: return __builtin_amdgcn_update_dpp(0, v, 0x102, 0xF, 0xF, true);
    default: return __builtin_amdgcn_update_dpp(0, v, 0x101, 0xF, 0xF, true);
  }
}
#define F2I(x) __builtin_bit_cast(int, (x))
#define I2F(x) __builtin_bit_cast(float, (x))
static inline float wave_tree_f(float v) {
  v += I2F(xrow_i(F2I(v)));
  v += I2F(shl_i(F2I(v), 8));
  v += I2F(shl_i(F2I(v), 4));
  v += I2F(shl_i(F2I(v), 2));
  v += I2F(shl_i(F2I(v), 1));
  return v;
}
static inline u32 wave_tree_u(u32 v) {
  v += (u32)xrow_i((int)v);
  v += (u32)shl_i((int)v, 8);
  v += (u32)shl_i((int)v, 4);
  v += (u32)shl_i((int)v, 2);
  v += (u32)shl_i((int)v, 1);
  return v;
}
/* The maximum of a wave (v >= 0: a source past the row reads 0), in every lane. */
static inline float wave_max_pos(float v) {
  v = __builtin_fmaxf(v, I2F(xrow_i(F2I(v))));
  v = __builtin_fmaxf(v, I2F(shl_i(F2I(v), 8)));
  v = __builtin_fmaxf(v, I2F(shl_i(F2I(v), 4)));
  v = __builtin_fmaxf(v, I2F(shl_i(F2I(v), 2)));
  v = __builtin_fmaxf(v, I2F(shl_i(F2I(v), 1)));
  return I2F(__builtin_amdgcn_readlane(F2I(v), 0));
}
/* The sum of a wave, in every lane. */
static inline float wave_sum_all(float v) { return I2F(__builtin_amdgcn_readlane(F2I(wave_tree_f(v)), 0)); }

/* Workgroup sum (all 256 threads); every thread gets the result. */
static LDS float red_sum[8];
static inline float wg_sum(float v) {
  v = wave_tree_f(v);
  if (lane() == 0u) red_sum[tid() >> 5] = v;
  barrier();
  float s = 0.0f;
  for (u32 w = 0; w < 8u; w++) s += red_sum[w];
  barrier();
  return s;
}
static LDS float red_max[8];
static inline float wg_max(float v) {
  for (u32 m = 16u; m >= 1u; m >>= 1) v = __builtin_fmaxf(v, shfl_xor_f(v, m));
  if (lane() == 0u) red_max[tid() >> 5] = v;
  barrier();
  float s = red_max[0];
  for (u32 w = 1; w < 8u; w++) s = __builtin_fmaxf(s, red_max[w]);
  barrier();
  return s;
}

/* ------------------------------------------------------------------ embed */

/* out[0..cols) = row tok of a Q4 matrix (GPU layout). One workgroup. */
KERNEL ie_embed_q4(G float *out, const G u32 *qw, const G f16 *qs, u32 nb, u32 tok) {
  for (u32 e = tid(); e < nb * 32u; e += NT) {
    const u32 b = e >> 5, j = e & 31u;
    const u32 w = qw[ie_q4_dst_word(tok, b, ie_q4_word_of(j), nb)];
    out[e] = (float)qs[ie_q4_dst_scale(tok, b, nb)] * (float)((int)ie_q4_nib(w, j) - 8);
  }
}

/* out[0..nb*32) = row tok of a Q8_0 matrix (GPU layout). One workgroup. */
KERNEL ie_embed_q8(G float *out, const G u32 *qw, const G f16 *qs, u32 nb, u32 tok) {
  for (u32 e = tid(); e < nb * 32u; e += NT) {
    const u32 b = e >> 5, j = e & 31u;
    const u32 w = qw[ie_q8_dst_word(tok, b, j >> 2, nb)];
    out[e] = (float)qs[ie_q8_dst_scale(tok, b, nb)] * (float)(i8)ie_byte(w, j & 3u);
  }
}

/* out = row tok of an f32 matrix. One workgroup. */
KERNEL ie_embed_f32(G float *out, const G float *f, u32 cols, u32 tok) {
  for (u32 e = tid(); e < cols; e += NT) out[e] = f[(unsigned long)tok * cols + e];
}

/* ---------------------------------------------------------------- rmsnorm */


/* ------------------------------------------------------------- quantize */

/* Q8 activations. q buffer: int8 values [32 nb], then float d [nb], then
 * u32 asum [nb]. quant_wave: lane j of a wave holds element j of block b;
 * every lane of the wave must call it (it shuffles). The fused kernels
 * (rmsnorm, attention, swiglu) call the same function, so their Q8 copy is
 * the one ie_quant_q8 would write. */
static inline void quant_wave(float v, u32 b, G u8 *q, u32 nb) {
  const u32 j = lane();
  const float amax = wave_max_pos(__builtin_fabsf(v));
  const float d = amax / 127.0f;
  const int qi = d != 0.0f ? (int)__builtin_roundf(v / d) : 0;
  ((G i8 *)q)[b * 32u + j] = (i8)qi;
  const u32 s = wave_tree_u((u32)qi); /* asum (ie_q8_sum semantics) */
  if (j == 0u) {
    ((G float *)(q + 32u * nb))[b] = d;
    ((G u32 *)(q + 36u * nb))[b] = s;
  }
}

/* One wave per block of 32, 8 blocks per workgroup. */
KERNEL ie_quant_q8(const G float *x, G u8 *q, u32 nb, u32 xs, u32 qs) {
  x = TOKC(x, xs), q = TOK(q, qs);
  const u32 b = wgid() * 8u + (tid() >> 5);
  if (b >= nb) return; /* whole waves leave together */
  quant_wave(x[b * 32u + lane()], b, q, nb);
}

/* y = x / sqrt(mean(x^2) + eps) * w. One workgroup. n is a multiple of 32.
 * If q is not NULL, also write the Q8 copy of y (fused quantize): wave w
 * does the blocks w, w + 8, ...; each lane stores and quantizes its own y. */
static inline void rmsnorm_wg(const G float *x, const G float *w, G float *y, u32 n, float eps, G u8 *q) {
  float ss = 0.0f;
  for (u32 i = tid(); i < n; i += NT) ss += x[i] * x[i];
  ss = wg_sum(ss);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)n + eps);
  const u32 nb = n >> 5;
  for (u32 b = tid() >> 5; b < nb; b += 8u) {
    const u32 i = b * 32u + lane();
    const float v = x[i] * s * w[i];
    y[i] = v;
    if (q) quant_wave(v, b, q, nb);
  }
}
/* The RMSNorm op: ceil(n / 256) workgroups. Every workgroup computes the
 * whole sum of squares (the same code, so the same value in every
 * workgroup); then wave v of workgroup g normalizes and quantizes block
 * g * 8 + v only. n is a multiple of 32. */
KERNEL ie_rmsnorm(const G float *x, const G float *w, G float *y, u32 n, float eps, G u8 *q, u32 xs, u32 ys, u32 qs) {
  x = TOKC(x, xs), y = TOK(y, ys), q = TOK(q, qs);
  float ss = 0.0f;
  for (u32 i = tid(); i < n; i += NT) ss += x[i] * x[i];
  ss = wg_sum(ss);
  const float s = 1.0f / __builtin_sqrtf(ss / (float)n + eps);
  const u32 nb = n >> 5, b = wgid() * 8u + (tid() >> 5);
  if (b < nb) { /* whole waves */
    const u32 i = b * 32u + lane();
    const float v = x[i] * s * w[i];
    y[i] = v;
    if (q) quant_wave(v, b, q, nb);
  }
}

/* ------------------------------------------------------------------ GEMV */

/* The fused RMSNorm of a GEMV result y (rows elements): every workgroup
 * makes its rows visible and counts itself; the last one (count[0] reaches
 * ngroups) computes ny = rmsnorm(y) * nw and its Q8 copy nq, then sets the
 * count back to 0 for the next GEMV. Every thread of the workgroup calls it. */
static LDS u32 gm_last;
static inline void gemv_norm_tail(const G float *y, u32 rows, const G float *nw, G float *ny, G u8 *nq, float eps,
                                  G u32 *count) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  /* the number of launched workgroups (the host may add idle ones): the
   * dispatch packet's grid_size_x (byte 12) over the workgroup size */
  const u32 ngroups = ((const __attribute__((address_space(4))) u32 *)__builtin_amdgcn_dispatch_ptr())[3] / NT;
  if (tid() == 0u) gm_last = __atomic_fetch_add(count, 1u, __ATOMIC_ACQ_REL) == ngroups - 1u;
  barrier();
  if (!gm_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  rmsnorm_wg(y, nw, ny, rows, eps, nq);
  if (tid() == 0u) *count = 0u;
}

/* The fused epilogue of a GEMV row (the CPU backend does the same, in the
 * same order): y = acc (+ bias[r]) then (res[r] +) y. */
static inline float epilogue(float acc, u32 r, const G float *bias, const G float *res) {
  float y = acc;
  if (bias) y = y + bias[r];
  if (res) y = res[r] + y;
  return y;
}

/* GEMV_U: blocks per lane per loop step. The loads of the GEMV_U blocks are
 * all issued before the first dot product, so a lane waits for memory once
 * per step, not once per block. A block past nb loads block t (valid) again
 * and adds 0.0f. The terms are added in block order, as before (and as
 * the CPU backend does). */
#ifndef GEMV_U
#define GEMV_U 2u
#endif


/* y = W x: W is Q4 (GPU layout: words qw, f16 scales qs), x is Q8 (as
 * written by ie_quant_q8). Grid: ie_gemv_ngroups(rows) workgroups of 8
 * waves; wave v of workgroup g does row ie_gemv_row(g, v); lane l does the
 * blocks ie_lane_blk(l, t) < nb; then the ie_tree wave reduction. */
KERNEL ie_gemv_q4q8(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                  const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps,
                  G u32 *count) {
  const u32 r = ie_gemv_row(wgid(), tid() >> 5), l = lane();
  if (r < rows) {
    const G u32 *aw = (const G u32 *)xq;
    const G float *da = (const G float *)(xq + 32u * nb);
    const G u32 *asum = (const G u32 *)(xq + 36u * nb);
    float acc = 0.0f;
    for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
      u32x4 w4[GEMV_U], a0[GEMV_U], a1[GEMV_U];
      float sc[GEMV_U];
      u32 as[GEMV_U];
      int ok[GEMV_U];
      for (u32 k = 0; k < GEMV_U; k++) {
        ok[k] = ie_lane_blk(l, t + k) < nb;
        const u32 b = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
        /* one 128-bit load of the 4 nibble words, two of the 8 activation words */
        w4[k] = *(const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb));
        a0[k] = *(const G u32x4 *)(aw + b * 8u);
        a1[k] = *(const G u32x4 *)(aw + b * 8u + 4u);
        sc[k] = (float)qs[ie_q4_dst_scale(r, b, nb)] * da[b];
        as[k] = asum[b];
      }
      for (u32 k = 0; k < GEMV_U; k++) {
        /* no branch: a branch lets the compiler sink the loads into it */
        const u32 wq[4] = {w4[k].x, w4[k].y, w4[k].z, w4[k].w};
        const u32 av[8] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w, a1[k].x, a1[k].y, a1[k].z, a1[k].w};
        const Mem qm = {wq}, am = {av};
        const int dot = (int)ie_q4q8_block(qm, 0u, am, 0u, as[k]);
        acc += ok[k] ? sc[k] * (float)dot : 0.0f; /* + 0.0f leaves acc unchanged */
      }
    }
    acc = wave_tree_f(acc);
    if (l == 0u) y[r] = epilogue(acc, r, bias, res);
  }
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* y = W x: W is Q8_0 (GPU layout: 8 int8 words per block in qw, f16 scales
 * qs), x is Q8. The same grid, lane split and reduction as ie_gemv_q4q8;
 * the block dot product is ie_q8q8_block (8 v_dot4_i32_iu8, both signed). */
KERNEL ie_gemv_q8q8(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                  const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps,
                  G u32 *count) {
  const u32 r = ie_gemv_row(wgid(), tid() >> 5), l = lane();
  if (r < rows) {
    const G u32 *aw = (const G u32 *)xq;
    const G float *da = (const G float *)(xq + 32u * nb);
    float acc = 0.0f;
    for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
      u32x4 w0[GEMV_U], w1[GEMV_U], a0[GEMV_U], a1[GEMV_U];
      float sc[GEMV_U];
      int ok[GEMV_U];
      for (u32 k = 0; k < GEMV_U; k++) {
        ok[k] = ie_lane_blk(l, t + k) < nb;
        const u32 b = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
        /* two 128-bit loads of the 8 weight words, two of the 8 activation words */
        const G u32 *wp = qw + ie_q8_dst_word(r, b, 0u, nb);
        w0[k] = *(const G u32x4 *)wp;
        w1[k] = *(const G u32x4 *)(wp + 4u);
        a0[k] = *(const G u32x4 *)(aw + b * 8u);
        a1[k] = *(const G u32x4 *)(aw + b * 8u + 4u);
        sc[k] = (float)qs[ie_q8_dst_scale(r, b, nb)] * da[b];
      }
      for (u32 k = 0; k < GEMV_U; k++) {
        const u32 wq[8] = {w0[k].x, w0[k].y, w0[k].z, w0[k].w, w1[k].x, w1[k].y, w1[k].z, w1[k].w};
        const u32 av[8] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w, a1[k].x, a1[k].y, a1[k].z, a1[k].w};
        const Mem qm = {wq}, am = {av};
        const int dot = (int)ie_q8q8_block(qm, 0u, am, 0u);
        acc += ok[k] ? sc[k] * (float)dot : 0.0f;
      }
    }
    acc = wave_tree_f(acc);
    if (l == 0u) y[r] = epilogue(acc, r, bias, res);
  }
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* The same GEMV for T <= 16 tokens (speculative decoding: verify several
 * tokens in one pass; prompt processing in chunks). Each weight block is loaded once and used for every
 * token. For each token the arithmetic and its order are those of
 * ie_gemv_q8q8, so token t's result is bitwise the result of a one-token
 * launch. xs: bytes between the Q8 activations of two tokens; ys: floats
 * between the outputs (and the residuals) of two tokens. */
#define GEMV_T 16u
KERNEL ie_gemv_q8q8_t(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                      const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  const u32 r = ie_gemv_row(wgid(), tid() >> 5), l = lane();
  if (r >= rows) return;
  float acc[GEMV_T];
  for (u32 j = 0; j < GEMV_T; j++) acc[j] = 0.0f;
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
    u32x4 w0[GEMV_U], w1[GEMV_U];
    float sw[GEMV_U];
    int ok[GEMV_U];
    u32 bk[GEMV_U];
    for (u32 k = 0; k < GEMV_U; k++) {
      ok[k] = ie_lane_blk(l, t + k) < nb;
      const u32 b = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
      bk[k] = b;
      const G u32 *wp = qw + ie_q8_dst_word(r, b, 0u, nb);
      w0[k] = *(const G u32x4 *)wp;
      w1[k] = *(const G u32x4 *)(wp + 4u);
      sw[k] = (float)qs[ie_q8_dst_scale(r, b, nb)];
    }
    for (u32 j = 0; j < GEMV_T; j++) {
      if (j >= T) break; /* uniform */
      const G u8 *xj = xq + j * xs;
      const G u32 *aw = (const G u32 *)xj;
      const G float *da = (const G float *)(xj + 32u * nb);
      u32x4 a0[GEMV_U], a1[GEMV_U];
      float sc[GEMV_U];
      for (u32 k = 0; k < GEMV_U; k++) {
        a0[k] = *(const G u32x4 *)(aw + bk[k] * 8u);
        a1[k] = *(const G u32x4 *)(aw + bk[k] * 8u + 4u);
        sc[k] = sw[k] * da[bk[k]];
      }
      for (u32 k = 0; k < GEMV_U; k++) {
        const u32 wq[8] = {w0[k].x, w0[k].y, w0[k].z, w0[k].w, w1[k].x, w1[k].y, w1[k].z, w1[k].w};
        const u32 av[8] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w, a1[k].x, a1[k].y, a1[k].z, a1[k].w};
        const Mem qm = {wq}, am = {av};
        const int dot = (int)ie_q8q8_block(qm, 0u, am, 0u);
        acc[j] += ok[k] ? sc[k] * (float)dot : 0.0f;
      }
    }
  }
  for (u32 j = 0; j < GEMV_T; j++) {
    if (j >= T) break;
    const float a = wave_tree_f(acc[j]);
    if (l == 0u) y[j * ys + r] = epilogue(a, r, bias, res ? res + j * rs : res);
  }
}

/* The multi-token GEMV for many tokens (prompt chunks): wave v of
 * workgroup g does the 4 rows 4 (8 g + v) .. + 3 (32 rows per workgroup),
 * so each activation load serves 4 rows (4 times less cache traffic than
 * ie_gemv_q8q8_t). For each (row, token) the arithmetic and its order are
 * those of ie_gemv_q8q8: the same lane blocks, the same block dot product,
 * the same reduction. */
#ifndef GEMV_R
#define GEMV_R 4u
#endif
KERNEL ie_gemv_q8q8_tr(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                       const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
#pragma clang fp contract(off) /* no fused multiply-add: the same rounding as ie_gemv_q8q8 */
  const u32 r0 = (wgid() * 8u + (tid() >> 5)) * GEMV_R, l = lane();
  if (r0 >= rows) return;
  float acc[GEMV_R][GEMV_T];
#pragma unroll
  for (u32 q = 0; q < GEMV_R; q++)
#pragma unroll
    for (u32 j = 0; j < GEMV_T; j++) acc[q][j] = 0.0f;
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
    const u32 b = ie_lane_blk(l, t);
    u32x4 w0[GEMV_R], w1[GEMV_R];
    float sw[GEMV_R];
#pragma unroll
    for (u32 q = 0; q < GEMV_R; q++) {
      const u32 r = r0 + q < rows ? r0 + q : r0; /* a row past the end: row r0 again, not stored */
      const G u32 *wp = qw + ie_q8_dst_word(r, b, 0u, nb);
      w0[q] = *(const G u32x4 *)wp;
      w1[q] = *(const G u32x4 *)(wp + 4u);
      sw[q] = (float)qs[ie_q8_dst_scale(r, b, nb)];
    }
#pragma unroll
    for (u32 j = 0; j < GEMV_T; j++) {
      if (j < T) { /* uniform */
        const G u8 *xj = xq + j * xs;
        const G u32 *aw = (const G u32 *)xj;
        const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u);
        const float da = ((const G float *)(xj + 32u * nb))[b];
        const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (u32 q = 0; q < GEMV_R; q++) {
          const u32 wq[8] = {w0[q].x, w0[q].y, w0[q].z, w0[q].w, w1[q].x, w1[q].y, w1[q].z, w1[q].w};
          const Mem qm = {wq}, am = {av};
          const int dot = (int)ie_q8q8_block(qm, 0u, am, 0u);
          const float sc = sw[q] * da;
          acc[q][j] = acc[q][j] + sc * (float)dot;
        }
      }
    }
  }
#pragma unroll
  for (u32 q = 0; q < GEMV_R; q++) {
    const u32 r = r0 + q;
#pragma unroll
    for (u32 j = 0; j < GEMV_T; j++) {
      if (j < T) {
        const float v = wave_tree_f(acc[q][j]);
        if (l == 0u && r < rows) y[j * ys + r] = epilogue(v, r, bias, res ? res + j * rs : res);
      }
    }
  }
}

/* y = W x for an f32 matrix (F16 weights dequantized at load):
 * the same grid; lane l does the elements ie_lane_blk(l, t) < cols. */
KERNEL ie_gemv_f32(const G float *w, const G float *x, G float *y, u32 rows, u32 cols, const G float *bias,
                   const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps, G u32 *count) {
  const u32 r = ie_gemv_row(wgid(), tid() >> 5), l = lane();
  if (r < rows) {
  const G float *row = w + (unsigned long)r * cols;
  float acc = 0.0f;
  for (u32 t = 0; ie_lane_blk(l, t) < cols; t++) acc += row[ie_lane_blk(l, t)] * x[ie_lane_blk(l, t)];
  acc = wave_tree_f(acc);
  if (l == 0u) y[r] = epilogue(acc, r, bias, res);
  }
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* ---------------------------------------------------------- elementwise */

KERNEL ie_bias(G float *a, const G float *w, u32 n) {
  const u32 i = wgid() * NT + tid();
  if (i < n) a[i] += w[i];
}

KERNEL ie_add(const G float *a, const G float *b, G float *c, u32 n) {
  const u32 i = wgid() * NT + tid();
  if (i < n) c[i] = a[i] + b[i];
}

/* a = silu(a) * b */
/* a = silu(a) * b. n is a multiple of 32, so a wave is one whole block (or
 * none). If q is not NULL, also write the Q8 copy of the result. */
KERNEL ie_swiglu(G float *a, const G float *b, u32 n, G u8 *q, u32 as, u32 qs) {
  a = TOK(a, as), b = TOKC(b, as), q = TOK(q, qs);
  const u32 i = wgid() * NT + tid();
  if (i >= n) return; /* whole waves leave together */
  const float g = a[i];
  const float v = g / (1.0f + __builtin_expf(-g)) * b[i];
  a[i] = v;
  if (q) quant_wave(v, i >> 5, q, n >> 5);
}

/* RoPE in place: nh heads of hd; cs/sn: the tables of this position.
 * neox = 0: pairs (2i, 2i+1) (llama); 1: pairs (i, i + hd/2) (qwen2). */
KERNEL ie_rope(G float *x, const G float *cs, const G float *sn, u32 nh, u32 hd, u32 neox) {
  const u32 h2 = hd >> 1, k = wgid() * NT + tid();
  if (k >= nh * h2) return;
  const u32 h = k / h2, i = k % h2;
  G float *v = x + h * hd;
  const u32 i0 = neox ? i : 2u * i, i1 = neox ? i + h2 : 2u * i + 1u;
  const float a = v[i0], b = v[i1];
  v[i0] = a * cs[i] - b * sn[i];
  v[i1] = a * sn[i] + b * cs[i];
}

/* RoPE of q (nq heads) and k (nkv heads) in place, then k and v into the
 * caches kc/vc (already offset to this layer and position). Threads
 * 0 .. (nq+nkv)*h2 - 1 rotate one pair each (a k pair also writes kc); the
 * next kvd threads copy v. */
KERNEL ie_rope_kv(G float *q, G float *k, const G float *v, G float *kc, G float *vc, const G float *cs,
                  const G float *sn, u32 nq, u32 nkv, u32 hd, u32 neox) {
  const u32 h2 = hd >> 1, t = wgid() * NT + tid(), nrot = (nq + nkv) * h2;
  if (t < nrot) {
    const int is_k = t >= nq * h2;
    const u32 u = is_k ? t - nq * h2 : t, h = u / h2, i = u % h2;
    G float *x = (is_k ? k : q) + h * hd;
    const u32 i0 = neox ? i : 2u * i, i1 = neox ? i + h2 : 2u * i + 1u;
    const float a = x[i0], b = x[i1];
    const float r0 = a * cs[i] - b * sn[i], r1 = a * sn[i] + b * cs[i];
    x[i0] = r0;
    x[i1] = r1;
    if (is_k) {
      kc[h * hd + i0] = r0;
      kc[h * hd + i1] = r1;
    }
  } else if (t < nrot + nkv * hd) {
    vc[t - nrot] = v[t - nrot];
  }
}

/* Store k and v (n floats each) into the caches at kc/vc (already offset to
 * the row of this layer and position). */
KERNEL ie_kv_store(const G float *k, const G float *v, G float *kc, G float *vc, u32 n) {
  const u32 i = wgid() * NT + tid();
  if (i < n) {
    kc[i] = k[i];
    vc[i] = v[i];
  }
}

/* --------------------------------------------------------------- attention */

/* One decode token. One workgroup per query head h (GQA: KV head h / grp).
 * kc, vc: the caches of this layer, [n_ctx][n_kv * hd]. sc: scores
 * [n_head][n_ctx]. Positions 0..pos. */
static inline float sigmoidf(float x) { return 1.0f / (1.0f + __builtin_expf(-x)); }

KERNEL ie_attn(const G float *q, const G float *kc, const G float *vc, G float *sc, G float *out, u32 pos,
               u32 n_ctx, u32 hd, u32 n_head, u32 n_kv, G u8 *oq, const G float *gate, u32 gstride) {
  const u32 h = wgid(), kh = h / (n_head / n_kv), kvd = n_kv * hd;
  const G float *qh = q + h * hd;
  G float *s = sc + h * n_ctx;
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  float mx = -__builtin_inff();
  for (u32 p = tid(); p <= pos; p += NT) {
    const G float *k = kc + (unsigned long)p * kvd + kh * hd;
    float d = 0.0f;
    for (u32 i = 0; i < hd; i++) d += qh[i] * k[i];
    d *= scale;
    s[p] = d;
    mx = __builtin_fmaxf(mx, d);
  }
  mx = wg_max(mx);
  float sum = 0.0f;
  for (u32 p = tid(); p <= pos; p += NT) {
    const float e = __builtin_expf(s[p] - mx);
    s[p] = e;
    sum += e;
  }
  sum = wg_sum(sum); /* has a barrier: the scores are visible to all threads */
  const float inv = 1.0f / sum;
  for (u32 i = tid(); i < hd; i += NT) {
    float acc = 0.0f;
    for (u32 p = 0; p <= pos; p++) acc += s[p] * vc[(unsigned long)p * kvd + kh * hd + i];
    float o = acc * inv;
    if (gate) o = o * sigmoidf(gate[h * gstride + i]);
    out[h * hd + i] = o;
    /* fused quantize: only when hd is a multiple of 32 and hd <= 256 (the
     * graph checks it), so the waves with i < hd hold whole blocks */
    if (oq) quant_wave(o, (h * hd + i) >> 5, oq, (n_head * hd) >> 5);
  }
}

/* Attention for one decode token, split over positions (flash decoding).
 * Workgroup g = h * nsplit + s: query head h (KV head h / (n_head / n_kv)),
 * positions s*ch .. min(s*ch + ch, pos + 1) - 1. Wave w does the positions
 * s*ch + w, + 8, ...: the 32 lanes compute q.k (lane l holds the elements
 * l, l + 32, ... of hd), then an online softmax update of (m, l, o[hd]).
 * The 8 waves merge in LDS and the workgroup writes (m, l, o) to part. The
 * last workgroup of head h (atomic count[h]) merges the nsplit partial
 * results, writes out = o / l and, if oq, its Q8 copy (hd % 32 == 0), and
 * sets count[h] back to 0 for the next layer. hd <= 256. */
static LDS float at_m[8], at_l[8];
static LDS float at_o[8][256];
static LDS u32 at_last;
/* the weights of the splits in the merge (nsplit <= IE_ATT_MAX_SPLIT) */
static LDS float at_w[2048];
KERNEL ie_attn_split(const G float *q, const G float *kc, const G float *vc, G float *part, G float *out, u32 pos,
                     u32 hd, u32 n_head, u32 n_kv, u32 nsplit, u32 ch, G u32 *count, G u8 *oq, const G float *gate,
                     u32 gstride) {
  const u32 h = wgid() / nsplit, s = wgid() % nsplit, kh = h / (n_head / n_kv), kvd = n_kv * hd;
  const u32 w = tid() >> 5, l = lane();
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  float qv[8], acc[8];
  for (u32 j = 0; j < 8u; j++) {
    const u32 i = l + 32u * j;
    qv[j] = i < hd ? q[h * hd + i] : 0.0f;
    acc[j] = 0.0f;
  }
  float m = -__builtin_inff(), sum = 0.0f;
  const u32 p0 = s * ch, p1 = p0 + ch < pos + 1u ? p0 + ch : pos + 1u;
  for (u32 p = p0 + w; p < p1; p += 8u) {
    const G float *k = kc + (unsigned long)p * kvd + kh * hd;
    const G float *v = vc + (unsigned long)p * kvd + kh * hd;
    float d = 0.0f;
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) d += qv[j] * k[l + 32u * j];
    d = wave_sum_all(d);
    d *= scale;
    const float mn = __builtin_fmaxf(m, d), c = __builtin_expf(m - mn), e = __builtin_expf(d - mn);
    sum = sum * c + e;
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) acc[j] = acc[j] * c + e * v[l + 32u * j];
    m = mn;
  }
  /* merge the 8 waves (a wave without positions has m = -inf, sum = 0) */
  if (l == 0u) at_m[w] = m, at_l[w] = sum;
  for (u32 j = 0; j < 8u; j++)
    if (l + 32u * j < hd) at_o[w][l + 32u * j] = acc[j];
  barrier();
  float M = at_m[0];
  for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, at_m[x]);
  G float *pp = part + (unsigned long)(h * nsplit + s) * (hd + 2u);
  if (tid() < hd) {
    float o = 0.0f;
    for (u32 x = 0; x < 8u; x++) o += at_o[x][tid()] * __builtin_expf(at_m[x] - M);
    pp[2u + tid()] = o;
  }
  if (tid() == 0u) {
    float L = 0.0f;
    for (u32 x = 0; x < 8u; x++) L += at_l[x] * __builtin_expf(at_m[x] - M);
    pp[0] = M;
    pp[1] = L;
  }
  /* every wave makes its stores visible to the device before the count */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (tid() == 0u) at_last = __atomic_fetch_add(&count[h], 1u, __ATOMIC_ACQ_REL) == nsplit - 1u;
  barrier();
  if (!at_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  /* merge: the whole workgroup finds M2 = max m_t and L2 = sum l_t w_t, with
   * the weight w_t = exp(m_t - M2) of split t computed once, in LDS */
  const G float *ph = part + (unsigned long)h * nsplit * (hd + 2u);
  float mv = -__builtin_inff();
  for (u32 t = tid(); t < nsplit; t += NT) mv = __builtin_fmaxf(mv, ph[t * (hd + 2u)]);
  const float M2 = wg_max(mv);
  float lv = 0.0f;
  for (u32 t = tid(); t < nsplit; t += NT) {
    const float wt = __builtin_expf(ph[t * (hd + 2u)] - M2);
    at_w[t] = wt;
    lv += ph[t * (hd + 2u) + 1u] * wt;
  }
  const float L2 = wg_sum(lv); /* its barrier also makes at_w visible */
  if (tid() < hd) { /* hd % 32 == 0 when oq is set: whole waves */
    float o = 0.0f;
    for (u32 t = 0; t < nsplit; t++) o += ph[t * (hd + 2u) + 2u + tid()] * at_w[t];
    o = o / L2;
    if (gate) o = o * sigmoidf(gate[h * gstride + tid()]); /* qwen35: the output gate */
    out[h * hd + tid()] = o;
    if (oq) quant_wave(o, (h * hd + tid()) >> 5, oq, (n_head * hd) >> 5);
  }
  if (tid() == 0u) count[h] = 0u;
}

/* ----------------------------------------------------------------- qwen35 */

/* Full attention, before the attention: workgroup g < nq: query head g, at
 * qkv + g * 2 hd (q of [q | gate]); nq <= g < nq + nkv: key head g - nq,
 * at qkv + 2 nq hd + (g - nq) hd; then nkv workgroups copy the value heads.
 * A q or k head: y = rmsnorm(x) * w (hd <= 256 values, thread t holds value
 * t), then RoPE of the first nrot values (pairs (i, i + nrot/2)); q goes to
 * qo + g * hd, k to the cache kc. */
static LDS float qk_buf[256];
KERNEL ie_qkn_rope_kv(const G float *qkv, G float *qo, G float *kc, G float *vc, const G float *qw, const G float *kw,
                      const G float *cs, const G float *sn, u32 nq, u32 nkv, u32 hd, u32 nrot, float eps, u32 ins,
                      u32 qos) {
  /* token y of a multi-token launch: position + y (KV rows and RoPE tables) */
  qkv = TOKC(qkv, ins), qo = TOK(qo, qos);
  kc = TOK(kc, nkv * hd * 4u), vc = TOK(vc, nkv * hd * 4u);
  cs = TOKC(cs, (nrot >> 1) * 4u), sn = TOKC(sn, (nrot >> 1) * 4u);
  const u32 g = wgid(), t = tid();
  if (g >= nq + nkv) { /* value head */
    const u32 h = g - nq - nkv;
    if (t < hd) vc[h * hd + t] = qkv[2u * nq * hd + nkv * hd + h * hd + t];
    return;
  }
  const int is_k = g >= nq;
  const u32 h = is_k ? g - nq : g;
  const G float *src = is_k ? qkv + 2u * nq * hd + h * hd : qkv + h * 2u * hd;
  const G float *w = is_k ? kw : qw;
  const float x = t < hd ? src[t] : 0.0f;
  const float ss = wg_sum(x * x);
  const float y = x * (1.0f / __builtin_sqrtf(ss / (float)hd + eps)) * (t < hd ? w[t] : 0.0f);
  if (t < hd) qk_buf[t] = y;
  barrier();
  float r = y;
  const u32 h2 = nrot >> 1;
  if (t < nrot) {
    const u32 i = t < h2 ? t : t - h2;
    const float a = qk_buf[i], b = qk_buf[i + h2];
    r = t < h2 ? a * cs[i] - b * sn[i] : a * sn[i] + b * cs[i];
  }
  if (t < hd) (is_k ? kc + h * hd : qo + h * hd)[t] = r;
}

/* Linear attention (Gated DeltaNet) of T <= 16 tokens at positions pos ..
 * pos + T - 1, one after the other; one workgroup per value head h (key
 * head kh = h % nk), state dims 128 x 128. Token t: in + t * is = [q k v (cd)
 * | z (inner) | beta (nv) | alpha (nv)], inner = nv * 128; output at out +
 * t * os, its Q8 copy at oq + t * qs. ring: the conv inputs of the last 32
 * positions [slot][cd] (slot = position % 32). S: GDN_SLOTS (4) state slots [slot][h][i][j]
 * (i: key dim, j: value dim); the state is read from slot c and the state
 * after token t is written to slot (c + t) % 4 (only for t >= wfrom: a prompt chunk keeps only its last state), so that a speculative step
 * can go back to the state after any of its tokens. Each thread reads and
 * writes only its own state values, so slot c can be overwritten. Thread t:
 * value dim j = t % 128, key dims i of half t / 128. The CPU backend (cpu.c,
 * gdn) does the same arithmetic for one token (slot c = 0). */
#define GDN_SLOTS 4u
#define GDN_RING 32u
static LDS float gd_q[128], gd_k[128], gd_v[128], gd_p[2][128], gd_r[8];
/* conv input of channel c at token u of this launch, or (before the launch)
 * from the ring; 0 before position 0 */
static inline float conv_in(const G float *in, u32 is, const G float *ring, u32 c, u32 t, u32 d, u32 pos, u32 cd) {
  if (d <= t) return in[(t - d) * is + c];
  const u32 p = pos + t;
  return p >= d ? ring[((p - d) & (GDN_RING - 1u)) * cd + c] : 0.0f;
}
static inline float conv_ch(const G float *in, u32 is, G float *ring, const G float *cw, u32 c, u32 t, u32 pos, u32 cd,
                            int writer) {
  float acc = 0.0f;
  for (u32 j = 0; j < 4u; j++) /* oldest first, as ggml_ssm_conv */
    acc += conv_in(in, is, ring, c, t, 3u - j, pos, cd) * cw[c * 4u + j];
  /* slot (pos + t) % GDN_RING: no workgroup reads it in this launch (reads
   * of the ring are of positions pos - 3 .. pos - 1, writes of pos .. pos +
   * 15: 19 slots < GDN_RING) */
  if (writer) ring[((pos + t) & (GDN_RING - 1u)) * cd + c] = in[t * is + c];
  return acc / (1.0f + __builtin_expf(-acc));
}
KERNEL ie_gdn(const G float *in, G float *out, G u8 *oq, G float *ring, G float *S, const G float *cw, const G float *dtb,
              const G float *sa, const G float *nw, u32 pos, u32 cd, u32 nk, u32 nv, float eps, u32 T, u32 is, u32 os,
              u32 qs, u32 c, u32 wfrom) {
  const u32 h = wgid(), t0 = tid(), j = t0 & 127u, half = t0 >> 7, kh = h % nk, w = t0 >> 5, inner = nv * 128u;
  const unsigned long slot = (unsigned long)nv * 16384u, own = (unsigned long)h * 16384u + half * 64u * 128u + j;
  /* the state loads first: they do not depend on the conv, so their
   * memory latency overlaps it */
  float st[64];
  for (u32 ii = 0; ii < 64u; ii++) st[ii] = pos ? S[c * slot + own + ii * 128u] : 0.0f;
  for (u32 t = 0; t < T; t++) {
    const G float *it = in + t * is;
    /* conv + SiLU: threads < 128 do q (channel kh*128 + j) and v (2 nk 128 +
     * h 128 + j); threads >= 128 do k (nk 128 + kh 128 + j). The q and k ring
     * slots are written by the workgroup h == kh only. */
    const float a = conv_ch(in, is, ring, cw, half ? nk * 128u + kh * 128u + j : kh * 128u + j, t, pos, cd, h == kh);
    const float vv = half ? 0.0f : conv_ch(in, is, ring, cw, 2u * nk * 128u + h * 128u + j, t, pos, cd, 1);
    /* L2 norm of q (waves 0-3) and k (waves 4-7): x / sqrt(sum x^2 + eps) */
    const float ss = wave_sum_all(a * a);
    if (lane() == 0u) gd_r[w] = ss;
    barrier();
    const float n2 = half ? gd_r[4] + gd_r[5] + gd_r[6] + gd_r[7] : gd_r[0] + gd_r[1] + gd_r[2] + gd_r[3];
    const float an = a * (1.0f / __builtin_sqrtf(n2 + eps));
    if (half) gd_k[j] = an;
    else gd_q[j] = an, gd_v[j] = vv;
    const float beta = sigmoidf(it[cd + inner + h]);
    const float al = it[cd + inner + nv + h] + dtb[h];
    const float sp = al > 20.0f ? al : __builtin_logf(1.0f + __builtin_expf(al));
    const float decay = __builtin_expf(sp * sa[h]);
    barrier();
    float pk = 0.0f;
    for (u32 ii = 0; ii < 64u; ii++) {
      st[ii] = st[ii] * decay;
      pk += st[ii] * gd_k[half * 64u + ii];
    }
    gd_p[half][j] = pk;
    barrier();
    const float delta = (gd_v[j] - (gd_p[0][j] + gd_p[1][j])) * beta;
    barrier();
    float po = 0.0f;
    G float *Sw = S + ((c + t) % GDN_SLOTS) * slot + own;
    for (u32 ii = 0; ii < 64u; ii++) {
      st[ii] += gd_k[half * 64u + ii] * delta;
      if (t >= wfrom) Sw[ii * 128u] = st[ii]; /* uniform */
      po += st[ii] * gd_q[half * 64u + ii];
    }
    gd_p[half][j] = po;
    barrier();
    const float o = (gd_p[0][j] + gd_p[1][j]) * (1.0f / __builtin_sqrtf(128.0f));
    /* gated RMSNorm over the 128 values of the head (waves 0-3) */
    const float s2 = wave_sum_all(o * o);
    if (lane() == 0u) gd_r[w] = s2;
    barrier();
    if (!half) { /* whole waves */
      const float n3 = gd_r[0] + gd_r[1] + gd_r[2] + gd_r[3];
      const float z = it[cd + h * 128u + j];
      const float y = o * (1.0f / __builtin_sqrtf(n3 / 128.0f + eps)) * nw[j] * (z / (1.0f + __builtin_expf(-z)));
      out[t * os + h * 128u + j] = y;
      if (oq) quant_wave(y, (h * 128u + j) >> 5, oq + t * qs, inner >> 5);
    }
    barrier(); /* the LDS of this token is read before the next token writes it */
  }
}

/* ------------------------------------------------------------------ argmax */

/* out[0] = the first index of the largest x. ng workgroups: workgroup g
 * scans i = g*NT + tid, step ng*NT, and writes its best (value, index) to
 * part[g]. The last workgroup to finish (atomic counter) combines the ng
 * partial results and sets the counter back to 0 for the next step.
 * Ties: the smaller index wins, as in the CPU backend. */
static LDS float am_v[NT];
static LDS u32 am_i[NT];
static LDS u32 am_last;
static inline int better(float v, u32 i, float bv, u32 bi) {
  return i != 0xFFFFFFFFu && (bi == 0xFFFFFFFFu || v > bv || (v == bv && i < bi));
}
static inline void wg_best(float *bv, u32 *bi) {
  am_v[tid()] = *bv;
  am_i[tid()] = *bi;
  barrier();
  for (u32 off = NT / 2u; off > 0u; off >>= 1) {
    if (tid() < off && better(am_v[tid() + off], am_i[tid() + off], am_v[tid()], am_i[tid()])) {
      am_v[tid()] = am_v[tid() + off];
      am_i[tid()] = am_i[tid() + off];
    }
    barrier();
  }
  *bv = am_v[0];
  *bi = am_i[0];
  barrier();
}
KERNEL ie_argmax(const G float *x, u32 n, G u32 *out, G u32 *part, G u32 *count, u32 ng) {
  float bv = -__builtin_inff();
  u32 bi = 0xFFFFFFFFu;
  for (u32 i = wgid() * NT + tid(); i < n; i += ng * NT)
    if (better(x[i], i, bv, bi)) bv = x[i], bi = i;
  wg_best(&bv, &bi);
  if (tid() == 0u) {
    part[2u * wgid()] = __builtin_bit_cast(u32, bv);
    part[2u * wgid() + 1u] = bi;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    am_last = __atomic_fetch_add(count, 1u, __ATOMIC_ACQ_REL) == ng - 1u;
  }
  barrier();
  if (!am_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  bv = -__builtin_inff(), bi = 0xFFFFFFFFu;
  for (u32 g = tid(); g < ng; g += NT) {
    const float v = __builtin_bit_cast(float, __atomic_load_n(&part[2u * g], __ATOMIC_RELAXED));
    const u32 i = __atomic_load_n(&part[2u * g + 1u], __ATOMIC_RELAXED);
    if (better(v, i, bv, bi)) bv = v, bi = i;
  }
  wg_best(&bv, &bi);
  if (tid() == 0u) {
    out[0] = bi;
    *count = 0u;
  }
}
