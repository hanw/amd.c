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
#define LDSP __attribute__((address_space(3))) /* LDS pointer targets */
#define KERNEL __attribute__((amdgpu_kernel, visibility("protected"))) void
#define NT 256u

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef unsigned char u8;
typedef unsigned short u16;
typedef signed char i8;
typedef _Float16 f16;
/* The KV cache element: fp16 (half the memory of f32: 128K positions of
 * Qwen3.8-27B in 9.1 GB), or f32 with -DIE_KV_F32. The attention kernels
 * read it as f32 (exact), so only the store rounds. The host finds the type
 * by the kernel ie_kv_f16 (present only in the fp16 build). */
#ifdef IE_KV_F32
typedef float kvt;
typedef float kv4 __attribute__((ext_vector_type(4)));
#else
typedef _Float16 kvt;
typedef _Float16 kv4 __attribute__((ext_vector_type(4)));
#endif
#define KV4(p, i) __builtin_convertvector(((const G kv4 *)(p))[i], f32x4) /* 4 elements as f32 */

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

/* Workgroup barrier for LDS only: global loads still in flight stay in
 * flight (barrier() waits for them and invalidates the L0 cache). For
 * kernels whose threads do not exchange data through global memory. */
static inline void lbarrier(void) {
  __asm__ volatile("s_wait_dscnt 0x0" ::: "memory");
  __builtin_amdgcn_s_barrier();
  __asm__ volatile("" ::: "memory");
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
  u32 i0 = tid();
  for (; i0 + 7u * NT < n; i0 += 8u * NT) { /* 8 loads in flight; the same order of the sums */
    float v[8];
#pragma unroll
    for (u32 k = 0; k < 8u; k++) v[k] = x[i0 + k * NT];
#pragma unroll
    for (u32 k = 0; k < 8u; k++) ss += v[k] * v[k];
  }
  for (; i0 < n; i0 += NT) ss += x[i0] * x[i0];
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

/* y = W x: W is MAT_Q4K (GGUF Q4_K repacked, src/model.h: the nibbles as
 * Q4_0 blocks, values 0..15; qs: (scale | min << 8) per sub-block of 32,
 * then (f16 d, f16 dmin) per super-block of 256), x is Q8. The same grid,
 * lanes and reduction as ie_gemv_q4q8. Per sub-block b:
 *   dq = sum q a = ie_q4k_dot (law q4k_dot),
 *   acc += da * ((d sc) dq - (dmin mn) asum)   (no fused multiply-add).
 * ie_gemv_q4kq8_tr does the same arithmetic in the same order per token. */
KERNEL ie_gemv_q4kq8(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                     const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps,
                     G u32 *count) {
#pragma clang fp contract(off)
  const u32 r = ie_gemv_row(wgid(), tid() >> 5), l = lane();
  if (r < rows) {
    const G u32 *aw = (const G u32 *)xq;
    const G float *da = (const G float *)(xq + 32u * nb);
    const G u32 *asum = (const G u32 *)(xq + 36u * nb);
    const G u16 *smr = qs + ie_q4k_sm(r, 0u, nb);
    const G f16 *ddr = (const G f16 *)qs + ie_q4k_dd(rows, r, 0u, nb);
    float acc = 0.0f;
    for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
      u32x4 w4[GEMV_U], a0[GEMV_U], a1[GEMV_U];
      u32 bk[GEMV_U], sm[GEMV_U], as[GEMV_U];
      float dv[GEMV_U], mv[GEMV_U], dav[GEMV_U];
      int ok[GEMV_U];
      for (u32 k = 0; k < GEMV_U; k++) { /* all loads of the step first */
        ok[k] = ie_lane_blk(l, t + k) < nb;
        bk[k] = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
        w4[k] = *(const G u32x4 *)(qw + ie_q4_dst_word(r, bk[k], 0u, nb));
        a0[k] = *(const G u32x4 *)(aw + bk[k] * 8u);
        a1[k] = *(const G u32x4 *)(aw + bk[k] * 8u + 4u);
        sm[k] = smr[bk[k]];
        dv[k] = (float)ddr[(bk[k] >> 3) * 2u], mv[k] = (float)ddr[(bk[k] >> 3) * 2u + 1u];
        as[k] = asum[bk[k]], dav[k] = da[bk[k]];
      }
      for (u32 k = 0; k < GEMV_U; k++) {
        const u32 wq[4] = {w4[k].x, w4[k].y, w4[k].z, w4[k].w};
        const u32 av[8] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w, a1[k].x, a1[k].y, a1[k].z, a1[k].w};
        const Mem qm = {wq}, am = {av};
        const int dq = (int)ie_q4k_dot(qm, 0u, am, 0u); /* sum q a (law q4k_dot) */
        /* the arithmetic of q4k_term */
        const float sc = dv[k] * (float)(sm[k] & 63u), mm = mv[k] * (float)(sm[k] >> 8);
        /* acc = fma(sc dq, da, acc); acc = fma(-mm, da asum, acc) (explicit fma: the same in every Q4K kernel) */
        const float x = dav[k] * (float)(int)as[k];
        const float a1 = __builtin_fmaf(sc * (float)dq, dav[k], acc), a2 = __builtin_fmaf(-mm, x, a1);
        acc = ok[k] ? a2 : acc;
      }
    }
    acc = wave_tree_f(acc);
    if (l == 0u) y[r] = epilogue(acc, r, bias, res);
  }
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* MAT_Q4K for T <= 16 tokens: 4 rows per wave (as ie_gemv_q8q8_tr); per
 * (row, token) the arithmetic and order of ie_gemv_q4kq8 (bitwise equal). */
#define Q4K_TR_BODY(QR, QT, QU) \
_Pragma("clang fp contract(off)") \
  const u32 r0 = (wgid() * 8u + (tid() >> 5)) * QR, l = lane(); \
  if (r0 >= rows) return; \
  float acc[QR][QT]; \
_Pragma("unroll") \
  for (u32 q = 0; q < QR; q++) \
_Pragma("unroll") \
    for (u32 j = 0; j < QT; j++) acc[q][j] = 0.0f; \
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t += QU) { \
_Pragma("unroll") \
  for (u32 u = 0; u < QU; u++) { /* QU blocks: the loads of all go out first */ \
    const int okb = ie_lane_blk(l, t + u) < nb; \
    const u32 b = okb ? ie_lane_blk(l, t + u) : ie_lane_blk(l, t); \
    /* per row, once per block: the 32 weights as int8 words (weight j = \
     * byte j % 4 of word j / 4, as the activations) and the two scales */ \
    u32 wv[QR][8]; \
    float sc[QR], mm[QR]; \
    if (!okb) continue; /* past nb (only some lanes, at the end) */ \
_Pragma("unroll") \
    for (u32 q = 0; q < QR; q++) { \
      const u32 r = r0 + q < rows ? r0 + q : r0; /* a row past the end: row r0 again, not stored */ \
      const u32x4 w4 = __builtin_nontemporal_load((const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb))); \
      const u32 x4[4] = {w4.x, w4.y, w4.z, w4.w}; \
_Pragma("unroll") \
      for (u32 k = 0; k < 4u; k++) wv[q][2u * k] = ie_q4k_lo(x4[k]), wv[q][2u * k + 1u] = ie_q4k_hi(x4[k]); /* once per block */ \
      const u32 sm = qs[ie_q4k_sm(r, b, nb)]; \
      const u32 dd = ie_q4k_dd(rows, r, b, nb); \
      sc[q] = (float)((const G f16 *)qs)[dd] * (float)(sm & 63u); \
      mm[q] = (float)((const G f16 *)qs)[dd + 1u] * (float)(sm >> 8); \
    } \
_Pragma("unroll") \
    for (u32 j = 0; j < QT; j++) { \
      if (j < T) { /* uniform */ \
        const G u8 *xj = xq + j * xs; \
        const G u32 *aw = (const G u32 *)xj; \
        const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u); \
        const float da = ((const G float *)(xj + 32u * nb))[b]; \
        const int as = (int)((const G u32 *)(xj + 36u * nb))[b]; \
        const float xa = da * (float)as; /* per token: shared by the rows */ \
        const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}; \
_Pragma("unroll") \
        for (u32 q = 0; q < QR; q++) { \
          const Mem um = {wv[q]}, am = {av}; \
          const int dq = (int)ie_q4k_dot_u(um, 0u, am, 0u); /* sum q a (laws q4k_dot_u, q4k_dot) */ \
          acc[q][j] = __builtin_fmaf(-mm[q], xa, __builtin_fmaf(sc[q] * (float)dq, da, acc[q][j])); \
        } \
      } \
    } \
  } \
  } \
_Pragma("unroll") \
  for (u32 q = 0; q < QR; q++) { \
    const u32 r = r0 + q; \
_Pragma("unroll") \
    for (u32 j = 0; j < QT; j++) { \
      if (j < T) { \
        const float v = wave_tree_f(acc[q][j]); \
        if (l == 0u && r < rows) y[j * ys + r] = epilogue(v, r, bias, res ? res + j * rs : res); \
      } \
    } \
  }
#define Q4K_TS_BODY(QR, QT, QU, TS) \
_Pragma("clang fp contract(off)") \
  const u32 r0 = (wgid() * (8u / TS) + (tid() >> 5) / TS) * QR, l = lane(); \
  if (r0 >= rows) return; \
  { const u32 j0_ = ((tid() >> 5) % TS) * QT; /* this wave: tokens j0_ .. j0_ + QT - 1 */ \
    xq += j0_ * xs, y += j0_ * ys, res = res ? res + j0_ * rs : res, T = T > j0_ ? T - j0_ : 0u; } \
  if (T == 0u) return; \
  float acc[QR][QT]; \
_Pragma("unroll") \
  for (u32 q = 0; q < QR; q++) \
_Pragma("unroll") \
    for (u32 j = 0; j < QT; j++) acc[q][j] = 0.0f; \
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t += QU) { \
_Pragma("unroll") \
  for (u32 u = 0; u < QU; u++) { /* QU blocks: the loads of all go out first */ \
    const int okb = ie_lane_blk(l, t + u) < nb; \
    const u32 b = okb ? ie_lane_blk(l, t + u) : ie_lane_blk(l, t); \
    /* per row, once per block: the 32 weights as int8 words (weight j = \
     * byte j % 4 of word j / 4, as the activations) and the two scales */ \
    u32 wv[QR][8]; \
    float sc[QR], mm[QR]; \
    if (!okb) continue; /* past nb (only some lanes, at the end) */ \
_Pragma("unroll") \
    for (u32 q = 0; q < QR; q++) { \
      const u32 r = r0 + q < rows ? r0 + q : r0; /* a row past the end: row r0 again, not stored */ \
      const u32x4 w4 = __builtin_nontemporal_load((const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb))); \
      const u32 x4[4] = {w4.x, w4.y, w4.z, w4.w}; \
_Pragma("unroll") \
      for (u32 k = 0; k < 4u; k++) wv[q][2u * k] = ie_q4k_lo(x4[k]), wv[q][2u * k + 1u] = ie_q4k_hi(x4[k]); /* once per block */ \
      const u32 sm = qs[ie_q4k_sm(r, b, nb)]; \
      const u32 dd = ie_q4k_dd(rows, r, b, nb); \
      sc[q] = (float)((const G f16 *)qs)[dd] * (float)(sm & 63u); \
      mm[q] = (float)((const G f16 *)qs)[dd + 1u] * (float)(sm >> 8); \
    } \
_Pragma("unroll") \
    for (u32 j = 0; j < QT; j++) { \
      if (j < T) { /* uniform */ \
        const G u8 *xj = xq + j * xs; \
        const G u32 *aw = (const G u32 *)xj; \
        const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u); \
        const float da = ((const G float *)(xj + 32u * nb))[b]; \
        const int as = (int)((const G u32 *)(xj + 36u * nb))[b]; \
        const float xa = da * (float)as; /* per token: shared by the rows */ \
        const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}; \
_Pragma("unroll") \
        for (u32 q = 0; q < QR; q++) { \
          const Mem um = {wv[q]}, am = {av}; \
          const int dq = (int)ie_q4k_dot_u(um, 0u, am, 0u); /* sum q a (laws q4k_dot_u, q4k_dot) */ \
          acc[q][j] = __builtin_fmaf(-mm[q], xa, __builtin_fmaf(sc[q] * (float)dq, da, acc[q][j])); \
        } \
      } \
    } \
  } \
  } \
_Pragma("unroll") \
  for (u32 q = 0; q < QR; q++) { \
    const u32 r = r0 + q; \
_Pragma("unroll") \
    for (u32 j = 0; j < QT; j++) { \
      if (j < T) { \
        const float v = wave_tree_f(acc[q][j]); \
        if (l == 0u && r < rows) y[j * ys + r] = epilogue(v, r, bias, res ? res + j * rs : res); \
      } \
    } \
  }
/* 6 <= T <= 8 tokens split over 2 waves of the workgroup (4 tokens per
 * wave): 8 rows per wave, 32 rows per workgroup, twice the workgroups of
 * ie_gemv_q4kq8_tr8 (more waves in flight: 6 to 14% faster at T = 8 on the
 * shapes of Qwen3.8-27B). The two waves of a row group read the same
 * weights (the second from the cache). Per (row, token) the order of
 * ie_gemv_q4kq8 (bitwise equal). */
KERNEL ie_gemv_q4kq8_ts2(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                         const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  Q4K_TS_BODY(8u, 4u, 1u, 2u)
}
KERNEL ie_gemv_q4kq8_tr(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                        const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  Q4K_TR_BODY(4u, 16u, 2u)
}
/* T <= 8 (speculative verify): 8 rows per wave, half the activation loads per weight byte */
KERNEL ie_gemv_q4kq8_tr8(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                         const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  Q4K_TR_BODY(8u, 8u, 1u)
}

/* y = W x: W is MAT_Q6K (GGUF Q6_K repacked, src/model.h), x is Q8. The
 * same grid, lanes and reduction as ie_gemv_q4q8. Per sub-block b: the half
 * dot products d0, d1 (ie_q6k_dots, law q6k_dots: the q - 32 bytes, signed dot4), then
 *   acc = fma(fma(d sc1, d1, (d sc0) d0), da, acc)
 * (explicit fma: the same in ie_gemv_q6kq8_tr and the CPU). */
KERNEL ie_gemv_q6kq8(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                     const G float *bias, const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps,
                     G u32 *count) {
#pragma clang fp contract(off)
  const u32 r = ie_gemv_row(wgid(), tid() >> 5), l = lane();
  if (r < rows) {
    const G u32 *aw = (const G u32 *)xq;
    const G float *da = (const G float *)(xq + 32u * nb);
    float acc = 0.0f;
    for (u32 t = 0; ie_lane_blk(l, t) < nb; t += GEMV_U) {
      u32x4 w4[GEMV_U], a0[GEMV_U], a1[GEMV_U];
      u32 h0[GEMV_U], h1[GEMV_U], sc[GEMV_U];
      float dv[GEMV_U], dav[GEMV_U];
      int ok[GEMV_U];
      for (u32 k = 0; k < GEMV_U; k++) { /* all loads of the step first */
        ok[k] = ie_lane_blk(l, t + k) < nb;
        const u32 b = ok[k] ? ie_lane_blk(l, t + k) : ie_lane_blk(l, t);
        w4[k] = *(const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb));
        const u32 hb = ie_q6k_hw(rows, r, b, 0u, nb);
        h0[k] = qw[hb], h1[k] = qw[hb + 1u];
        a0[k] = *(const G u32x4 *)(aw + b * 8u);
        a1[k] = *(const G u32x4 *)(aw + b * 8u + 4u);
        sc[k] = qs[ie_q6k_sc(r, b, nb)];
        dv[k] = (float)((const G f16 *)qs)[ie_q6k_d(rows, r, b, nb)];
        dav[k] = da[b];
      }
      for (u32 k = 0; k < GEMV_U; k++) {
        const u32 wq[4] = {w4[k].x, w4[k].y, w4[k].z, w4[k].w}, hq[2] = {h0[k], h1[k]};
        const u32 av[8] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w, a1[k].x, a1[k].y, a1[k].z, a1[k].w};
        const Mem qm = {wq}, hm = {hq}, am = {av};
        const int d0 = (int)ie_q6k_dots(qm, 0u, hm, 0u, am, 0u, 0u); /* law q6k_dots */
        const int d1 = (int)ie_q6k_dots(qm, 0u, hm, 1u, am, 0u, 1u);
        const float s0 = dv[k] * (float)(int)(signed char)(sc[k] & 255u), s1 = dv[k] * (float)(int)(signed char)(sc[k] >> 8);
        const float a2 = __builtin_fmaf(__builtin_fmaf(s1, (float)d1, s0 * (float)d0), dav[k], acc);
        acc = ok[k] ? a2 : acc;
      }
    }
    acc = wave_tree_f(acc);
    if (l == 0u) y[r] = epilogue(acc, r, bias, res);
  }
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* MAT_Q6K for T <= QT tokens, QR rows per wave (as the Q4K _tr kernels):
 * per (row, token) the arithmetic and order of ie_gemv_q6kq8 (bitwise
 * equal). The q - 32 bytes of a row are common to the tokens: computed once. */
#define Q6K_TR_BODY(QR, QT) \
  _Pragma("clang fp contract(off)") \
  const u32 r0 = (wgid() * 8u + (tid() >> 5)) * QR, l = lane(); \
  if (r0 >= rows) return; \
  float acc[QR][QT]; \
  _Pragma("unroll") for (u32 q = 0; q < QR; q++) \
    _Pragma("unroll") for (u32 j = 0; j < QT; j++) acc[q][j] = 0.0f; \
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) { \
    const u32 b = ie_lane_blk(l, t); \
    u32 sv[QR][2][4]; /* the q - 32 words of each half, made once per block */ \
    float s0[QR], s1[QR]; \
    _Pragma("unroll") for (u32 q = 0; q < QR; q++) { \
      const u32 r = r0 + q < rows ? r0 + q : r0; /* a row past the end: row r0 again, not stored */ \
      const u32x4 w4 = __builtin_nontemporal_load((const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb))); \
      const u32 x4[4] = {w4.x, w4.y, w4.z, w4.w}; \
      const u32 hb = ie_q6k_hw(rows, r, b, 0u, nb); \
      const u32 hq[2] = {__builtin_nontemporal_load(qw + hb), __builtin_nontemporal_load(qw + hb + 1u)}; \
      _Pragma("unroll") for (u32 hh = 0; hh < 2u; hh++) \
        _Pragma("unroll") for (u32 k = 0; k < 4u; k++) sv[q][hh][k] = ie_q6k_sw(x4[k], hq[hh], hh, k); \
      const u32 sc = qs[ie_q6k_sc(r, b, nb)]; \
      const float d = (float)((const G f16 *)qs)[ie_q6k_d(rows, r, b, nb)]; \
      s0[q] = d * (float)(int)(signed char)(sc & 255u), s1[q] = d * (float)(int)(signed char)(sc >> 8); \
    } \
    _Pragma("unroll") for (u32 j = 0; j < QT; j++) { \
      if (j < T) { /* uniform */ \
        const G u8 *xj = xq + j * xs; \
        const G u32 *aw = (const G u32 *)xj; \
        const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u); \
        const float da = ((const G float *)(xj + 32u * nb))[b]; \
        const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}; \
        const Mem am = {av}; \
        _Pragma("unroll") for (u32 q = 0; q < QR; q++) { \
          const Mem s0m = {sv[q][0]}, s1m = {sv[q][1]}; \
          const int d0 = (int)ie_q6k_dots_s(s0m, 0u, am, 0u, 0u); /* laws q6k_dots_s, q6k_dots */ \
          const int d1 = (int)ie_q6k_dots_s(s1m, 0u, am, 0u, 1u); \
          acc[q][j] = __builtin_fmaf(__builtin_fmaf(s1[q], (float)d1, s0[q] * (float)d0), da, acc[q][j]); \
        } \
      } \
    } \
  } \
  _Pragma("unroll") for (u32 q = 0; q < QR; q++) { \
    const u32 r = r0 + q; \
    _Pragma("unroll") for (u32 j = 0; j < QT; j++) { \
      if (j < T) { \
        const float v = wave_tree_f(acc[q][j]); \
        if (l == 0u && r < rows) y[j * ys + r] = epilogue(v, r, bias, res ? res + j * rs : res); \
      } \
    } \
  }
KERNEL ie_gemv_q6kq8_tr(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                        const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  Q6K_TR_BODY(4u, 16u)
}
KERNEL ie_gemv_q6kq8_tr8(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                         const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  Q6K_TR_BODY(8u, 8u)
}
#define Q6K_TS_BODY(QR, QT, TS) \
  _Pragma("clang fp contract(off)") \
  const u32 r0 = (wgid() * (8u / TS) + (tid() >> 5) / TS) * QR, l = lane(); \
  if (r0 >= rows) return; \
  { const u32 j0_ = ((tid() >> 5) % TS) * QT; /* this wave: tokens j0_ .. j0_ + QT - 1 */ \
    xq += j0_ * xs, y += j0_ * ys, res = res ? res + j0_ * rs : res, T = T > j0_ ? T - j0_ : 0u; } \
  if (T == 0u) return; \
  float acc[QR][QT]; \
  _Pragma("unroll") for (u32 q = 0; q < QR; q++) \
    _Pragma("unroll") for (u32 j = 0; j < QT; j++) acc[q][j] = 0.0f; \
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) { \
    const u32 b = ie_lane_blk(l, t); \
    u32 sv[QR][2][4]; /* the q - 32 words of each half, made once per block */ \
    float s0[QR], s1[QR]; \
    _Pragma("unroll") for (u32 q = 0; q < QR; q++) { \
      const u32 r = r0 + q < rows ? r0 + q : r0; /* a row past the end: row r0 again, not stored */ \
      const u32x4 w4 = __builtin_nontemporal_load((const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb))); \
      const u32 x4[4] = {w4.x, w4.y, w4.z, w4.w}; \
      const u32 hb = ie_q6k_hw(rows, r, b, 0u, nb); \
      const u32 hq[2] = {__builtin_nontemporal_load(qw + hb), __builtin_nontemporal_load(qw + hb + 1u)}; \
      _Pragma("unroll") for (u32 hh = 0; hh < 2u; hh++) \
        _Pragma("unroll") for (u32 k = 0; k < 4u; k++) sv[q][hh][k] = ie_q6k_sw(x4[k], hq[hh], hh, k); \
      const u32 sc = qs[ie_q6k_sc(r, b, nb)]; \
      const float d = (float)((const G f16 *)qs)[ie_q6k_d(rows, r, b, nb)]; \
      s0[q] = d * (float)(int)(signed char)(sc & 255u), s1[q] = d * (float)(int)(signed char)(sc >> 8); \
    } \
    _Pragma("unroll") for (u32 j = 0; j < QT; j++) { \
      if (j < T) { /* uniform */ \
        const G u8 *xj = xq + j * xs; \
        const G u32 *aw = (const G u32 *)xj; \
        const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u); \
        const float da = ((const G float *)(xj + 32u * nb))[b]; \
        const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}; \
        const Mem am = {av}; \
        _Pragma("unroll") for (u32 q = 0; q < QR; q++) { \
          const Mem s0m = {sv[q][0]}, s1m = {sv[q][1]}; \
          const int d0 = (int)ie_q6k_dots_s(s0m, 0u, am, 0u, 0u); /* laws q6k_dots_s, q6k_dots */ \
          const int d1 = (int)ie_q6k_dots_s(s1m, 0u, am, 0u, 1u); \
          acc[q][j] = __builtin_fmaf(__builtin_fmaf(s1[q], (float)d1, s0[q] * (float)d0), da, acc[q][j]); \
        } \
      } \
    } \
  } \
  _Pragma("unroll") for (u32 q = 0; q < QR; q++) { \
    const u32 r = r0 + q; \
    _Pragma("unroll") for (u32 j = 0; j < QT; j++) { \
      if (j < T) { \
        const float v = wave_tree_f(acc[q][j]); \
        if (l == 0u && r < rows) y[j * ys + r] = epilogue(v, r, bias, res ? res + j * rs : res); \
      } \
    } \
  }
/* 6 <= T <= 8: the tokens split over 2 waves (as ie_gemv_q4kq8_ts2); bitwise equal to ie_gemv_q6kq8 */
KERNEL ie_gemv_q6kq8_ts2(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                         const G float *bias, const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  Q6K_TS_BODY(8u, 4u, 2u)
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
#ifndef TR_NT
#define TR_NT 1
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
      if (TR_NT) { /* weights are read once: keep them out of the caches the activations live in */
        w0[q] = __builtin_nontemporal_load((const G u32x4 *)wp);
        w1[q] = __builtin_nontemporal_load((const G u32x4 *)(wp + 4u));
      } else {
        w0[q] = *(const G u32x4 *)wp;
        w1[q] = *(const G u32x4 *)(wp + 4u);
      }
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


/* Prompt chunks of many tokens: Y = W X with the matrix instruction
 * v_wmma_i32_16x16x16_iu8 (int8 x int8 -> int32, 16 x 16 x 16 per wave).
 * Lane layout (gfx12, checked by tools/wmma_test.c): A: lane l holds row
 * l % 16, K bytes 8 (l / 16) .. + 7; B: lane l holds token l % 16, the same
 * K bytes; D: lane l, element v = row 8 (l / 16) + v, token l % 16.
 *
 * Workgroup tile: 128 rows x 64 tokens. K goes in chunks of GM_KB blocks of
 * 32: the workgroup copies the weight tile (128 rows x GM_KB x 32 bytes,
 * 16-byte loads, each row's bytes are contiguous), the activation tile and
 * both scale tiles into LDS; then wave w does rows 32 (w % 4) .. + 31 and
 * tokens 32 (w / 4) .. + 31: 2 x 2 WMMA tiles, two WMMA (K = 16 each) per
 * block give the exact int32 block dot product of each (row, token), then
 * acc += (scale_w * scale_x) * dot, the blocks in order. The token tiles of
 * one row tile are next to each other in the grid, so they read W from the
 * cache: W comes from memory about once. The float sums are in a different
 * order than in ie_gemv_q8q8, so the results are close to (about 1e-7
 * relative), not bitwise equal to, the one-token results. */
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
#ifndef GM_KB
#define GM_KB 4u /* blocks of 32 per K chunk */
#endif
/* Workgroup tile GM_R rows x GM_T tokens; each wave GM_WI x GM_WJ WMMA tiles
 * (16 x 16 each); the 8 waves form a (GM_R / 16 GM_WI) x (GM_T / 16 GM_WJ)
 * grid. src/hip.c uses the same GM_R and GM_T for the launch grid. */
#ifndef GM_R
#define GM_R 128u
#endif
#ifndef GM_T
#define GM_T 64u
#endif
#ifndef GM_WI
#define GM_WI 2u
#endif
#ifndef GM_WJ
#define GM_WJ 2u
#endif
#define GM_WROWS (GM_R / (16u * GM_WI)) /* waves along the rows */
#define GM_LS (GM_KB * 32u + 16u) /* LDS row stride, bytes (padded: fewer bank conflicts, 16-byte aligned) */
#ifndef GM_NBUF
#define GM_NBUF 1u /* LDS stages: 2 = double buffer (one barrier per K chunk; measured slower: half the workgroups per CU) */
#endif
static LDS u8 gm_w[GM_NBUF][GM_R * GM_LS];
static LDS u8 gm_x[GM_NBUF][GM_T * GM_LS];
static LDS float gm_sw[GM_NBUF][GM_KB * GM_R], gm_sx[GM_NBUF][GM_T * GM_KB]; /* gm_sw[k][row]: 8 rows in 2 16-byte reads */
typedef float f8 __attribute__((ext_vector_type(8)));
#define GM_NSX ((GM_T * GM_KB + NT - 1u) / NT)
KERNEL ie_gemm_q8(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, const G float *bias,
                  const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  const u32 ntt = (T + GM_T - 1u) / GM_T, tt = wgid() % ntt, rt = wgid() / ntt, t0 = tid(), w = t0 >> 5, l = lane();
  const u32 rbase = rt * GM_R, tbase = tt * GM_T, h = l >> 4;
  const u32 wr = 16u * GM_WI * (w % GM_WROWS), wt = 16u * GM_WJ * (w / GM_WROWS); /* this wave's rows and tokens */
  f8 acc[GM_WI][GM_WJ];
#pragma unroll
  for (u32 i = 0; i < GM_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GM_WJ; j++) acc[i][j] = (f8)(0.0f);
  /* Software pipeline: the global loads of K chunk n + 1 go out (into
   * registers) before the WMMA of chunk n, and (GM_NBUF = 2) are stored into
   * the other LDS buffer after it: one barrier per chunk. Thread t0 copies
   * the pieces t0 + 256 i. */
  u32x4 rw[GM_R * 8u / NT], rx[GM_T * 8u / NT];
  float rsw[GM_R * GM_KB / NT], rsx[GM_NSX];
#define GM_FETCH(kb0_)                                                                                          \
  do {                                                                                                          \
    const u32 kbf = (kb0_), nkf = nb - kbf < GM_KB ? nb - kbf : GM_KB;                                          \
    _Pragma("unroll") for (u32 i = 0; i < GM_R * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i, r = p >> 3, pc = p & 7u, gr = rbase + r < rows ? rbase + r : rows - 1u;        \
      if (pc < 2u * nkf) rw[i] = *(const G u32x4 *)(qw + ie_q8_dst_word(gr, kbf, 0u, nb) + pc * 4u);            \
    }                                                                                                           \
    _Pragma("unroll") for (u32 i = 0; i < GM_T * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i, tk = p >> 3, pc = p & 7u, gt = tbase + tk < T ? tbase + tk : T - 1u;           \
      if (pc < 2u * nkf) rx[i] = *(const G u32x4 *)(xq + (unsigned long)gt * xs + kbf * 32u + pc * 16u);       \
    }                                                                                                           \
    _Pragma("unroll") for (u32 i = 0; i < GM_R * GM_KB / NT; i++) {                                             \
      const u32 p = t0 + NT * i, k = p / GM_R, r = p % GM_R, gr = rbase + r < rows ? rbase + r : rows - 1u;     \
      if (k < nkf) rsw[i] = (float)qs[ie_q8_dst_scale(gr, kbf + k, nb)];                                       \
    }                                                                                                           \
    _Pragma("unroll") for (u32 i = 0; i < GM_NSX; i++) {                                                        \
      const u32 p = t0 + NT * i, tk = p / GM_KB, k = p % GM_KB, gt = tbase + tk < T ? tbase + tk : T - 1u;      \
      if (p < GM_T * GM_KB && k < nkf) rsx[i] = ((const G float *)(xq + (unsigned long)gt * xs + 32u * nb))[kbf + k]; \
    }                                                                                                           \
  } while (0)
#define GM_STORE(bf_, kb0_)                                                                                     \
  do {                                                                                                          \
    const u32 nks = nb - (kb0_) < GM_KB ? nb - (kb0_) : GM_KB;                                                  \
    _Pragma("unroll") for (u32 i = 0; i < GM_R * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i;                                                                                \
      if ((p & 7u) < 2u * nks) *(LDSP u32x4 *)(gm_w[bf_] + (p >> 3) * GM_LS + (p & 7u) * 16u) = rw[i];          \
    }                                                                                                           \
    _Pragma("unroll") for (u32 i = 0; i < GM_T * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i;                                                                                \
      if ((p & 7u) < 2u * nks) *(LDSP u32x4 *)(gm_x[bf_] + (p >> 3) * GM_LS + (p & 7u) * 16u) = rx[i];          \
    }                                                                                                           \
    _Pragma("unroll") for (u32 i = 0; i < GM_R * GM_KB / NT; i++) gm_sw[bf_][t0 + NT * i] = rsw[i];            \
    _Pragma("unroll") for (u32 i = 0; i < GM_NSX; i++)                                                          \
      if (t0 + NT * i < GM_T * GM_KB) gm_sx[bf_][t0 + NT * i] = rsx[i];                                         \
  } while (0)
  GM_FETCH(0u);
  GM_STORE(0u, 0u);
  barrier();
  u32 bf = 0;
  for (u32 kb0 = 0; kb0 < nb; kb0 += GM_KB) {
    const u32 nk = nb - kb0 < GM_KB ? nb - kb0 : GM_KB;
    const int more = kb0 + GM_KB < nb;
    if (more) GM_FETCH(kb0 + GM_KB); /* in flight during the math below */
    const LDSP u8 *lw = gm_w[bf], *lx = gm_x[bf];
    const LDSP float *lsw = gm_sw[bf], *lsx = gm_sx[bf];
    for (u32 k = 0; k < nk; k++) {
      v2i a0[GM_WI], a1[GM_WI], b0[GM_WJ], b1[GM_WJ];
#pragma unroll
      for (u32 i = 0; i < GM_WI; i++) {
        const LDSP u8 *pa = lw + (wr + 16u * i + (l & 15u)) * GM_LS + k * 32u + 8u * h;
        a0[i] = *(const LDSP v2i *)pa, a1[i] = *(const LDSP v2i *)(pa + 16u);
      }
#pragma unroll
      for (u32 j = 0; j < GM_WJ; j++) {
        const LDSP u8 *pb = lx + (wt + 16u * j + (l & 15u)) * GM_LS + k * 32u + 8u * h;
        b0[j] = *(const LDSP v2i *)pb, b1[j] = *(const LDSP v2i *)(pb + 16u);
      }
      /* all WMMA of the block first (independent tiles), then the scaling.
       * The int32 sums start at the bits of 12582912.0f (0x4B400000): the
       * result read as float is exactly 12582912 + dot (|dot| <= 32 * 127 *
       * 127 < 2^22), so one exact subtraction gives the dot as a float. */
      const v8i zm = (v8i)(0x4B400000);
      v8i c[GM_WI][GM_WJ];
#pragma unroll
      for (u32 i = 0; i < GM_WI; i++)
#pragma unroll
        for (u32 j = 0; j < GM_WJ; j++) c[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0[i], 1, b0[j], zm, 0);
#pragma unroll
      for (u32 i = 0; i < GM_WI; i++)
#pragma unroll
        for (u32 j = 0; j < GM_WJ; j++) c[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1[i], 1, b1[j], c[i][j], 0);
      f8 sw[GM_WI];
#pragma unroll
      for (u32 i = 0; i < GM_WI; i++) sw[i] = *(const LDSP f8 *)(lsw + k * GM_R + wr + 16u * i + 8u * h);
#pragma unroll
      for (u32 j = 0; j < GM_WJ; j++) {
        const float dx = lsx[(wt + 16u * j + (l & 15u)) * GM_KB + k];
#pragma unroll
        for (u32 i = 0; i < GM_WI; i++) acc[i][j] += (sw[i] * dx) * (__builtin_bit_cast(f8, c[i][j]) - 12582912.0f);
      }
    }
    if (GM_NBUF == 2u) {
      if (more) GM_STORE(bf ^ 1u, kb0 + GM_KB); /* the other buffer: nobody reads it in this chunk */
      barrier();
      bf ^= 1u;
    } else {
      barrier(); /* the tiles are read before the next chunk overwrites them */
      if (more) GM_STORE(0u, kb0 + GM_KB);
      barrier();
    }
  }
#pragma unroll
  for (u32 i = 0; i < GM_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GM_WJ; j++) {
      const u32 t = tbase + wt + 16u * j + (l & 15u);
      if (t >= T) continue;
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = rbase + wr + 16u * i + 8u * h + v;
        if (r < rows) y[t * ys + r] = epilogue(acc[i][j][v], r, bias, res ? res + t * rs : res);
      }
    }
#undef GM_FETCH
#undef GM_STORE
}

/* ie_gemm_q8r: the same product, weights straight from memory into the
 * registers of the wave that uses them (no LDS for W). Each wave owns
 * GR_WI x 16 rows and all GR_T tokens of the workgroup tile, so no other
 * wave needs its weights; only the activations (used by all waves) and the
 * scales go through LDS (double buffered: one barrier per K chunk). The
 * weights of the next K chunk are loaded into registers during the math of
 * the current one.
 * K order inside a 32-byte block: lane half h takes bytes 16 h .. 16 h + 7
 * for the first WMMA and 16 h + 8 .. + 15 for the second (one 16-byte read
 * per lane and block). A and B use the same order and each byte is used
 * once, so the block dot product is the same exact int32. */
#ifndef GR_WI
#define GR_WI 2u /* row tiles (16 rows) per wave */
#endif
#ifndef GR_T
#define GR_T 64u /* tokens per workgroup tile (all waves) */
#endif
#ifndef GR_KB
#define GR_KB 4u
#endif
#define GR_R (8u * 16u * GR_WI) /* rows per workgroup */
#define GR_WJ (GR_T / 16u)
#define GR_SPT (GR_R >= NT ? GR_R / NT : 1u) /* weight scale rows per thread */
#define GR_LS (GR_KB * 32u + 16u) /* LDS row stride, bytes */
static LDS u8 gr_x[2][GR_T * GR_LS] __attribute__((aligned(16)));
static LDS float gr_sw[2][GR_KB * GR_R] __attribute__((aligned(32))), gr_sx[2][GR_T * GR_KB];
KERNEL ie_gemm_q8r(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, const G float *bias,
                   const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  /* nb % GR_KB == 0 (src/hip.c checks): no partial K chunk */
  const u32 ntt = (T + GR_T - 1u) / GR_T, tt = wgid() % ntt, rt = wgid() / ntt, t0 = tid(), w = t0 >> 5, l = lane();
  const u32 rbase = rt * GR_R, tbase = tt * GR_T, h = l >> 4, wr = 16u * GR_WI * w;
  f8 acc[GR_WI][GR_WJ];
#pragma unroll
  for (u32 i = 0; i < GR_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GR_WJ; j++) acc[i][j] = (f8)(0.0f);
  /* the addresses of chunk 0 (rows and tokens past the end are clamped:
   * computed, not stored); chunk kb0 adds kb0 * 32 bytes (scales: kb0) */
  const G u8 *wp[GR_WI];
#pragma unroll
  for (u32 i = 0; i < GR_WI; i++) {
    const u32 r = rbase + wr + 16u * i + (l & 15u);
    wp[i] = (const G u8 *)qw + (unsigned long)(r < rows ? r : rows - 1u) * nb * 32u + 16u * h;
  }
  const G u8 *xp[GR_T * 8u / NT];
#pragma unroll
  for (u32 i = 0; i < GR_T * 8u / NT; i++) {
    const u32 p = t0 + NT * i, tk = p >> 3, gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = xq + (unsigned long)gt * xs + (p & 7u) * 16u;
  }
  const u32 sr = rbase + t0 * GR_SPT; /* scale rows of this thread: GR_SPT consecutive (threads t0 * GR_SPT < GR_R) */
  const G f16 *swp = qs + (unsigned long)(sr < rows ? sr : rows - 1u) * nb; /* idle threads: a valid row */
  const u32 sxt = tbase + t0 / GR_KB;
  const G float *sxp = (const G float *)(xq + (unsigned long)(sxt < T ? sxt : T - 1u) * xs + 32u * nb) + t0 % GR_KB;
  u32x4 wa[GR_KB][GR_WI], wn[GR_KB][GR_WI]; /* weights: current chunk, next chunk */
  u32x4 rx[GR_T * 8u / NT];
  f16 rsw[GR_SPT][GR_KB];
  float rsx;
#define GR_FETCH_W(dst_, kb0_)                                                                                  \
  _Pragma("unroll") for (u32 k = 0; k < GR_KB; k++)                                                             \
    _Pragma("unroll") for (u32 i = 0; i < GR_WI; i++) dst_[k][i] = *(const G u32x4 *)(wp[i] + ((kb0_) + k) * 32u);
#define GR_FETCH_X(kb0_)                                                                                        \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GR_T * 8u / NT; i++) rx[i] = *(const G u32x4 *)(xp[i] + (kb0_) * 32u); \
    _Pragma("unroll") for (u32 q = 0; q < GR_SPT; q++) {                                                        \
      const u32 r = sr + q < rows ? q : 0u;                                                                     \
      _Pragma("unroll") for (u32 k = 0; k < GR_KB; k++) rsw[q][k] = swp[(unsigned long)r * nb + (kb0_) + k];   \
    }                                                                                                           \
    rsx = t0 < GR_T * GR_KB ? sxp[(kb0_)] : 0.0f;                                                               \
  } while (0)
#define GR_STORE_X(bf_)                                                                                         \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GR_T * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i;                                                                                \
      *(LDSP u32x4 *)(gr_x[bf_] + (p >> 3) * GR_LS + (p & 7u) * 16u) = rx[i];                                   \
    }                                                                                                           \
    if (t0 * GR_SPT < GR_R) _Pragma("unroll") for (u32 q = 0; q < GR_SPT; q++)                                   \
      _Pragma("unroll") for (u32 k = 0; k < GR_KB; k++) gr_sw[bf_][k * GR_R + t0 * GR_SPT + q] = (float)rsw[q][k]; \
    if (t0 < GR_T * GR_KB) gr_sx[bf_][t0] = rsx;                                                                \
  } while (0)
  GR_FETCH_W(wa, 0u);
  GR_FETCH_X(0u);
  GR_STORE_X(0u);
  barrier();
  u32 bf = 0;
#pragma unroll 1
  for (u32 kb0 = 0; kb0 < nb; kb0 += GR_KB) {
    const int more = kb0 + GR_KB < nb;
    if (more) { /* in flight during the math below */
      GR_FETCH_W(wn, kb0 + GR_KB);
      GR_FETCH_X(kb0 + GR_KB);
    }
    const LDSP u8 *lx = gr_x[bf];
    const LDSP float *lsw = gr_sw[bf], *lsx = gr_sx[bf];
    const u32 nk = nb - kb0 < GR_KB ? nb - kb0 : GR_KB; /* = GR_KB; the branch keeps the blocks apart for the scheduler */
#pragma unroll
    for (u32 k = 0; k < GR_KB; k++) {
      if (k >= nk) break;
      u32x4 b[GR_WJ];
#pragma unroll
      for (u32 j = 0; j < GR_WJ; j++) b[j] = *(const LDSP u32x4 *)(lx + (16u * j + (l & 15u)) * GR_LS + k * 32u + 16u * h);
      const v8i zm = (v8i)(0x4B400000); /* see ie_gemm_q8 */
      float dx[GR_WJ];
#pragma unroll
      for (u32 j = 0; j < GR_WJ; j++) dx[j] = lsx[(16u * j + (l & 15u)) * GR_KB + k];
#pragma unroll
      for (u32 i = 0; i < GR_WI; i++) { /* one row tile at a time (fewer registers) */
        v8i c[GR_WJ];
        const v2i a0 = {(int)wa[k][i].x, (int)wa[k][i].y}, a1 = {(int)wa[k][i].z, (int)wa[k][i].w};
#pragma unroll
        for (u32 j = 0; j < GR_WJ; j++) {
          const v2i b0 = {(int)b[j].x, (int)b[j].y};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0, zm, 0);
        }
#pragma unroll
        for (u32 j = 0; j < GR_WJ; j++) {
          const v2i b1 = {(int)b[j].z, (int)b[j].w};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1, c[j], 0);
        }
        const f8 sw = *(const LDSP f8 *)(lsw + k * GR_R + wr + 16u * i + 8u * h);
#pragma unroll
        for (u32 j = 0; j < GR_WJ; j++) acc[i][j] += (sw * dx[j]) * (__builtin_bit_cast(f8, c[j]) - 12582912.0f);
      }
      __builtin_amdgcn_sched_barrier(0); /* keep the blocks apart: the scheduler otherwise overlaps them and spills */
    }
    if (more) GR_STORE_X(bf ^ 1u); /* nobody reads that buffer in this chunk */
    lbarrier();
    bf ^= 1u;
#pragma unroll
    for (u32 k = 0; k < GR_KB; k++)
#pragma unroll
      for (u32 i = 0; i < GR_WI; i++) wa[k][i] = wn[k][i];
  }
#pragma unroll
  for (u32 i = 0; i < GR_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GR_WJ; j++) {
      const u32 t = tbase + 16u * j + (l & 15u);
      if (t >= T) continue;
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = rbase + wr + 16u * i + 8u * h + v;
        if (r < rows) y[t * ys + r] = epilogue(acc[i][j][v], r, bias, res ? res + t * rs : res);
      }
    }
#undef GR_FETCH_W
#undef GR_FETCH_X
#undef GR_STORE_X
}

/* ie_gemm_q4kr: ie_gemm_q8r for MAT_Q4K weights (nb % 8 == 0). Each lane
 * reads the whole 16-byte nibble block and takes the low (lane half 0:
 * weights 0..15) or high nibbles (weights 16..31), values 0..15: the WMMA
 * gives dq = sum q a exactly; then acc += da ((d sc) dq - (dmin mn) asum).
 * Not bitwise equal to ie_gemv_q4kq8 (another order of the float sums). */
#ifndef GK_WI
#define GK_WI 1u /* row tiles per wave (fewer registers than ie_gemm_q8r: 4 more VALU terms) */
#endif
#ifndef GK_T
#define GK_T 64u
#endif
#define GK_KB 4u
#define GK_R (8u * 16u * GK_WI)
#define GK_WJ (GK_T / 16u)
#define GK_LS (GK_KB * 32u + 16u)
static LDS u8 gk_x[2][GK_T * GK_LS] __attribute__((aligned(16)));
static LDS float gk_sw[2][GK_KB * GK_R] __attribute__((aligned(32))), gk_sx[2][GK_T * GK_KB];
static LDS float gk_mw[2][GK_KB * GK_R] __attribute__((aligned(32)));
/* the min term - sum over blocks of (dmin mn) (da asum) is a small matrix
 * product (K = the blocks): fp16 operands per chunk (row r: -dmin mn of
 * its GK_KB blocks; token t: da asum), one fp16 WMMA per chunk adds it to
 * the f32 sums. fp16 rounding: relative 2^-11 on each of the two factors
 * (far below the 4-bit weight and 8-bit activation rounding). */
typedef _Float16 hk8 __attribute__((ext_vector_type(8)));
typedef _Float16 hk4 __attribute__((ext_vector_type(4)));
static LDS hk4 gk_mh[2][GK_R], gk_sh[2][GK_T];
KERNEL ie_gemm_q4kr(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, const G float *bias,
                   const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  /* nb % GK_KB == 0 (src/hip.c checks): no partial K chunk */
  const u32 ntt = (T + GK_T - 1u) / GK_T, tt = wgid() % ntt, rt = wgid() / ntt, t0 = tid(), w = t0 >> 5, l = lane();
  const u32 rbase = rt * GK_R, tbase = tt * GK_T, h = l >> 4, wr = 16u * GK_WI * w;
  f8 acc[GK_WI][GK_WJ];
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GK_WJ; j++) acc[i][j] = (f8)(0.0f);
  /* the addresses of chunk 0 (rows and tokens past the end are clamped:
   * computed, not stored); chunk kb0 adds kb0 * 32 bytes (scales: kb0) */
  const G u8 *wp[GK_WI];
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++) {
    const u32 r = rbase + wr + 16u * i + (l & 15u);
    wp[i] = (const G u8 *)qw + (unsigned long)(r < rows ? r : rows - 1u) * nb * 16u; /* the whole 16-byte nibble block */
  }
  const G u8 *xp[GK_T * 8u / NT];
#pragma unroll
  for (u32 i = 0; i < GK_T * 8u / NT; i++) {
    const u32 p = t0 + NT * i, tk = p >> 3, gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = xq + (unsigned long)gt * xs + (p & 7u) * 16u;
  }
  const u32 sr = rbase + t0 * 1u; /* scale rows of this thread: 1u consecutive (threads t0 * 1u < GK_R) */
  const u32 srr = sr < rows ? sr : rows - 1u; /* idle threads: a valid row */
  const G u16 *swp = qs + (unsigned long)srr * nb;
  const G f16 *ddp = (const G f16 *)qs + (unsigned long)rows * nb + (unsigned long)srr * (nb >> 3) * 2u;
  const u32 sxt = tbase + t0 / GK_KB;
  const G float *sxp = (const G float *)(xq + (unsigned long)(sxt < T ? sxt : T - 1u) * xs + 32u * nb) + t0 % GK_KB;
  const G int *sap = (const G int *)(xq + (unsigned long)(sxt < T ? sxt : T - 1u) * xs + 36u * nb) + t0 % GK_KB;
  u32x4 wa[GK_KB][GK_WI], wn[GK_KB][GK_WI]; /* weights: current chunk, next chunk */
  u32x4 rx[GK_T * 8u / NT];
  u16 rsw[GK_KB];
  f16 rdd[2];
  float rsx, rsa;
#define GK_FETCH_W(dst_, kb0_)                                                                                  \
  _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++)                                                             \
    _Pragma("unroll") for (u32 i = 0; i < GK_WI; i++) dst_[k][i] = *(const G u32x4 *)(wp[i] + ((kb0_) + k) * 16u);
#define GK_FETCH_X(kb0_)                                                                                        \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GK_T * 8u / NT; i++) rx[i] = *(const G u32x4 *)(xp[i] + (kb0_) * 32u); \
    _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++) rsw[k] = swp[(kb0_) + k];                                 \
    rdd[0] = ddp[((kb0_) >> 3) * 2u], rdd[1] = ddp[((kb0_) >> 3) * 2u + 1u]; /* GK_KB divides 8: one super-block */ \
    rsa = t0 < GK_T * GK_KB ? (float)sap[(kb0_)] : 0.0f;                                                       \
    rsx = t0 < GK_T * GK_KB ? sxp[(kb0_)] : 0.0f;                                                               \
  } while (0)
#define GK_STORE_X(bf_)                                                                                         \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GK_T * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i;                                                                                \
      *(LDSP u32x4 *)(gk_x[bf_] + (p >> 3) * GK_LS + (p & 7u) * 16u) = rx[i];                                   \
    }                                                                                                           \
    if (t0 < GK_R) {                                                                                            \
      hk4 mh_;                                                                                                  \
      _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++) {                                                       \
        gk_sw[bf_][k * GK_R + t0] = (float)rdd[0] * (float)(rsw[k] & 63u);                                     \
        mh_[k] = (f16)(-((float)rdd[1] * (float)(rsw[k] >> 8)));                                                \
      }                                                                                                         \
      gk_mh[bf_][t0] = mh_;                                                                                     \
    }                                                                                                           \
    if (t0 < GK_T * GK_KB) ((LDSP f16 *)gk_sh[bf_])[t0] = (f16)(rsa * rsx);                                    \
    if (t0 < GK_T * GK_KB) gk_sx[bf_][t0] = rsx;                                                                \
  } while (0)
  GK_FETCH_W(wa, 0u);
  GK_FETCH_X(0u);
  GK_STORE_X(0u);
  barrier();
  u32 bf = 0;
#pragma unroll 1
  for (u32 kb0 = 0; kb0 < nb; kb0 += GK_KB) {
    const int more = kb0 + GK_KB < nb;
    if (more) { /* in flight during the math below */
      GK_FETCH_W(wn, kb0 + GK_KB);
      GK_FETCH_X(kb0 + GK_KB);
    }
    const LDSP u8 *lx = gk_x[bf];
    const LDSP float *lsw = gk_sw[bf], *lsx = gk_sx[bf];
    const u32 nk = nb - kb0 < GK_KB ? nb - kb0 : GK_KB; /* = GK_KB; the branch keeps the blocks apart for the scheduler */
#pragma unroll
    for (u32 k = 0; k < GK_KB; k++) {
      if (k >= nk) break;
      u32x4 b[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) b[j] = *(const LDSP u32x4 *)(lx + (16u * j + (l & 15u)) * GK_LS + k * 32u + 16u * h);
      const v8i zm = (v8i)(0x4B400000); /* see ie_gemm_q8 */
      float dx[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) dx[j] = lsx[(16u * j + (l & 15u)) * GK_KB + k];
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) { /* one row tile at a time (fewer registers) */
        v8i c[GK_WJ];
        /* lane half h: weights 16 h .. 16 h + 15 = the low (h = 0) or high nibbles of the 16 bytes */
        const u32x4 nb4 = h ? (wa[k][i] >> 4) & 0x0F0F0F0Fu : wa[k][i] & 0x0F0F0F0Fu;
        const v2i a0 = {(int)nb4.x, (int)nb4.y}, a1 = {(int)nb4.z, (int)nb4.w};
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) {
          const v2i b0 = {(int)b[j].x, (int)b[j].y};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0, zm, 0);
        }
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) {
          const v2i b1 = {(int)b[j].z, (int)b[j].w};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1, c[j], 0);
        }
        const f8 sw = *(const LDSP f8 *)(lsw + k * GK_R + wr + 16u * i + 8u * h);
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) acc[i][j] += (sw * dx[j]) * (__builtin_bit_cast(f8, c[j]) - 12582912.0f); /* (da d sc) dq */
      }
      __builtin_amdgcn_sched_barrier(0); /* keep the blocks apart: the scheduler otherwise overlaps them and spills */
    }
    { /* the min term of the chunk: lane half 0 holds K = blocks 0 .. GK_KB - 1, half 1 zeros */
      const hk4 z4 = (hk4)(0.0f);
      hk8 bm[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) {
        const hk4 v = h ? z4 : gk_sh[bf][16u * j + (l & 15u)];
        bm[j] = (hk8){v[0], v[1], v[2], v[3], 0, 0, 0, 0};
      }
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) {
        const hk4 v = h ? z4 : gk_mh[bf][wr + 16u * i + (l & 15u)];
        const hk8 am = {v[0], v[1], v[2], v[3], 0, 0, 0, 0};
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(am, bm[j], acc[i][j]);
      }
    }
    if (more) GK_STORE_X(bf ^ 1u); /* nobody reads that buffer in this chunk */
    lbarrier();
    bf ^= 1u;
#pragma unroll
    for (u32 k = 0; k < GK_KB; k++)
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) wa[k][i] = wn[k][i];
  }
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GK_WJ; j++) {
      const u32 t = tbase + 16u * j + (l & 15u);
      if (t >= T) continue;
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = rbase + wr + 16u * i + 8u * h + v;
        if (r < rows) y[t * ys + r] = epilogue(acc[i][j][v], r, bias, res ? res + t * rs : res);
      }
    }
#undef GK_FETCH_W
#undef GK_FETCH_X
#undef GK_STORE_X
}

/* Q4_K x fp16 activations for prompt chunks (ie_gemm_q4k_h). The Q8
 * activations of the chunk first become fp16 (ie_q8_f16: q d, exact up to
 * the fp16 rounding); the weights are dequantized to fp16 in LDS (w = d sc q
 * - dmin m, once per weight and workgroup) and fp16 WMMA sums in f32 over
 * all of K: no per-block scale math on the sums (ie_gemm_q4kr: 3 VALU per
 * output element and block). Workgroup = 64 rows x 512 tokens, so a chunk
 * of up to 512 tokens reads each weight once; wave w: the 64 rows x tokens
 * 64 w .. + 63 (4 x 4 tiles of 16 x 16, 128 f32 registers). Per block of 32
 * (K step): the activations (512 x 32 fp16) and the dequantized weights
 * (64 x 32) to LDS, the next block's loads in registers during the math.
 * Rows % 64 == 0, nb % 8 == 0 (the host checks). Measured against
 * ie_gemm_q4kr (one harness, tools/tilelang/ab_q4k.py, 512 tokens): 34816 x
 * 5120 2.206 vs 1.953 ms, 5120 x 17408 1.312 vs 0.966 ms; TileLang's kernel
 * with the same tile (512 x 64 x 32) 1.83 ms: off by default (IE_GEMM_H=1). */
KERNEL ie_q8_f16(const G u8 *q, G f16 *xh, u32 nb, u32 qs, u32 hs) {
  q = TOKC(q, qs), xh = TOK(xh, hs);
  const u32 i = wgid() * NT + tid();
  if (i >= nb * 32u) return;
  const float d = ((const G float *)(q + 32u * nb))[i >> 5];
  xh[i] = (f16)((float)((const G i8 *)q)[i] * d);
}
#define QH_R 64u
#define QH_T 512u
#define QH_S 40u /* LDS row stride, halves (80 bytes: the 16 rows of a lane group hit distinct banks) */
static LDS f16 qh_x[QH_T * QH_S] __attribute__((aligned(16)));
static LDS f16 qh_w[QH_R * QH_S] __attribute__((aligned(16)));
typedef unsigned int u32x2v __attribute__((ext_vector_type(2)));
static inline __attribute__((always_inline)) void gemm_q4k_h(const G u32 *qw, const G u16 *qs, const G f16 *xh, G float *y, u32 rows,
                                                            u32 nb, const G float *bias, const G float *res, u32 T, u32 hs, u32 ys,
                                                            u32 rs, const u32 QT) {
  const u32 WT = QT / 8u, NJ = WT / 16u, NL = QT * 4u / NT; /* tokens per wave, its token tiles, 16-byte loads per thread */
  const u32 ntt = (T + QT - 1u) / QT, tt = wgid() % ntt, rt = wgid() / ntt;
  const u32 r0 = rt * QH_R, t0 = tt * QT, w = __builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 hl = l >> 4, rl = l & 15u, wt0 = t0 + WT * w;
  const int won = wt0 < T; /* wave-uniform: this wave has tokens */
  f8 acc[4][4]; /* [row tile][token tile < NJ] */
#pragma unroll
  for (u32 i = 0; i < 4u; i++)
#pragma unroll
    for (u32 j = 0; j < 4u; j++) acc[i][j] = (f8)(0.0f);
  /* loaders: activations, 16 bytes (8 halves) per chunk c = tid + 256 n: token c / 4, part c % 4 */
  const G f16 *xp[8];
#pragma unroll
  for (u32 n = 0; n < NL; n++) {
    const u32 c = tid() + NT * n, tk = t0 + (c >> 2);
    xp[n] = xh + (unsigned long)(tk < T ? tk : T - 1u) * hs + (c & 3u) * 8u;
  }
  /* weights: thread = row tid / 4, quarter q4 = tid % 4: weights 8 q4 .. + 7 of each block
   * (q4 < 2: low nibbles of bytes 8 q4 .. ; q4 >= 2: high nibbles of bytes 8 (q4 - 2) ..) */
  const u32 wr = tid() >> 2, q4 = tid() & 3u;
  const G u8 *wp = (const G u8 *)qw + (unsigned long)(r0 + wr) * nb * 16u + (q4 & 1u) * 8u;
  const G u16 *sp = qs + (unsigned long)(r0 + wr) * nb;
  const G f16 *dp = (const G f16 *)qs + (unsigned long)rows * nb + (unsigned long)(r0 + wr) * (nb >> 3) * 2u;
  u32x4 xr[8];
  u32x2v wb;
  u32 sm;
  float dd, dm;
#define QH_LOAD(kb_)                                                                            \
  do {                                                                                          \
    _Pragma("unroll") for (u32 n = 0; n < NL; n++) xr[n] = *(const G u32x4 *)(xp[n] + (kb_) * 32u); \
    wb = *(const G u32x2v *)(wp + (kb_) * 16u);                                                 \
    sm = sp[(kb_)];                                                                             \
    dd = (float)dp[((kb_) >> 3) * 2u], dm = (float)dp[((kb_) >> 3) * 2u + 1u];                  \
  } while (0)
  QH_LOAD(0u);
#pragma unroll 1
  for (u32 kb = 0; kb < nb; kb++) {
#pragma unroll
    for (u32 n = 0; n < NL; n++) {
      const u32 c = tid() + NT * n;
      *(LDSP u32x4 *)(qh_x + (c >> 2) * QH_S + (c & 3u) * 8u) = xr[n];
    }
    {
      const float a = dd * (float)(sm & 63u), mb = dm * (float)((sm >> 8) & 255u);
      hk8 wv;
#pragma unroll
      for (u32 e = 0; e < 8u; e++) {
        const u32 byte = ((e < 4u ? wb.x : wb.y) >> (8u * (e & 3u))) & 255u;
        const u32 qv = q4 >= 2u ? byte >> 4 : byte & 15u;
        wv[e] = (f16)(a * (float)qv - mb);
      }
      *(LDSP hk8 *)(qh_w + wr * QH_S + q4 * 8u) = wv;
    }
    barrier();
    if (kb + 1u < nb) QH_LOAD(kb + 1u); /* in flight during the math */
    if (won) {
      hk8 a[4][2];
#pragma unroll
      for (u32 i = 0; i < 4u; i++)
#pragma unroll
        for (u32 kk = 0; kk < 2u; kk++) a[i][kk] = *(const LDSP hk8 *)(qh_w + (16u * i + rl) * QH_S + 16u * kk + 8u * hl);
#pragma unroll
      for (u32 j = 0; j < NJ; j++) {
        hk8 bx[2];
#pragma unroll
        for (u32 kk = 0; kk < 2u; kk++) bx[kk] = *(const LDSP hk8 *)(qh_x + (WT * w + 16u * j + rl) * QH_S + 16u * kk + 8u * hl);
#pragma unroll
        for (u32 i = 0; i < 4u; i++)
#pragma unroll
          for (u32 kk = 0; kk < 2u; kk++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i][kk], bx[kk], acc[i][j]);
      }
    }
    barrier();
  }
#undef QH_LOAD
  if (!won) return;
#pragma unroll
  for (u32 j = 0; j < NJ; j++) {
    const u32 t = wt0 + 16u * j + rl;
    if (t >= T) continue;
#pragma unroll
    for (u32 i = 0; i < 4u; i++)
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = r0 + 16u * i + 8u * hl + v;
        y[(unsigned long)t * ys + r] = epilogue(acc[i][j][v], r, bias, res ? res + (unsigned long)t * rs : res);
      }
  }
}
KERNEL ie_gemm_q4k_h(const G u32 *qw, const G u16 *qs, const G f16 *xh, G float *y, u32 rows, u32 nb,
                     const G float *bias, const G float *res, u32 T, u32 hs, u32 ys, u32 rs) {
  gemm_q4k_h(qw, qs, xh, y, rows, nb, bias, res, T, hs, ys, rs, 512u);
}
KERNEL ie_gemm_q4k_h256(const G u32 *qw, const G u16 *qs, const G f16 *xh, G float *y, u32 rows, u32 nb,
                        const G float *bias, const G float *res, u32 T, u32 hs, u32 ys, u32 rs) {
  gemm_q4k_h(qw, qs, xh, y, rows, nb, bias, res, T, hs, ys, rs, 256u);
}

/* ie_gemm_q4kr_sb: ie_gemm_q4kr for activations whose 8 blocks of a
 * super-block of 256 share one scale da (ie_requant_sb). Then the sum over
 * the blocks of a chunk is exact in int32: sum sc dq (sc: the 6-bit block
 * scales, dq <= 60960 in 24 bits, v_mad_i32_i24), and one float step per
 * chunk acc += (d da) sum: about 1.25 VALU per element and block instead of
 * 3. The min term as ie_gemm_q4kr. Measured (8K prefill, 2026-10-10): 5904
 * -> 5785 ms (2%), perplexity 4K 5.7357 -> 5.7354; off by default
 * (IE_Q4K_SB=1): the GEMM is also bound by its LDS loads and latency (without
 * any scale math it is only 16% faster). */
static LDS int gk_sc[2][GK_KB * GK_R] __attribute__((aligned(32)));
static LDS float gk_dd[2][GK_R] __attribute__((aligned(32)));
static inline v8i mul24v(v8i a, v8i b) { /* the operands fit in 24 bits (signed): v_mul_i32_i24 */
  return ((a << 8) >> 8) * ((b << 8) >> 8);
}
/* Requantize Q8 activations so that the 8 blocks of each super-block of 256
 * share one scale: dequantize (q d), then quantize with D = max |v| / 127
 * of the 256 values (as llama.cpp's Q8_K). One wave per super-block (lane
 * l: block l / 4, elements 8 (l % 4) ..), grid y = tokens. Same layout:
 * int8 [32 nb], float d [nb] (all D), u32 asum [nb]. nb % 8 == 0. */
KERNEL ie_requant_sb(const G u8 *qin, G u8 *qout, u32 nb, u32 is, u32 os) {
  qin = TOKC(qin, is), qout = TOK(qout, os);
  const u32 sb = wgid() * 8u + (tid() >> 5), l = lane();
  if (8u * sb >= nb) return; /* whole waves */
  const u32 blk = 8u * sb + (l >> 2), e0 = 8u * (l & 3u);
  const float d = ((const G float *)(qin + 32u * nb))[blk];
  const G i8 *src = (const G i8 *)qin + blk * 32u + e0;
  float v[8], m = 0.0f;
#pragma unroll
  for (u32 e = 0; e < 8u; e++) v[e] = (float)src[e] * d, m = __builtin_fmaxf(m, __builtin_fabsf(v[e]));
  m = wave_max_pos(m);
  const float D = m / 127.0f;
  G i8 *dst = (G i8 *)qout + blk * 32u + e0;
  float sum = 0.0f;
#pragma unroll
  for (u32 e = 0; e < 8u; e++) {
    const int q = D != 0.0f ? (int)__builtin_roundf(v[e] / D) : 0;
    dst[e] = (i8)q;
    sum += (float)q;
  }
  sum += shfl_xor_f(sum, 1u);
  sum += shfl_xor_f(sum, 2u);
  if ((l & 3u) == 0u) {
    ((G float *)(qout + 32u * nb))[blk] = D;
    ((G u32 *)(qout + 36u * nb))[blk] = (u32)(int)sum;
  }
}
KERNEL ie_gemm_q4kr_sb(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, const G float *bias,
                   const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  /* nb % GK_KB == 0 (src/hip.c checks): no partial K chunk */
  const u32 ntt = (T + GK_T - 1u) / GK_T, tt = wgid() % ntt, rt = wgid() / ntt, t0 = tid(), w = t0 >> 5, l = lane();
  const u32 rbase = rt * GK_R, tbase = tt * GK_T, h = l >> 4, wr = 16u * GK_WI * w;
  f8 acc[GK_WI][GK_WJ];
  v8i acc_i[GK_WI][GK_WJ];
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GK_WJ; j++) acc[i][j] = (f8)(0.0f), acc_i[i][j] = (v8i)(0);
  /* the addresses of chunk 0 (rows and tokens past the end are clamped:
   * computed, not stored); chunk kb0 adds kb0 * 32 bytes (scales: kb0) */
  const G u8 *wp[GK_WI];
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++) {
    const u32 r = rbase + wr + 16u * i + (l & 15u);
    wp[i] = (const G u8 *)qw + (unsigned long)(r < rows ? r : rows - 1u) * nb * 16u; /* the whole 16-byte nibble block */
  }
  const G u8 *xp[GK_T * 8u / NT];
#pragma unroll
  for (u32 i = 0; i < GK_T * 8u / NT; i++) {
    const u32 p = t0 + NT * i, tk = p >> 3, gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = xq + (unsigned long)gt * xs + (p & 7u) * 16u;
  }
  const u32 sr = rbase + t0 * 1u; /* scale rows of this thread: 1u consecutive (threads t0 * 1u < GK_R) */
  const u32 srr = sr < rows ? sr : rows - 1u; /* idle threads: a valid row */
  const G u16 *swp = qs + (unsigned long)srr * nb;
  const G f16 *ddp = (const G f16 *)qs + (unsigned long)rows * nb + (unsigned long)srr * (nb >> 3) * 2u;
  const u32 sxt = tbase + t0 / GK_KB;
  const G float *sxp = (const G float *)(xq + (unsigned long)(sxt < T ? sxt : T - 1u) * xs + 32u * nb) + t0 % GK_KB;
  const G int *sap = (const G int *)(xq + (unsigned long)(sxt < T ? sxt : T - 1u) * xs + 36u * nb) + t0 % GK_KB;
  u32x4 wa[GK_KB][GK_WI], wn[GK_KB][GK_WI]; /* weights: current chunk, next chunk */
  u32x4 rx[GK_T * 8u / NT];
  u16 rsw[GK_KB];
  f16 rdd[2];
  float rsx, rsa;
#define GK_FETCH_W(dst_, kb0_)                                                                                  \
  _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++)                                                             \
    _Pragma("unroll") for (u32 i = 0; i < GK_WI; i++) dst_[k][i] = *(const G u32x4 *)(wp[i] + ((kb0_) + k) * 16u);
#define GK_FETCH_X(kb0_)                                                                                        \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GK_T * 8u / NT; i++) rx[i] = *(const G u32x4 *)(xp[i] + (kb0_) * 32u); \
    _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++) rsw[k] = swp[(kb0_) + k];                                 \
    rdd[0] = ddp[((kb0_) >> 3) * 2u], rdd[1] = ddp[((kb0_) >> 3) * 2u + 1u]; /* GK_KB divides 8: one super-block */ \
    rsa = t0 < GK_T * GK_KB ? (float)sap[(kb0_)] : 0.0f;                                                       \
    rsx = t0 < GK_T * GK_KB ? sxp[(kb0_)] : 0.0f;                                                               \
  } while (0)
#define GK_STORE_X(bf_)                                                                                         \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GK_T * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i;                                                                                \
      *(LDSP u32x4 *)(gk_x[bf_] + (p >> 3) * GK_LS + (p & 7u) * 16u) = rx[i];                                   \
    }                                                                                                           \
    if (t0 < GK_R) {                                                                                            \
      hk4 mh_;                                                                                                  \
      _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++) {                                                       \
        gk_sc[bf_][k * GK_R + t0] = (int)(rsw[k] & 63u);                                                        \
        mh_[k] = (f16)(-((float)rdd[1] * (float)(rsw[k] >> 8)));                                                \
      }                                                                                                         \
      gk_mh[bf_][t0] = mh_;                                                                                     \
      gk_dd[bf_][t0] = (float)rdd[0];                                                                           \
    }                                                                                                           \
    if (t0 < GK_T * GK_KB) ((LDSP f16 *)gk_sh[bf_])[t0] = (f16)(rsa * rsx);                                    \
    if (t0 < GK_T * GK_KB) gk_sx[bf_][t0] = rsx;                                                                \
  } while (0)
  GK_FETCH_W(wa, 0u);
  GK_FETCH_X(0u);
  GK_STORE_X(0u);
  barrier();
  u32 bf = 0;
#pragma unroll 1
  for (u32 kb0 = 0; kb0 < nb; kb0 += GK_KB) {
    const int more = kb0 + GK_KB < nb;
    if (more) { /* in flight during the math below */
      GK_FETCH_W(wn, kb0 + GK_KB);
      GK_FETCH_X(kb0 + GK_KB);
    }
    const LDSP u8 *lx = gk_x[bf];
    const LDSP float *lsx = gk_sx[bf];
    const LDSP int *lsc = gk_sc[bf];
    const u32 nk = nb - kb0 < GK_KB ? nb - kb0 : GK_KB; /* = GK_KB; the branch keeps the blocks apart for the scheduler */
#pragma unroll
    for (u32 k = 0; k < GK_KB; k++) {
      if (k >= nk) break;
      u32x4 b[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) b[j] = *(const LDSP u32x4 *)(lx + (16u * j + (l & 15u)) * GK_LS + k * 32u + 16u * h);
      const v8i zm = (v8i)(0);
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) { /* one row tile at a time (fewer registers) */
        v8i c[GK_WJ];
        /* lane half h: weights 16 h .. 16 h + 15 = the low (h = 0) or high nibbles of the 16 bytes */
        const u32x4 nb4 = h ? (wa[k][i] >> 4) & 0x0F0F0F0Fu : wa[k][i] & 0x0F0F0F0Fu;
        const v2i a0 = {(int)nb4.x, (int)nb4.y}, a1 = {(int)nb4.z, (int)nb4.w};
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) {
          const v2i b0 = {(int)b[j].x, (int)b[j].y};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0, zm, 0);
        }
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) {
          const v2i b1 = {(int)b[j].z, (int)b[j].w};
          c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1, c[j], 0);
        }
        const v8i scv = *(const LDSP v8i *)(lsc + k * GK_R + wr + 16u * i + 8u * h);
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) acc_i[i][j] += mul24v(c[j], scv); /* sc dq, exact */
      }
    }
#pragma unroll
    for (u32 i = 0; i < GK_WI; i++) { /* the chunk's scale: d of the rows x the shared da of the tokens */
      const f8 dd = *(const LDSP f8 *)(gk_dd[bf] + wr + 16u * i + 8u * h);
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) {
        const float dx0 = lsx[(16u * j + (l & 15u)) * GK_KB];
        acc[i][j] += (dd * dx0) * __builtin_convertvector(acc_i[i][j], f8);
        acc_i[i][j] = (v8i)(0);
      }
    }
    { /* the min term of the chunk: lane half 0 holds K = blocks 0 .. GK_KB - 1, half 1 zeros */
      const hk4 z4 = (hk4)(0.0f);
      hk8 bm[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) {
        const hk4 v = h ? z4 : gk_sh[bf][16u * j + (l & 15u)];
        bm[j] = (hk8){v[0], v[1], v[2], v[3], 0, 0, 0, 0};
      }
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) {
        const hk4 v = h ? z4 : gk_mh[bf][wr + 16u * i + (l & 15u)];
        const hk8 am = {v[0], v[1], v[2], v[3], 0, 0, 0, 0};
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(am, bm[j], acc[i][j]);
      }
    }
    if (more) GK_STORE_X(bf ^ 1u); /* nobody reads that buffer in this chunk */
    lbarrier();
    bf ^= 1u;
#pragma unroll
    for (u32 k = 0; k < GK_KB; k++)
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) wa[k][i] = wn[k][i];
  }
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GK_WJ; j++) {
      const u32 t = tbase + 16u * j + (l & 15u);
      if (t >= T) continue;
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = rbase + wr + 16u * i + 8u * h + v;
        if (r < rows) y[t * ys + r] = epilogue(acc[i][j][v], r, bias, res ? res + t * rs : res);
      }
    }
#undef GK_FETCH_W
#undef GK_FETCH_X
#undef GK_STORE_X
}

/* ie_gemm_q6kr: ie_gemm_q4kr for MAT_Q6K weights. A operands: the q - 32
 * bytes (ie_q6k_sw), in the standard K order so that each WMMA covers one
 * half (one int8 scale): c0 = sum over weights 0..15, c1 = over 16..31;
 * then acc += da ((d sc0) c0 + (d sc1) c1). */
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
KERNEL ie_gemm_q6kr(const G u32 *qw, const G u16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, const G float *bias,
                   const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  /* nb % GK_KB == 0 (src/hip.c checks): no partial K chunk */
  const u32 ntt = (T + GK_T - 1u) / GK_T, tt = wgid() % ntt, rt = wgid() / ntt, t0 = tid(), w = t0 >> 5, l = lane();
  const u32 rbase = rt * GK_R, tbase = tt * GK_T, h = l >> 4, wr = 16u * GK_WI * w;
  f8 acc[GK_WI][GK_WJ];
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GK_WJ; j++) acc[i][j] = (f8)(0.0f);
  /* the addresses of chunk 0 (rows and tokens past the end are clamped:
   * computed, not stored); chunk kb0 adds kb0 * 32 bytes (scales: kb0) */
  const G u8 *wp[GK_WI];
  const G u32 *hp[GK_WI];
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++) {
    const u32 r = rbase + wr + 16u * i + (l & 15u), rc = r < rows ? r : rows - 1u;
    wp[i] = (const G u8 *)qw + (unsigned long)rc * nb * 16u; /* the 16-byte nibble block */
    hp[i] = qw + ie_q6k_hw(rows, rc, 0u, 0u, nb);             /* the 2 high-bits words */
  }
  const G u8 *xp[GK_T * 8u / NT];
#pragma unroll
  for (u32 i = 0; i < GK_T * 8u / NT; i++) {
    const u32 p = t0 + NT * i, tk = p >> 3, gt = tbase + tk < T ? tbase + tk : T - 1u;
    xp[i] = xq + (unsigned long)gt * xs + (p & 7u) * 16u;
  }
  const u32 sr = rbase + t0 * 1u; /* scale rows of this thread: 1u consecutive (threads t0 * 1u < GK_R) */
  const u32 srr = sr < rows ? sr : rows - 1u; /* idle threads: a valid row */
  const G u16 *swp = qs + ie_q6k_sc(srr, 0u, nb);
  const G f16 *ddp = (const G f16 *)qs + ie_q6k_d(rows, srr, 0u, nb);
  const u32 sxt = tbase + t0 / GK_KB;
  const G float *sxp = (const G float *)(xq + (unsigned long)(sxt < T ? sxt : T - 1u) * xs + 32u * nb) + t0 % GK_KB;
  u32x4 wa[GK_KB][GK_WI], wn[GK_KB][GK_WI]; /* weights: current chunk, next chunk */
  u32x2 ha[GK_KB][GK_WI], hn[GK_KB][GK_WI]; /* their high bits */
  u32x4 rx[GK_T * 8u / NT];
  u16 rsw[GK_KB];
  f16 rdd;
  float rsx;
#define GK_FETCH_W(dst_, kb0_)                                                                                  \
  _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++)                                                             \
    _Pragma("unroll") for (u32 i = 0; i < GK_WI; i++) {                                                        \
      dst_[k][i] = *(const G u32x4 *)(wp[i] + ((kb0_) + k) * 16u);                                              \
      GK_H(dst_)[k][i] = *(const G u32x2 *)(hp[i] + ((kb0_) + k) * 2u);                                            \
    }
#define GK_FETCH_X(kb0_)                                                                                        \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GK_T * 8u / NT; i++) rx[i] = *(const G u32x4 *)(xp[i] + (kb0_) * 32u); \
    _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++) rsw[k] = swp[(kb0_) + k];                                 \
    rdd = ddp[(kb0_) >> 3]; /* GK_KB divides 8: one super-block */                                            \
    rsx = t0 < GK_T * GK_KB ? sxp[(kb0_)] : 0.0f;                                                               \
  } while (0)
#define GK_STORE_X(bf_)                                                                                         \
  do {                                                                                                          \
    _Pragma("unroll") for (u32 i = 0; i < GK_T * 8u / NT; i++) {                                                \
      const u32 p = t0 + NT * i;                                                                                \
      *(LDSP u32x4 *)(gk_x[bf_] + (p >> 3) * GK_LS + (p & 7u) * 16u) = rx[i];                                   \
    }                                                                                                           \
    if (t0 < GK_R) _Pragma("unroll") for (u32 k = 0; k < GK_KB; k++) {                                       \
      gk_sw[bf_][k * GK_R + t0] = (float)rdd * (float)(int)(signed char)(rsw[k] & 255u); /* d sc0 */           \
      gk_mw[bf_][k * GK_R + t0] = (float)rdd * (float)(int)(signed char)(rsw[k] >> 8);   /* d sc1 */           \
    }                                                                                                           \
    if (t0 < GK_T * GK_KB) gk_sx[bf_][t0] = rsx;                                                                \
  } while (0)
#define GK_H(d_) GK_H_##d_
#define GK_H_wa ha
#define GK_H_wn hn
  GK_FETCH_W(wa, 0u);
  GK_FETCH_X(0u);
  GK_STORE_X(0u);
  barrier();
  u32 bf = 0;
#pragma unroll 1
  for (u32 kb0 = 0; kb0 < nb; kb0 += GK_KB) {
    const int more = kb0 + GK_KB < nb;
    if (more) { /* in flight during the math below */
      GK_FETCH_W(wn, kb0 + GK_KB);
      GK_FETCH_X(kb0 + GK_KB);
    }
    const LDSP u8 *lx = gk_x[bf];
    const LDSP float *lsw = gk_sw[bf], *lsx = gk_sx[bf], *lmw = gk_mw[bf];
    const u32 nk = nb - kb0 < GK_KB ? nb - kb0 : GK_KB; /* = GK_KB; the branch keeps the blocks apart for the scheduler */
#pragma unroll
    for (u32 k = 0; k < GK_KB; k++) {
      if (k >= nk) break;
      /* the standard K order: WMMA 1 = weights 0..15 (half 0), lane half h
       * bytes 8h .. 8h + 7; WMMA 2 = weights 16..31 (half 1), bytes 16 + 8h .. */
      v2i b0[GK_WJ], b1[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) {
        const LDSP u8 *pb = lx + (16u * j + (l & 15u)) * GK_LS + k * 32u + 8u * h;
        b0[j] = *(const LDSP v2i *)pb, b1[j] = *(const LDSP v2i *)(pb + 16u);
      }
      const v8i zm = (v8i)(0x4B400000); /* see ie_gemm_q8 */
      float dx[GK_WJ];
#pragma unroll
      for (u32 j = 0; j < GK_WJ; j++) dx[j] = lsx[(16u * j + (l & 15u)) * GK_KB + k];
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) { /* one row tile at a time (fewer registers) */
        v8i c0[GK_WJ], c1[GK_WJ];
        /* lane half h: the q - 32 bytes (ie_q6k_sw) of words 2h, 2h + 1 of each half */
        const u32 wk[4] = {wa[k][i].x, wa[k][i].y, wa[k][i].z, wa[k][i].w};
        const v2i a0 = {(int)ie_q6k_sw(wk[2u * h], ha[k][i].x, 0u, 2u * h), (int)ie_q6k_sw(wk[2u * h + 1u], ha[k][i].x, 0u, 2u * h + 1u)};
        const v2i a1 = {(int)ie_q6k_sw(wk[2u * h], ha[k][i].y, 1u, 2u * h), (int)ie_q6k_sw(wk[2u * h + 1u], ha[k][i].y, 1u, 2u * h + 1u)};
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) c0[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0[j], zm, 0);
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) c1[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1[j], zm, 0);
        const f8 s0 = *(const LDSP f8 *)(lsw + k * GK_R + wr + 16u * i + 8u * h);
        const f8 s1 = *(const LDSP f8 *)(lmw + k * GK_R + wr + 16u * i + 8u * h);
#pragma unroll
        for (u32 j = 0; j < GK_WJ; j++) { /* da ((d sc0) d0 + (d sc1) d1) */
          f8 in = s0 * (__builtin_bit_cast(f8, c0[j]) - 12582912.0f);
          in += s1 * (__builtin_bit_cast(f8, c1[j]) - 12582912.0f);
          acc[i][j] += dx[j] * in;
        }
      }
      __builtin_amdgcn_sched_barrier(0); /* keep the blocks apart: the scheduler otherwise overlaps them and spills */
    }
    if (more) GK_STORE_X(bf ^ 1u); /* nobody reads that buffer in this chunk */
    lbarrier();
    bf ^= 1u;
#pragma unroll
    for (u32 k = 0; k < GK_KB; k++)
#pragma unroll
      for (u32 i = 0; i < GK_WI; i++) wa[k][i] = wn[k][i], ha[k][i] = hn[k][i];
  }
#pragma unroll
  for (u32 i = 0; i < GK_WI; i++)
#pragma unroll
    for (u32 j = 0; j < GK_WJ; j++) {
      const u32 t = tbase + 16u * j + (l & 15u);
      if (t >= T) continue;
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = rbase + wr + 16u * i + 8u * h + v;
        if (r < rows) y[t * ys + r] = epilogue(acc[i][j][v], r, bias, res ? res + t * rs : res);
      }
    }
#undef GK_FETCH_W
#undef GK_FETCH_X
#undef GK_STORE_X
#undef GK_H
#undef GK_H_wa
#undef GK_H_wn
}

/* The same product with fp16 WMMA (v_wmma_f32_16x16x16_f16, f32 sums):
 * while staging the tiles in LDS, each weight becomes fp16 (int8 x its
 * block scale) and each activation too (int8 x its block scale), so the
 * WMMA sums need no per-block scaling. Tile: 128 rows x 64 tokens, K in
 * chunks of GH_K = 64; waves as in ie_gemm_q8. Lane layout as the int8
 * WMMA (8 values per lane). fp16 keeps 11 significant bits: each product
 * is rounded once to fp16 (relative error <= 2^-11) before the f32 sums. */
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
#define GH_K 64u
#define GH_LS (GH_K + 8u) /* LDS row stride in halves (16-byte aligned rows, fewer bank conflicts) */
static LDS f16 gh_w[GM_R * GH_LS];
static LDS f16 gh_x[GM_T * GH_LS];
KERNEL ie_gemm_h(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb, const G float *bias,
                 const G float *res, u32 T, u32 xs, u32 ys, u32 rs) {
  const u32 ntt = (T + GM_T - 1u) / GM_T, tt = wgid() % ntt, rt = wgid() / ntt, t0 = tid(), w = t0 >> 5, l = lane();
  const u32 rbase = rt * GM_R, tbase = tt * GM_T, h = l >> 4;
  const u32 wr = 32u * (w & 3u), wt = 32u * (w >> 2);
  const u32 K = nb * 32u;
  v8f acc[2][2];
  for (u32 i = 0; i < 2u; i++)
    for (u32 j = 0; j < 2u; j++) acc[i][j] = (v8f)(0.0f);
  for (u32 k0 = 0; k0 < K; k0 += GH_K) { /* K is a multiple of 32; a last chunk of 32 is padded with zeros */
    /* weights: 128 rows x 4 pieces of 16 int8 (one piece: half a block) */
    for (u32 p = t0; p < GM_R * 4u; p += NT) {
      const u32 r = p >> 2, pc = p & 3u, gr = rbase + r < rows ? rbase + r : rows - 1u, kk = k0 + pc * 16u;
      h8 lo = (h8)(0.0f), hi = (h8)(0.0f);
      if (kk < K) {
        const u32x4 q = *(const G u32x4 *)(qw + ie_q8_dst_word(gr, kk >> 5, (kk & 31u) >> 2, nb));
        const float sc = (float)qs[ie_q8_dst_scale(gr, kk >> 5, nb)];
        const u32 wd[4] = {q.x, q.y, q.z, q.w};
        for (u32 e = 0; e < 8u; e++) lo[e] = (f16)(sc * (float)(i8)ie_byte(wd[e >> 2], e & 3u));
        for (u32 e = 0; e < 8u; e++) hi[e] = (f16)(sc * (float)(i8)ie_byte(wd[2u + (e >> 2)], e & 3u));
      }
      *(LDSP h8 *)(gh_w + r * GH_LS + pc * 16u) = lo;
      *(LDSP h8 *)(gh_w + r * GH_LS + pc * 16u + 8u) = hi;
    }
    /* activations: 64 tokens x 4 pieces */
    for (u32 p = t0; p < GM_T * 4u; p += NT) {
      const u32 tk = p >> 2, pc = p & 3u, gt = tbase + tk < T ? tbase + tk : T - 1u, kk = k0 + pc * 16u;
      h8 lo = (h8)(0.0f), hi = (h8)(0.0f);
      if (kk < K) {
        const G u8 *xt = xq + (unsigned long)gt * xs;
        const u32x4 q = *(const G u32x4 *)(xt + kk);
        const float sc = ((const G float *)(xt + 32u * nb))[kk >> 5];
        const u32 wd[4] = {q.x, q.y, q.z, q.w};
        for (u32 e = 0; e < 8u; e++) lo[e] = (f16)(sc * (float)(i8)ie_byte(wd[e >> 2], e & 3u));
        for (u32 e = 0; e < 8u; e++) hi[e] = (f16)(sc * (float)(i8)ie_byte(wd[2u + (e >> 2)], e & 3u));
      }
      *(LDSP h8 *)(gh_x + tk * GH_LS + pc * 16u) = lo;
      *(LDSP h8 *)(gh_x + tk * GH_LS + pc * 16u + 8u) = hi;
    }
    barrier();
    for (u32 ks = 0; ks < GH_K; ks += 16u) {
      h8 a[2], bb[2];
      for (u32 i = 0; i < 2u; i++) {
        a[i] = *(const LDSP h8 *)(gh_w + (wr + 16u * i + (l & 15u)) * GH_LS + ks + 8u * h);
        bb[i] = *(const LDSP h8 *)(gh_x + (wt + 16u * i + (l & 15u)) * GH_LS + ks + 8u * h);
      }
      for (u32 i = 0; i < 2u; i++)
        for (u32 j = 0; j < 2u; j++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], bb[j], acc[i][j]);
    }
    barrier();
  }
  for (u32 i = 0; i < 2u; i++)
    for (u32 j = 0; j < 2u; j++) {
      const u32 t = tbase + wt + 16u * j + (l & 15u);
      if (t >= T) continue;
      for (u32 v = 0; v < 8u; v++) {
        const u32 r = rbase + wr + 16u * i + 8u * h + v;
        if (r < rows) y[t * ys + r] = epilogue(acc[i][j][v], r, bias, res ? res + t * rs : res);
      }
    }
}

/* y = W x for an f32 matrix (F16 weights dequantized at load):
 * the same grid; lane l does the elements ie_lane_blk(l, t) < cols. */
/* One row per workgroup (small matrices, e.g. 48 x 5120: latency bound;
 * one row per wave gave 6 workgroups, 20 dependent load rounds each).
 * Thread t takes columns t, t + 256, ..., 8 loads of each in flight; then
 * the wave tree and the workgroup sum (wg_sum: waves 0..7 in order). */
static inline float gemv_f32_row(const G float *row, const G float *x, u32 cols) {
  float acc = 0.0f;
  u32 c = tid();
  for (; c + 7u * NT < cols; c += 8u * NT) {
    float wv[8], xv[8];
#pragma unroll
    for (u32 k = 0; k < 8u; k++) wv[k] = row[c + k * NT], xv[k] = x[c + k * NT];
#pragma unroll
    for (u32 k = 0; k < 8u; k++) acc += wv[k] * xv[k];
  }
  for (; c < cols; c += NT) acc += row[c] * x[c];
  return wg_sum(acc);
}
KERNEL ie_gemv_f32(const G float *w, const G float *x, G float *y, u32 rows, u32 cols, const G float *bias,
                   const G float *res, const G float *nw, G float *ny, G u8 *nq, float eps, G u32 *count) {
  const u32 r = wgid();
  if (r < rows) {
    const float acc = gemv_f32_row(w + (unsigned long)r * cols, x, cols);
    if (tid() == 0u) y[r] = epilogue(acc, r, bias, res);
  }
  if (nw) gemv_norm_tail(y, rows, nw, ny, nq, eps, count);
}

/* ie_gemv_f32 for T tokens in one launch (grid y = token; byte strides xs,
 * ys, rs between tokens): the same arithmetic per token. */
KERNEL ie_gemv_f32_t(const G float *w, const G float *x, G float *y, u32 rows, u32 cols, const G float *bias,
                     const G float *res, u32 xs, u32 ys, u32 rs) {
  x = TOKC(x, xs), y = TOK(y, ys), res = TOK(res, rs);
  const u32 r = wgid();
  if (r >= rows) return;
  const float acc = gemv_f32_row(w + (unsigned long)r * cols, x, cols);
  if (tid() == 0u) y[r] = epilogue(acc, r, bias, res);
}

/* ie_gemv_f32_t for prompt chunks (many tokens, not bitwise equal to
 * ie_gemv_f32): one row per wave, 8 rows per workgroup. */
KERNEL ie_gemv_f32_tw(const G float *w, const G float *x, G float *y, u32 rows, u32 cols, const G float *bias,
                      const G float *res, u32 xs, u32 ys, u32 rs) {
  x = TOKC(x, xs), y = TOK(y, ys), res = TOK(res, rs);
  const u32 r = ie_gemv_row(wgid(), tid() >> 5), l = lane();
  if (r >= rows) return;
  const G float *row = w + (unsigned long)r * cols;
  float acc = 0.0f;
  u32 t = 0;
  for (; ie_lane_blk(l, t + 7u) < cols; t += 8u) {
    float wv[8], xv[8];
#pragma unroll
    for (u32 k = 0; k < 8u; k++) wv[k] = row[ie_lane_blk(l, t + k)], xv[k] = x[ie_lane_blk(l, t + k)];
#pragma unroll
    for (u32 k = 0; k < 8u; k++) acc += wv[k] * xv[k];
  }
  for (; ie_lane_blk(l, t) < cols; t++) acc += row[ie_lane_blk(l, t)] * x[ie_lane_blk(l, t)];
  acc = wave_tree_f(acc);
  if (l == 0u) y[r] = epilogue(acc, r, bias, res);
}

/* f32 matrix x prompt chunk, for small matrices (the 96 x 5120 projections
 * of qwen35): workgroup (g, tt) = rows 8 g .. + 7 (one per wave) and tokens
 * 8 tt .. + 7. Per chunk of 256 columns the 8 tokens' x go to LDS once for
 * the 8 waves; lane l takes the columns l, l + 32, .. of its row, each
 * weight used for the 8 tokens. Then a wave sum per token. Not bitwise equal
 * to ie_gemv_f32. */
#define GS_T 8u
#define GS_C 256u
static LDS float gs_x[GS_T][GS_C];
KERNEL ie_gemm_f32_s(const G float *w, const G float *x, G float *y, u32 rows, u32 cols, const G float *bias,
                     const G float *res, u32 xs, u32 ys, u32 rs, u32 T) {
  const u32 ntt = (T + GS_T - 1u) / GS_T, tt = wgid() % ntt, g = wgid() / ntt, wv = tid() >> 5, l = lane();
  const u32 r = 8u * g + wv, t0 = GS_T * tt;
  const G float *row = w + (unsigned long)(r < rows ? r : rows - 1u) * cols;
  float acc[GS_T];
#pragma unroll
  for (u32 t = 0; t < GS_T; t++) acc[t] = 0.0f;
  for (u32 c0 = 0; c0 < cols; c0 += GS_C) {
    const u32 nc = cols - c0 < GS_C ? cols - c0 : GS_C;
#pragma unroll
    for (u32 n = 0; n < GS_T * GS_C / NT; n++) { /* 8 tokens x 256 columns */
      const u32 i = tid() + NT * n, t = i / GS_C, c = i % GS_C, tk = t0 + t < T ? t0 + t : T - 1u;
      gs_x[t][c] = c < nc ? ((const G float *)((const G u8 *)x + (unsigned long)tk * xs))[c0 + c] : 0.0f;
    }
    barrier();
#pragma unroll
    for (u32 k = 0; k < GS_C / 32u; k++) {
      const u32 c = l + 32u * k;
      const float wk = c < nc ? row[c0 + c] : 0.0f;
#pragma unroll
      for (u32 t = 0; t < GS_T; t++) acc[t] += wk * gs_x[t][c];
    }
    barrier();
  }
#pragma unroll
  for (u32 t = 0; t < GS_T; t++) {
    const float a = wave_tree_f(acc[t]);
    const u32 tk = t0 + t;
    if (l == 0u && r < rows && tk < T)
      ((G float *)((G u8 *)y + (unsigned long)tk * ys))[r] = epilogue(a, r, bias, res ? (const G float *)((const G u8 *)res + (unsigned long)tk * rs) : res);
  }
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
KERNEL ie_rope_kv(G float *q, G float *k, const G float *v, G kvt *kc, G kvt *vc, const G float *cs,
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
KERNEL ie_kv_store(const G float *k, const G float *v, G kvt *kc, G kvt *vc, u32 n) {
  const u32 i = wgid() * NT + tid();
  if (i < n) {
    kc[i] = k[i];
    vc[i] = v[i];
  }
}

/* --------------------------------------------------------------- attention */

#ifndef IE_KV_F32
KERNEL ie_kv_f16(void) {} /* marker: the KV cache is fp16 */
#endif

/* One decode token. One workgroup per query head h (GQA: KV head h / grp).
 * kc, vc: the caches of this layer, [n_ctx][n_kv * hd]. sc: scores
 * [n_head][n_ctx]. Positions 0..pos. */
static inline float sigmoidf(float x) { return 1.0f / (1.0f + __builtin_expf(-x)); }

KERNEL ie_attn(const G float *q, const G kvt *kc, const G kvt *vc, G float *sc, G float *out, u32 pos,
               u32 n_ctx, u32 hd, u32 n_head, u32 n_kv, G u8 *oq, const G float *gate, u32 gstride) {
  const u32 h = wgid(), kh = h / (n_head / n_kv), kvd = n_kv * hd;
  const G float *qh = q + h * hd;
  G float *s = sc + h * n_ctx;
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  float mx = -__builtin_inff();
  for (u32 p = tid(); p <= pos; p += NT) {
    const G kvt *k = kc + (unsigned long)p * kvd + kh * hd;
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
static LDS float at_w[4096];
static inline void attn_split_body(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out,
                                   u32 pos, u32 hd, u32 n_head, u32 n_kv, u32 nsplit, u32 ch, G u32 *count, G u8 *oq,
                                   const G float *gate, u32 gstride, u32 h, u32 s) {
  const u32 kh = h / (n_head / n_kv), kvd = n_kv * hd;
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
    const G kvt *k = kc + (unsigned long)p * kvd + kh * hd;
    const G kvt *v = vc + (unsigned long)p * kvd + kh * hd;
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
KERNEL ie_attn_split(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos,
                     u32 hd, u32 n_head, u32 n_kv, u32 nsplit, u32 ch, G u32 *count, G u8 *oq, const G float *gate,
                     u32 gstride) {
  attn_split_body(q, kc, vc, part, out, pos, hd, n_head, n_kv, nsplit, ch, count, oq, gate, gstride, wgid() / nsplit,
                  wgid() % nsplit);
}

/* ie_attn_split for T tokens in one launch (speculative verify; grid y =
 * token j at position pos + j): per token the splits, the arithmetic and the
 * merge of ie_attn_split at that position (bitwise equal). ch: ie_att_ch. Grid x: n_head *
 * smax (smax: the splits of the last token); a workgroup past the splits of
 * its token returns at once. Per token: its part area (pstride floats), its
 * counters (n_head) and the byte strides of q, out, oq and gate. */
KERNEL ie_attn_split_t(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos,
                       u32 hd, u32 n_head, u32 n_kv, u32 smax, u32 ch, G u32 *count, G u8 *oq, const G float *gate,
                       u32 gstride, u32 qst, u32 ost, u32 oqst, u32 gst, u32 pstride) {
  const u32 j = __builtin_amdgcn_workgroup_id_y(), pj = pos + j;
  const u32 nsplit = (pj + ch) / ch; /* ie_att_nsplit (src/model.h): ch does not depend on the position */
  const u32 h = wgid() / smax, s = wgid() % smax;
  if (s >= nsplit) return;
  attn_split_body((const G float *)((const G u8 *)q + (unsigned long)j * qst), kc, vc, part + (unsigned long)j * pstride,
                  (G float *)((G u8 *)out + (unsigned long)j * ost), pj, hd, n_head, n_kv, nsplit, ch, count + j * n_head,
                  oq ? oq + (unsigned long)j * oqst : oq, gate ? (const G float *)((const G u8 *)gate + (unsigned long)j * gst) : gate,
                  gstride, h, s);
}

/* Split attention with the K/V read once per KV head (ie_attn_kv): for T
 * tokens at positions pos .. pos + T - 1 and the G = n_head / n_kv query
 * heads of one KV head. Workgroup g = (kh * ngs + sg) * nb + b: KV head kh,
 * the splits sg * spw .. sg * spw + spw - 1 (split s: positions s*ch ..
 * s*ch + ch - 1, as ie_attn_split), pairs 8b .. 8b + 7 of the G * T
 * (query head, token) pairs of kh (pair i: token i / G, query head
 * kh * G + i % G). K and V go to LDS, 8 positions at a time (16-byte loads,
 * the next 8 positions in flight during the math); wave w does pair 8b + w.
 * Bitwise equal per (head, token) to attn_split_body at that position: wave
 * x of ie_attn_split does the positions s*ch + x, + 8, ... of split s; here
 * one wave keeps 8 streams x = 0 .. 7 with the same positions in the same
 * order and the same arithmetic, merges them in the order of the LDS merge
 * (x = 0 .. 7) and writes the split's (M, L, o) to the same place. A
 * workgroup does spw splits one after the other (the splits are not
 * changed; spw only sets the work per workgroup), then adds its number of
 * splits to count[j * n_head + h]; the workgroup that completes the count
 * merges the splits with all 256 threads, as attn_split_body (the same sums
 * in the same order, its loads issued 8 at a time). LDS: the K/V buffer also
 * holds the merge weights (16 KB in all). hd <= 256, hd % 4 == 0,
 * ch % 8 == 0, kc and vc 8-byte aligned. */
#define AKV_P 8u
static LDS float akv_kv[2][AKV_P][256]; /* K, V; in the merge: the weights of the splits (<= 4096) */
static LDS u32 akv_last[8];
KERNEL ie_attn_kv(const G float *q, const G kvt *kc, const G kvt *vc, G float *part, G float *out, u32 pos, u32 hd,
                  u32 n_head, u32 n_kv, u32 smax, u32 ch, G u32 *count, G u8 *oq, const G float *gate, u32 gstride, u32 qst,
                  u32 ost, u32 oqst, u32 gst, u32 pstride, u32 T, u32 nb, u32 spw) {
  const u32 grp = n_head / n_kv, kvd = n_kv * hd, ngs = (smax + spw - 1u) / spw;
  const u32 b = wgid() % nb, sg = (wgid() / nb) % ngs, kh = wgid() / (nb * ngs);
  const u32 w = tid() >> 5, l = lane();
  const u32 pi = 8u * b + w;                  /* this wave's pair */
  const u32 j = pi / grp, h = kh * grp + pi % grp;
  const u32 pj = pos + j, nsplit = (pj + ch) / ch; /* ie_att_nsplit for token j */
  const u32 s0 = sg * spw, s1 = s0 + spw < smax ? s0 + spw : smax;
  const int on = pi < grp * T && s0 < nsplit; /* wave-uniform: this pair has splits here */
  const u32 ndone = on ? (s1 < nsplit ? s1 : nsplit) - s0 : 0u; /* its splits in this workgroup */
  /* the workgroup's positions: up to the last token's */
  const u32 P0 = s0 * ch, PE = s1 * ch < pos + T ? s1 * ch : pos + T;
  if (P0 >= PE) return; /* no token has these splits (workgroup-uniform) */
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  const G float *qh = on ? (const G float *)((const G u8 *)q + (unsigned long)j * qst) + h * hd : q;
  float qv[8];
  for (u32 jj = 0; jj < 8u; jj++) {
    const u32 i = l + 32u * jj;
    qv[jj] = on && i < hd ? qh[i] : 0.0f;
  }
  float sm[8], ss[8], sa[8][8]; /* the 8 streams (the waves of ie_attn_split) of the current split */
  const u32 h4 = hd >> 2, n4 = AKV_P * h4; /* float4 per row, per pass (<= 512) */
  f32x4 kr[2], vr[2];
#define AKV_LOAD(c)                                                                          \
  for (u32 u = 0; u < 2u; u++) {                                                             \
    const u32 i4 = tid() + NT * u, r = i4 / h4, e4 = i4 - r * h4;                           \
    kr[u] = vr[u] = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};                                          \
    if (i4 < n4 && (c) + r < PE) {                                                           \
      const unsigned long o = ((unsigned long)((c) + r) * kvd + kh * hd) / 4u + e4;          \
      kr[u] = KV4(kc, o);                                                                    \
      vr[u] = KV4(vc, o);                                                                    \
    }                                                                                        \
  }
  AKV_LOAD(P0)
  for (u32 c = P0; c < PE; c += AKV_P) {
    const u32 s = c / ch, sp0 = s * ch, sp1 = sp0 + ch; /* the split of these 8 positions */
    if (c == sp0) /* a new split: new streams */
      for (u32 x = 0; x < 8u; x++) {
        sm[x] = -__builtin_inff(), ss[x] = 0.0f;
        for (u32 jj = 0; jj < 8u; jj++) sa[x][jj] = 0.0f;
      }
    for (u32 u = 0; u < 2u; u++) {
      const u32 i4 = tid() + NT * u, r = i4 / h4, e4 = i4 - r * h4;
      if (i4 < n4) {
        *(LDSP f32x4 *)&akv_kv[0][r][4u * e4] = kr[u];
        *(LDSP f32x4 *)&akv_kv[1][r][4u * e4] = vr[u];
      }
    }
    barrier();
    if (c + AKV_P < PE) { AKV_LOAD(c + AKV_P) }
    const int live = on && s < nsplit; /* this pair has split s */
    const u32 p1 = sp1 < pj + 1u ? sp1 : pj + 1u; /* this pair's end in split s */
    if (live)
#pragma unroll
      for (u32 r = 0; r < AKV_P; r++) {
        /* no branch: the 8 streams are independent, so their math can
         * interleave; a position past this pair's end keeps the old state */
        const u32 p = c + r, x = r; /* stream (p - sp0) % 8; c - sp0 is a multiple of 8 */
        const int ok = p < p1;
        float d = 0.0f;
        for (u32 jj = 0; jj < 8u; jj++)
          if (l + 32u * jj < hd) d += qv[jj] * akv_kv[0][r][l + 32u * jj];
        d = wave_sum_all(d);
        d *= scale;
        const float mn = __builtin_fmaxf(sm[x], d), cc = __builtin_expf(sm[x] - mn), e = __builtin_expf(d - mn);
        const float s2 = ss[x] * cc + e;
        ss[x] = ok ? s2 : ss[x];
        for (u32 jj = 0; jj < 8u; jj++)
          if (l + 32u * jj < hd) {
            const float a2 = sa[x][jj] * cc + e * akv_kv[1][r][l + 32u * jj];
            sa[x][jj] = ok ? a2 : sa[x][jj];
          }
        sm[x] = ok ? mn : sm[x];
      }
    barrier();
    /* the end of split s (for this workgroup): merge the 8 streams as the LDS
     * merge of attn_split_body and write the split's (M, L, o) */
    if (live && (c + AKV_P >= sp1 || c + AKV_P >= PE)) {
      float M = sm[0];
      for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, sm[x]);
      G float *pp = part + (unsigned long)j * pstride + (unsigned long)(h * nsplit + s) * (hd + 2u);
      for (u32 jj = 0; jj < 8u; jj++) {
        const u32 i = l + 32u * jj;
        if (i < hd) {
          float o = 0.0f;
          for (u32 x = 0; x < 8u; x++) o += sa[x][jj] * __builtin_expf(sm[x] - M);
          pp[2u + i] = o;
        }
      }
      if (l == 0u) {
        float L = 0.0f;
        for (u32 x = 0; x < 8u; x++) L += ss[x] * __builtin_expf(sm[x] - M);
        pp[0] = M;
        pp[1] = L;
      }
    }
  }
#undef AKV_LOAD
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  barrier();
  if (l == 0u)
    akv_last[w] = ndone ? __atomic_fetch_add(&count[j * n_head + h], ndone, __ATOMIC_ACQ_REL) + ndone == nsplit : 0u;
  barrier();
  /* the merges this workgroup completed (all 256 threads, as attn_split_body) */
  LDSP float *wv = &akv_kv[0][0][0]; /* the weights of the splits (the K/V buffer is free now) */
  for (u32 x = 0; x < 8u; x++) {
    if (!akv_last[x]) continue;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    const u32 px = 8u * b + x, jx = px / grp, hx = kh * grp + px % grp, ns = (pos + jx + ch) / ch;
    const G float *ph = part + (unsigned long)jx * pstride + (unsigned long)hx * ns * (hd + 2u);
    float mv = -__builtin_inff();
    for (u32 t = tid(); t < ns; t += NT) mv = __builtin_fmaxf(mv, ph[t * (hd + 2u)]);
    const float M2 = wg_max(mv);
    float lv = 0.0f;
    for (u32 t = tid(); t < ns; t += NT) {
      const float wt = __builtin_expf(ph[t * (hd + 2u)] - M2);
      wv[t] = wt;
      lv += ph[t * (hd + 2u) + 1u] * wt;
    }
    const float L2 = wg_sum(lv); /* its barrier also makes wv visible */
    G float *ox = (G float *)((G u8 *)out + (unsigned long)jx * ost);
    G u8 *oqx = oq ? oq + (unsigned long)jx * oqst : oq;
    const G float *gx = gate ? (const G float *)((const G u8 *)gate + (unsigned long)jx * gst) : gate;
    if (tid() < hd) {
      float o = 0.0f;
      u32 t = 0;
      for (; t + 8u <= ns; t += 8u) { /* 8 loads in flight; the sum in the order t = 0, 1, ... */
        float pv[8];
        for (u32 u = 0; u < 8u; u++) pv[u] = ph[(t + u) * (hd + 2u) + 2u + tid()];
        for (u32 u = 0; u < 8u; u++) o += pv[u] * wv[t + u];
      }
      for (; t < ns; t++) o += ph[t * (hd + 2u) + 2u + tid()] * wv[t];
      o = o / L2;
      if (gx) o = o * sigmoidf(gx[hx * gstride + tid()]);
      ox[hx * hd + tid()] = o;
      if (oqx) quant_wave(o, (hx * hd + tid()) >> 5, oqx, (n_head * hd) >> 5);
    }
    if (tid() == 0u) count[jx * n_head + hx] = 0u;
    barrier(); /* wv is used again by the next merge */
  }
}

/* ----------------------------------------------------------------- qwen35 */

/* Full attention, before the attention: workgroup g < nq: query head g, at
 * qkv + g * 2 hd (q of [q | gate]); nq <= g < nq + nkv: key head g - nq,
 * at qkv + 2 nq hd + (g - nq) hd; then nkv workgroups copy the value heads.
 * A q or k head: y = rmsnorm(x) * w (hd <= 256 values, thread t holds value
 * t), then RoPE of the first nrot values (pairs (i, i + nrot/2)); q goes to
 * qo + g * hd, k to the cache kc. */
static LDS float qk_buf[256];
KERNEL ie_qkn_rope_kv(const G float *qkv, G float *qo, G kvt *kc, G kvt *vc, const G float *qw, const G float *kw,
                      const G float *cs, const G float *sn, u32 nq, u32 nkv, u32 hd, u32 nrot, float eps, u32 ins,
                      u32 qos) {
  /* token y of a multi-token launch: position + y (KV rows and RoPE tables) */
  qkv = TOKC(qkv, ins), qo = TOK(qo, qos);
  kc = TOK(kc, nkv * hd * (u32)sizeof(kvt)), vc = TOK(vc, nkv * hd * (u32)sizeof(kvt));
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
  if (t < hd) {
    if (is_k) kc[h * hd + t] = (kvt)r;
    else qo[h * hd + t] = r;
  }
}

/* Linear attention (Gated DeltaNet) of T <= 16 tokens at positions pos ..
 * pos + T - 1, one after the other; one workgroup per value head h (key
 * head kh = h % nk), state dims 128 x 128. Token t: in + t * is = [q k v (cd)
 * | z (inner) | beta (nv) | alpha (nv)], inner = nv * 128; output at out +
 * t * os, its Q8 copy at oq + t * qs. ring: the conv inputs of the last 32
 * positions [slot][cd] (slot = position % 32). S: ns (4 or 8) state slots [slot][h][i][j]
 * (i: key dim, j: value dim); the state is read from slot c and the state
 * after token t is written to slot (c + t) % ns (only for t >= wfrom: a prompt chunk keeps only its last state), so that a speculative step
 * can go back to the state after any of its tokens. Each thread reads and
 * writes only its own state values, so slot c can be overwritten. Thread t:
 * value dim j = t % 128, key dims i of half t / 128. The CPU backend (cpu.c,
 * gdn) does the same arithmetic for one token (slot c = 0). */
#define GDN_RING 32u
static LDS float gd_q[128], gd_k[128], gd_r[8];
/* conv input of channel c at token u of this launch, or (before the launch)
 * from the ring; 0 before position 0 */
static inline float conv_in(const G float *in, u32 is, const G float *ring, u32 c, u32 t, u32 d, u32 pos, u32 cd) {
  if (d <= t) return in[(t - d) * is + c];
  const u32 p = pos + t;
  return p >= d ? ring[((p - d) & (GDN_RING - 1u)) * cd + c] : 0.0f;
}
static inline float conv_ch(const G float *in, u32 is, G float *ring, const G float *cw, u32 c, u32 t, u32 pos, u32 cd,
                            int writer) { /* writer = 0: the host stores the ring after the launch (ie_ring_store) */
  float acc = 0.0f;
  for (u32 j = 0; j < 4u; j++) /* oldest first, as ggml_ssm_conv */
    acc += conv_in(in, is, ring, c, t, 3u - j, pos, cd) * cw[c * 4u + j];
  /* slot (pos + t) % GDN_RING: no workgroup reads it in this launch (reads
   * of the ring are of positions pos - 3 .. pos - 1, writes of pos .. pos +
   * 15: 19 slots < GDN_RING) */
  if (writer) ring[((pos + t) & (GDN_RING - 1u)) * cd + c] = in[t * is + c];
  return acc / (1.0f + __builtin_expf(-acc));
}
/* Linear attention for 1 .. 4 tokens (decode, speculative verify): one
 * workgroup per value head, the gated RMSNorm and the Q8 copy fused (no
 * second launch). Thread t: value dim j = t % 128, key dims i of half t /
 * 128. Same state slots and ring as ie_gdn. */
static LDS float gd1_v[128], gd1_p[2][128];
KERNEL ie_gdn1(const G float *in, G float *out, G u8 *oq, G float *ring, G float *S, const G float *cw, const G float *dtb,
               const G float *sa, const G float *nw, u32 pos, u32 cd, u32 nk, u32 nv, float eps, u32 T, u32 is, u32 os,
               u32 qs, u32 c, u32 wfrom, u32 rw, u32 ns) {
  const u32 h = wgid(), t0 = tid(), j = t0 & 127u, half = t0 >> 7, kh = h % nk, w = t0 >> 5, inner = nv * 128u;
  const unsigned long slot = (unsigned long)nv * 16384u, own = (unsigned long)h * 16384u + half * 64u * 128u + j;
  float st[64];
  for (u32 ii = 0; ii < 64u; ii++) st[ii] = pos ? S[c * slot + own + ii * 128u] : 0.0f;
  for (u32 t = 0; t < T; t++) {
    const G float *it = in + t * is;
    const float a = conv_ch(in, is, ring, cw, half ? nk * 128u + kh * 128u + j : kh * 128u + j, t, pos, cd, rw && h == kh);
    const float vv = half ? 0.0f : conv_ch(in, is, ring, cw, 2u * nk * 128u + h * 128u + j, t, pos, cd, rw);
    const float ss = wave_sum_all(a * a);
    if (lane() == 0u) gd_r[w] = ss;
    barrier();
    const float n2 = half ? gd_r[4] + gd_r[5] + gd_r[6] + gd_r[7] : gd_r[0] + gd_r[1] + gd_r[2] + gd_r[3];
    const float an = a * (1.0f / __builtin_sqrtf(n2 + eps));
    if (half) gd_k[j] = an;
    else gd_q[j] = an, gd1_v[j] = vv;
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
    gd1_p[half][j] = pk;
    barrier();
    const float delta = (gd1_v[j] - (gd1_p[0][j] + gd1_p[1][j])) * beta;
    barrier();
    float po = 0.0f;
    G float *Sw = S + ((c + t) % ns) * slot + own;
    for (u32 ii = 0; ii < 64u; ii++) {
      st[ii] += gd_k[half * 64u + ii] * delta;
      if (t >= wfrom) Sw[ii * 128u] = st[ii]; /* uniform */
      po += st[ii] * gd_q[half * 64u + ii];
    }
    gd1_p[half][j] = po;
    barrier();
    const float o = (gd1_p[0][j] + gd1_p[1][j]) * (1.0f / __builtin_sqrtf(128.0f));
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
    barrier();
  }
}

/* (ie_gdn) Workgroup (h, cq): value head h, value columns 32 cq .. 32 cq + 31
 * (4 workgroups per head: more workgroups than heads keeps the GPU busy).
 * Wave w holds the state rows (key dims) 16 w .. 16 w + 15 of these columns:
 * lane l, column j = 32 cq + l, 16 state values in registers. The conv, L2
 * norm and gates are computed by every workgroup of the head (cheap); the
 * output o (before the gated norm) goes to out, and ie_gnorm then does the
 * gated RMSNorm, which needs the whole head. */
#define GD_SPLIT 4u
static LDS float gd_p8[8][32], gd_v32[32];
KERNEL ie_gdn(const G float *in, G float *out, G float *ring, G float *S, const G float *cw, const G float *dtb,
              const G float *sa, u32 pos, u32 cd, u32 nk, u32 nv, float eps, u32 T, u32 is, u32 os, u32 c, u32 wfrom,
              u32 rw, u32 ns) {
  const u32 h = wgid() / GD_SPLIT, cq = wgid() % GD_SPLIT, t0 = tid(), w = t0 >> 5, l = lane(), kh = h % nk;
  const u32 j128 = t0 & 127u, half = t0 >> 7, inner = nv * 128u;
  const unsigned long slot = (unsigned long)nv * 16384u, own = (unsigned long)h * 16384u + (16u * w) * 128u + 32u * cq + l;
  /* the state loads first: they do not depend on the conv, so their
   * memory latency overlaps it */
  float st[16];
  for (u32 ii = 0; ii < 16u; ii++) st[ii] = pos ? S[c * slot + own + ii * 128u] : 0.0f;
  for (u32 t = 0; t < T; t++) {
    const G float *it = in + t * is;
    /* conv + SiLU: threads < 128 do q (channel kh*128 + j128), threads >=
     * 128 do k; threads < 32 also v (this workgroup's 32 columns). Ring
     * writers: q and k by workgroup (kh, 0), v by its workgroup. */
    const float a = conv_ch(in, is, ring, cw, half ? nk * 128u + kh * 128u + j128 : kh * 128u + j128, t, pos, cd,
                            rw && h == kh && cq == 0u);
    const float vv = t0 < 32u ? conv_ch(in, is, ring, cw, 2u * nk * 128u + h * 128u + 32u * cq + t0, t, pos, cd, rw) : 0.0f;
    /* L2 norm of q (waves 0-3) and k (waves 4-7): x / sqrt(sum x^2 + eps) */
    const float ss = wave_sum_all(a * a);
    if (lane() == 0u) gd_r[w] = ss;
    barrier();
    const float n2 = half ? gd_r[4] + gd_r[5] + gd_r[6] + gd_r[7] : gd_r[0] + gd_r[1] + gd_r[2] + gd_r[3];
    const float an = a * (1.0f / __builtin_sqrtf(n2 + eps));
    if (half) gd_k[j128] = an;
    else gd_q[j128] = an;
    if (t0 < 32u) gd_v32[t0] = vv;
    const float beta = sigmoidf(it[cd + inner + h]);
    const float al = it[cd + inner + nv + h] + dtb[h];
    const float sp = al > 20.0f ? al : __builtin_logf(1.0f + __builtin_expf(al));
    const float decay = __builtin_expf(sp * sa[h]);
    barrier();
    float pk = 0.0f;
    for (u32 ii = 0; ii < 16u; ii++) {
      st[ii] = st[ii] * decay;
      pk += st[ii] * gd_k[16u * w + ii];
    }
    gd_p8[w][l] = pk;
    barrier();
    float kv = 0.0f;
    for (u32 x = 0; x < 8u; x++) kv += gd_p8[x][l];
    const float delta = (gd_v32[l] - kv) * beta;
    barrier();
    float po = 0.0f;
    G float *Sw = S + ((c + t) % ns) * slot + own;
    for (u32 ii = 0; ii < 16u; ii++) {
      st[ii] += gd_k[16u * w + ii] * delta;
      if (t >= wfrom) Sw[ii * 128u] = st[ii]; /* uniform */
      po += st[ii] * gd_q[16u * w + ii];
    }
    gd_p8[w][l] = po;
    barrier();
    if (w == 0u) {
      float o = 0.0f;
      for (u32 x = 0; x < 8u; x++) o += gd_p8[x][l];
      out[t * os + h * 128u + 32u * cq + l] = o * (1.0f / __builtin_sqrtf(128.0f));
    }
    barrier(); /* the LDS of this token is read before the next token writes it */
  }
}

/* The gated RMSNorm of the linear attention output, in place: per value
 * head (128 values) y = rmsnorm(o) * nw * silu(z), z from the input row (in
 * + cd); and the Q8 copy. Wave w of workgroup (g, token y) does head 8 g +
 * w; lane l holds the values l, l + 32, l + 64, l + 96 (one per Q8 block). */
KERNEL ie_gnorm(G float *out, const G float *in, const G float *nw, G u8 *oq, u32 nv, u32 cd, float eps, u32 is, u32 os,
                u32 qs) {
  out = TOK(out, os), in = TOKC(in, is), oq = TOK(oq, qs);
  const u32 h = wgid() * 8u + (tid() >> 5), l = lane();
  if (h >= nv) return; /* whole waves */
  float o[4], s2 = 0.0f;
  for (u32 q = 0; q < 4u; q++) o[q] = out[h * 128u + 32u * q + l], s2 += o[q] * o[q];
  s2 = wave_sum_all(s2);
  const float r = 1.0f / __builtin_sqrtf(s2 / 128.0f + eps);
  for (u32 q = 0; q < 4u; q++) {
    const u32 jj = 32u * q + l;
    const float z = in[cd + h * 128u + jj];
    const float y = o[q] * r * nw[jj] * (z / (1.0f + __builtin_expf(-z)));
    out[h * 128u + jj] = y;
    if (oq) quant_wave(y, (h * 128u + 32u * q) >> 5, oq, (nv * 128u) >> 5);
  }
}

/* ---- Linear attention of a prompt chunk, the chunked (parallel) form ----
 * For a block of GC tokens with start state S0 (S[i][j], i key, j value),
 * gamma_t = the product of the decays g of tokens 1 .. t of the block:
 *   delta_t = beta_t (v_t - gamma_t S0^T k_t - sum_{s<t} (gamma_t/gamma_s) (k_t.k_s) delta_s)
 * so Delta = U - W S0 with U = Tm diag(beta) V, W = Tm diag(beta gamma) K,
 * Tm = (I + L)^-1, L[t][s] = beta_t (gamma_t/gamma_s) k_t.k_s (s < t): U and
 * W do not depend on S0. Then
 *   O = diag(gamma) Q S0 + M Delta,  M[t][s] = (gamma_t/gamma_s) q_t.k_s (s <= t)
 *   S_new = gamma_n S0 + K^T diag(gamma_n / gamma) Delta.
 * (tools/gdn_chunk.py checks the identity against the token recurrence.)
 * Scratch (global, per run): qk[t][nk][256] (q (times 1/sqrt(128)) | k, L2
 * normed), vv[t][nv][128], bg[t][nv][2] (beta, log g), U, W [t][nv][128],
 * Mb[block][nv][GC][GC], Gc[t][nv] (cumulative log g in the block). */
#define GC 32u
typedef float f4 __attribute__((ext_vector_type(4)));
/* 1: conv + SiLU of all channels, L2 norms, beta and log g; workgroup y = token y */
KERNEL ie_gdn_prep(const G float *in, const G float *ring, const G float *cw, const G float *dtb, const G float *sa,
                   u32 pos, u32 cd, u32 nk, u32 nv, float eps, u32 is, G float *qk, G float *vv, G float *bg) {
  const u32 t = tokid(), inner = nv * 128u, nq = nk * 128u;
  const G float *it = in + t * is;
  for (u32 c = tid(); c < cd; c += NT) {
    float acc = 0.0f;
    for (u32 j = 0; j < 4u; j++) acc += conv_in(in, is, ring, c, t, 3u - j, pos, cd) * cw[c * 4u + j];
    const float y = acc / (1.0f + __builtin_expf(-acc));
    if (c < 2u * nq) {
      const u32 isk = c >= nq, cc = isk ? c - nq : c;
      qk[((unsigned long)t * nk + cc / 128u) * 256u + isk * 128u + cc % 128u] = y;
    } else {
      vv[(unsigned long)t * inner + (c - 2u * nq)] = y;
    }
  }
  for (u32 h = tid(); h < nv; h += NT) {
    const float al = it[cd + inner + nv + h] + dtb[h];
    const float sp = al > 20.0f ? al : __builtin_logf(1.0f + __builtin_expf(al));
    bg[((unsigned long)t * nv + h) * 2u] = 1.0f / (1.0f + __builtin_expf(-it[cd + inner + h]));
    bg[((unsigned long)t * nv + h) * 2u + 1u] = sp * sa[h];
  }
  barrier(); /* the conv outputs of this workgroup are visible to it */
  const u32 w = tid() >> 5, l = lane();
  for (u32 hh = w; hh < 2u * nk; hh += 8u) { /* q of key head hh / 2 (even) or its k (odd) */
    G float *x = qk + ((unsigned long)t * nk + hh / 2u) * 256u + (hh & 1u) * 128u;
    float v4[4], ss = 0.0f;
    for (u32 q = 0; q < 4u; q++) v4[q] = x[l + 32u * q], ss += v4[q] * v4[q];
    ss = wave_sum_all(ss);
    const float r = (1.0f / __builtin_sqrtf(ss + eps)) * ((hh & 1u) ? 1.0f : 1.0f / __builtin_sqrtf(128.0f));
    for (u32 q = 0; q < 4u; q++) x[l + 32u * q] = v4[q] * r;
  }
}

/* 2: U, W and M of block b for value head h (workgroup h + nv b) */
static LDS float wy_k[GC][132] __attribute__((aligned(16))), wy_q[GC][132] __attribute__((aligned(16))), wy_L[GC][GC] __attribute__((aligned(16))), wy_G[GC], wy_b[GC], wy_lg[GC];
/* rows of 132: 16-byte reads of 8 lanes (8 rows) fall on different banks */
KERNEL ie_gdn_wy(const G float *qk, const G float *vv, const G float *bg, G float *U, G float *W, G float *Mb, G float *Gc,
                 u32 T, u32 nk, u32 nv) {
  const u32 h = wgid() % nv, b = wgid() / nv, t0 = b * GC, n = T - t0 < GC ? T - t0 : GC, kh = h % nk, th = tid();
  {
    f4 q4[4], k4[4];
#pragma unroll
    for (u32 k = 0; k < 4u; k++) { /* float4 e: row e / 32, dims 4 (e % 32) .. + 3 */
      const u32 e = th + NT * k, r = e >> 5;
      const G float *src = qk + ((unsigned long)(t0 + r) * nk + kh) * 256u + 4u * (e & 31u);
      q4[k] = r < n ? *(const G f4 *)src : (f4){0.0f, 0.0f, 0.0f, 0.0f};
      k4[k] = r < n ? *(const G f4 *)(src + 128u) : (f4){0.0f, 0.0f, 0.0f, 0.0f};
    }
#pragma unroll
    for (u32 k = 0; k < 4u; k++) {
      const u32 e = th + NT * k;
      *(LDSP f4 *)&wy_q[e >> 5][4u * (e & 31u)] = q4[k];
      *(LDSP f4 *)&wy_k[e >> 5][4u * (e & 31u)] = k4[k];
    }
  }
  if (th < GC) {
    wy_b[th] = th < n ? bg[((unsigned long)(t0 + th) * nv + h) * 2u] : 0.0f;
    wy_lg[th] = th < n ? bg[((unsigned long)(t0 + th) * nv + h) * 2u + 1u] : 0.0f;
  }
  lbarrier();
  if (th < GC) { /* cumulative log decay in the block */
    float G0 = 0.0f;
    for (u32 r = 0; r <= th; r++) G0 += wy_lg[r];
    wy_G[th] = G0;
    if (th < n) Gc[(unsigned long)(t0 + th) * nv + h] = G0;
  }
  lbarrier();
  /* L (strictly lower) and M (lower with the diagonal): thread e does pairs e, e + 256, .. */
  for (u32 e = th; e < GC * GC; e += NT) {
    const u32 r = e / GC, c = e % GC;
    float kk = 0.0f, qq = 0.0f;
    if (r < n && c <= r) {
      float kk2 = 0.0f, qq2 = 0.0f;
#pragma unroll 4
      for (u32 i = 0; i < 128u; i += 8u) {
        const f4 a0 = *(const LDSP f4 *)&wy_k[r][i], a1 = *(const LDSP f4 *)&wy_k[r][i + 4u];
        const f4 q0 = *(const LDSP f4 *)&wy_q[r][i], q1 = *(const LDSP f4 *)&wy_q[r][i + 4u];
        const f4 c0 = *(const LDSP f4 *)&wy_k[c][i], c1 = *(const LDSP f4 *)&wy_k[c][i + 4u];
        kk += a0.x * c0.x + a0.y * c0.y + a0.z * c0.z + a0.w * c0.w;
        kk2 += a1.x * c1.x + a1.y * c1.y + a1.z * c1.z + a1.w * c1.w;
        qq += q0.x * c0.x + q0.y * c0.y + q0.z * c0.z + q0.w * c0.w;
        qq2 += q1.x * c1.x + q1.y * c1.y + q1.z * c1.z + q1.w * c1.w;
      }
      kk += kk2, qq += qq2;
    }
    const float dec = r < n && c <= r ? __builtin_expf(wy_G[r] - wy_G[c]) : 0.0f;
    wy_L[r][c] = c < r ? wy_b[r] * dec * kk : 0.0f;
    Mb[(((unsigned long)b * nv + h) * GC + r) * GC + c] = dec * qq;
  }
  lbarrier();
  /* forward substitution, one column per thread: column th of [beta V | beta gamma K] */
  float x[GC];
  const int isw = th >= 128u;
  const u32 j = th & 127u;
#pragma unroll
  for (u32 r = 0; r < GC; r++) {
    x[r] = 0.0f;
    if (r < n) { /* uniform */
      float acc = isw ? wy_b[r] * __builtin_expf(wy_G[r]) * wy_k[r][j] : wy_b[r] * vv[((unsigned long)(t0 + r) * nv + h) * 128u + j];
#pragma unroll
      for (u32 c = 0; c < r; c++) acc -= wy_L[r][c] * x[c];
      x[r] = acc;
      (isw ? W : U)[((unsigned long)(t0 + r) * nv + h) * 128u + j] = acc;
    }
  }
}

/* 3: the blocks in order, for value head h and its value columns 32 cq ..
 * 32 cq + 31 (workgroup 4 h + cq): the columns of the state do not depend on
 * each other, so 4 workgroups share a head. Wave w owns the 4 columns
 * 32 cq + 4 w + a (a < 4); lane l holds the state of the key dims 4 l + k
 * (k < 4): st[4 k + a] = S[4 l + k][32 cq + 4 w + a]. A product X S0 (X:
 * 32 rows of 128) is then a sum over the 32 lanes, done as a reduce-scatter
 * with DPP: no exchange between waves, and the LDS reads of X rows are
 * 16 bytes per lane, consecutive (no broadcast). Writes the output o (before
 * the gated norm) of every token and the state after the last token to
 * slot (c + T - 1) % ns. */
static LDS float sq_w[GC][128] __attribute__((aligned(16))), sq_q[GC][128] __attribute__((aligned(16))),
    sq_k[GC][128] __attribute__((aligned(16))), sq_d[8][GC][4] __attribute__((aligned(16))), sq_m[GC][GC + 1u], sq_G[GC];
static inline int xor_i(int v, u32 m) { /* the value of lane (this lane ^ m) */
  switch (m) { /* DPP row_xmask:m = 0x160 + m */
    case 16: return xrow_i(v);
    case 8: return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xF, 0xF, true);
    case 4: return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xF, 0xF, true);
    case 2: return __builtin_amdgcn_update_dpp(0, v, 0x162, 0xF, 0xF, true);
    default: return __builtin_amdgcn_update_dpp(0, v, 0x161, 0xF, 0xF, true);
  }
}
/* p[a] = sum over the 128 key dims of x[r][i] S[i][col a] for the row r =
 * rho(l) = 16 (l & 1) + (l >> 1) of this lane (a permutation of the rows):
 * per lane, partial sums over its 4 key dims, for 16 rows at a time (fewer
 * registers); 4 halving steps (step m: the lanes with bit m keep the upper
 * half of the rows and send the lower half to lane ^ m) leave row l >> 1 in
 * the lanes l and l ^ 1, which then add their values. */
static inline void sq_prod(const LDSP float (*x)[128], const float *st, u32 l, float *p) {
#pragma unroll
  for (u32 hf = 0; hf < 2u; hf++) {
    float acc[64];
#pragma unroll
    for (u32 r = 0; r < 16u; r++) {
      const f4 v = *(const LDSP f4 *)&x[16u * hf + r][4u * l];
#pragma unroll
      for (u32 a = 0; a < 4u; a++) acc[4u * r + a] = v.x * st[a] + v.y * st[4u + a] + v.z * st[8u + a] + v.w * st[12u + a];
    }
#pragma unroll
    for (u32 m = 16u, R = 16u; m >= 2u; m >>= 1, R >>= 1) {
      const int up = (l & m) != 0u;
#pragma unroll
      for (u32 r = 0; r < R / 2u; r++)
#pragma unroll
        for (u32 a = 0; a < 4u; a++) {
          const float lo = acc[4u * r + a], hi = acc[4u * (r + R / 2u) + a];
          acc[4u * r + a] = (up ? hi : lo) + I2F(xor_i(F2I(up ? lo : hi), m));
        }
    }
#pragma unroll
    for (u32 a = 0; a < 4u; a++) {
      const float v = acc[a] + I2F(xor_i(F2I(acc[a]), 1u));
      if ((l & 1u) == hf) p[a] = v;
    }
  }
}
/* 32 rows of 128 floats (rows r < n; row r at src + r * rs) into registers:
 * thread th holds the float4 number th + NT k (row (th + NT k) / 32) */
static inline void sq_ld(f4 *x, const G float *src, unsigned long rs, u32 n, u32 th) {
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 e = th + NT * k, r = e >> 5;
    x[k] = r < n ? *(const G f4 *)(src + r * rs + 4u * (e & 31u)) : (f4){0.0f, 0.0f, 0.0f, 0.0f};
  }
}
static inline void sq_st(LDSP float (*dst)[128], const f4 *x, u32 th) {
#pragma unroll
  for (u32 k = 0; k < 4u; k++) {
    const u32 e = th + NT * k;
    *(LDSP f4 *)&dst[e >> 5][4u * (e & 31u)] = x[k];
  }
}
KERNEL ie_gdn_seq(const G float *qk, const G float *U, const G float *W, const G float *Mb, const G float *Gc, G float *out,
                  u32 os, G float *S, u32 c, u32 ns, u32 pos, u32 T, u32 nk, u32 nv) {
  const u32 h = wgid() / GD_SPLIT, cq = wgid() % GD_SPLIT, kh = h % nk, th = tid(), w = th >> 5, l = lane();
  const u32 col = 32u * cq + 4u * w, rl = 16u * (l & 1u) + (l >> 1); /* this wave's first column; this lane's row (sq_prod) */
  const unsigned long slot = (unsigned long)nv * 16384u, own = (unsigned long)h * 16384u + (4u * l) * 128u + col;
  const unsigned long wrs = (unsigned long)nv * 128u, qrs = (unsigned long)nk * 256u;
  float st[16];
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 a = 0; a < 4u; a++) st[4u * k + a] = pos ? S[c * slot + own + k * 128u + a] : 0.0f;
  /* the next block's inputs are loaded into registers one block ahead */
  f4 rw[4], rq[4], rk[4];
  float ru[4], rm[4], rg;
  {
    const u32 n = T < GC ? T : GC;
    sq_ld(rw, W + (unsigned long)h * 128u, wrs, n, th);
    sq_ld(rq, qk + (unsigned long)kh * 256u, qrs, n, th);
    sq_ld(rk, qk + (unsigned long)kh * 256u + 128u, qrs, n, th);
#pragma unroll
    for (u32 a = 0; a < 4u; a++) ru[a] = rl < n ? U[(unsigned long)rl * wrs + h * 128u + col + a] : 0.0f;
#pragma unroll
    for (u32 k = 0; k < 4u; k++) rm[k] = Mb[(unsigned long)h * GC * GC + th + NT * k];
    rg = th < n ? Gc[(unsigned long)th * nv + h] : 0.0f;
  }
  for (u32 b = 0, t0 = 0; t0 < T; b++, t0 += GC) {
    const u32 n = T - t0 < GC ? T - t0 : GC, t1 = t0 + GC, n1 = t1 < T ? (T - t1 < GC ? T - t1 : GC) : 0u;
    sq_st(sq_w, rw, th);
    sq_st(sq_q, rq, th);
    sq_st(sq_k, rk, th);
#pragma unroll
    for (u32 k = 0; k < 4u; k++) {
      const u32 e = th + NT * k;
      sq_m[e >> 5][e & 31u] = rm[k];
    }
    if (th < GC) sq_G[th] = rg;
    float u[4];
#pragma unroll
    for (u32 a = 0; a < 4u; a++) u[a] = ru[a];
    lbarrier();
    if (n1) { /* uniform: prefetch the next block */
      sq_ld(rw, W + (unsigned long)t1 * wrs + h * 128u, wrs, n1, th);
      sq_ld(rq, qk + (unsigned long)t1 * qrs + kh * 256u, qrs, n1, th);
      sq_ld(rk, qk + (unsigned long)t1 * qrs + kh * 256u + 128u, qrs, n1, th);
#pragma unroll
      for (u32 a = 0; a < 4u; a++) ru[a] = rl < n1 ? U[(unsigned long)(t1 + rl) * wrs + h * 128u + col + a] : 0.0f;
#pragma unroll
      for (u32 k = 0; k < 4u; k++) rm[k] = Mb[(((unsigned long)(b + 1u) * nv + h) * GC) * GC + th + NT * k];
      rg = th < n1 ? Gc[(unsigned long)(t1 + th) * nv + h] : 0.0f;
    }
    /* Delta = U - W S0, row rl of the block (0 for rows >= n) */
    float p[4], d[4];
    sq_prod(sq_w, st, l, p);
#pragma unroll
    for (u32 a = 0; a < 4u; a++) d[a] = rl < n ? u[a] - p[a] : 0.0f;
    *(LDSP f4 *)&sq_d[w][rl][0] = (f4){d[0], d[1], d[2], d[3]}; /* this wave's rows of Delta */
    /* O = diag(gamma) Q S0 + M Delta (M is 0 above the diagonal) */
    sq_prod(sq_q, st, l, p);
    const float gl = sq_G[rl];
    float o[4];
#pragma unroll
    for (u32 a = 0; a < 4u; a++) o[a] = __builtin_expf(gl) * p[a];
#pragma unroll 8
    for (u32 s2 = 0; s2 < GC; s2++) {
      const float mm = sq_m[rl][s2];
      const f4 dv = *(const LDSP f4 *)&sq_d[w][s2][0];
      o[0] += mm * dv.x, o[1] += mm * dv.y, o[2] += mm * dv.z, o[3] += mm * dv.w;
    }
    if (rl < n) {
      G float *op = out + (t0 + rl) * os + h * 128u + col;
#pragma unroll
      for (u32 a = 0; a < 4u; a++) op[a] = o[a];
    }
    /* S_new = gamma_n S0 + K^T diag(gamma_n / gamma) Delta */
    const float Gn = sq_G[n - 1u], eg = __builtin_expf(Gn), sc = __builtin_expf(Gn - gl);
    /* the reads of sq_d above are done (one wave: LDS operations in order) */
    *(LDSP f4 *)&sq_d[w][rl][0] = (f4){d[0] * sc, d[1] * sc, d[2] * sc, d[3] * sc};
#pragma unroll
    for (u32 i = 0; i < 16u; i++) st[i] *= eg;
#pragma unroll 4
    for (u32 r = 0; r < GC; r++) {
      const f4 kv = *(const LDSP f4 *)&sq_k[r][4u * l];
      const f4 dv = *(const LDSP f4 *)&sq_d[w][r][0];
      const float kk[4] = {kv.x, kv.y, kv.z, kv.w}, dd[4] = {dv.x, dv.y, dv.z, dv.w};
#pragma unroll
      for (u32 k = 0; k < 4u; k++)
#pragma unroll
        for (u32 a = 0; a < 4u; a++) st[4u * k + a] += kk[k] * dd[a];
    }
    lbarrier(); /* the LDS of this block is read before the next block writes it */
  }
  G float *Sw = S + ((c + T - 1u) % ns) * slot + own;
#pragma unroll
  for (u32 k = 0; k < 4u; k++)
#pragma unroll
    for (u32 a = 0; a < 4u; a++) Sw[k * 128u + a] = st[4u * k + a];
}

/* The conv inputs of the last 3 of T tokens (positions pos + T - 3 ..) into
 * the ring, after an ie_gdn launch with rw = 0 (a prompt chunk: more tokens
 * than the ring has slots, so ie_gdn cannot write the ring safely while
 * other workgroups still read it). Thread i: channel i % cd of token T - 3
 * + i / cd. */
KERNEL ie_ring_store(const G float *in, G float *ring, u32 pos, u32 T, u32 cd, u32 is) {
  const u32 i = wgid() * NT + tid(), n = T < 3u ? T : 3u;
  if (i >= n * cd) return;
  const u32 t = T - n + i / cd, c = i % cd;
  ring[((pos + t) & (GDN_RING - 1u)) * cd + c] = in[t * is + c];
}

/* Attention of a prompt chunk in one launch: workgroup (h, y) = query head h
 * of token y (position pos0 + y), all positions 0 .. pos0 + y; wave w does
 * the positions w, w + 8, .. with an online softmax, then the 8 waves merge
 * in LDS (as ie_attn_split, with one split). Output gate and Q8 copy as
 * ie_attn_split. Strides (bytes) between the tokens: qs (q), os (out), oqs
 * (oq), gs (gate). hd <= 256, hd % 32 == 0 when oq is set. */
KERNEL ie_attn_pf(const G float *q, const G kvt *kc, const G kvt *vc, G float *out, u32 pos0, u32 hd, u32 n_head,
                  u32 n_kv, G u8 *oq, const G float *gate, u32 gstride, u32 qs, u32 os, u32 oqs, u32 gs) {
  q = TOKC(q, qs), out = TOK(out, os), oq = TOK(oq, oqs), gate = TOK(gate, gs);
  const u32 h = wgid(), pos = pos0 + tokid(), kh = h / (n_head / n_kv), kvd = n_kv * hd;
  const u32 w = tid() >> 5, l = lane();
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  float qv[8], acc[8];
  for (u32 j = 0; j < 8u; j++) {
    const u32 i = l + 32u * j;
    qv[j] = i < hd ? q[h * hd + i] : 0.0f;
    acc[j] = 0.0f;
  }
  float m = -__builtin_inff(), sum = 0.0f;
  for (u32 p = w; p <= pos; p += 8u) {
    const G kvt *k = kc + (unsigned long)p * kvd + kh * hd;
    const G kvt *v = vc + (unsigned long)p * kvd + kh * hd;
    float d = 0.0f;
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) d += qv[j] * k[l + 32u * j];
    d = wave_sum_all(d) * scale;
    const float mn = __builtin_fmaxf(m, d), c = __builtin_expf(m - mn), e = __builtin_expf(d - mn);
    sum = sum * c + e;
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) acc[j] = acc[j] * c + e * v[l + 32u * j];
    m = mn;
  }
  if (l == 0u) at_m[w] = m, at_l[w] = sum;
  for (u32 j = 0; j < 8u; j++)
    if (l + 32u * j < hd) at_o[w][l + 32u * j] = acc[j];
  barrier();
  float M = at_m[0];
  for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, at_m[x]);
  float L = 0.0f;
  for (u32 x = 0; x < 8u; x++) L += at_l[x] * (at_m[x] == -__builtin_inff() ? 0.0f : __builtin_expf(at_m[x] - M));
  if (tid() < hd) { /* whole waves when oq is set */
    float o = 0.0f;
    for (u32 x = 0; x < 8u; x++)
      if (at_m[x] != -__builtin_inff()) o += at_o[x][tid()] * __builtin_expf(at_m[x] - M);
    o = o / L;
    if (gate) o = o * sigmoidf(gate[h * gstride + tid()]);
    out[h * hd + tid()] = o;
    if (oq) quant_wave(o, (h * hd + tid()) >> 5, oq, (n_head * hd) >> 5);
  }
}

/* RMSNorm of many tokens (prompt chunks): workgroup y = token y does the
 * whole row (each element read once for the sum of squares, once for the
 * output), wave w the blocks w, w + 8, .. . Same arithmetic as rmsnorm_wg. */
KERNEL ie_rmsnorm_t(const G float *x, const G float *w, G float *y, u32 n, float eps, G u8 *q, u32 xs, u32 ys, u32 qs) {
  x = TOKC(x, xs), y = TOK(y, ys), q = TOK(q, qs);
  rmsnorm_wg(x, w, y, n, eps, q);
}

/* Attention of a prompt chunk, grouped: workgroup (g, y) = KV head g of
 * token y (position pos0 + y) and its GA = n_head / n_kv query heads (<= 8).
 * Each K and V row is read once for all GA heads (6 for Qwen3.8-27B), and
 * the GA dot products are independent (more work in flight per wave). Wave
 * w does the positions w, w + 8, ..; then the waves merge one head at a
 * time in LDS. As ie_attn_pf otherwise. */
#define GA_MAX 8u
static LDS float ag_m[8][GA_MAX], ag_l[8][GA_MAX];
KERNEL ie_attn_pfg(const G float *q, const G kvt *kc, const G kvt *vc, G float *out, u32 pos0, u32 hd, u32 n_head,
                   u32 n_kv, G u8 *oq, const G float *gate, u32 gstride, u32 qs, u32 os, u32 oqs, u32 gs) {
  q = TOKC(q, qs), out = TOK(out, os), oq = TOK(oq, oqs), gate = TOK(gate, gs);
  const u32 kh = wgid(), pos = pos0 + tokid(), ga = n_head / n_kv, kvd = n_kv * hd;
  const u32 w = tid() >> 5, l = lane();
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  float qv[GA_MAX][8], acc[GA_MAX][8], m[GA_MAX], sum[GA_MAX];
#pragma unroll
  for (u32 g = 0; g < GA_MAX; g++) {
    m[g] = -__builtin_inff(), sum[g] = 0.0f;
    for (u32 j = 0; j < 8u; j++) {
      const u32 i = l + 32u * j;
      qv[g][j] = g < ga && i < hd ? q[(kh * ga + g) * hd + i] : 0.0f;
      acc[g][j] = 0.0f;
    }
  }
  for (u32 p = w; p <= pos; p += 8u) {
    const G kvt *k = kc + (unsigned long)p * kvd + kh * hd;
    const G kvt *v = vc + (unsigned long)p * kvd + kh * hd;
    float kr[8], vr[8];
    for (u32 j = 0; j < 8u; j++) {
      kr[j] = l + 32u * j < hd ? k[l + 32u * j] : 0.0f;
      vr[j] = l + 32u * j < hd ? v[l + 32u * j] : 0.0f;
    }
#pragma unroll
    for (u32 g = 0; g < GA_MAX; g++) {
      if (g < ga) { /* uniform */
      float d = 0.0f;
      for (u32 j = 0; j < 8u; j++) d += qv[g][j] * kr[j];
      d = wave_sum_all(d) * scale;
      const float mn = __builtin_fmaxf(m[g], d), c = __builtin_expf(m[g] - mn), e = __builtin_expf(d - mn);
      sum[g] = sum[g] * c + e;
      for (u32 j = 0; j < 8u; j++) acc[g][j] = acc[g][j] * c + e * vr[j];
      m[g] = mn;
      }
    }
  }
  if (l == 0u) {
#pragma unroll
    for (u32 g = 0; g < GA_MAX; g++) ag_m[w][g] = m[g], ag_l[w][g] = sum[g];
  }
#pragma unroll
  for (u32 g = 0; g < GA_MAX; g++) {
    if (g >= ga) continue; /* uniform */
    for (u32 j = 0; j < 8u; j++)
      if (l + 32u * j < hd) at_o[w][l + 32u * j] = acc[g][j];
    barrier();
    float M = ag_m[0][g];
    for (u32 x = 1; x < 8u; x++) M = __builtin_fmaxf(M, ag_m[x][g]);
    float L = 0.0f;
    for (u32 x = 0; x < 8u; x++) L += ag_m[x][g] == -__builtin_inff() ? 0.0f : ag_l[x][g] * __builtin_expf(ag_m[x][g] - M);
    if (tid() < hd) {
      float o = 0.0f;
      for (u32 x = 0; x < 8u; x++)
        if (ag_m[x][g] != -__builtin_inff()) o += at_o[x][tid()] * __builtin_expf(ag_m[x][g] - M);
      o = o / L;
      const u32 h = kh * ga + g;
      if (gate) o = o * sigmoidf(gate[h * gstride + tid()]);
      out[h * hd + tid()] = o;
      if (oq) quant_wave(o, (h * hd + tid()) >> 5, oq, (n_head * hd) >> 5);
    }
    barrier(); /* at_o is read before the next head writes it */
  }
}

/* Attention of a prompt chunk with fp16 WMMA (flash attention): workgroup
 * (kh, part, tile) = KV head kh, output columns FA_DC part .. + FA_DC - 1
 * of each head, and the 16 tokens 16 tile .. + 15 of the chunk (positions
 * pos0 + t). Wave w < GA = n_head / n_kv (<= 8) does query head kh * GA + w
 * for those 16 tokens; all waves load. Per block of FA_NB = 32 key
 * positions: K (fp16, key rows, all of hd) and the part's columns of V
 * (fp16, transposed: one row per column) go to LDS once for the GA heads and
 * 16 tokens; then
 *   S = (Q * scale) K^T     16 x 32, two WMMA tiles, K-dim = hd (fp16 x fp16 -> f32)
 *   online softmax per token row in f32 (causal mask: key > token position -> -inf)
 *   O = O * c + P V         16 x FA_DC, P as fp16 through LDS (D layout -> A layout)
 * The output is split in parts so that a wave holds Q (64 registers) and
 * its part of O (64): S is computed once per part. WMMA lane layout (gfx12,
 * tools/wmma_test.c): A lane l = row l % 16, K 8 (l / 16) .. + 7; B (given
 * as N x K) lane l = column l % 16, the same K; D lane l, element v = row
 * 8 (l / 16) + v, column l % 16. The row sums stay per lane until the end
 * (c is the same in every lane of a row). Not bitwise equal to ie_attn_pfg:
 * Q, K, V and P are rounded to fp16 (relative error <= 2^-11 each), the sums
 * are f32 in another order. hd % 32 == 0, hd <= 256. Output and its fused Q8
 * copy as ie_attn_pfg (the Q8 copy by the workgroup of the last part). */
#define FA_NB 32u
#define FA_DC 128u
#define FA_KS (256u + 8u) /* LDS row stride of K, halves */
#define FA_VS (FA_NB + 8u) /* LDS row stride of V^T and of P, halves */
static LDS f16 fa_k[FA_NB * FA_KS];
static LDS f16 fa_vt[FA_DC * FA_VS];
static LDS f16 fa_p[8u][16u * FA_VS];
KERNEL ie_attn_fa(const G float *q, const G kvt *kc, const G kvt *vc, G float *out, u32 pos0, u32 hd, u32 n_head,
                  u32 n_kv, const G float *gate, u32 gstride, u32 qs, u32 os, u32 gs, u32 T) {
  const u32 ntile = (T + 15u) / 16u, nparts = (hd + FA_DC - 1u) / FA_DC;
  const u32 tile = wgid() % ntile, part = (wgid() / ntile) % nparts, kh = wgid() / (ntile * nparts), t0 = 16u * tile;
  const u32 ga = n_head / n_kv, kvd = n_kv * hd, w = tid() >> 5, l = lane(), hl = l >> 4, rl = l & 15u;
  const u32 nks = hd / 16u, h = kh * ga + w, c0 = part * FA_DC, nc = hd - c0 < FA_DC ? hd - c0 : FA_DC;
  const int on = w < ga; /* wave-uniform */
  const u32 tlast = t0 + 15u < T ? t0 + 15u : T - 1u, kend = pos0 + tlast + 1u; /* keys 0 .. kend - 1 */
  const float scale = 1.0f / __builtin_sqrtf((float)hd);
  /* Q (scaled, fp16) in the A layout: token t0 + rl, elements 16 ks + 8 hl .. + 7 */
  h8 qa[16];
  {
    const u32 tq = t0 + rl < T ? t0 + rl : T - 1u;
    const G float *qr = (const G float *)((const G u8 *)q + (unsigned long)tq * qs) + h * hd;
#pragma unroll
    for (u32 ks = 0; ks < 16u; ks++)
      for (u32 e = 0; e < 8u; e++) qa[ks][e] = on && ks < nks ? (f16)(qr[16u * ks + 8u * hl + e] * scale) : (f16)0.0f;
  }
  v8f o[FA_DC / 16u], zero = (v8f)(0.0f);
  float m[8], ls[8];
#pragma unroll
  for (u32 nt = 0; nt < FA_DC / 16u; nt++) o[nt] = zero;
  for (u32 v = 0; v < 8u; v++) m[v] = -__builtin_inff(), ls[v] = 0.0f;
  const u32 h4 = hd >> 2, n4 = FA_NB * h4, c4 = nc >> 2; /* float4 per key row, per block; of the part */
  for (u32 kb = 0; kb < kend; kb += FA_NB) {
    /* K (all of hd) and V (the part's columns) of keys kb .. kb + 31 to LDS (fp16); keys >= kend as zeros */
    for (u32 i4 = tid(); i4 < n4; i4 += NT) {
      const u32 r = i4 / h4, e4 = i4 - r * h4, p = kb + r;
      f32x4 kv = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      if (p < kend) kv = KV4(kc, ((unsigned long)p * kvd + kh * hd) / 4u + e4);
      LDSP f16 *kd = fa_k + r * FA_KS + 4u * e4;
      kd[0] = (f16)kv.x, kd[1] = (f16)kv.y, kd[2] = (f16)kv.z, kd[3] = (f16)kv.w;
    }
    for (u32 i4 = tid(); i4 < FA_NB * c4; i4 += NT) {
      const u32 r = i4 / c4, e4 = i4 - r * c4, p = kb + r;
      f32x4 vv = (f32x4){0.0f, 0.0f, 0.0f, 0.0f};
      if (p < kend) vv = KV4(vc, ((unsigned long)p * kvd + kh * hd + c0) / 4u + e4);
      fa_vt[(4u * e4 + 0u) * FA_VS + r] = (f16)vv.x;
      fa_vt[(4u * e4 + 1u) * FA_VS + r] = (f16)vv.y;
      fa_vt[(4u * e4 + 2u) * FA_VS + r] = (f16)vv.z;
      fa_vt[(4u * e4 + 3u) * FA_VS + r] = (f16)vv.w;
    }
    barrier();
    if (on) {
      /* S = Q K^T: two 16 x 16 tiles (keys kb + 16 j + rl) */
      v8f s[2];
#pragma unroll
      for (u32 j = 0; j < 2u; j++) {
        s[j] = zero;
#pragma unroll
        for (u32 ks = 0; ks < 16u; ks++)
          if (ks < nks) {
            const h8 kb8 = *(const LDSP h8 *)(fa_k + (16u * j + rl) * FA_KS + 16u * ks + 8u * hl);
            s[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa[ks], kb8, s[j]);
          }
      }
      /* online softmax of rows 8 hl + v (tokens t0 + 8 hl + v) */
      float c[8];
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 tp = pos0 + t0 + 8u * hl + v; /* this row's position */
        const u32 k0 = kb + rl, k1 = kb + 16u + rl;
        const float s0 = k0 <= tp ? s[0][v] : -__builtin_inff(), s1 = k1 <= tp ? s[1][v] : -__builtin_inff();
        float rm = __builtin_fmaxf(s0, s1);
        for (u32 x = 1u; x < 16u; x <<= 1) rm = __builtin_fmaxf(rm, shfl_xor_f(rm, x)); /* the 16 lanes of the row */
        const float mn = __builtin_fmaxf(m[v], rm);
        c[v] = __builtin_expf(m[v] - mn);
        const float p0 = __builtin_expf(s0 - mn), p1 = __builtin_expf(s1 - mn);
        ls[v] = ls[v] * c[v] + p0 + p1;
        m[v] = mn;
        LDSP f16 *pr = fa_p[w] + (8u * hl + v) * FA_VS;
        pr[rl] = (f16)p0;
        pr[16u + rl] = (f16)p1;
      }
      __asm__ volatile("s_wait_dscnt 0x0" ::: "memory"); /* this wave's P stores before its loads */
      h8 pa[2];
      for (u32 kk = 0; kk < 2u; kk++) pa[kk] = *(const LDSP h8 *)(fa_p[w] + rl * FA_VS + 16u * kk + 8u * hl);
#pragma unroll
      for (u32 nt = 0; nt < FA_DC / 16u; nt++) {
        if (16u * nt >= nc) continue;
        for (u32 v = 0; v < 8u; v++) o[nt][v] *= c[v];
#pragma unroll
        for (u32 kk = 0; kk < 2u; kk++) {
          const h8 vb = *(const LDSP h8 *)(fa_vt + (16u * nt + rl) * FA_VS + 16u * kk + 8u * hl);
          o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pa[kk], vb, o[nt]);
        }
      }
    }
    barrier();
  }
  if (on) {
    float inv[8];
    for (u32 v = 0; v < 8u; v++) {
      float x = ls[v];
      for (u32 s = 1u; s < 16u; s <<= 1) x += shfl_xor_f(x, s);
      inv[v] = 1.0f / x;
    }
#pragma unroll
    for (u32 nt = 0; nt < FA_DC / 16u; nt++) {
      if (16u * nt >= nc) continue;
      for (u32 v = 0; v < 8u; v++) {
        const u32 tk = t0 + 8u * hl + v, d = c0 + 16u * nt + rl;
        if (tk >= T) continue;
        float x = o[nt][v] * inv[v];
        if (gate) x = x * sigmoidf(((const G float *)((const G u8 *)gate + (unsigned long)tk * gs))[h * gstride + d]);
        ((G float *)((G u8 *)out + (unsigned long)tk * os))[h * hd + d] = x;
      }
    }
  }
}

/* ie_attn_fa with S computed once (hd == 256): workgroup (kh, tile) = KV
 * head kh and the 16 tokens 16 tile .. + 15, all 256 output columns. Per
 * block of FA_NB keys: K and all of V^T to LDS; phase A: wave w < GA
 * computes S = Q K^T of head kh * GA + w (Q in its registers), the online
 * softmax of its rows, and writes P (fp16) and the rescale c of each row to
 * LDS; phase B: the GA * 16 output tiles (head, 16 columns) are spread
 * over all 8 waves, in registers: FA2_OA per wave w < GA (it also holds Q),
 * the rest on the other 8 - GA waves (<= FA2_OB each). Each does O = O * c
 * + P V for its tiles. At the end wave w < GA writes 1 / l of its rows to
 * LDS for the owners of the tiles. The two kinds of waves run separate
 * copies of the loop (fa2_run, the same barriers), so that their registers
 * are allocated apart. Per block, all K and V loads are issued before the
 * LDS stores (one wait for the memory, not one per load). The same arithmetic
 * per output element as ie_attn_fa
 * (S, P, the sums in the same order), so the same result. Needs GA <= 6. */
/* 4 KV elements as fp16 (the fp16 cache: as stored) */
static inline hk4 kv_h4(const G kvt *p, unsigned long i4) {
#ifdef IE_KV_F32
  return __builtin_convertvector(((const G kv4 *)p)[i4], hk4);
#else
  return ((const G kv4 *)p)[i4];
#endif
}
#define FA2_OA 12u
#define FA2_OB 12u
#define FA2_VS (FA_NB + 8u)
static LDS f16 fa2_vt[256u * FA2_VS];
static LDS float fa2_c[8u][16u];
static inline __attribute__((always_inline)) void fa2_run(const G float *q, const G kvt *kc, const G kvt *vc,
                                                         G float *out, u32 pos0, u32 n_head, u32 n_kv,
                                                         const G float *gate, u32 gstride, u32 qs, u32 os, u32 gs,
                                                         u32 T, const int dos, const u32 OT, u32 tb, u32 tn) {
  const u32 ntile = (T + 15u) / 16u, tile = wgid() % ntile, kh = wgid() / ntile, t0 = 16u * tile;
  const u32 hd = 256u, ga = n_head / n_kv, kvd = n_kv * hd, w = __builtin_amdgcn_readfirstlane(tid() >> 5), l = lane();
  const u32 hl = l >> 4, rl = l & 15u;
  const u32 tlast = t0 + 15u < T ? t0 + 15u : T - 1u, kend = pos0 + tlast + 1u; /* keys 0 .. kend - 1 */
  h8 qa[16];
  if (dos) {
    const float scale = 1.0f / 16.0f; /* 1 / sqrt(256) */
    const u32 tq = t0 + rl < T ? t0 + rl : T - 1u;
    const G float *qr = (const G float *)((const G u8 *)q + (unsigned long)tq * qs) + (kh * ga + w) * hd;
#pragma unroll
    for (u32 ks = 0; ks < 16u; ks++)
#pragma unroll
      for (u32 e = 0; e < 8u; e++) qa[ks][e] = (f16)(qr[16u * ks + 8u * hl + e] * scale);
  }
  v8f o[FA2_OB], zero = (v8f)(0.0f);
  float m[8], ls[8];
#pragma unroll
  for (u32 t = 0; t < OT; t++) o[t] = zero;
  #pragma unroll
  for (u32 v = 0; v < 8u; v++) m[v] = -__builtin_inff(), ls[v] = 0.0f;
  for (u32 kb = 0; kb < kend; kb += FA_NB) {
    /* opaque copies of the lane numbers: the LDS addresses are computed in the loop, not kept in registers */
    u32 hlo = hl, rlo = rl;
    __asm__ volatile("" : "+v"(hlo), "+v"(rlo));
    { /* all loads of the block first (one wait), then the LDS stores: K rows as fp16 (8-byte
       * stores); V^T from 4 keys x 4 columns per job, transposed in registers */
      hk4 xk[FA_NB * 64u / NT], xv[(FA_NB / 4u) * 64u / NT][4];
#pragma unroll
      for (u32 n = 0; n < FA_NB * 64u / NT; n++) {
        const u32 i4 = tid() + NT * n, r = i4 >> 6, e4 = i4 & 63u, p = kb + r;
        xk[n] = (hk4)(0.0f);
        if (p < kend) xk[n] = kv_h4(kc, ((unsigned long)p * kvd + kh * hd) / 4u + e4);
      }
#pragma unroll
      for (u32 n = 0; n < (FA_NB / 4u) * 64u / NT; n++) {
        const u32 j = tid() + NT * n, kq = j >> 6, e4 = j & 63u;
#pragma unroll
        for (u32 u = 0; u < 4u; u++) {
          const u32 p = kb + 4u * kq + u;
          xv[n][u] = (hk4)(0.0f);
          if (p < kend) xv[n][u] = kv_h4(vc, ((unsigned long)p * kvd + kh * hd) / 4u + e4);
        }
      }
#pragma unroll
      for (u32 n = 0; n < FA_NB * 64u / NT; n++) {
        const u32 i4 = tid() + NT * n, r = i4 >> 6, e4 = i4 & 63u;
        *(LDSP hk4 *)(fa_k + r * FA_KS + 4u * e4) = xk[n];
      }
#pragma unroll
      for (u32 n = 0; n < (FA_NB / 4u) * 64u / NT; n++) {
        const u32 j = tid() + NT * n, kq = j >> 6, e4 = j & 63u;
#pragma unroll
        for (u32 c = 0; c < 4u; c++)
          *(LDSP hk4 *)(fa2_vt + (4u * e4 + c) * FA2_VS + 4u * kq) = (hk4){xv[n][0][c], xv[n][1][c], xv[n][2][c], xv[n][3][c]};
      }
    }
    barrier();
    if (dos) { /* phase A */
      v8f s[2];
#pragma unroll
      for (u32 j = 0; j < 2u; j++) {
        s[j] = zero;
#pragma unroll
        for (u32 ks = 0; ks < 16u; ks++) {
          const h8 kb8 = *(const LDSP h8 *)(fa_k + (16u * j + rlo) * FA_KS + 16u * ks + 8u * hlo);
          s[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa[ks], kb8, s[j]);
        }
      }
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 tp = pos0 + t0 + 8u * hlo + v;
        const u32 k0 = kb + rlo, k1 = kb + 16u + rlo;
        const float s0 = k0 <= tp ? s[0][v] : -__builtin_inff(), s1 = k1 <= tp ? s[1][v] : -__builtin_inff();
        float rm = __builtin_fmaxf(s0, s1);
#pragma unroll
        for (u32 x = 1u; x < 16u; x <<= 1) rm = __builtin_fmaxf(rm, shfl_xor_f(rm, x));
        const float mn = __builtin_fmaxf(m[v], rm);
        const float c = __builtin_expf(m[v] - mn);
        const float p0 = __builtin_expf(s0 - mn), p1 = __builtin_expf(s1 - mn);
        ls[v] = ls[v] * c + p0 + p1;
        m[v] = mn;
        LDSP f16 *pr = fa_p[w] + (8u * hlo + v) * FA_VS;
        pr[rlo] = (f16)p0;
        pr[16u + rlo] = (f16)p1;
        if (rlo == 0u) fa2_c[w][8u * hlo + v] = c;
      }
    }
    barrier();
    h8 pa[2];
    float cr[8];
#pragma unroll
    for (u32 t = 0; t < OT; t++) { /* phase B: tiles tb .. tb + tn - 1 (head i / 16, columns 16 (i % 16) ..) */
      if (t < tn) {
        const u32 i = tb + t, hh = i >> 4, nt = i & 15u;
        if (t == 0u || nt == 0u) { /* a new head (uniform): its P and row rescales */
#pragma unroll
          for (u32 kk = 0; kk < 2u; kk++) pa[kk] = *(const LDSP h8 *)(fa_p[hh] + rlo * FA_VS + 16u * kk + 8u * hlo);
#pragma unroll
          for (u32 v = 0; v < 8u; v++) cr[v] = fa2_c[hh][8u * hlo + v];
        }
#pragma unroll
        for (u32 v = 0; v < 8u; v++) o[t][v] *= cr[v];
#pragma unroll
        for (u32 kk = 0; kk < 2u; kk++) {
          const h8 vb = *(const LDSP h8 *)(fa2_vt + (16u * nt + rlo) * FA2_VS + 16u * kk + 8u * hlo);
          o[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pa[kk], vb, o[t]);
        }
      }
    }
    barrier();
  }
  u32 hle = hl, rle = rl; /* opaque: the epilogue's addresses are not computed before the loop */
  __asm__ volatile("" : "+v"(hle), "+v"(rle));
  if (dos)
#pragma unroll
    for (u32 v = 0; v < 8u; v++) {
      float x = ls[v];
#pragma unroll
      for (u32 s = 1u; s < 16u; s <<= 1) x += shfl_xor_f(x, s);
      if (rl == 0u) fa2_c[w][8u * hl + v] = 1.0f / x;
    }
  barrier();
#pragma unroll
  for (u32 t = 0; t < OT; t++)
    if (t < tn) {
      const u32 i = tb + t, hh = i >> 4, nt = i & 15u, h = kh * ga + hh;
#pragma unroll
      for (u32 v = 0; v < 8u; v++) {
        const u32 tk = t0 + 8u * hle + v, d = 16u * nt + rle;
        if (tk >= T) continue;
        float x = o[t][v] * fa2_c[hh][8u * hle + v];
        if (gate) x = x * sigmoidf(((const G float *)((const G u8 *)gate + (unsigned long)tk * gs))[h * gstride + d]);
        ((G float *)((G u8 *)out + (unsigned long)tk * os))[h * hd + d] = x;
      }
    }
}
KERNEL ie_attn_fa2(const G float *q, const G kvt *kc, const G kvt *vc, G float *out, u32 pos0, u32 hd, u32 n_head,
                   u32 n_kv, const G float *gate, u32 gstride, u32 qs, u32 os, u32 gs, u32 T) {
  (void)hd;
  const u32 ga = n_head / n_kv, w = __builtin_amdgcn_readfirstlane(tid() >> 5), ntot = 16u * ga;
  const u32 na = FA2_OA * ga < ntot ? FA2_OA * ga : ntot; /* the tiles of the S waves */
  const u32 nbw = 8u - ga, per = (ntot - na + nbw - 1u) / nbw; /* the others: per wave (<= FA2_OB) */
  if (w < ga) {
    const u32 tb = w * FA2_OA, tn = tb < na ? (na - tb < FA2_OA ? na - tb : FA2_OA) : 0u;
    fa2_run(q, kc, vc, out, pos0, n_head, n_kv, gate, gstride, qs, os, gs, T, 1, FA2_OA, tb, tn);
  } else {
    const u32 tb = na + (w - ga) * per, tn = tb < ntot ? (ntot - tb < per ? ntot - tb : per) : 0u;
    fa2_run(q, kc, vc, out, pos0, n_head, n_kv, gate, gstride, qs, os, gs, T, 0, FA2_OB, tb, tn);
  }
}

/* The Q8 copy of ie_attn_fa's output (fused into ie_attn_pfg, a separate
 * pass here: the parts of a head are in different workgroups). Workgroup y
 * = token, wave = block of 32, as ie_quant_q8. */
KERNEL ie_attn_fa_q8(const G float *out, G u8 *oq, u32 nb, u32 os, u32 oqs) {
  out = TOKC(out, os), oq = TOK(oq, oqs);
  const u32 b = wgid() * 8u + (tid() >> 5);
  if (b >= nb) return;
  quant_wave(out[b * 32u + lane()], b, oq, nb);
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
/* out[0] = the first index of the largest x; out[1] (float bits) = its
 * softmax probability 1 / sum_i exp(x_i - max) (the MTP draft confidence).
 * part: 3 words per workgroup (max, index, sum of exp(x - max) over its
 * elements). The argmax itself is as before (the sums do not change it). */
KERNEL ie_argmax(const G float *x, u32 n, G u32 *out, G u32 *part, G u32 *count, u32 ng) {
  float bv = -__builtin_inff();
  u32 bi = 0xFFFFFFFFu;
  for (u32 i = wgid() * NT + tid(); i < n; i += ng * NT)
    if (better(x[i], i, bv, bi)) bv = x[i], bi = i;
  wg_best(&bv, &bi);
  float se = 0.0f;
  for (u32 i = wgid() * NT + tid(); i < n; i += ng * NT) se += __builtin_expf(x[i] - bv);
  se = wg_sum(se);
  if (tid() == 0u) {
    part[3u * wgid()] = __builtin_bit_cast(u32, bv);
    part[3u * wgid() + 1u] = bi;
    part[3u * wgid() + 2u] = __builtin_bit_cast(u32, se);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    am_last = __atomic_fetch_add(count, 1u, __ATOMIC_ACQ_REL) == ng - 1u;
  }
  barrier();
  if (!am_last) return;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  bv = -__builtin_inff(), bi = 0xFFFFFFFFu;
  for (u32 g = tid(); g < ng; g += NT) {
    const float v = __builtin_bit_cast(float, __atomic_load_n(&part[3u * g], __ATOMIC_RELAXED));
    const u32 i = __atomic_load_n(&part[3u * g + 1u], __ATOMIC_RELAXED);
    if (better(v, i, bv, bi)) bv = v, bi = i;
  }
  wg_best(&bv, &bi);
  float S = 0.0f;
  for (u32 g = tid(); g < ng; g += NT) {
    const float v = __builtin_bit_cast(float, __atomic_load_n(&part[3u * g], __ATOMIC_RELAXED));
    const float sg = __builtin_bit_cast(float, __atomic_load_n(&part[3u * g + 2u], __ATOMIC_RELAXED));
    if (sg > 0.0f) S += sg * __builtin_expf(v - bv);
  }
  S = wg_sum(S);
  if (tid() == 0u) {
    out[0] = bi;
    out[1] = __builtin_bit_cast(u32, 1.0f / S);
    *count = 0u;
  }
}

/* ------------------------------------------------------------ top K (sampling) */

/* The K largest logits of each row (K <= TK_MAX), for sampling: the host
 * then needs K (value, index) pairs instead of the whole row. Order: value
 * descending; equal values: the smaller index first (as the host's
 * selection). Two launches:
 *   ie_topk_part (grid ng x rows): workgroup g takes the indices
 *     [g cs, (g + 1) cs) of its row (cs <= TK_CS), keeps them in LDS and
 *     takes its K best one at a time (best of the workgroup, then removed);
 *   ie_topk_merge (grid rows): the ng K candidates of a row, the same way. */
#define TK_MAX 64u
#define TK_CS 4096u
static LDS float tk_v[TK_CS];
static LDS u32 tk_i[TK_CS];
static LDS float tk_wv[8];
static LDS u32 tk_wi[8], tk_wp[8];
/* (v, i) better than (bv, bi): larger value, or equal value and smaller index; i = ~0: none */
static inline int tk_better(float v, u32 i, float bv, u32 bi) {
  return i != 0xFFFFFFFFu && (bi == 0xFFFFFFFFu || v > bv || (v == bv && i < bi));
}
/* the best (value, index, LDS position) of the workgroup, in every thread */
static inline void tk_wg_best(float *bv, u32 *bi, u32 *bp) {
#pragma unroll
  for (u32 m = 16u; m > 0u; m >>= 1) {
    const float ov = shfl_xor_f(*bv, m);
    const u32 oi = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ m) * 4u), (int)*bi);
    const u32 op = (u32)__builtin_amdgcn_ds_bpermute((int)((lane() ^ m) * 4u), (int)*bp);
    if (tk_better(ov, oi, *bv, *bi)) *bv = ov, *bi = oi, *bp = op;
  }
  if (lane() == 0u) tk_wv[tid() >> 5] = *bv, tk_wi[tid() >> 5] = *bi, tk_wp[tid() >> 5] = *bp;
  lbarrier();
  float v = tk_wv[0];
  u32 i = tk_wi[0], p = tk_wp[0];
  for (u32 w = 1; w < 8u; w++)
    if (tk_better(tk_wv[w], tk_wi[w], v, i)) v = tk_wv[w], i = tk_wi[w], p = tk_wp[w];
  *bv = v, *bi = i, *bp = p;
}
/* the K best of the n (value, index) pairs in tk_v / tk_i, to ov / oi */
static inline void tk_select(u32 n, u32 K, G float *ov, G u32 *oi) {
  for (u32 k = 0; k < K; k++) {
    float bv = -__builtin_inff();
    u32 bi = 0xFFFFFFFFu, bp = 0u;
    for (u32 p = tid(); p < n; p += NT)
      if (tk_better(tk_v[p], tk_i[p], bv, bi)) bv = tk_v[p], bi = tk_i[p], bp = p;
    tk_wg_best(&bv, &bi, &bp);
    if (tid() == 0u) {
      ov[k] = bv, oi[k] = bi;
      if (bi != 0xFFFFFFFFu) tk_i[bp] = 0xFFFFFFFFu; /* taken */
    }
    lbarrier(); /* the removal is visible; tk_wv free again */
  }
}
KERNEL ie_topk_part(const G float *x, u32 n, u32 xs, u32 K, u32 cs, u32 ng, G float *cv, G u32 *ci) {
  const u32 r = __builtin_amdgcn_workgroup_id_y(), g = wgid();
  const G float *row = x + (unsigned long)r * xs;
  const u32 i0 = g * cs;
  for (u32 p = tid(); p < cs; p += NT) {
    const u32 i = i0 + p;
    tk_v[p] = i < n ? row[i] : -__builtin_inff();
    tk_i[p] = i < n ? i : 0xFFFFFFFFu;
  }
  lbarrier();
  tk_select(cs, K, cv + ((unsigned long)r * ng + g) * K, ci + ((unsigned long)r * ng + g) * K);
}
KERNEL ie_topk_merge(const G float *cv, const G u32 *ci, u32 nc, u32 K, G float *ov, G u32 *oi) {
  const u32 r = wgid();
  for (u32 p = tid(); p < nc; p += NT) tk_v[p] = cv[(unsigned long)r * nc + p], tk_i[p] = ci[(unsigned long)r * nc + p];
  lbarrier();
  tk_select(nc, K, ov + (unsigned long)r * K, oi + (unsigned long)r * K);
}
