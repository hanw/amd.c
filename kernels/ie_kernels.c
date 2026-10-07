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

/* Workgroup barrier with the memory ordering of LDS and global memory. */
static inline void barrier(void) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

/* Value of lane (this lane + off) of the wave: ds_bpermute (byte address). */
static inline float shfl_down_f(float v, u32 off) {
  return __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((int)((lane() + off) * 4u), __builtin_bit_cast(int, v)));
}
static inline u32 shfl_down_u(u32 v, u32 off) {
  return (u32)__builtin_amdgcn_ds_bpermute((int)((lane() + off) * 4u), (int)v);
}
static inline float shfl_xor_f(float v, u32 m) {
  return __builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute((int)((lane() ^ m) * 4u), __builtin_bit_cast(int, v)));
}

/* The wave reduction of ie_tree: step m (1..5) adds the value of lane
 * i + (32 >> m), i.e. shuffle down by 16, 8, 4, 2, 1. Lane 0 holds the sum
 * (law wave_sum). Other lanes hold partial values. */
static inline float wave_tree_f(float v) {
  for (u32 m = 1u; m <= 5u; m++) v += shfl_down_f(v, 32u >> m);
  return v;
}
static inline u32 wave_tree_u(u32 v) {
  for (u32 m = 1u; m <= 5u; m++) v += shfl_down_u(v, 32u >> m);
  return v;
}

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
  float amax = __builtin_fabsf(v);
  for (u32 m = 16u; m >= 1u; m >>= 1) amax = __builtin_fmaxf(amax, shfl_xor_f(amax, m));
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
KERNEL ie_quant_q8(const G float *x, G u8 *q, u32 nb) {
  const u32 b = wgid() * 8u + (tid() >> 5);
  if (b >= nb) return; /* whole waves leave together */
  quant_wave(x[b * 32u + lane()], b, q, nb);
}

/* y = x / sqrt(mean(x^2) + eps) * w. One workgroup. n is a multiple of 32.
 * If q is not NULL, also write the Q8 copy of y (fused quantize): wave w
 * does the blocks w, w + 8, ...; each lane stores and quantizes its own y. */
KERNEL ie_rmsnorm(const G float *x, const G float *w, G float *y, u32 n, float eps, G u8 *q) {
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

/* ------------------------------------------------------------------ GEMV */

/* The fused epilogue of a GEMV row (the CPU backend does the same, in the
 * same order): y = acc (+ bias[r]) then (res[r] +) y. */
static inline float epilogue(float acc, u32 r, const G float *bias, const G float *res) {
  float y = acc;
  if (bias) y = y + bias[r];
  if (res) y = res[r] + y;
  return y;
}

/* y = W x: W is Q4 (GPU layout: words qw, f16 scales qs), x is Q8 (as
 * written by ie_quant_q8). Grid: ie_gemv_ngroups(rows) workgroups of 8
 * waves; wave v of workgroup g does row ie_gemv_row(g, v); lane l does the
 * blocks ie_lane_blk(l, t) < nb; then the ie_tree wave reduction. */
KERNEL ie_gemv_q4q8(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                  const G float *bias, const G float *res) {
  const u32 r = ie_gemv_row(wgid(), tid() >> 5), l = lane();
  if (r >= rows) return;
  const G u32 *aw = (const G u32 *)xq;
  const G float *da = (const G float *)(xq + 32u * nb);
  const G u32 *asum = (const G u32 *)(xq + 36u * nb);
  float acc = 0.0f;
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
    const u32 b = ie_lane_blk(l, t);
    /* One 128-bit load of the 4 nibble words of the block (16-byte aligned),
     * two 128-bit loads of the 8 activation words. */
    const u32x4 w4 = *(const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb));
    const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u);
    const u32x4 a1 = *(const G u32x4 *)(aw + b * 8u + 4u);
    const u32 wq[4] = {w4.x, w4.y, w4.z, w4.w};
    const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const Mem qm = {wq}, am = {av};
    const int dot = (int)ie_q4q8_block(qm, 0u, am, 0u, asum[b]);
    acc += ((float)qs[ie_q4_dst_scale(r, b, nb)] * da[b]) * (float)dot;
  }
  acc = wave_tree_f(acc);
  if (l == 0u) y[r] = epilogue(acc, r, bias, res);
}

/* y = W x: W is Q8_0 (GPU layout: 8 int8 words per block in qw, f16 scales
 * qs), x is Q8. The same grid, lane split and reduction as ie_gemv_q4q8;
 * the block dot product is ie_q8q8_block (8 v_dot4_i32_iu8, both signed). */
KERNEL ie_gemv_q8q8(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb,
                  const G float *bias, const G float *res) {
  const u32 r = ie_gemv_row(wgid(), tid() >> 5), l = lane();
  if (r >= rows) return;
  const G u32 *aw = (const G u32 *)xq;
  const G float *da = (const G float *)(xq + 32u * nb);
  float acc = 0.0f;
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
    const u32 b = ie_lane_blk(l, t);
    /* Two 128-bit loads of the 8 weight words (32-byte aligned blocks), two
     * of the 8 activation words. */
    const G u32 *wp = qw + ie_q8_dst_word(r, b, 0u, nb);
    const u32x4 w0 = *(const G u32x4 *)wp, w1 = *(const G u32x4 *)(wp + 4u);
    const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u);
    const u32x4 a1 = *(const G u32x4 *)(aw + b * 8u + 4u);
    const u32 wq[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const Mem qm = {wq}, am = {av};
    const int dot = (int)ie_q8q8_block(qm, 0u, am, 0u);
    acc += ((float)qs[ie_q8_dst_scale(r, b, nb)] * da[b]) * (float)dot;
  }
  acc = wave_tree_f(acc);
  if (l == 0u) y[r] = epilogue(acc, r, bias, res);
}

/* y = W x for an f32 matrix (F16 weights dequantized at load):
 * the same grid; lane l does the elements ie_lane_blk(l, t) < cols. */
KERNEL ie_gemv_f32(const G float *w, const G float *x, G float *y, u32 rows, u32 cols, const G float *bias,
                   const G float *res) {
  const u32 r = ie_gemv_row(wgid(), tid() >> 5), l = lane();
  if (r >= rows) return;
  const G float *row = w + (unsigned long)r * cols;
  float acc = 0.0f;
  for (u32 t = 0; ie_lane_blk(l, t) < cols; t++) acc += row[ie_lane_blk(l, t)] * x[ie_lane_blk(l, t)];
  acc = wave_tree_f(acc);
  if (l == 0u) y[r] = epilogue(acc, r, bias, res);
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
KERNEL ie_swiglu(G float *a, const G float *b, u32 n, G u8 *q) {
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
KERNEL ie_attn(const G float *q, const G float *kc, const G float *vc, G float *sc, G float *out, u32 pos,
               u32 n_ctx, u32 hd, u32 n_head, u32 n_kv, G u8 *oq) {
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
    const float o = acc * inv;
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
KERNEL ie_attn_split(const G float *q, const G float *kc, const G float *vc, G float *part, G float *out, u32 pos,
                     u32 hd, u32 n_head, u32 n_kv, u32 nsplit, u32 ch, G u32 *count, G u8 *oq) {
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
    for (u32 mm = 16u; mm >= 1u; mm >>= 1) d += shfl_xor_f(d, mm);
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
  const G float *ph = part + (unsigned long)h * nsplit * (hd + 2u);
  float M2 = -__builtin_inff(), L2 = 0.0f;
  for (u32 t = 0; t < nsplit; t++) M2 = __builtin_fmaxf(M2, ph[t * (hd + 2u)]);
  for (u32 t = 0; t < nsplit; t++) L2 += ph[t * (hd + 2u) + 1u] * __builtin_expf(ph[t * (hd + 2u)] - M2);
  if (tid() < hd) { /* hd % 32 == 0 when oq is set: whole waves */
    float o = 0.0f;
    for (u32 t = 0; t < nsplit; t++) o += ph[t * (hd + 2u) + 2u + tid()] * __builtin_expf(ph[t * (hd + 2u)] - M2);
    o = o / L2;
    out[h * hd + tid()] = o;
    if (oq) quant_wave(o, (h * hd + tid()) >> 5, oq, (n_head * hd) >> 5);
  }
  if (tid() == 0u) count[h] = 0u;
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
