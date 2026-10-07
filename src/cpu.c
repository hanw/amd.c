/* cpu.c -- the CPU backend. It runs the same algorithm as the GPU kernels
 * (kernels/ie_kernels.c), with the same work split, so that the two agree up to float
 * rounding:
 *  - activations are quantized per block of 32: d = amax / 127,
 *    q = round(x / d), stored as words + d + asum (the sum of the 32 q);
 *  - GEMV row r: lane l (l < 32) adds, for blocks b = ie_lane_blk(l, t) < nb,
 *    d_w * d_a * (float)(int32)ie_q4q8_block(...); then the 32 lane sums are
 *    added in the order of ie_tree (shuffle down by 16, 8, 4, 2, 1).
 * All index and integer arithmetic comes from core/ie_core.h. */
#include <math.h>
#include <string.h>
#include <time.h>

#include "../core/ie_core.h"
#include "backend.h"
#include "common.h"

typedef struct {
  backend base;
  const model *m;
  const graph *g;
  uint8_t *arena;
  float *kc, *vc; /* KV cache: [layer][pos][n_kv * hd] */
} cpu_backend;

#define BUF(b, id) ((void *)((b)->arena + (b)->g->bufs[id].off))

/* ---------------------------------------------------------------- kernels */

void cpu_quant_q8(const float *x, uint8_t *q, uint32_t nb) {
  int8_t *qv = (int8_t *)q;
  float *d = (float *)(q + 32u * nb);
  u32 *asum = (u32 *)(q + 36u * nb);
  for (uint32_t b = 0; b < nb; b++) {
    float amax = 0.0f;
    for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[32 * b + j]));
    float db = amax / 127.0f;
    u32 s = 0;
    for (int j = 0; j < 32; j++) {
      int v = db != 0.0f ? (int)roundf(x[32 * b + j] / db) : 0;
      qv[32 * b + j] = (int8_t)v;
      s += (u32)v;
    }
    d[b] = db;
    asum[b] = s;
  }
}

/* The wave tree reduction of ie_tree on 32 lane values (lane 0's result). */
static float wave_tree(float v[32]) {
  for (u32 m = 1; m <= 5; m++) {
    u32 off = 32u >> m; /* 16, 8, 4, 2, 1 */
    for (u32 i = 0; i < off; i++) v[i] = v[i] + v[i + off];
  }
  return v[0];
}

void cpu_gemv_q4(const mat *w, const uint8_t *xq, float *y) {
  const u32 nb = w->nb;
  const Mem qw = {w->qw}, aw = {(const u32 *)xq};
  const float *da = (const float *)(xq + 32u * nb);
  const u32 *asum = (const u32 *)(xq + 36u * nb);
  const u32 ng = ie_gemv_ngroups(w->rows);
  for (u32 g = 0; g < ng; g++)
    for (u32 v = 0; v < ie_rows_per_wg(); v++) {
      const u32 r = ie_gemv_row(g, v);
      if (r >= w->rows) continue;
      float lane[32];
      for (u32 l = 0; l < ie_wave(); l++) {
        float acc = 0.0f;
        for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
          const u32 b = ie_lane_blk(l, t);
          const int32_t dot = (int32_t)ie_q4q8_block(qw, ie_q4_dst_word(r, b, 0, nb), aw, b * 8u, asum[b]);
          const float dw = ie_f16_to_f32(w->qs[ie_q4_dst_scale(r, b, nb)]);
          acc += (dw * da[b]) * (float)dot;
        }
        lane[l] = acc;
      }
      y[r] = wave_tree(lane);
    }
}

/* Q8_0 matrix x Q8 activations: the same grid, lanes and reduction as Q4;
 * the block dot product is ie_q8q8_block (law q8q8_block). */
void cpu_gemv_q8(const mat *w, const uint8_t *xq, float *y) {
  const u32 nb = w->nb;
  const Mem qw = {w->qw}, aw = {(const u32 *)xq};
  const float *da = (const float *)(xq + 32u * nb);
  const u32 ng = ie_gemv_ngroups(w->rows);
  for (u32 g = 0; g < ng; g++)
    for (u32 v = 0; v < ie_rows_per_wg(); v++) {
      const u32 r = ie_gemv_row(g, v);
      if (r >= w->rows) continue;
      float lane[32];
      for (u32 l = 0; l < ie_wave(); l++) {
        float acc = 0.0f;
        for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
          const u32 b = ie_lane_blk(l, t);
          const int32_t dot = (int32_t)ie_q8q8_block(qw, ie_q8_dst_word(r, b, 0, nb), aw, b * 8u);
          const float dw = ie_f16_to_f32(w->qs[ie_q8_dst_scale(r, b, nb)]);
          acc += (dw * da[b]) * (float)dot;
        }
        lane[l] = acc;
      }
      y[r] = wave_tree(lane);
    }
}

/* f32 matrix: the same grid and lane split over elements. */
static void cpu_gemv_f32(const mat *w, const float *x, float *y) {
  for (u32 r = 0; r < w->rows; r++) {
    float lane[32];
    const float *row = w->f + (size_t)r * w->cols;
    for (u32 l = 0; l < 32; l++) {
      float acc = 0.0f;
      for (u32 t = 0; ie_lane_blk(l, t) < w->cols; t++) acc += row[ie_lane_blk(l, t)] * x[ie_lane_blk(l, t)];
      lane[l] = acc;
    }
    y[r] = wave_tree(lane);
  }
}

/* Row r of a matrix as floats (the embedding). */
static void mat_row(const mat *w, u32 r, float *out) {
  if (w->kind == MAT_F32) {
    memcpy(out, w->f + (size_t)r * w->cols, w->cols * 4);
    return;
  }
  const Mem qw = {w->qw};
  for (u32 b = 0; b < w->nb; b++) {
    const float d = ie_f16_to_f32(w->qs[ie_q4_dst_scale(r, b, w->nb)]);
    for (u32 j = 0; j < 32; j++) {
      const u32 nib = ie_q4_nib(IE_LOAD(qw, ie_q4_dst_word(r, b, ie_q4_word_of(j), w->nb)), j);
      out[32 * b + j] = d * (float)((int)nib - 8);
    }
  }
}

static void rmsnorm(const float *x, const float *wt, float *y, u32 n, float eps) {
  float ss = 0.0f;
  for (u32 i = 0; i < n; i++) ss += x[i] * x[i];
  const float s = 1.0f / sqrtf(ss / (float)n + eps);
  for (u32 i = 0; i < n; i++) y[i] = x[i] * s * wt[i];
}

static void rope(float *x, u32 nh, u32 hd, int neox, const float *c, const float *s) {
  const u32 h2 = hd / 2;
  for (u32 h = 0; h < nh; h++) {
    float *v = x + h * hd;
    for (u32 i = 0; i < h2; i++) {
      const u32 i0 = neox ? i : 2 * i, i1 = neox ? i + h2 : 2 * i + 1;
      const float a = v[i0], b = v[i1];
      v[i0] = a * c[i] - b * s[i];
      v[i1] = a * s[i] + b * c[i];
    }
  }
}

/* Attention of one decode token. Scores buffer: [n_head][n_ctx]. */
static void attention(const model *m, const float *q, const float *kc, const float *vc, u32 pos, u32 n_ctx, float *sc,
                      float *out) {
  const u32 hd = m->hd, kvd = m->n_kv * hd, grp = m->n_head / m->n_kv;
  const float scale = 1.0f / sqrtf((float)hd);
  for (u32 h = 0; h < m->n_head; h++) {
    const float *qh = q + h * hd;
    const u32 kh = h / grp; /* GQA: the KV head of query head h */
    float *s = sc + (size_t)h * n_ctx;
    float mx = -INFINITY;
    for (u32 p = 0; p <= pos; p++) {
      const float *k = kc + (size_t)p * kvd + kh * hd;
      float d = 0.0f;
      for (u32 i = 0; i < hd; i++) d += qh[i] * k[i];
      s[p] = d * scale;
      mx = fmaxf(mx, s[p]);
    }
    float sum = 0.0f;
    for (u32 p = 0; p <= pos; p++) sum += (s[p] = expf(s[p] - mx));
    const float inv = 1.0f / sum;
    float *o = out + h * hd;
    for (u32 i = 0; i < hd; i++) {
      float acc = 0.0f;
      for (u32 p = 0; p <= pos; p++) acc += s[p] * vc[(size_t)p * kvd + kh * hd + i];
      o[i] = acc * inv;
    }
  }
}

/* ---------------------------------------------------------------- backend */

static uint32_t cpu_step(backend *bk, uint32_t tok, uint32_t pos, float *logits, double *ms);
static void cpu_close(backend *bk);

backend *cpu_open(const model *m, const graph *g) {
  cpu_backend *b = calloc(1, sizeof *b);
  b->base.step = cpu_step;
  b->base.close = cpu_close;
  b->m = m;
  b->g = g;
  b->arena = ie_alloc(g->arena);
  const size_t kv = (size_t)m->n_layer * g->n_ctx * m->n_kv * m->hd;
  b->kc = ie_alloc(kv * 4);
  b->vc = ie_alloc(kv * 4);
  return &b->base;
}

static void cpu_close(backend *bk) {
  cpu_backend *b = (cpu_backend *)bk;
  free(b->arena);
  free(b->kc);
  free(b->vc);
  free(b);
}

static uint32_t cpu_step(backend *bk, uint32_t tok, uint32_t pos, float *logits, double *ms) {
  cpu_backend *B = (cpu_backend *)bk;
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  const model *m = B->m;
  const graph *g = B->g;
  const u32 h2 = m->hd / 2, kvd = m->n_kv * m->hd;
  if (tok >= m->vocab) ie_die("token %u >= vocab %u", tok, m->vocab);
  if (pos >= g->n_ctx) ie_die("position %u >= context %u", pos, g->n_ctx);
  for (u32 i = 0; i < g->n_ops; i++) {
    const op *o = &g->ops[i];
    float *a = o->a >= 0 ? BUF(B, o->a) : NULL, *bb = o->b >= 0 ? BUF(B, o->b) : NULL;
    uint8_t *qo = o->qo >= 0 ? (uint8_t *)BUF(B, o->qo) : NULL;
    const float *res = o->res >= 0 ? (const float *)BUF(B, o->res) : NULL;
    switch (o->kind) {
      case OP_EMBED: mat_row(&m->tok, tok, a); break;
      case OP_RMSNORM:
        rmsnorm(a, o->v->f, bb, o->n, m->eps);
        if (qo) cpu_quant_q8(bb, qo, o->n / 32);
        break;
      case OP_QUANT: cpu_quant_q8(a, (uint8_t *)bb, o->n / 32); break;
      case OP_GEMV_Q4:
      case OP_GEMV_Q8:
      case OP_GEMV_F32:
        if (o->kind == OP_GEMV_Q4) cpu_gemv_q4(o->w, (const uint8_t *)a, bb);
        else if (o->kind == OP_GEMV_Q8) cpu_gemv_q8(o->w, (const uint8_t *)a, bb);
        else cpu_gemv_f32(o->w, a, bb);
        /* the fused epilogue, in the order of the GPU kernels */
        for (u32 j = 0; j < o->w->rows; j++) {
          float y = bb[j];
          if (o->v) y = y + o->v->f[j];
          if (res) y = res[j] + y;
          bb[j] = y;
        }
        break;
      case OP_BIAS:
        for (u32 j = 0; j < o->n; j++) a[j] += o->v->f[j];
        break;
      case OP_ADD: {
        float *c = BUF(B, o->c);
        for (u32 j = 0; j < o->n; j++) c[j] = a[j] + bb[j];
        break;
      }
      case OP_ROPE:
        rope(a, o->nh, m->hd, m->rope == ROPE_NEOX, g->rope_cos + (size_t)pos * h2, g->rope_sin + (size_t)pos * h2);
        break;
      case OP_ROPE_KV: {
        const float *cs = g->rope_cos + (size_t)pos * h2, *sn = g->rope_sin + (size_t)pos * h2;
        rope(a, o->nh, m->hd, m->rope == ROPE_NEOX, cs, sn);
        rope(bb, m->n_kv, m->hd, m->rope == ROPE_NEOX, cs, sn);
        const size_t base = ((size_t)o->layer * g->n_ctx + pos) * kvd;
        memcpy(B->kc + base, bb, kvd * 4);
        memcpy(B->vc + base, BUF(B, o->c), kvd * 4);
        break;
      }
      case OP_KV: {
        const size_t base = ((size_t)o->layer * g->n_ctx + pos) * kvd;
        memcpy(B->kc + base, a, kvd * 4);
        memcpy(B->vc + base, bb, kvd * 4);
        break;
      }
      case OP_ATTN: {
        const size_t base = (size_t)o->layer * g->n_ctx * kvd;
        float *sc = BUF(B, o->c);
        attention(m, a, B->kc + base, B->vc + base, pos, g->n_ctx, sc, bb);
        if (qo) cpu_quant_q8(bb, qo, o->n / 32);
        break;
      }
      case OP_SWIGLU:
        for (u32 j = 0; j < o->n; j++) a[j] = a[j] / (1.0f + expf(-a[j])) * bb[j];
        if (qo) cpu_quant_q8(a, qo, o->n / 32);
        break;
      case OP_ARGMAX: {
        u32 best = 0;
        for (u32 j = 1; j < o->n; j++)
          if (a[j] > a[best]) best = j;
        *(u32 *)bb = best;
        break;
      }
    }
  }
  if (logits) memcpy(logits, BUF(B, g->logits), m->vocab * 4);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  if (ms) *ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
  return *(u32 *)BUF(B, g->argmax);
}

