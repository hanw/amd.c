/* model.c -- load Llama / Qwen2 from GGUF, repack Q4_0 into the verified GPU
 * layout, build the op list of one decode step. */
#include "model.h"

#include <math.h>
#include <string.h>

#include "../core/ie_core.h"
#include "common.h"

/* ---------------------------------------------------------------- loading */

static const gguf_tensor *need_tensor(gguf_file *g, const char *name) {
  const gguf_tensor *t = gguf_tensor_find(g, name);
  if (!t) ie_die("missing tensor %s", name);
  if (!t->data) ie_die("tensor %s: unsupported type %u", name, t->type);
  return t;
}

/* Dequantize n elements of tensor data (F32 / F16 / Q8_0 / Q4_0) to float. */
static void dequant(const gguf_tensor *t, uint64_t first, uint64_t n, float *out) {
  const uint8_t *p = t->data;
  switch (t->type) {
    case GGML_F32: memcpy(out, p + first * 4, n * 4); break;
    case GGML_F16:
      for (uint64_t i = 0; i < n; i++) {
        uint16_t h;
        memcpy(&h, p + (first + i) * 2, 2);
        out[i] = ie_f16_to_f32(h);
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
    default: ie_die("tensor %s: type %s is not supported here", t->name, ggml_type_name(t->type));
  }
}

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

/* Load matrix `name` of rows x cols (GGUF ne = [cols, rows]). A Q8_0 matrix
 * keeps its integer form (MAT_Q8) when q8_ok, else it becomes f32. */
static void load_mat(mat *w, gguf_file *g, const char *name, uint32_t rows, uint32_t cols, uint64_t *bytes, int q8_ok) {
  const gguf_tensor *t = need_tensor(g, name);
  memset(w, 0, sizeof *w);
  if (t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != 1 || t->ne[3] != 1)
    ie_die("tensor %s: shape [%llu, %llu], expected [%u, %u]", name, (unsigned long long)t->ne[0],
           (unsigned long long)t->ne[1], cols, rows);
  w->rows = rows;
  w->cols = cols;
  if (t->type == GGML_Q4_0) {
    w->kind = MAT_Q4;
    w->nb = cols / 32;
    repack_q4(w, t->data);
    *bytes += (uint64_t)rows * w->nb * 18;
  } else if (t->type == GGML_Q6_K && q8_ok && ie_q8_sizes_ok(rows, cols / 32)) {
    /* Q6_K: no integer kernel; requantize to Q8_0 at load (an
     * approximation: Q8_0 has one scale per 32 weights, Q6_K one per 16) */
    w->kind = MAT_Q8;
    w->nb = cols / 32;
    float *row = ie_alloc((size_t)cols * 4);
    uint8_t *q8 = ie_alloc((size_t)rows * w->nb * 34);
    for (uint32_t r = 0; r < rows; r++) {
      dequant(t, (uint64_t)r * cols, cols, row);
      quant_q8_0_bytes(row, cols, q8 + (size_t)r * w->nb * 34);
    }
    repack_q8(w, q8);
    free(row), free(q8);
    *bytes += (uint64_t)rows * w->nb * 34;
  } else if (t->type == GGML_Q8_0 && q8_ok && ie_q8_sizes_ok(rows, cols / 32)) {
    w->kind = MAT_Q8;
    w->nb = cols / 32;
    repack_q8(w, t->data);
    *bytes += (uint64_t)rows * w->nb * 34;
  } else {
    w->kind = MAT_F32;
    w->f = ie_alloc((size_t)rows * cols * 4);
    dequant(t, 0, (uint64_t)rows * cols, w->f);
    *bytes += (uint64_t)rows * cols * 4;
  }
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
                &L->wlqkv, &L->wz, &L->wbeta, &L->walpha, &L->win};
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

void model_load(model *m, gguf_file *g) {
  memset(m, 0, sizeof *m);
  char *arch = gguf_get_str(g, "general.architecture");
  if (!arch) ie_die("missing general.architecture");
  if (strcmp(arch, "llama") == 0) m->arch = ARCH_LLAMA, m->rope = ROPE_NORM;
  else if (strcmp(arch, "qwen2") == 0) m->arch = ARCH_QWEN2, m->rope = ROPE_NEOX;
  else if (strcmp(arch, "qwen35") == 0) m->arch = ARCH_QWEN35, m->rope = ROPE_NEOX;
  else ie_die("architecture %s is not supported (llama, qwen2, qwen35)", arch);
  const int q35 = m->arch == ARCH_QWEN35;

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
  /* qwen35: a Q8_0 embedding stays Q8_0 (1.3 GB for the 27B model; as f32
   * it would be 5 GB) */
  load_mat(&m->tok, g, "token_embd.weight", m->vocab, m->dim, &wb, q35);
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
  for (uint32_t i = 0; i < m->n_layer; i++) {
    layer *L = &m->l[i];
    char n[128];
#define NAME(s) (snprintf(n, sizeof n, "blk.%u." s, i), n)
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
      concat_mats(&L->win, in4, 4);
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

void model_free(model *m) {
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
  const int quant_in = w->kind == MAT_Q4 || w->kind == MAT_Q8;
  op *o = emit(B, w->kind == MAT_Q4 ? OP_GEMV_Q4 : w->kind == MAT_Q8 ? OP_GEMV_Q8 : OP_GEMV_F32, quant_in ? xq : xf, y, -1);
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
  if (!(nf && nf[0] == '1')) {
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
    if (q35) {
      qsrc = new_buf(g, "q", qd * 4);
      o = emit(B, OP_QKN_ROPE_KV, qkv, qsrc, -1), o->n = qd, o->nh = m->n_head;
      o->v = &L->q_norm, o->nv = &L->k_norm;
    } else {
      o = emit(B, OP_ROPE_KV, qkv, qkv, qkv), o->n = qd, o->nh = m->n_head;
      o->boff = qd * 4, o->coff = (qd + kvd) * 4;
    }
    int att = new_buf(g, "attn", qd * 4);
    const uint32_t sc_cpu = m->n_head * n_ctx, sc_gpu = m->n_head * ie_att_max_split(n_ctx) * (m->hd + 2);
    int sc = new_buf(g, "scores", (sc_cpu > sc_gpu ? sc_cpu : sc_gpu) * 4);
    o = emit(B, OP_ATTN, qsrc, att, sc), o->n = qd;
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

void graph_build(graph *g, const model *m, uint32_t n_ctx, uint32_t T, int mtp) {
  memset(g, 0, sizeof *g);
  g->bufs = calloc(MAX_BUFS, sizeof(buf));
  g->n_ctx = n_ctx;
  g->T = T ? T : 1;
  g->h_out = g->mtp_h = g->mtp_g = g->mtp_logits = g->mtp_argmax = -1;
  gb B = {g, 0, -1};
  uint32_t dim = m->dim;
  if (mtp && !m->has_mtp) ie_die("the model has no MTP head");

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
        matvec_into(&B, &L->walpha, xn, xq, lin, (cd + inner + m->n_vh) * 4, "alpha", NULL, -1);
      }
      int go = new_buf(g, "gdn", inner * 4);
      o = emit(&B, OP_GDN, lin, go, -1), o->n = inner;
      int gq = L->wo.kind != MAT_F32 ? fuse_quant(&B, inner, "gdn_q8") : -1;
      x = matvec(&B, &L->wo, go, gq, "x", NULL, x); /* x = x + Wout gdn */
    } else {
    x = attn_block(&B, m, L, x, xn, xq, n_ctx);
    }
    x = ffn_block(&B, m, L, x);
    if (l + 1 < m->n_layer) {
      const layer *N = &m->l[l + 1];
      xn = fuse_norm(&B, &N->attn_norm, dim, "attn_in", layer_in_q8(N), "attn_in_q8", &xq);
    } else {
      xn = fuse_norm(&B, &m->out_norm, dim, "out_in", m->out.kind != MAT_F32, "out_in_q8", &xq);
    }
  }
  B.layer = -1;
  g->logits = matvec(&B, &m->out, xn, xq, "logits", NULL, -1);
  g->argmax = new_buf(g, "argmax", 4);
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
    g->mtp_logits = matvec(&B, &m->out, g->mtp_g, gq2, "mtp_logits", NULL, -1);
    g->mtp_argmax = new_buf(g, "mtp_argmax", 4);
    emit(&B, OP_ARGMAX, g->mtp_logits, g->mtp_argmax, -1)->n = m->vocab;
  }
  /* The host reads (or copies) these buffers between runs: they live
   * through the whole op list. */
  const int keep[] = {g->logits, g->argmax, g->h_out, g->mtp_h, g->mtp_g, g->mtp_logits, g->mtp_argmax};
  for (unsigned k = 0; k < sizeof keep / sizeof keep[0]; k++) touch(g, keep[k], 0), touch(g, keep[k], g->n_ops);

  /* bytes of weights read per token (GEMV + one embedding row) */
  uint64_t wb = 0;
  for (uint32_t i = 0; i < g->n_main; i++) {
    const op *p = &g->ops[i];
    if (p->kind == OP_GEMV_Q4) wb += (uint64_t)p->w->rows * p->w->nb * 18;
    if (p->kind == OP_GEMV_F32) wb += (uint64_t)p->w->rows * p->w->cols * 4;
    if (p->kind == OP_GEMV_Q8) wb += (uint64_t)p->w->rows * p->w->nb * 34;
  }
  g->weight_bytes_per_token = wb;

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
  memset(g, 0, sizeof *g);
}
