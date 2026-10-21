/* model.h -- Llama / Qwen2 models from GGUF, and the op list of one decode
 * step. Both backends (cpu.c, hip.c) run the same op list on the same
 * arena offsets. */
#ifndef IE_MODEL_H
#define IE_MODEL_H

#include <stdint.h>

#include "gguf.h"

enum { ARCH_LLAMA = 0, ARCH_QWEN2 = 1, ARCH_QWEN35 = 2, ARCH_FLINT = 3 };
enum { ROPE_NORM = 0, ROPE_NEOX = 1 }; /* NORM: pairs (2i, 2i+1); NEOX: (i, i + hd/2) */

/* A weight matrix: rows x cols, y = W x. */
enum { MAT_Q4 = 0, MAT_F32 = 1, MAT_Q8 = 2, MAT_Q4K = 3, MAT_Q6K = 4, MAT_Q5K = 5, MAT_IQ4XS = 6 };
typedef struct {
  int kind;
  uint32_t rows, cols, nb; /* nb = cols / 32 (Q4 and Q8) */
  /* MAT_Q4: the verified GPU layout (core/ie_core.h): rows*nb*4 nibble words
   * and rows*nb f16 scales. */
  uint32_t *qw;
  uint16_t *qs;
  /* MAT_Q8: the verified Q8_0 GPU layout (ie_q8_dst_word, ie_q8_dst_scale):
   * rows*nb*8 int8 words in qw and rows*nb f16 scales in qs. */
  /* MAT_Q4K (GGUF Q4_K, sub-blocks of 32 in super-blocks of 256): qw as
   * MAT_Q4 (rows*nb*4 nibble words, the nibbles of each sub-block in the
   * Q4_0 order, values 0..15); qs: rows*nb u16 (6-bit scale | 6-bit min << 8)
   * of each sub-block, then rows*(nb/8)*2 u16 (f16 d, f16 dmin) of each
   * super-block. Weight = d*sc*q - dmin*mn (as ggml). Index maps: ie_q4k_* in core/ie_core.h. */
  /* MAT_Q6K (GGUF Q6_K): qw = the low 4 bits as MAT_Q4 nibble words, then
   * 2 words of high bits per sub-block (ie_q6k_hw); qs = (sc0 | sc1 << 8)
   * (int8) per sub-block, then f16 d per super-block (ie_q6k_sc, ie_q6k_d). */
  /* MAT_Q5K, MAT_IQ4XS: core/ie_ext.h. */
  /* MAT_F32: rows*cols floats (dequantized from F32/F16, or Q8_0 for the embedding). */
  float *f;
  /* Experts (flint MoE): n_exp > 1 matrices of rows x cols, one after the
   * other in qw, qs (or f); expert e starts at e * mat_qw_words(), e *
   * mat_qs_n() (or e * rows * cols floats). n_exp 0 or 1: one matrix. */
  uint32_t n_exp;
  /* Streaming load (ie_mat_sink set): the device copies of qw and qs (or
   * f in d0); the host arrays are then freed (NULL). */
  void *d0, *d1;
} mat;

typedef struct {
  uint32_t n;
  float *f;
} vec;


typedef struct {
  vec attn_norm, ffn_norm;
  mat wq, wk, wv, wo, wgate, wup, wdown;
  vec bq, bk, bv; /* n == 0 if absent */
  /* Merged matrices (rows == 0 if not merged: the parts differ in kind).
   * wqkv = [wq; wk; wv] with bias bqkv; wgu = [wgate; wup]. When merged, the
   * parts are freed (their rows stay set, their arrays are NULL). */
  mat wqkv, wgu;
  vec bqkv;
  /* qwen35 (Qwen3.5 / 3.8): rec = 1 for a linear attention layer (Gated
   * DeltaNet), 0 for a full attention layer. ffn_norm holds
   * post_attention_norm. Full attention: wq has 2 * n_head * hd rows, per
   * head [q (hd) | gate (hd)]; q_norm and k_norm are per head RMSNorm
   * weights (hd). Linear attention: wlqkv (conv_dim rows: q | k | v), wz
   * (inner), wbeta, walpha (n_vh rows each), merged into win = [wlqkv; wz;
   * wbeta; walpha] when they have one kind; conv = ssm_conv1d [conv_dim][4];
   * wo = ssm_out. */
  int rec;
  uint32_t kvi, sti; /* index among the attention layers (KV cache) / the linear layers (state) */
  vec q_norm, k_norm;
  mat wlqkv, wz, wbeta, walpha, win;
  vec conv, dt_bias, ssm_a, ssm_norm;
  /* flint: win = sliding window (tokens, the current one included; 0: full
   * attention), norope = no RoPE in this layer, sinks = the learned logit
   * per head added to the softmax denominator. moe: wr = router (n_exp rows,
   * f32), egu = [gate_e; up_e] per expert (or eg, eu when their kinds
   * differ), ed = down per expert, ebias = the router bias (selection only;
   * n == 0: none); the shared expert is wgate, wup, wdown (wgu). */
  uint32_t swin;
  int norope, moe;
  vec sinks, ebias;
  mat wr, egu, eg, eu, ed;
} layer;

/* DFlash 2 drafter (z-lab/dflash, a separate GGUF, architecture "dflash"):
 * a block of B tokens [t(P), mask x (B-1)] at positions P .. P+B-1 runs
 * through n_layer Qwen3 layers whose attention also sees the context: the
 * keys and values of fc(the hidden states of n_tl target layers) of the
 * earlier positions (non-causal, sliding window). Two-tap dynamic
 * convolutions along the block wrap the attention and the FFN. The token
 * embedding and the output matrix are the target's. A selector (host)
 * picks one path through the top-k candidates of each position. */
typedef struct {
  vec attn_norm, ffn_norm, q_norm, k_norm;
  vec aconv_base, fconv_base; /* [2][kernel][dim]: [0] before, [1] after the sublayer */
  mat wq, wk, wv, wo, wgate, wup, wdown, wgu, aconv, fconv; /* aconv, fconv: dim -> 2*kernel*groups */
} df_layer;
typedef struct {
  uint32_t n_layer, dim, ffn, n_head, n_kv, hd, block, window, mask, top_k, rank, kernel, group;
  uint32_t n_tl, tl[8]; /* target layers (0-based): their outputs, concatenated, are fc's input */
  float eps, rope_base;
  vec enc_norm, out_norm;
  mat fc, sel_h;
  df_layer *l;
  const void *sel_prev, *sel_next; /* gguf_tensor of the selector codebooks (host, the file mapped) */
} dflash;

typedef struct {
  int arch, rope;
  uint32_t n_layer, dim, ffn, n_head, n_kv, hd, vocab, ctx_train;
  float eps, rope_base;
  uint32_t n_rot;        /* rotated dims per head (hd, or fewer: qwen35 partial RoPE) */
  uint32_t n_kvl, n_rec; /* layers with a KV cache / with a linear attention state */
  /* qwen35 linear attention: key heads, value heads, head dim (keys and
   * values), conv window; conv_dim = 2 * n_kh * sd + n_vh * sd. */
  uint32_t n_kh, n_vh, sd, d_conv, conv_dim;
  /* flint MoE: experts, experts per token, expert FFN, shared expert FFN;
   * the routing weights are normalized and scaled by escale */
  uint32_t n_exp, n_used, eff, sff;
  float escale;
  mat tok;   /* token_embd (Q4 repacked or F32), rows = vocab, cols = dim */
  mat out;   /* output; may share the arrays of tok (tied) */
  int tied;
  vec out_norm;
  layer *l; /* n_layer layers, then (has_mtp) the MTP layer l[n_layer] */
  /* qwen35 multi-token prediction (MTP, "nextn") head: one full attention
   * layer l[n_layer] (its own KV cache, kvi = n_kvl - 1) after
   * x = eh_proj [rmsnorm(embed(token)) * enorm | rmsnorm(h) * hnorm]; then
   * rmsnorm(.) * head_norm (this is also the h of the next draft) and the
   * output matrix. */
  int has_mtp;
  mat eh;
  /* the output matrix requantized to Q4_0 for the MTP drafts only (rows ==
   * 0: the drafts use out). Half the bytes of Q8_0; the verify step still
   * uses out, so the output is unchanged. */
  mat out_draft;
  /* the draft head's rows: tokens 0 .. dv_n - 1, then dv_s .. (IE_DRAFT_VOCAB; 0: all rows, row = token) */
  uint32_t dv_n, dv_s;
  vec enorm, hnorm, head_norm;
  dflash *df; /* the DFlash 2 drafter (NULL: none) */
  /* tokenizer */
  int tok_gpt2; /* 1: GPT2 byte-level, 0: SentencePiece */
  uint32_t n_tokens;
  char **tokens;
  uint64_t weight_bytes; /* bytes of all repacked/dequantized weights */
} model;

/* Load the MTP head (blk.N.nextn.*) from g (the model file or a separate
 * file); returns 0 if g has none. */
int model_load_mtp(model *m, gguf_file *g);
/* Load a DFlash 2 drafter from g (kept open: the selector reads its
 * codebooks); returns 0 if g is not one. */
int model_load_dflash(model *m, gguf_file *g);
/* Row tok of a selector codebook (which 0: predecessor, 1: successor). */
void model_df_code(const model *m, int which, uint32_t tok, float *out);
/* Make m->out_draft from the output matrix of g (the model file). */
void model_make_draft_head(model *m, gguf_file *g);
/* The token of row i of the draft head. */
static inline uint32_t model_draft_tok(const model *m, uint32_t i) { return m->dv_n && i >= m->dv_n ? m->dv_s + (i - m->dv_n) : i; }
/* If set, model_load gives every finished matrix to this function (for
 * example: copy it to the GPU and free the host arrays), so that the host
 * never holds all the weights at once. */
extern void (*ie_mat_sink)(mat *w);
void model_load(model *m, gguf_file *g);
void model_free(model *m);
/* Repack GGUF Q4_0 bytes (w->rows, w->cols, w->nb set) into w->qw, w->qs. */
void repack_q4(mat *w, const uint8_t *src);
/* A matrix of rows x cols from GGUF bytes of type ttype, as model_load
 * makes it (Q8_0 kept as Q8; K and IQ kinds native where supported). */
void model_mat_from_gguf(mat *w, uint32_t ttype, const uint8_t *src, uint32_t rows, uint32_t cols);
/* Repack GGUF Q8_0 bytes into w->qw, w->qs (the verified Q8_0 layout). */
void repack_q8(mat *w, const uint8_t *src);
/* The text of token id t (malloc'ed, decoded bytes; *len set). */
char *model_detok(const model *m, uint32_t t, uint32_t *len);

/* ---- the op list of one decode step ---- */
enum {
  OP_EMBED,   /* a = x <- row tok of token_embd */
  OP_RMSNORM, /* b = rmsnorm(a) * w */
  OP_QUANT,   /* b = Q8(a) */
  OP_GEMV_Q4, /* b = W a (a is a Q8 buffer) */
  OP_GEMV_Q4K, /* the same for a MAT_Q4K matrix */
  OP_GEMV_Q6K, /* the same for a MAT_Q6K matrix */
  OP_GEMV_F32,/* b = W a (a is f32) */
  OP_GEMV_Q8, /* b = W a, W is Q8_0 (a is a Q8 buffer) */
  OP_BIAS,    /* a += w */
  OP_ADD,     /* c = a + b */
  OP_ROPE,    /* rotate a in place: nh heads of hd */
  OP_KV,      /* store a (k) and b (v) at the step position of layer */
  OP_ROPE_KV, /* rotate a (q, nh heads) and b (k, n_kv heads); store b and c (v) in the KV cache of layer */
  OP_ATTN,    /* b = attention(q = a, the KV cache of layer), scores in c */
  OP_SWIGLU,  /* a = silu(a) * b */
  OP_ARGMAX,  /* b (u32) = argmax(a) */
  /* qwen35 full attention: a = [q|gate per head (2 hd) x nh | k | v]; b = q
   * (nh x hd): per head RMSNorm (v = q_norm, nv = k_norm), RoPE of the
   * first n_rot dims (NEOX pairs), k and v into the KV cache of layer. */
  OP_QKN_ROPE_KV,
  /* qwen35 linear attention (Gated DeltaNet) of one token: a = [q k v
   * (conv_dim) | z (inner) | beta (n_vh) | alpha (n_vh)], b = the output
   * (inner, after the gated RMSNorm), Q8 copy in qo. State of layer. */
  OP_GDN,
  /* b (at boff) = a: n floats per token */
  OP_COPY,
  /* DFlash: b = conv(a) along the tokens of the run: b[t] = sum_{o<2}
   * (base[o] + dyn[t][o]) * a[t-o] per channel (a[-1] = 0); v = base
   * [2][2][dim], c = dyn ([2][2][groups] per token), nh = which half (0 or
   * 1); res >= 0: b = res + conv(a) */
  OP_DF_CONV,
  /* DFlash: a = [q (n_head hd) | k | v] per token (q absent: nh == 0):
   * per head RMSNorm (v: q_norm, nv: k_norm), RoPE (NEOX, all hd dims),
   * k and v into the drafter KV cache of layer at pos + t */
  OP_DF_QKV,
  /* DFlash: b = attention(q = a) over the drafter KV cache of layer: query
   * at pos + t sees the positions within the window up to pos + T - 1 */
  OP_DF_ATTN,
  OP_GEMV_Q5K,   /* b = W a, MAT_Q5K (a is a Q8 buffer) */
  OP_GEMV_IQ4XS, /* b = W a, MAT_IQ4XS */
  /* flint MoE. ROUTE: b = [nh expert ids (u32) | nh weights (f32)] from the
   * router logits a (n experts): sigmoid, top nh by sigmoid + bias (v, or
   * none), weights = sigmoid / sum * m->escale. */
  OP_MOE_ROUTE,
  /* b + s bstr = W_e (a + s astr) for slot s < nh, e = expert id s of the
   * route buffer c (astr 0: one input for all slots); w->kind: Q8, Q4K, Q5K,
   * Q6K, IQ4XS (a is Q8) or F32 */
  OP_GEMV_EXP,
  /* per slot s < nh: g = a + s astr (floats [gate n | up n]): g = silu(g) * up;
   * the Q8 copy of slot s at qo + s q8_bytes(n / 32) */
  OP_MOE_SWIGLU,
  /* b = res + sum over slots s of weight_s (a + s n floats) (c: the route) */
  OP_MOE_SUM,
};

typedef struct {
  int kind;
  int a, b, c;     /* buffer ids (-1: none) */
  uint32_t n;      /* elements (of a) */
  uint32_t nh;     /* rope: heads */
  int layer;
  const mat *w;
  const vec *v;
  /* Byte offsets into the buffers a, b, c (a buffer can hold several
   * vectors, for example q, k and v). */
  uint32_t aoff, boff, coff;
  /* GEMV only: after the result (with bias and residual) is complete,
   * nout = rmsnorm(result) * nv and, if nq >= 0, its Q8 copy in nq. The GPU
   * does this in the last workgroup to finish. */
  const vec *nv;
  int nout, nq;
  /* Fused work (-1: none):
   *  qo:  also write the Q8 copy of this op's f32 result (RMSNORM: b,
   *       ATTN: b, SWIGLU: a), as OP_QUANT would.
   *  res: GEMV only: add this f32 buffer to the result (residual). The GEMV
   *       also adds v (the bias) when v is set. */
  int qo, res;
  /* ATTN only: if gt >= 0, out *= sigmoid(gate), gate of head h at float
   * gtoff / 4 + h * gstride of buffer gt (qwen35). */
  int gt;
  uint32_t gtoff, gstride;
  /* QKN_ROPE_KV: qstr = floats from one q head to the next (qwen35: 2 hd,
   * flint: hd), norope: no RoPE. ATTN: win (0: all positions), sk = the
   * sinks (NULL: none). GEMV_EXP, MOE_SWIGLU: astr, bstr = byte strides of
   * the slots. */
  uint32_t qstr, win, astr, bstr;
  int norope;
  const vec *sk;
} op;

/* Activation buffers. Each has a size in bytes and, after planning, an
 * arena offset and a lifetime [first, last] in op indices. */
#define MAX_BUFS 4096
/* GPU attention (ie_attn_split): positions per workgroup at decode position
 * pos (positions 0 .. pos). Measured on the R9700: chunks of 8 or 16 for
 * short contexts were not faster at 64 tokens and slower at 512 (the last
 * workgroup merges more partial results), so the chunk is 32. */
static inline uint32_t ie_att_ch(uint32_t pos) { (void)pos; return 32u; }
static inline uint32_t ie_att_nsplit(uint32_t pos) { return (pos + ie_att_ch(pos)) / ie_att_ch(pos); }
/* The most splits for any position below n_ctx. The scores buffer of OP_ATTN
 * also holds the partial results: n_head * ie_att_max_split(n_ctx) * (hd + 2)
 * floats. ie_attn_split needs hd <= IE_ATT_MAX_HD. */
static inline uint32_t ie_att_max_split(uint32_t n_ctx) {
  uint32_t mx = 1;
  for (uint32_t p = 0; p < n_ctx; p++)
    if (ie_att_nsplit(p) > mx) mx = ie_att_nsplit(p);
  return mx;
}
#define IE_ATT_MAX_HD 256u
/* ie_attn_split keeps the weights of up to this many splits in LDS. */
#define IE_ATT_MAX_SPLIT 4096u
typedef struct {
  const char *name;
  /* stride: bytes per token (256-aligned); size = stride * graph.T; or
   * stride 0: one part that all tokens share (new_buf_shared) */
  uint32_t stride, size, off, first, last;
} buf;

typedef struct {
  op *ops;
  uint32_t n_ops;
  buf *bufs;
  uint32_t n_bufs;
  uint32_t arena; /* bytes */
  int x, logits, argmax; /* buffer ids that the host reads */
  uint32_t n_ctx;
  /* T: the most tokens one run of the op list does (speculative decoding:
   * 1 + drafts). Each buffer holds T tokens (buf.stride apart). */
  uint32_t T;
  /* The ops [0, n_main) are the model; [n_main, n_ops) the MTP head (if
   * built). h_out: the final normed hidden state of the model (the MTP h
   * input of the next position); mtp_h: the MTP h input (the host copies
   * rows into it); mtp_g: the MTP normed output (the h of the next draft). */
  uint32_t n_main;
  /* first op of the output head (output matrix, argmax) of the model and of
   * the MTP head: a prompt chunk runs the model head for its last token
   * only, and the MTP head not at all */
  uint32_t i_head, i_mtp_head;
  int h_out, mtp_h, mtp_g, mtp_logits, mtp_argmax;
  /* the MTP ops end at n_mtp; DFlash (m->df set): the context ops [i_dfc,
   * i_dfb) (fc of df_feat rows, their keys and values into the drafter KV
   * cache), the block ops [i_dfb, n_ops) (df_logits: the draft head's
   * logits, df_sel: the selector's hidden projection). df_feat: the outputs
   * of the target layers df->tl, written by the model ops. */
  uint32_t n_mtp, i_dfc, i_dfb;
  int df_feat, df_logits, df_sel;
  float *rope_cos, *rope_sin; /* [n_ctx][n_rot/2] */
  /* The KV cache of attention layer i (kvi): kv_len[i] positions (position
   * p in slot p % kv_len[i]; n_ctx, or a ring for a sliding window layer)
   * from entry kv_off[i] (in units of n_kv * hd values); kv_total entries */
  uint32_t *kv_len;
  uint64_t *kv_off, kv_total;
  uint64_t weight_bytes_per_token;
} graph;

/* T: tokens per run (1, or 1 + drafts); mtp: also build the MTP ops (and
 * the DFlash ops when m->df is set). */
void graph_build(graph *gr, const model *m, uint32_t n_ctx, uint32_t T, int mtp);
void graph_free(graph *gr);
/* The ring of a sliding window layer: the smallest power of two >= win +
 * max(IE_KV_SLACK, graph T) (room for the tokens of one run), at most n_ctx. */
#define IE_KV_SLACK 16u
/* u32 words of qw and u16 entries of qs of one rows x cols matrix (one expert). */
uint64_t mat_qw_words(int kind, uint32_t rows, uint32_t cols);
uint64_t mat_qs_n(int kind, uint32_t rows, uint32_t cols);
/* Expert e of w as a matrix of its own (the arrays point into w's). */
mat mat_expert(const mat *w, uint32_t e);

/* plan.c: give each buffer an arena offset; checks with ie_plan_check and
 * exits if the check fails. */
void plan_arena(graph *gr, int verbose);

#endif
