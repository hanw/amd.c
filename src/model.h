/* model.h -- Llama / Qwen2 models from GGUF, and the op list of one decode
 * step. Both backends (cpu.c, hip.c) run the same op list on the same
 * arena offsets. */
#ifndef IE_MODEL_H
#define IE_MODEL_H

#include <stdint.h>

#include "gguf.h"

enum { ARCH_LLAMA = 0, ARCH_QWEN2 = 1, ARCH_QWEN35 = 2 };
enum { ROPE_NORM = 0, ROPE_NEOX = 1 }; /* NORM: pairs (2i, 2i+1); NEOX: (i, i + hd/2) */

/* A weight matrix: rows x cols, y = W x. */
enum { MAT_Q4 = 0, MAT_F32 = 1, MAT_Q8 = 2 };
typedef struct {
  int kind;
  uint32_t rows, cols, nb; /* nb = cols / 32 (Q4 and Q8) */
  /* MAT_Q4: the verified GPU layout (core/ie_core.h): rows*nb*4 nibble words
   * and rows*nb f16 scales. */
  uint32_t *qw;
  uint16_t *qs;
  /* MAT_Q8: the verified Q8_0 GPU layout (ie_q8_dst_word, ie_q8_dst_scale):
   * rows*nb*8 int8 words in qw and rows*nb f16 scales in qs. */
  /* MAT_F32: rows*cols floats (dequantized from F32/F16, or Q8_0 for the embedding). */
  float *f;
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
} layer;

typedef struct {
  int arch, rope;
  uint32_t n_layer, dim, ffn, n_head, n_kv, hd, vocab, ctx_train;
  float eps, rope_base;
  uint32_t n_rot;        /* rotated dims per head (hd, or fewer: qwen35 partial RoPE) */
  uint32_t n_kvl, n_rec; /* layers with a KV cache / with a linear attention state */
  /* qwen35 linear attention: key heads, value heads, head dim (keys and
   * values), conv window; conv_dim = 2 * n_kh * sd + n_vh * sd. */
  uint32_t n_kh, n_vh, sd, d_conv, conv_dim;
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
  vec enorm, hnorm, head_norm;
  /* tokenizer */
  int tok_gpt2; /* 1: GPT2 byte-level, 0: SentencePiece */
  uint32_t n_tokens;
  char **tokens;
  uint64_t weight_bytes; /* bytes of all repacked/dequantized weights */
} model;

/* Load the MTP head (blk.N.nextn.*) from g (the model file or a separate
 * file); returns 0 if g has none. */
int model_load_mtp(model *m, gguf_file *g);
/* Make m->out_draft from the output matrix of g (the model file). */
void model_make_draft_head(model *m, gguf_file *g);
/* If set, model_load gives every finished matrix to this function (for
 * example: copy it to the GPU and free the host arrays), so that the host
 * never holds all the weights at once. */
extern void (*ie_mat_sink)(mat *w);
void model_load(model *m, gguf_file *g);
void model_free(model *m);
/* Repack GGUF Q4_0 bytes (w->rows, w->cols, w->nb set) into w->qw, w->qs. */
void repack_q4(mat *w, const uint8_t *src);
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
#define IE_ATT_MAX_SPLIT 2048u
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
  float *rope_cos, *rope_sin; /* [n_ctx][n_rot/2] */
  uint64_t weight_bytes_per_token;
} graph;

/* T: tokens per run (1, or 1 + drafts); mtp: also build the MTP ops. */
void graph_build(graph *gr, const model *m, uint32_t n_ctx, uint32_t T, int mtp);
void graph_free(graph *gr);

/* plan.c: give each buffer an arena offset; checks with ie_plan_check and
 * exits if the check fails. */
void plan_arena(graph *gr, int verbose);

#endif
