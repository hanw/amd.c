/* model.h -- Llama / Qwen2 models from GGUF, and the op list of one decode
 * step. Both backends (cpu.c, hip.c) run the same op list on the same
 * arena offsets. */
#ifndef IE_MODEL_H
#define IE_MODEL_H

#include <stdint.h>

#include "gguf.h"

enum { ARCH_LLAMA = 0, ARCH_QWEN2 = 1 };
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
} layer;

typedef struct {
  int arch, rope;
  uint32_t n_layer, dim, ffn, n_head, n_kv, hd, vocab, ctx_train;
  float eps, rope_base;
  mat tok;   /* token_embd (Q4 repacked or F32), rows = vocab, cols = dim */
  mat out;   /* output; may share the arrays of tok (tied) */
  int tied;
  vec out_norm;
  layer *l;
  /* tokenizer */
  int tok_gpt2; /* 1: GPT2 byte-level, 0: SentencePiece */
  uint32_t n_tokens;
  char **tokens;
  uint64_t weight_bytes; /* bytes of all repacked/dequantized weights */
} model;

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
  uint32_t size, off, first, last;
} buf;

typedef struct {
  op *ops;
  uint32_t n_ops;
  buf *bufs;
  uint32_t n_bufs;
  uint32_t arena; /* bytes */
  int x, logits, argmax; /* buffer ids that the host reads */
  uint32_t n_ctx;
  float *rope_cos, *rope_sin; /* [n_ctx][hd/2] */
  uint64_t weight_bytes_per_token;
} graph;

void graph_build(graph *gr, const model *m, uint32_t n_ctx);
void graph_free(graph *gr);

/* plan.c: give each buffer an arena offset; checks with ie_plan_check and
 * exits if the check fails. */
void plan_arena(graph *gr, int verbose);

#endif
