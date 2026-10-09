/* gen.h -- generation on a backend: sampling, the prompt in chunks
 * (prefill), speculative decoding with the MTP head, and gen_run (one
 * request, tokens given to a callback as they are made). ie-run (main.c)
 * and ie-serve (serve.c) use it. */
#ifndef IE_GEN_H
#define IE_GEN_H

#include "backend.h"
#include "model.h"

/* ---- sampling (temperature, top-k, top-p) ----
 * temp = 0: greedy (argmax). Otherwise p ~ softmax(logits / temp) over the
 * top_k largest logits (top_k = 0: all; but top_p < 1 with top_k = 0 keeps
 * the 1000 largest: the nucleus is cut from those), then the smallest set
 * of the largest whose probability reaches top_p. */
typedef struct {
  float temp, top_p;
  uint32_t top_k;
  uint64_t rng;
  uint32_t *idx; /* scratch: candidate ids */
  float *pr;     /* scratch: their weights */
  float *cv;     /* the GPU top K: values (16 rows x 64) */
  uint32_t *ci;  /* and ids */
} sampler;

/* Perplexity: the sum of -log p(next prompt token) at positions >= first. */
typedef struct {
  int on;
  uint32_t first, n;
  double nll;
} ppl_acc;

/* Called with new output tokens, in order. Returns nonzero to end the
 * generation (a stop token, or the client is gone). */
typedef int (*gen_emit)(void *ctx, const uint32_t *toks, uint32_t n);

double now_ms(void);
/* The number of candidates: top_k, or (top_p < 1) the 1000 largest, or all. */
uint32_t sample_k(const sampler *sp, uint32_t V);
/* Sample from n candidates (values v, ids id), value descending. */
uint32_t sample_cands(sampler *sp, const float *v, const uint32_t *id, uint32_t n);
/* The top K candidates of rows r0 .. r0 + n - 1 of the logits on the GPU:
 * 1 if done, 0 if the backend cannot (then the host selects). */
int gpu_cands(const sampler *sp, backend *b, const graph *g, uint32_t r0, uint32_t n, uint32_t V, float *cv, uint32_t *ci);
/* Sample one token from vocab logits lg (greedy if temp <= 0). */
uint32_t sample(sampler *sp, const float *lg, uint32_t V);
/* The prompt in chunks (GPU). Returns the greedy token after the prompt;
 * *hrow: the row of g->h_out that holds h(n-1). pa may be NULL. */
uint32_t prefill(backend *b, const graph *g, uint32_t vocab, const uint32_t *toks, uint32_t n, uint32_t chunk, int mtp,
                 ppl_acc *pa, uint32_t *hrow);
/* Speculative decoding with the MTP head (GPU). emit (or NULL) gets each
 * output token after the prompt; when it returns nonzero, the generation
 * ends. Returns the number of tokens in out. */
uint32_t spec_generate(backend *b, const model *m, const graph *g, uint32_t *out, uint32_t n_prompt, uint32_t total, uint32_t D,
                       int64_t eos, int stop, uint32_t first, uint32_t hrow, sampler *sp, uint32_t V, double *gen_ms,
                       uint32_t *n_steps, uint32_t *n_acc, float pmin, uint32_t *n_drafted, gen_emit emit, void *ectx,
                       const int64_t *stops, uint32_t n_stops, uint32_t *p_done, uint32_t *h_row);
/* stops (n_stops ids, or NULL): a draft that is a stop token is not accepted,
 * so the model does not run past the stop token (as in plain decoding); the
 * output is the same. p_done (or NULL): the positions done at the end (the
 * state is after out[0 .. *p_done)); h_row: the row of h_out that holds h(*p_done - 1). */

/* ---- one request (ie-serve) ---- */
typedef struct {
  float temp, top_p; /* temp <= 0: greedy */
  uint32_t top_k;
  uint64_t seed;
  uint32_t max_new; /* the most output tokens (>= 1) */
  uint32_t draft;   /* 0: no MTP; 1 .. 7: drafts per step (GPU, the graph built with the MTP head) */
  float pmin;       /* --draft-pmin */
  uint32_t chunk;   /* prompt chunk (GPU) */
  int cache;        /* use and update the prompt cache (needs backend.snap) */
  const int64_t *stops; /* the stop token ids (n_stops; -1 entries are ignored) */
  uint32_t n_stops;
  gen_emit emit;
  void *ctx;
} gen_params;

typedef struct {
  uint32_t n_prompt, n_out; /* n_out: tokens made (also those after a stop) */
  double prefill_ms, gen_ms;
  uint32_t steps, acc, drafted; /* MTP */
  uint32_t cached; /* prompt tokens taken from the prompt cache (not computed again) */
  int cache_hit;   /* the snapshot used: -1 none, 0 the last prompt, 1 the last prompt and answer */
} gen_stats;

typedef struct {
  backend *b;
  const model *m;
  const graph *g;
  sampler sp;   /* scratch arrays allocated by gen_init */
  float *logits; /* vocab floats */
  /* The prompt cache: two snapshots of the last request, after its first
   * snap_n[k] tokens snap_toks[k]: k = 0, the prompt but its last token;
   * k = 1, the prompt and the answer (all tokens that were run). The KV
   * caches still hold those positions: later runs write only higher ones. */
  uint32_t *snap_toks[2], snap_n[2], snap_cap[2];
} gen_ctx;

void gen_init(gen_ctx *c, backend *b, const model *m, const graph *g);
void gen_free(gen_ctx *c);
/* Run the prompt (positions 0 .. n_prompt - 1) and generate up to
 * p->max_new tokens. With p->cache, a prompt that starts with the cached
 * tokens continues from the snapshot (the result is close to, but not
 * bitwise the same as, the full prompt: the chunks differ), and the snapshot
 * is then taken for this prompt. The caller checks that n_prompt + max_new
 * + draft <= the graph's n_ctx. */
void gen_run(gen_ctx *c, const uint32_t *prompt, uint32_t n_prompt, const gen_params *p, gen_stats *st);

#endif
