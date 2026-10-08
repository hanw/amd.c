/* gen.c -- generation on a backend (see gen.h). The sampling, prefill and
 * spec_generate code came from main.c unchanged, except the emit callback. */
#include <math.h>
#include <string.h>
#include <time.h>

#include "common.h"
#include "gen.h"

double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}


static double rng_u01(sampler *sp) { /* xorshift64*, 53 bits */
  sp->rng ^= sp->rng >> 12, sp->rng ^= sp->rng << 25, sp->rng ^= sp->rng >> 27;
  return (double)((sp->rng * 2685821657736338717ull) >> 11) / 9007199254740992.0;
}
static uint32_t sample_argmax(const float *lg, uint32_t V) {
  uint32_t best = 0;
  for (uint32_t i = 1; i < V; i++)
    if (lg[i] > lg[best]) best = i;
  return best;
}
/* The number of candidates: top_k, or (top_p < 1) the 1000 largest, or all. */
uint32_t sample_k(const sampler *sp, uint32_t V) {
  uint32_t K = sp->top_k ? sp->top_k : (sp->top_p < 1.0f ? 1000u : V);
  return K > V ? V : K;
}
/* Sample from n candidates (values v, ids id) in the order value
 * descending, equal values smaller id first: weights exp((v - v[0]) /
 * temp), the nucleus top_p, one draw. v may alias sp->pr. */
uint32_t sample_cands(sampler *sp, const float *v, const uint32_t *id, uint32_t n) {
  const float mx = v[0];
  double z = 0;
  for (uint32_t i = 0; i < n; i++) z += (sp->pr[i] = expf((v[i] - mx) / sp->temp));
  if (sp->top_p < 1.0f && n > 1) { /* the weights are already descending: keep the nucleus */
    double c = 0;
    uint32_t m = 0;
    while (m < n && c < sp->top_p * z) c += sp->pr[m++];
    n = m, z = c;
  }
  double u = rng_u01(sp) * z;
  for (uint32_t i = 0; i < n; i++) {
    u -= sp->pr[i];
    if (u <= 0) return id[i];
  }
  return id[n - 1];
}
/* The candidates of rows r0 .. r0 + n - 1 of the logits on the GPU (top K):
 * 1 if done (cv, ci: n * K entries), 0 if the backend cannot (then the
 * host reads the rows and selects). */
int gpu_cands(const sampler *sp, backend *b, const graph *g, uint32_t r0, uint32_t n, uint32_t V, float *cv, uint32_t *ci) {
  const uint32_t K = sample_k(sp, V);
  if (!b->topk || K >= V || getenv("IE_HOST_TOPK")) return 0;
  return b->topk(b, g->logits, r0, n, V, K, cv, ci);
}
uint32_t sample(sampler *sp, const float *lg, uint32_t V) {
  if (sp->temp <= 0.0f) return sample_argmax(lg, V);
  const uint32_t K = sample_k(sp, V);
  uint32_t n = 0;
  if (K == V) {
    for (uint32_t i = 0; i < V; i++) sp->idx[n++] = i;
  } else { /* the K largest: a min-heap of K ids */
    for (uint32_t i = 0; i < V; i++) {
      if (n < K) {
        uint32_t c = n++;
        sp->idx[c] = i;
        while (c && lg[sp->idx[(c - 1) / 2]] > lg[sp->idx[c]]) {
          uint32_t pp = (c - 1) / 2, t = sp->idx[pp];
          sp->idx[pp] = sp->idx[c], sp->idx[c] = t, c = pp;
        }
      } else if (lg[i] > lg[sp->idx[0]]) {
        sp->idx[0] = i;
        uint32_t c = 0;
        for (;;) {
          uint32_t l = 2 * c + 1, r = l + 1, m = c;
          if (l < n && lg[sp->idx[l]] < lg[sp->idx[m]]) m = l;
          if (r < n && lg[sp->idx[r]] < lg[sp->idx[m]]) m = r;
          if (m == c) break;
          uint32_t t = sp->idx[m];
          sp->idx[m] = sp->idx[c], sp->idx[c] = t, c = m;
        }
      }
    }
  }
  if (K < V) { /* the candidates in the order of the GPU top K: value descending, equal values smaller index first */
    for (uint32_t i = 0; i < n; i++) sp->pr[i] = lg[sp->idx[i]];
    for (uint32_t i = 1; i < n; i++) { /* insertion sort: n <= K */
      const float v = sp->pr[i];
      const uint32_t id = sp->idx[i];
      uint32_t j = i;
      while (j && (sp->pr[j - 1] < v || (sp->pr[j - 1] == v && sp->idx[j - 1] > id))) sp->pr[j] = sp->pr[j - 1], sp->idx[j] = sp->idx[j - 1], j--;
      sp->pr[j] = v, sp->idx[j] = id;
    }
    return sample_cands(sp, sp->pr, sp->idx, n);
  }
  float mx = lg[sp->idx[0]];
  for (uint32_t i = 1; i < n; i++) mx = lg[sp->idx[i]] > mx ? lg[sp->idx[i]] : mx;
  double z = 0;
  for (uint32_t i = 0; i < n; i++) z += (sp->pr[i] = expf((lg[sp->idx[i]] - mx) / sp->temp));
  if (sp->top_p < 1.0f && n > 1) { /* sort by weight, keep the nucleus */
    for (uint32_t i = 1; i < n; i++) { /* insertion sort: n <= top_k (small) */
      float w = sp->pr[i];
      uint32_t id = sp->idx[i], j = i;
      while (j && sp->pr[j - 1] < w) sp->pr[j] = sp->pr[j - 1], sp->idx[j] = sp->idx[j - 1], j--;
      sp->pr[j] = w, sp->idx[j] = id;
    }
    double c = 0;
    uint32_t m = 0;
    while (m < n && c < sp->top_p * z) c += sp->pr[m++];
    n = m, z = c;
  }
  double u = rng_u01(sp) * z;
  for (uint32_t i = 0; i < n; i++) {
    u -= sp->pr[i];
    if (u <= 0) return sp->idx[i];
  }
  return sp->idx[n - 1];
}


static void ppl_add(ppl_acc *pa, const float *lg, uint32_t vocab, uint32_t next) {
  double mx = lg[0], z = 0;
  for (uint32_t i = 1; i < vocab; i++) mx = lg[i] > mx ? lg[i] : mx;
  for (uint32_t i = 0; i < vocab; i++) z += exp((double)lg[i] - mx);
  pa->nll += mx + log(z) - lg[next];
  pa->n++;
}

/* The prompt in chunks of up to `chunk` tokens (GPU): one model run per
 * chunk reads the weights once for all its tokens. With mtp, the MTP head
 * also fills its KV cache for every prompt position q (the pair (h(q-1),
 * token q), h(-1) = 0, as llama.cpp). pa: perplexity of the prompt (needs
 * every row of logits). Returns the greedy token after the prompt; *hrow:
 * the row of g->h_out that holds h(n-1). */
uint32_t prefill(backend *b, const graph *g, uint32_t vocab, const uint32_t *toks, uint32_t n, uint32_t chunk,
                        int mtp, ppl_acc *pa, uint32_t *hrow) {
  uint32_t *a = malloc(chunk * 4), *d = malloc(chunk * 4), last = 0, T = 1;
  float *lg = pa && pa->on ? malloc((size_t)chunk * g->bufs[g->logits].stride) : NULL;
  b->all_logits = lg != NULL;
  if (mtp) b->copy_rows(b, g->mtp_h, 0, -1, 0, 1); /* h(-1) = 0 */
  for (uint32_t p0 = 0; p0 < n; p0 += T) {
    T = n - p0 < chunk ? n - p0 : chunk;
    b->run(b, 0, toks + p0, T, p0, 1, a, NULL);
    b->accept(b, T - 1);
    last = a[T - 1];
    if (lg) {
      b->read_rows(b, g->logits, 0, T, lg);
      const uint32_t stride = g->bufs[g->logits].stride / 4u;
      for (uint32_t t = 0; t < T; t++)
        if (p0 + t >= pa->first && p0 + t + 1 < n) ppl_add(pa, lg + (size_t)t * stride, vocab, toks[p0 + t + 1]);
    }
    if (mtp) { /* MTP positions p0 .. p0+T-1: h rows [h(p0-1) (row 0, from before), h(p0) .. h(p0+T-2)] */
      if (T > 1) b->copy_rows(b, g->mtp_h, 1, g->h_out, 0, T - 1);
      b->run(b, 1, toks + p0, T, p0, 1, d, NULL); /* KV catch-up only: its drafts are not used */
      b->copy_rows(b, g->mtp_h, 0, g->h_out, T - 1, 1);
    }
  }
  free(lg), free(a), free(d);
  *hrow = T - 1;
  return last;
}

/* Greedy decoding with the MTP head drafting D tokens per step (GPU).
 * Positions < P are done in the model and in the MTP KV cache (MTP position
 * q holds the pair (h(q-1), token q), h = the model's normed final hidden
 * state). A step: (1) the MTP head catches up the tokens accepted in the last
 * step with their true h and drafts d1 from (h(P-1), t(P)); (2) for D >= 2 it
 * drafts d2 from (its own h, d1), and so on; (3) the model runs [t(P), d1, .., dD] at
 * P .. P + D in one pass (the weights read once) and gives the greedy token
 * a_j after each; (4) d_j is accepted while d_j == a_(j-1). So the output is
 * exactly the plain greedy output. The linear attention state after the
 * last accepted token is in slot (slot + k) % 4 (ie_gdn). out[0 .. n_prompt)
 * is the prompt; returns the number of tokens in out. */
uint32_t spec_generate(backend *b, const model *m, const graph *g, uint32_t *out, uint32_t n_prompt, uint32_t total, uint32_t D,
                       int64_t eos, int stop, uint32_t first, uint32_t hrow, sampler *sp, uint32_t V, double *gen_ms,
                       uint32_t *n_steps, uint32_t *n_acc, float pmin, uint32_t *n_drafted, gen_emit emit, void *ectx) {
  /* With sampling (temp > 0): the model samples its token a_j after each
   * of t(P), d1, .., dD from its own logits, and d_(j+1) is accepted while
   * it equals a_j ("sample and match"). The drafts are fixed (the MTP
   * argmax), so each emitted token is a sample of the model's distribution
   * given the tokens before it: the output distribution is the model's. */
  float *lg = sp->temp > 0.0f ? malloc((size_t)(D + 1) * g->bufs[g->logits].stride) : NULL;
  double sample_ms = 0; /* host time of the sampling (with the reads of the logits or of the top K) */
  uint32_t a[16], d[16], tk[16];
  /* the prompt is done (prefill): the model and the MTP KV cache for
   * positions < P; first: the greedy token at P; h(P-1) in h_out row hrow */
  uint32_t P = n_prompt, n = n_prompt, k = 0, tP = first;
  out[n++] = tP;
  /* out[n_prompt .. sent) went to emit; a nonzero return ends the loop */
  uint32_t sent = n_prompt;
  int ended = 0;
#define SPEC_EMIT() (emit && !ended && n > sent ? (emit(ectx, out + sent, n - sent) ? (ended = 1) : (sent = n, 0)) : 0)
  const double t0 = now_ms();
  *n_steps = 0, *n_acc = 0, *n_drafted = 0;
  while (n < total && !(stop && (int64_t)out[n - 1] == eos)) {
    if (SPEC_EMIT()) break;
    /* (1) catch up positions P-k .. P-1 and draft at P: tokens out[P-k .. P].
     * A draft is kept only while the MTP head's probability of it is at
     * least pmin: nd drafts (0 .. D) this step. */
    float pr[16];
    uint32_t nd = 0;
    b->copy_rows(b, g->mtp_h, 0, g->h_out, hrow, k + 1);
    hrow = 0;
    b->run(b, 1, &out[P - k], k + 1, P - k, 0, tk, pr);
    if (pr[k] >= pmin) d[1] = model_draft_tok(m, tk[k]), nd = 1;
    for (uint32_t j = 2; j <= D && nd == j - 1; j++) { /* (2) */
      b->copy_rows(b, g->mtp_h, 0, g->mtp_g, j == 2 ? k : 0, 1);
      b->run(b, 1, &d[j - 1], 1, P + j - 1, 0, tk, pr);
      if (pr[0] >= pmin) d[j] = model_draft_tok(m, tk[0]), nd = j;
    }
    /* (3) verify */
    tk[0] = tP;
    for (uint32_t j = 1; j <= nd; j++) tk[j] = d[j];
    b->run(b, 0, tk, nd + 1, P, 0, a, NULL);
    const double ts0 = now_ms();
    if (lg) { /* sample instead of the argmax; only the tokens up to the first mismatch matter */
      const uint32_t K = sample_k(sp, V);
      if (gpu_cands(sp, b, g, 0, nd + 1, V, sp->cv, sp->ci)) {
        for (uint32_t j = 0; j <= nd; j++) {
          a[j] = sample_cands(sp, sp->cv + (size_t)j * K, sp->ci + (size_t)j * K, K);
          if (j < nd && a[j] != d[j + 1]) break;
        }
      } else {
        b->read_rows(b, g->logits, 0, nd + 1, lg);
        const uint32_t st = g->bufs[g->logits].stride / 4u;
        for (uint32_t j = 0; j <= nd; j++) {
          a[j] = sample(sp, lg + (size_t)j * st, V);
          if (j < nd && a[j] != d[j + 1]) break;
        }
      }
    }
    sample_ms += now_ms() - ts0;
    /* (4) accept */
    k = 0;
    while (k < nd && d[k + 1] == a[k]) k++;
    if (getenv("IE_SPEC_TRACE")) {
      fprintf(stderr, "step P=%u t=%u drafts", P, tP);
      for (uint32_t j = 1; j <= nd; j++) fprintf(stderr, " %u", d[j]);
      fprintf(stderr, " model");
      for (uint32_t j = 0; j <= nd; j++) fprintf(stderr, " %u", a[j]);
      fprintf(stderr, " -> k=%u\n", k);
    }
    *n_drafted += nd;
    for (uint32_t j = 1; j <= k && n < total; j++) out[n++] = d[j];
    if (n < total) out[n++] = a[k];
    b->accept(b, k); /* the linear attention state after token k */
    P += k + 1;
    tP = a[k];
    (*n_steps)++, *n_acc += k;
    if (stop)
      for (uint32_t j = n_prompt; j < n; j++)
        if ((int64_t)out[j] == eos) { n = j + 1; break; }
  }
  SPEC_EMIT(); /* the last step's tokens (the result does not matter now) */
#undef SPEC_EMIT
  *gen_ms = now_ms() - t0;
  if (lg) fprintf(stderr, "sampling: %.1f ms in all (%.3f ms per step)\n", sample_ms, *n_steps ? sample_ms / *n_steps : 0.0);
  free(lg);
  return n;
}
/* ---- one request (ie-serve) ---- */

void gen_init(gen_ctx *c, backend *b, const model *m, const graph *g) {
  memset(c, 0, sizeof *c);
  c->b = b, c->m = m, c->g = g;
  c->sp.idx = malloc((size_t)m->vocab * 4);
  c->sp.pr = malloc((size_t)m->vocab * 4);
  c->sp.cv = malloc(16 * 64 * 4);
  c->sp.ci = malloc(16 * 64 * 4);
  /* read_rows writes a full row (buf.stride bytes, 256-aligned), step writes vocab floats */
  size_t n = g->bufs[g->logits].stride;
  if (n < (size_t)m->vocab * 4) n = (size_t)m->vocab * 4;
  c->logits = malloc(n);
  if (!c->sp.idx || !c->sp.pr || !c->sp.cv || !c->sp.ci || !c->logits) ie_die("gen_init: out of memory");
}

void gen_free(gen_ctx *c) {
  free(c->sp.idx), free(c->sp.pr), free(c->sp.cv), free(c->sp.ci), free(c->logits);
  memset(c, 0, sizeof *c);
}

void gen_run(gen_ctx *c, const uint32_t *prompt, uint32_t n_prompt, const gen_params *p, gen_stats *st) {
  backend *b = c->b;
  const graph *g = c->g;
  sampler *sp = &c->sp;
  const uint32_t V = c->m->vocab;
  sp->temp = p->temp, sp->top_p = p->top_p, sp->top_k = p->top_k;
  sp->rng = p->seed * 0x9E3779B97F4A7C15ull + 1; /* as ie-run --seed */
  memset(st, 0, sizeof *st);
  st->n_prompt = n_prompt;

  /* the prompt: positions 0 .. n_prompt - 1 (position 0 starts a new
   * sequence: the linear attention state and the conv ring read as zero) */
  uint32_t first, hrow = 0;
  const double t0 = now_ms();
  if (b->run) { /* GPU: chunks, as ie-run */
    first = prefill(b, g, V, prompt, n_prompt, p->chunk, p->draft > 0, NULL, &hrow);
    if (sp->temp > 0.0f) {
      if (gpu_cands(sp, b, g, hrow, 1, V, sp->cv, sp->ci)) first = sample_cands(sp, sp->cv, sp->ci, sample_k(sp, V));
      else {
        b->read_rows(b, g->logits, hrow, 1, c->logits);
        first = sample(sp, c->logits, V);
      }
    }
  } else { /* CPU: one token per step */
    double ms;
    for (uint32_t pos = 0; pos + 1 < n_prompt; pos++) b->step(b, prompt[pos], pos, NULL, &ms);
    first = b->step(b, prompt[n_prompt - 1], n_prompt - 1, sp->temp > 0.0f ? c->logits : NULL, &ms);
    if (sp->temp > 0.0f) first = sample(sp, c->logits, V);
  }
  st->prefill_ms = now_ms() - t0;

  if (p->draft) { /* MTP: the same loop as ie-run --draft */
    const uint32_t total = n_prompt + p->max_new;
    uint32_t *out = malloc((size_t)total * 4);
    if (!out) ie_die("gen_run: out of memory");
    memcpy(out, prompt, (size_t)n_prompt * 4);
    const uint32_t n = spec_generate(b, c->m, g, out, n_prompt, total, p->draft, -1, 0, first, hrow, sp, V, &st->gen_ms,
                                     &st->steps, &st->acc, p->pmin, &st->drafted, p->emit, p->ctx);
    st->n_out = n - n_prompt;
    free(out);
    return;
  }

  /* one token per step, as the ie-run loop */
  const double t1 = now_ms();
  uint32_t tok = first, pos = n_prompt;
  st->n_out = 1;
  if (!p->emit(p->ctx, &tok, 1)) {
    for (; st->n_out < p->max_new; st->n_out++, pos++) {
      const int gk = sp->temp > 0.0f && b->topk && sample_k(sp, V) <= 64u && !getenv("IE_HOST_TOPK");
      double ms;
      uint32_t next = b->step(b, tok, pos, (sp->temp > 0.0f && !gk) ? c->logits : NULL, &ms);
      if (sp->temp > 0.0f) {
        if (gk && gpu_cands(sp, b, g, 0, 1, V, sp->cv, sp->ci)) next = sample_cands(sp, sp->cv, sp->ci, sample_k(sp, V));
        else {
          if (gk) b->read_rows(b, g->logits, 0, 1, c->logits);
          next = sample(sp, c->logits, V);
        }
      }
      tok = next;
      if (p->emit(p->ctx, &tok, 1)) { st->n_out++; break; }
    }
  }
  st->gen_ms = now_ms() - t1;
}
