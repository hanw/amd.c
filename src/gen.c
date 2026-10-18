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

/* Prompt lookup: the drafts from the text so far. t[0 .. n) are the tokens
 * (t[n-1]: the token at the step's position). Finds the earlier place whose
 * tokens before it match the longest suffix of t (up to LOOKUP_NMAX tokens,
 * the latest place on a tie) and copies up to D tokens after that place to
 * d[1 ..]. Returns the number of drafts; *mlen: the match length (0: none). */
enum { LOOKUP_NMAX = 16 };

/* The most tokens of a verify run: 8 (7 MTP drafts), or 1 + IE_LOOKUP_MAX
 * (default 15, up to 15) with the prompt lookup on; the GPU backend keeps
 * this many linear attention states. */
uint32_t spec_verify_max(void) {
  const char *e = getenv("IE_LOOKUP");
  if (e && e[0] == '0') return 8u;
  e = getenv("IE_LOOKUP_MAX");
  uint32_t n = e ? (uint32_t)atoi(e) : 15u;
  if (n > 15u) n = 15u;
  return n + 1u > 8u ? n + 1u : 8u;
}

/* The positions past the generated tokens that a verify run may use. */
uint32_t spec_room(uint32_t draft) {
  const uint32_t r = spec_verify_max() - 1u;
  return draft ? (r > draft ? r : draft) : 0u;
}
static uint32_t lookup_draft(const uint32_t *t, uint32_t n, uint32_t D, uint32_t *d, uint32_t *mlen) {
  uint32_t best = 0, bi = 0;
  *mlen = 0;
  if (n < 2) return 0;
  const uint32_t last = t[n - 1];
  for (uint32_t i = n - 1; i-- > 0;) {
    if (t[i] != last) continue;
    uint32_t L = 1;
    while (L < LOOKUP_NMAX && L <= i && t[i - L] == t[n - 1 - L]) L++;
    if (L > best) {
      best = L, bi = i;
      if (best == LOOKUP_NMAX) break;
    }
  }
  if (!best) return 0;
  uint32_t c = n - 1 - bi; /* tokens after the place */
  if (c > D) c = D;
  for (uint32_t j = 0; j < c; j++) d[j + 1] = t[bi + 1 + j];
  *mlen = best;
  return c;
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
                       uint32_t *n_steps, uint32_t *n_acc, float pmin, uint32_t *n_drafted, gen_emit emit, void *ectx,
                       const int64_t *stops, uint32_t n_stops, uint32_t *p_done, uint32_t *h_row) {
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
  /* IE_SPEC_TIME=1: the host time of the drafts and of the verify runs (each run ends with a read of its argmax, so it waits for the GPU) */
  const int stime = getenv("IE_SPEC_TIME") != NULL;
  double t_draft = 0, t_verify = 0, tq;
  /* prompt lookup: IE_LOOKUP=0 off; IE_LOOKUP_MIN: the shortest match used
   * (default 2); IE_LOOKUP_LONG: the match that skips the rest of the MTP
   * chain (default 16); IE_LOOKUP_AGREE: the MTP drafts that must agree
   * (default 2); IE_LOOKUP_MAX: lookup drafts per step (default 15) */
  const char *ev = getenv("IE_LOOKUP");
  const int lk_on = !(ev && ev[0] == '0');
  ev = getenv("IE_LOOKUP_MIN");
  const uint32_t lk_min = ev ? (uint32_t)atoi(ev) : 2u;
  ev = getenv("IE_LOOKUP_LONG");
  const uint32_t lk_long = ev ? (uint32_t)atoi(ev) : 16u;
  ev = getenv("IE_LOOKUP_AGREE");
  const uint32_t lk_agree = ev ? (uint32_t)atoi(ev) : 2u;
  uint32_t lk_n = 0;
  uint32_t lk_max = spec_verify_max() - 1u; /* lookup drafts per step: the graph's room */
  if (lk_max > g->T - 1u) lk_max = g->T - 1u;
  if (lk_max < D) lk_max = D;
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
    tq = stime ? now_ms() : 0;
    b->copy_rows(b, g->mtp_h, 0, g->h_out, hrow, k + 1);
    hrow = 0;
    b->run(b, 1, &out[P - k], k + 1, P - k, 0, tk, pr);
    if (pr[k] >= pmin) d[1] = model_draft_tok(m, tk[k]), nd = 1;
    /* Prompt lookup (IE_LOOKUP, default on): an earlier place in the text
     * that ends like the text now (a match of ml >= lk_min tokens) gives up
     * to lk_max drafts: its next tokens. A verify of 16 tokens costs about
     * 1.6 times one of 8, so they are used only where the MTP chain agrees:
     * (a) ml >= lk_long: when the chain's first lk_agree drafts are the
     *     lookup's, the lookup's drafts are used and the rest of the chain is
     *     not run; (b) else, after the chain: when its nd >= lk_agree drafts
     *     are all the lookup's, the lookup's next tokens are added. The verify
     *     step is the same, so the output is the same; only the drafts change. */
    uint32_t ml = 0, ld[16], nl = lk_on ? lookup_draft(out, n, lk_max, ld, &ml) : 0;
    if (nl && (ml < lk_min || nl <= lk_agree)) nl = 0;
    /* the chain so far differs from the lookup: no lookup this step */
#define LK_DIFF() (nd && d[nd] != ld[nd])
    if (nl && LK_DIFF()) nl = 0;
    for (uint32_t j = 2; j <= D && nd == j - 1; j++) { /* (2) */
      if (nl && ml >= lk_long && nd >= lk_agree) { /* (a) */
        for (uint32_t i = nd + 1; i <= nl; i++) d[i] = ld[i];
        nd = nl, lk_n++;
        goto verify;
      }
      b->copy_rows(b, g->mtp_h, 0, g->mtp_g, j == 2 ? k : 0, 1);
      b->run(b, 1, &d[j - 1], 1, P + j - 1, 0, tk, pr);
      if (pr[0] >= pmin) d[j] = model_draft_tok(m, tk[0]), nd = j;
      if (nl && LK_DIFF()) nl = 0;
    }
#undef LK_DIFF
    if (nl > nd && nd >= lk_agree) { /* (b) */
      for (uint32_t i = nd + 1; i <= nl; i++) d[i] = ld[i];
      nd = nl, lk_n++;
    }
  verify: /* (3) verify */
    if (stime) t_draft += now_ms() - tq, tq = now_ms();
    tk[0] = tP;
    for (uint32_t j = 1; j <= nd; j++) tk[j] = d[j];
    b->run(b, 0, tk, nd + 1, P, 0, a, NULL);
    if (stime) t_verify += now_ms() - tq;
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
    while (k < nd && d[k + 1] == a[k]) {
      int st = 0;
      for (uint32_t s = 0; s < n_stops; s++) st |= stops[s] >= 0 && (int64_t)d[k + 1] == stops[s];
      if (st) break; /* the stop token is given out as a[k], not run */
      k++;
    }
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
  if (p_done) *p_done = P, *h_row = hrow ? hrow : k; /* the last verify's row k holds h(P - 1) */
  if (lk_on && !emit && *n_steps) fprintf(stderr, "lookup: %u of %u steps drafted from the text\n", lk_n, *n_steps);
  if (stime && *n_steps)
    fprintf(stderr, "spec time: drafts %.2f ms/step, verify %.2f ms/step (%u steps)\n", t_draft / *n_steps, t_verify / *n_steps, *n_steps);
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
  free(c->sp.idx), free(c->sp.pr), free(c->sp.cv), free(c->sp.ci), free(c->logits), free(c->snap_toks[0]), free(c->snap_toks[1]);
  memset(c, 0, sizeof *c);
}

/* The prompt tokens toks[start .. end) at their positions, in chunks (GPU),
 * as prefill does but from any start: the MTP h input for start (row 0 of
 * mtp_h) is already set. Returns the greedy token after the last one;
 * *hrow: the row of h_out that holds h(end - 1). */
static uint32_t prefill_range(backend *b, const graph *g, const uint32_t *toks, uint32_t start, uint32_t end, uint32_t chunk,
                              int mtp, uint32_t *hrow) {
  uint32_t a[512], d[512], last = 0, T = 1;
  if (chunk > 512) chunk = 512;
  b->all_logits = 0;
  for (uint32_t p0 = start; p0 < end; p0 += T) {
    T = end - p0 < chunk ? end - p0 : chunk;
    b->run(b, 0, toks + p0, T, p0, 1, a, NULL);
    b->accept(b, T - 1);
    last = a[T - 1];
    if (mtp) {
      if (T > 1) b->copy_rows(b, g->mtp_h, 1, g->h_out, 0, T - 1);
      b->run(b, 1, toks + p0, T, p0, 1, d, NULL);
      b->copy_rows(b, g->mtp_h, 0, g->h_out, T - 1, 1);
    }
  }
  *hrow = T - 1;
  return last;
}

/* Snapshot k: save the state, and keep the tokens a[0 .. na) and b[0 .. nb) that it is after. */
static void cache_save(gen_ctx *c, int k, int hbuf, const uint32_t *a, uint32_t na, const uint32_t *b, uint32_t nb) {
  const uint32_t n = na + nb;
  if (n > c->snap_cap[k]) {
    c->snap_cap[k] = n + 1024;
    c->snap_toks[k] = realloc(c->snap_toks[k], (size_t)c->snap_cap[k] * 4);
    if (!c->snap_toks[k]) ie_die("gen: out of memory");
  }
  c->b->snap(c->b, 1, k, hbuf);
  memcpy(c->snap_toks[k], a, (size_t)na * 4);
  if (nb) memcpy(c->snap_toks[k] + na, b, (size_t)nb * 4);
  c->snap_n[k] = n;
}
/* 1 if the snapshot k can start this prompt: its tokens begin the prompt, and at least one more token follows */
static int cache_fits(const gen_ctx *c, int k, const uint32_t *prompt, uint32_t n) {
  return c->snap_n[k] && n > c->snap_n[k] && !memcmp(prompt, c->snap_toks[k], (size_t)c->snap_n[k] * 4);
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
   * sequence: the linear attention state and the conv ring read as zero).
   * The prompt cache: if this prompt starts with the cached snap_n tokens
   * (and has at least one more), restore the snapshot and start at snap_n. */
  const int mtp = p->draft > 0, cache = p->cache && b->snap, hbuf = mtp ? g->mtp_h : -1;
  uint32_t S = 0;
  st->cache_hit = -1;
  if (cache)
    for (int k = 1; k >= 0; k--) /* the longer one first */
      if (cache_fits(c, k, prompt, n_prompt)) { st->cache_hit = k, S = c->snap_n[k]; break; }
  st->cached = S;
  if (cache && st->cache_hit != 1 && c->snap_n[1] && getenv("IE_SERVE_DEBUG")) { /* why the last answer did not fit */
    uint32_t i = 0;
    while (i < c->snap_n[1] && i < n_prompt && prompt[i] == c->snap_toks[1][i]) i++;
    fprintf(stderr, "cache: snapshot 1 (%u tokens) differs from the prompt (%u tokens) at %u:", c->snap_n[1], n_prompt, i);
    for (uint32_t k = i > 3 ? i - 3 : 0; k < i + 5 && k < c->snap_n[1]; k++) fprintf(stderr, " %u", c->snap_toks[1][k]);
    fprintf(stderr, " | prompt:");
    for (uint32_t k = i > 3 ? i - 3 : 0; k < i + 5 && k < n_prompt; k++) fprintf(stderr, " %u", prompt[k]);
    fprintf(stderr, "\n");
  }
  uint32_t first, hrow = 0;
  const double t0 = now_ms();
  if (S) b->snap(b, 0, st->cache_hit, hbuf);
  if (cache) c->snap_n[1] = 0; /* this request changes the positions after the prompt */
  else c->snap_n[0] = c->snap_n[1] = 0; /* and a run without the cache changes all positions */
  if (b->run) { /* GPU: chunks */
    if (!S && mtp) b->copy_rows(b, g->mtp_h, 0, -1, 0, 1); /* h(-1) = 0 */
    if (!cache) first = prefill_range(b, g, prompt, 0, n_prompt, p->chunk, mtp, &hrow); /* as ie-run */
    else { /* all but the last token, the snapshot, then the last token */
      if (n_prompt - 1 > S) prefill_range(b, g, prompt, S, n_prompt - 1, p->chunk, mtp, &hrow);
      if (n_prompt - 1 > S) cache_save(c, 0, hbuf, prompt, n_prompt - 1, NULL, 0);
      else if (st->cache_hit == 1) cache_save(c, 0, hbuf, prompt, n_prompt - 1, NULL, 0); /* snapshot 0 is of another prompt */
      first = prefill_range(b, g, prompt, n_prompt - 1, n_prompt, p->chunk, mtp, &hrow);
    }
    if (sp->temp > 0.0f) {
      if (gpu_cands(sp, b, g, hrow, 1, V, sp->cv, sp->ci)) first = sample_cands(sp, sp->cv, sp->ci, sample_k(sp, V));
      else {
        b->read_rows(b, g->logits, hrow, 1, c->logits);
        first = sample(sp, c->logits, V);
      }
    }
  } else { /* CPU: one token per step */
    double ms;
    for (uint32_t pos = S; pos + 1 < n_prompt; pos++) b->step(b, prompt[pos], pos, NULL, &ms);
    if (cache && (n_prompt - 1 > S || st->cache_hit == 1)) cache_save(c, 0, -1, prompt, n_prompt - 1, NULL, 0);
    first = b->step(b, prompt[n_prompt - 1], n_prompt - 1, sp->temp > 0.0f ? c->logits : NULL, &ms);
    if (sp->temp > 0.0f) first = sample(sp, c->logits, V);
  }
  st->prefill_ms = now_ms() - t0;

  if (p->draft) { /* MTP: the same loop as ie-run --draft */
    const uint32_t total = n_prompt + p->max_new;
    uint32_t *out = malloc((size_t)total * 4);
    if (!out) ie_die("gen_run: out of memory");
    memcpy(out, prompt, (size_t)n_prompt * 4);
    uint32_t done = 0, hr = 0;
    const uint32_t n = spec_generate(b, c->m, g, out, n_prompt, total, p->draft, -1, 0, first, hrow, sp, V, &st->gen_ms,
                                     &st->steps, &st->acc, p->pmin, &st->drafted, p->emit, p->ctx, p->stops, p->n_stops, &done, &hr);
    st->n_out = n - n_prompt;
    if (cache && done > n_prompt) { /* snapshot 1: after the prompt and the answer tokens that were run */
      b->copy_rows(b, g->mtp_h, 0, g->h_out, hr, 1); /* the MTP h input of the next position */
      cache_save(c, 1, hbuf, out, done, NULL, 0);
    }
    free(out);
    return;
  }

  /* one token per step, as the ie-run loop */
  const double t1 = now_ms();
  uint32_t tok = first, pos = n_prompt, *ov = cache ? malloc((size_t)p->max_new * 4) : NULL;
  if (ov) ov[0] = first;
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
      if (ov) ov[st->n_out] = tok;
      if (p->emit(p->ctx, &tok, 1)) { st->n_out++; break; }
    }
  }
  st->gen_ms = now_ms() - t1;
  /* snapshot 1: the last token given out was not run; all before it were */
  if (ov && st->n_out > 1) cache_save(c, 1, hbuf, prompt, n_prompt, ov, st->n_out - 1);
  free(ov);
}
