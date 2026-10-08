/* main.c -- ie-run: greedy decode from token ids.
 *   ie-run MODEL.gguf [--backend cpu|gpu] [--tokens 1,2,3] [--n 32]
 *          [--ctx N] [--dump-logits FILE] [--hsaco FILE] [--info]
 * --dump-logits writes one line per step: "pos token l0 l1 ... l(vocab-1)". */
#include <math.h>
#include <string.h>
#include <time.h>

#include "backend.h"
#include "common.h"
#include "gguf.h"
#include "model.h"

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

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
} sampler;
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
static uint32_t sample(sampler *sp, const float *lg, uint32_t V) {
  if (sp->temp <= 0.0f) return sample_argmax(lg, V);
  uint32_t K = sp->top_k ? sp->top_k : (sp->top_p < 1.0f ? 1000u : V);
  if (K > V) K = V;
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

/* Perplexity: the sum of -log p(next prompt token) at positions >= first. */
typedef struct {
  int on;
  uint32_t first, n;
  double nll;
} ppl_acc;
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
static uint32_t prefill(backend *b, const graph *g, uint32_t vocab, const uint32_t *toks, uint32_t n, uint32_t chunk,
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
static uint32_t spec_generate(backend *b, const graph *g, uint32_t *out, uint32_t n_prompt, uint32_t total, uint32_t D,
                              int64_t eos, int stop, uint32_t first, uint32_t hrow, sampler *sp, uint32_t V, double *gen_ms,
                              uint32_t *n_steps, uint32_t *n_acc, float pmin, uint32_t *n_drafted) {
  /* With sampling (temp > 0): the model samples its token a_j after each
   * of t(P), d1, .., dD from its own logits, and d_(j+1) is accepted while
   * it equals a_j ("sample and match"). The drafts are fixed (the MTP
   * argmax), so each emitted token is a sample of the model's distribution
   * given the tokens before it: the output distribution is the model's. */
  float *lg = sp->temp > 0.0f ? malloc((size_t)(D + 1) * g->bufs[g->logits].stride) : NULL;
  uint32_t a[16], d[16], tk[16];
  /* the prompt is done (prefill): the model and the MTP KV cache for
   * positions < P; first: the greedy token at P; h(P-1) in h_out row hrow */
  uint32_t P = n_prompt, n = n_prompt, k = 0, tP = first;
  out[n++] = tP;
  const double t0 = now_ms();
  *n_steps = 0, *n_acc = 0, *n_drafted = 0;
  while (n < total && !(stop && (int64_t)out[n - 1] == eos)) {
    /* (1) catch up positions P-k .. P-1 and draft at P: tokens out[P-k .. P].
     * A draft is kept only while the MTP head's probability of it is at
     * least pmin: nd drafts (0 .. D) this step. */
    float pr[16];
    uint32_t nd = 0;
    b->copy_rows(b, g->mtp_h, 0, g->h_out, hrow, k + 1);
    hrow = 0;
    b->run(b, 1, &out[P - k], k + 1, P - k, 0, tk, pr);
    if (pr[k] >= pmin) d[1] = tk[k], nd = 1;
    for (uint32_t j = 2; j <= D && nd == j - 1; j++) { /* (2) */
      b->copy_rows(b, g->mtp_h, 0, g->mtp_g, j == 2 ? k : 0, 1);
      b->run(b, 1, &d[j - 1], 1, P + j - 1, 0, tk, pr);
      if (pr[0] >= pmin) d[j] = tk[0], nd = j;
    }
    /* (3) verify */
    tk[0] = tP;
    for (uint32_t j = 1; j <= nd; j++) tk[j] = d[j];
    b->run(b, 0, tk, nd + 1, P, 0, a, NULL);
    if (lg) { /* sample instead of the argmax; only the tokens up to the first mismatch matter */
      b->read_rows(b, g->logits, 0, nd + 1, lg);
      const uint32_t st = g->bufs[g->logits].stride / 4u;
      for (uint32_t j = 0; j <= nd; j++) {
        a[j] = sample(sp, lg + (size_t)j * st, V);
        if (j < nd && a[j] != d[j + 1]) break;
      }
    }
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
  *gen_ms = now_ms() - t0;
  free(lg);
  return n;
}

static void usage(void) {
  fprintf(stderr,
          "usage: ie-run MODEL.gguf [--backend cpu|gpu] [--tokens 1,2,3] [--n 32] [--ctx N]\n"
          "              [--dump-logits FILE] [--hsaco build/ie_kernels.hsaco] [--info]\n"
          "              [--tokens-file FILE] [--ppl [--ppl-first N]] [--stop]\n"
          "              [--draft D [--mtp FILE] [--draft-pmin P]] [--chunk N]\n"
          "  --stop: end the generation at the end-of-sequence token\n"
          "  --draft D: speculative decoding with the model's MTP head, up to D = 1 to 7 drafts per step (GPU);\n"
          "             --draft-pmin P: stop drafting when the MTP probability of a draft is below P (default 0.6);\n"
          "             --mtp FILE: the MTP head from FILE (default: the model file)\n"
          "  --chunk N: GPU, the prompt in chunks of up to N tokens (1 to 512, default 256)\n"
          "  --temp T [--top-k K] [--top-p P] [--seed S]: sampling (default: greedy)\n");
  exit(2);
}

int main(int argc, char **argv) {
  const char *mtp_path = NULL;
  uint32_t draft = 0, chunk = 256, top_k = 0;
  float temp = 0.0f, top_p = 1.0f, pmin = 0.6f;
  uint64_t seed = 1;
  const char *path = NULL, *be = "cpu", *toks = "1", *dump = NULL, *hsaco = "build/ie_kernels.hsaco", *tfile = NULL;
  int n_gen = 32, info = 0, ppl = 0, stop = 0;
  uint32_t ppl_first = 0;
  uint32_t n_ctx = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--backend") && i + 1 < argc) be = argv[++i];
    else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) toks = argv[++i];
    else if (!strcmp(argv[i], "--n") && i + 1 < argc) n_gen = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) n_ctx = (uint32_t)atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump = argv[++i];
    else if (!strcmp(argv[i], "--hsaco") && i + 1 < argc) hsaco = argv[++i];
    else if (!strcmp(argv[i], "--info")) info = 1;
    else if (!strcmp(argv[i], "--tokens-file") && i + 1 < argc) tfile = argv[++i];
    else if (!strcmp(argv[i], "--ppl")) ppl = 1;
    else if (!strcmp(argv[i], "--stop")) stop = 1;
    else if (!strcmp(argv[i], "--draft") && i + 1 < argc) draft = (uint32_t)atoi(argv[++i]);
    else if (!strcmp(argv[i], "--mtp") && i + 1 < argc) mtp_path = argv[++i];
    else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) chunk = (uint32_t)atoi(argv[++i]);
    else if (!strcmp(argv[i], "--draft-pmin") && i + 1 < argc) pmin = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = (uint32_t)atoi(argv[++i]);
    else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) top_p = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "--ppl-first") && i + 1 < argc) ppl_first = (uint32_t)atoi(argv[++i]);
    else if (argv[i][0] != '-' && !path) path = argv[i];
    else usage();
  }
  if (!path || n_gen < 0) usage();

  /* --tokens-file: token ids separated by commas or white space */
  char *tbuf = NULL;
  if (tfile) {
    FILE *tf = fopen(tfile, "rb");
    if (!tf) ie_die("cannot read %s", tfile);
    fseek(tf, 0, SEEK_END);
    long tn = ftell(tf);
    fseek(tf, 0, SEEK_SET);
    tbuf = malloc((size_t)tn + 1);
    if (fread(tbuf, 1, (size_t)tn, tf) != (size_t)tn) ie_die("cannot read %s", tfile);
    tbuf[tn] = 0;
    fclose(tf);
    for (char *c = tbuf; *c; c++)
      if (*c == ' ' || *c == '\n' || *c == '\t' || *c == '\r') *c = ',';
    while (tn > 0 && tbuf[tn - 1] == ',') tbuf[--tn] = 0;
    toks = tbuf;
  }
  /* --ppl: the prompt only (no generation); score the next-token
   * probability at each position >= ppl_first */
  if (ppl) n_gen = 0;

  /* prompt token ids */
  uint32_t n_prompt = 0, cap = 16, *prompt = malloc(cap * 4);
  for (const char *p = toks; *p;) {
    char *end;
    unsigned long v = strtoul(p, &end, 10);
    if (end == p) ie_die("bad --tokens list: %s", toks);
    if (n_prompt == cap) prompt = realloc(prompt, (cap *= 2) * 4);
    prompt[n_prompt++] = (uint32_t)v;
    p = *end == ',' ? end + 1 : end;
  }
  if (n_prompt == 0) ie_die("empty prompt");

  gguf_file g;
  gguf_open(&g, path);
  /* GPU: weights go to the device during the load (IE_STREAM=0: after) */
  const char *se = getenv("IE_STREAM");
  if (!strcmp(be, "gpu") && !(se && se[0] == '0')) gpu_stream_init();
  if (info) gguf_print(&g, stderr);
  model m;
  model_load(&m, &g);
  gguf_file g2;
  if (draft) {
    if (draft > 7 || ppl || dump || strcmp(be, "gpu")) ie_die("--draft: 1 to 7, GPU only, not with --ppl or --dump-logits");
    if (mtp_path) gguf_open(&g2, mtp_path);
    if (!model_load_mtp(&m, mtp_path ? &g2 : &g)) ie_die("no MTP head (blk.N.nextn.*) in %s", mtp_path ? mtp_path : path);
  }
  uint32_t need = n_prompt + (uint32_t)n_gen + draft;
  if (n_ctx == 0) n_ctx = need;
  if (n_ctx < need) ie_die("--ctx %u is smaller than prompt + n = %u", n_ctx, need);
  fprintf(stderr, "model: %s, %u layers, dim %u, ffn %u, heads %u/%u (hd %u), vocab %u, rope %s, weights %.2f MB\n",
          m.arch == ARCH_QWEN35 ? "qwen35" : m.arch == ARCH_QWEN2 ? "qwen2" : "llama", m.n_layer, m.dim, m.ffn, m.n_head, m.n_kv, m.hd, m.vocab,
          m.rope == ROPE_NEOX ? "neox" : "norm", m.weight_bytes / 1e6);

  if (chunk < 1 || chunk > 512) ie_die("--chunk: 1 to 512");
  const int gpu = !strcmp(be, "gpu"), use_pf = gpu && !dump;
  if (!use_pf) chunk = 1;
  graph gr;
  graph_build(&gr, &m, n_ctx, draft + 1 > chunk ? draft + 1 : chunk, draft > 0);
  plan_arena(&gr, 1);

  backend *b = NULL;
  if (!strcmp(be, "cpu")) b = cpu_open(&m, &gr);
  else if (!strcmp(be, "gpu")) b = gpu_open(&m, &gr, hsaco);
  else usage();

  FILE *df = dump ? fopen(dump, "w") : NULL;
  if (dump && !df) ie_die("cannot write %s", dump);
  float *logits = malloc(m.vocab * 4);
  int64_t eos_v = -1;
  if (!gguf_get_int(&g, "tokenizer.ggml.eos_token_id", &eos_v)) eos_v = -1;
  uint32_t total = n_prompt + (uint32_t)n_gen, tok = prompt[0], *out = malloc(total * 4);
  double ms_sum = 0, ms_gen = 0, nll = 0;
  uint32_t n_nll = 0;
  uint32_t n_timed = 0;
  sampler sp = {temp, top_p, top_k, seed * 0x9E3779B97F4A7C15ull + 1, malloc((size_t)m.vocab * 4), malloc((size_t)m.vocab * 4)};
  if (temp > 0.0f && (ppl || dump)) ie_die("--temp: not with --ppl or --dump-logits");
  uint32_t start = 0, hrow = 0;
  if (use_pf) { /* the prompt in chunks (prefill), then one token per step */
    ppl_acc pa = {ppl, ppl_first, 0, 0};
    const double t0 = now_ms();
    uint32_t first = prefill(b, &gr, m.vocab, prompt, n_prompt, chunk, draft > 0, &pa, &hrow);
    if (temp > 0.0f) { /* sample the first token from the last prompt row */
      float *lg = malloc(gr.bufs[gr.logits].stride);
      b->read_rows(b, gr.logits, hrow, 1, lg);
      first = sample(&sp, lg, m.vocab);
      free(lg);
    }
    const double pms = now_ms() - t0;
    fprintf(stderr, "prefill: %u tokens in %.1f ms, %.1f tokens/s (chunks of up to %u)\n", n_prompt, pms,
            1e3 * n_prompt / pms, chunk);
    nll = pa.nll, n_nll = pa.n;
    memcpy(out, prompt, n_prompt * 4);
    start = n_prompt, tok = first;
    if (stop && n_prompt < total && (int64_t)first == eos_v) out[n_prompt] = first, total = n_prompt + 1, start = total;
  }
  if (draft && start < total) {
    double gms = 0;
    uint32_t steps = 0, acc = 0, drafted = 0;
    total = spec_generate(b, &gr, out, n_prompt, total, draft, eos_v, stop, tok, hrow, &sp, m.vocab, &gms, &steps, &acc,
                          pmin, &drafted);
    const uint32_t gen = total - n_prompt - 1; /* the tokens after the first */
    fprintf(stderr, "mtp: %u steps, %u drafts accepted of %u (%.1f%%), %.2f tokens per step\n", steps, acc, drafted,
            drafted ? 100.0 * acc / drafted : 0.0, steps ? (double)gen / steps : 0.0);
    if (gen) fprintf(stderr, "gpu: %u tokens in %.1f ms, %.3f ms/token, %.1f tokens/s (speculative, %u drafts)\n", gen, gms,
                     gms / gen, 1e3 * gen / gms, draft);
  }
  for (uint32_t pos = start; !draft && pos < total; pos++) {
    double ms = 0;
    out[pos] = tok;
    if (pos + 1 == total) break; /* the last token needs no step */
    uint32_t next = b->step(b, tok, pos, (df || ppl || temp > 0.0f) ? logits : NULL, &ms);
    if (temp > 0.0f && pos + 1 >= n_prompt) next = sample(&sp, logits, m.vocab);
    if (ppl && pos >= ppl_first && pos + 1 < n_prompt) { /* -log p(prompt[pos + 1]) */
      double mx = logits[0], z = 0;
      for (uint32_t i = 1; i < m.vocab; i++) mx = logits[i] > mx ? logits[i] : mx;
      for (uint32_t i = 0; i < m.vocab; i++) z += exp((double)logits[i] - mx);
      nll += mx + log(z) - logits[prompt[pos + 1]];
      n_nll++;
    }
    ms_sum += ms;
    if (pos >= 1) ms_gen += ms, n_timed++; /* skip the first (warm-up) step */
    if (df) {
      fprintf(df, "%u %u", pos, tok);
      for (uint32_t i = 0; i < m.vocab; i++) fprintf(df, " %.9g", logits[i]);
      fprintf(df, "\n");
    }
    tok = pos + 1 < n_prompt ? prompt[pos + 1] : next;
    if (stop && pos + 1 >= n_prompt && (int64_t)tok == eos_v) { /* keep the eos token, then end */
      out[pos + 1] = tok;
      total = pos + 2;
      break;
    }
  }
  if (df) fclose(df);

  printf("ids:");
  for (uint32_t i = 0; i < total; i++) printf(" %u", out[i]);
  printf("\ntext: ");
  for (uint32_t i = 0; i < total; i++) {
    uint32_t len;
    char *s = model_detok(&m, out[i], &len);
    fwrite(s, 1, len, stdout);
    free(s);
  }
  printf("\n");
  if (n_timed) {
    double per = ms_gen / n_timed;
    fprintf(stderr, "%s: %u steps, %.3f ms/token, %.1f tokens/s, weights read %.2f MB/token, %.2f GB/s\n", be,
            n_timed, per, 1e3 / per, gr.weight_bytes_per_token / 1e6, gr.weight_bytes_per_token / (per * 1e6));
  }
  if (ppl && n_nll)
    printf("ppl: %u tokens scored, mean nll %.6f, perplexity %.4f\n", n_nll, nll / n_nll, exp(nll / n_nll));
  (void)ms_sum;
  free(tbuf);
  b->close(b);
  free(logits), free(out), free(prompt), free(sp.idx), free(sp.pr);
  graph_free(&gr);
  model_free(&m);
  gguf_close(&g);
  return 0;
}
