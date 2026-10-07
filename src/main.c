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
                              int64_t eos, int stop, double *gen_ms, uint32_t *n_steps, uint32_t *n_acc) {
  uint32_t a[5], d[5], tk[5];
  /* the prompt: the model and the MTP head, one token at a time */
  for (uint32_t p = 0; p < n_prompt; p++) {
    if (p == 0) b->copy_rows(b, g->mtp_h, 0, -1, 0, 1); /* h(-1) = 0, as llama.cpp */
    else b->copy_rows(b, g->mtp_h, 0, g->h_out, 0, 1);
    b->run(b, 1, &out[p], 1, p, 0, d);
    b->run(b, 0, &out[p], 1, p, 0, a);
  }
  uint32_t P = n_prompt, n = n_prompt, k = 0, slot = 0, tP = a[0];
  out[n++] = tP;
  const double t0 = now_ms();
  *n_steps = 0, *n_acc = 0;
  while (n < total && !(stop && (int64_t)out[n - 1] == eos)) {
    /* (1) catch up positions P-k .. P-1 and draft at P: tokens out[P-k .. P] */
    b->copy_rows(b, g->mtp_h, 0, g->h_out, 0, k + 1);
    b->run(b, 1, &out[P - k], k + 1, P - k, 0, tk);
    d[1] = tk[k];
    for (uint32_t j = 2; j <= D; j++) { /* (2) */
      b->copy_rows(b, g->mtp_h, 0, g->mtp_g, j == 2 ? k : 0, 1);
      b->run(b, 1, &d[j - 1], 1, P + j - 1, 0, tk);
      d[j] = tk[0];
    }
    /* (3) verify */
    tk[0] = tP;
    for (uint32_t j = 1; j <= D; j++) tk[j] = d[j];
    b->run(b, 0, tk, D + 1, P, slot, a);
    /* (4) accept */
    k = 0;
    while (k < D && d[k + 1] == a[k]) k++;
    if (getenv("IE_SPEC_TRACE")) {
      fprintf(stderr, "step P=%u slot=%u t=%u drafts", P, slot, tP);
      for (uint32_t j = 1; j <= D; j++) fprintf(stderr, " %u", d[j]);
      fprintf(stderr, " model");
      for (uint32_t j = 0; j <= D; j++) fprintf(stderr, " %u", a[j]);
      fprintf(stderr, " -> k=%u\n", k);
    }
    for (uint32_t j = 1; j <= k && n < total; j++) out[n++] = d[j];
    if (n < total) out[n++] = a[k];
    slot = (slot + k) % 4u; /* GDN_SLOTS */
    P += k + 1;
    tP = a[k];
    (*n_steps)++, *n_acc += k;
    if (stop)
      for (uint32_t j = n_prompt; j < n; j++)
        if ((int64_t)out[j] == eos) { n = j + 1; break; }
  }
  *gen_ms = now_ms() - t0;
  return n;
}

static void usage(void) {
  fprintf(stderr,
          "usage: ie-run MODEL.gguf [--backend cpu|gpu] [--tokens 1,2,3] [--n 32] [--ctx N]\n"
          "              [--dump-logits FILE] [--hsaco build/ie_kernels.hsaco] [--info]\n"
          "              [--tokens-file FILE] [--ppl [--ppl-first N]] [--stop]\n"
          "              [--draft D [--mtp FILE]]\n"
          "  --stop: end the generation at the end-of-sequence token\n"
          "  --draft D: speculative decoding with the model's MTP head, D = 1 to 3 drafts per step (GPU);\n"
          "             --mtp FILE: the MTP head from FILE (default: the model file)\n");
  exit(2);
}

int main(int argc, char **argv) {
  const char *mtp_path = NULL;
  uint32_t draft = 0;
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
    if (draft > 3 || ppl || dump || strcmp(be, "gpu")) ie_die("--draft: 1 to 3, GPU only, not with --ppl or --dump-logits");
    if (mtp_path) gguf_open(&g2, mtp_path);
    if (!model_load_mtp(&m, mtp_path ? &g2 : &g)) ie_die("no MTP head (blk.N.nextn.*) in %s", mtp_path ? mtp_path : path);
  }
  uint32_t need = n_prompt + (uint32_t)n_gen + draft;
  if (n_ctx == 0) n_ctx = need;
  if (n_ctx < need) ie_die("--ctx %u is smaller than prompt + n = %u", n_ctx, need);
  fprintf(stderr, "model: %s, %u layers, dim %u, ffn %u, heads %u/%u (hd %u), vocab %u, rope %s, weights %.2f MB\n",
          m.arch == ARCH_QWEN35 ? "qwen35" : m.arch == ARCH_QWEN2 ? "qwen2" : "llama", m.n_layer, m.dim, m.ffn, m.n_head, m.n_kv, m.hd, m.vocab,
          m.rope == ROPE_NEOX ? "neox" : "norm", m.weight_bytes / 1e6);

  graph gr;
  graph_build(&gr, &m, n_ctx, draft + 1, draft > 0);
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
  if (draft) {
    memcpy(out, prompt, n_prompt * 4);
    double gms = 0;
    uint32_t steps = 0, acc = 0;
    total = spec_generate(b, &gr, out, n_prompt, total, draft, eos_v, stop, &gms, &steps, &acc);
    const uint32_t gen = total - n_prompt - 1; /* the tokens after the first */
    fprintf(stderr, "mtp: %u steps, %u drafts accepted of %u (%.1f%%), %.2f tokens per step\n", steps, acc, steps * draft,
            steps ? 100.0 * acc / (steps * draft) : 0.0, steps ? (double)gen / steps : 0.0);
    if (gen) fprintf(stderr, "gpu: %u tokens in %.1f ms, %.3f ms/token, %.1f tokens/s (speculative, %u drafts)\n", gen, gms,
                     gms / gen, 1e3 * gen / gms, draft);
  }
  for (uint32_t pos = 0; !draft && pos < total; pos++) {
    double ms = 0;
    out[pos] = tok;
    if (pos + 1 == total) break; /* the last token needs no step */
    uint32_t next = b->step(b, tok, pos, (df || ppl) ? logits : NULL, &ms);
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
  free(logits), free(out), free(prompt);
  graph_free(&gr);
  model_free(&m);
  gguf_close(&g);
  return 0;
}
