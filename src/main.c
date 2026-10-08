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
#include "gen.h"
#include "model.h"

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
    const char *dh = getenv("IE_DRAFT_HEAD"); /* q8: the drafts use the Q8_0 output matrix */
    if (!(dh && !strcmp(dh, "q8"))) model_make_draft_head(&m, &g);
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
  sampler sp = {temp, top_p, top_k, seed * 0x9E3779B97F4A7C15ull + 1, malloc((size_t)m.vocab * 4), malloc((size_t)m.vocab * 4),
                malloc(16 * 64 * 4), malloc(16 * 64 * 4)};
  if (temp > 0.0f && (ppl || dump)) ie_die("--temp: not with --ppl or --dump-logits");
  uint32_t start = 0, hrow = 0;
  if (use_pf) { /* the prompt in chunks (prefill), then one token per step */
    ppl_acc pa = {ppl, ppl_first, 0, 0};
    const double t0 = now_ms();
    uint32_t first = prefill(b, &gr, m.vocab, prompt, n_prompt, chunk, draft > 0, &pa, &hrow);
    if (temp > 0.0f) { /* sample the first token from the last prompt row */
      if (gpu_cands(&sp, b, &gr, hrow, 1, m.vocab, sp.cv, sp.ci)) {
        first = sample_cands(&sp, sp.cv, sp.ci, sample_k(&sp, m.vocab));
      } else {
        float *lg = malloc(gr.bufs[gr.logits].stride);
        b->read_rows(b, gr.logits, hrow, 1, lg);
        first = sample(&sp, lg, m.vocab);
        free(lg);
      }
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
    total = spec_generate(b, &m, &gr, out, n_prompt, total, draft, eos_v, stop, tok, hrow, &sp, m.vocab, &gms, &steps, &acc,
                          pmin, &drafted, NULL, NULL);
    const uint32_t gen = total - n_prompt - 1; /* the tokens after the first */
    fprintf(stderr, "mtp: %u steps, %u drafts accepted of %u (%.1f%%), %.2f tokens per step\n", steps, acc, drafted,
            drafted ? 100.0 * acc / drafted : 0.0, steps ? (double)gen / steps : 0.0);
    if (gen) fprintf(stderr, "gpu: %u tokens in %.1f ms, %.3f ms/token, %.1f tokens/s (speculative, %u drafts)\n", gen, gms,
                     gms / gen, 1e3 * gen / gms, draft);
  }
  const double tw0 = now_ms();
  const uint32_t pw0 = start;
  for (uint32_t pos = start; !draft && pos < total; pos++) {
    double ms = 0;
    out[pos] = tok;
    if (pos + 1 == total) break; /* the last token needs no step */
    const int gk = temp > 0.0f && !df && !ppl && b->topk && sample_k(&sp, m.vocab) <= 64u && !getenv("IE_HOST_TOPK");
    uint32_t next = b->step(b, tok, pos, (df || ppl || (temp > 0.0f && !gk)) ? logits : NULL, &ms);
    if (temp > 0.0f && pos + 1 >= n_prompt) {
      if (gk && gpu_cands(&sp, b, &gr, 0, 1, m.vocab, sp.cv, sp.ci)) next = sample_cands(&sp, sp.cv, sp.ci, sample_k(&sp, m.vocab));
      else {
        if (gk) b->read_rows(b, gr.logits, 0, 1, logits);
        next = sample(&sp, logits, m.vocab);
      }
    }
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
  if (!draft && temp > 0.0f && total > pw0 + 1) /* the host's view: steps and sampling */
    fprintf(stderr, "wall: %.3f ms/token with sampling\n", (now_ms() - tw0) / (total - pw0 - 1));

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
  free(logits), free(out), free(prompt), free(sp.idx), free(sp.pr), free(sp.cv), free(sp.ci);
  graph_free(&gr);
  model_free(&m);
  gguf_close(&g);
  return 0;
}
