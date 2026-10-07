/* main.c -- ie-run: greedy decode from token ids.
 *   ie-run MODEL.gguf [--backend cpu|gpu] [--tokens 1,2,3] [--n 32]
 *          [--ctx N] [--dump-logits FILE] [--hsaco FILE] [--info]
 * --dump-logits writes one line per step: "pos token l0 l1 ... l(vocab-1)". */
#include <string.h>

#include "backend.h"
#include "common.h"
#include "gguf.h"
#include "model.h"

static void usage(void) {
  fprintf(stderr,
          "usage: ie-run MODEL.gguf [--backend cpu|gpu] [--tokens 1,2,3] [--n 32] [--ctx N]\n"
          "              [--dump-logits FILE] [--hsaco build/ie_kernels.hsaco] [--info]\n");
  exit(2);
}

int main(int argc, char **argv) {
  const char *path = NULL, *be = "cpu", *toks = "1", *dump = NULL, *hsaco = "build/ie_kernels.hsaco";
  int n_gen = 32, info = 0;
  uint32_t n_ctx = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--backend") && i + 1 < argc) be = argv[++i];
    else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) toks = argv[++i];
    else if (!strcmp(argv[i], "--n") && i + 1 < argc) n_gen = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) n_ctx = (uint32_t)atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump = argv[++i];
    else if (!strcmp(argv[i], "--hsaco") && i + 1 < argc) hsaco = argv[++i];
    else if (!strcmp(argv[i], "--info")) info = 1;
    else if (argv[i][0] != '-' && !path) path = argv[i];
    else usage();
  }
  if (!path || n_gen < 0) usage();

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
  if (info) gguf_print(&g, stderr);
  model m;
  model_load(&m, &g);
  uint32_t need = n_prompt + (uint32_t)n_gen;
  if (n_ctx == 0) n_ctx = need;
  if (n_ctx < need) ie_die("--ctx %u is smaller than prompt + n = %u", n_ctx, need);
  fprintf(stderr, "model: %s, %u layers, dim %u, ffn %u, heads %u/%u (hd %u), vocab %u, rope %s, weights %.2f MB\n",
          m.arch == ARCH_QWEN2 ? "qwen2" : "llama", m.n_layer, m.dim, m.ffn, m.n_head, m.n_kv, m.hd, m.vocab,
          m.rope == ROPE_NEOX ? "neox" : "norm", m.weight_bytes / 1e6);

  graph gr;
  graph_build(&gr, &m, n_ctx);
  plan_arena(&gr, 1);

  backend *b = NULL;
  if (!strcmp(be, "cpu")) b = cpu_open(&m, &gr);
  else if (!strcmp(be, "gpu")) b = gpu_open(&m, &gr, hsaco);
  else usage();

  FILE *df = dump ? fopen(dump, "w") : NULL;
  if (dump && !df) ie_die("cannot write %s", dump);
  float *logits = malloc(m.vocab * 4);
  uint32_t total = n_prompt + (uint32_t)n_gen, tok = prompt[0], *out = malloc(total * 4);
  double ms_sum = 0, ms_gen = 0;
  uint32_t n_timed = 0;
  for (uint32_t pos = 0; pos < total; pos++) {
    double ms = 0;
    out[pos] = tok;
    if (pos + 1 == total) break; /* the last token needs no step */
    uint32_t next = b->step(b, tok, pos, df ? logits : NULL, &ms);
    ms_sum += ms;
    if (pos >= 1) ms_gen += ms, n_timed++; /* skip the first (warm-up) step */
    if (df) {
      fprintf(df, "%u %u", pos, tok);
      for (uint32_t i = 0; i < m.vocab; i++) fprintf(df, " %.9g", logits[i]);
      fprintf(df, "\n");
    }
    tok = pos + 1 < n_prompt ? prompt[pos + 1] : next;
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
  (void)ms_sum;
  b->close(b);
  free(logits), free(out), free(prompt);
  graph_free(&gr);
  model_free(&m);
  gguf_close(&g);
  return 0;
}
