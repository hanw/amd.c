/* cache_check.c -- check the prompt cache of gen_run (ie-serve).
 *   cache_check MODEL.gguf [--backend cpu|gpu] [--hsaco FILE] [--draft D] [--greedy]
 * Runs a prompt A, then A + its answer + X (the cache continues from the
 * end of the answer), then the same prompt again (the cache continues from
 * its last token), then the same prompt without the cache. Sampled with one
 * seed. On the
 * CPU (one token per step) the three answers must be the same; on the GPU
 * the prompt chunks differ, so they may differ a little. */
#include <stdio.h>
#include <string.h>

#include "../src/backend.h"
#include "../src/common.h"
#include "../src/gen.h"
#include "../src/gguf.h"
#include "../src/model.h"

typedef struct {
  uint32_t v[4096], n;
} toks;
static int collect(void *ctx, const uint32_t *t, uint32_t n) {
  toks *o = ctx;
  for (uint32_t i = 0; i < n && o->n < 4096; i++) o->v[o->n++] = t[i];
  return 0;
}
static void show(const char *what, const gen_stats *st, const toks *o) {
  printf("%-28s prompt %4u, cached %4u (snapshot %2d), answer:", what, st->n_prompt, st->cached, st->cache_hit);
  for (uint32_t i = 0; i < o->n && i < 12; i++) printf(" %u", o->v[i]);
  printf("%s\n", o->n > 12 ? " ..." : "");
  (void)0;
}

int main(int argc, char **argv) {
  const char *path = NULL, *be = "cpu", *hsaco = "build/ie_kernels.hsaco";
  uint32_t draft = 0;
  int greedy = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--backend") && i + 1 < argc) be = argv[++i];
    else if (!strcmp(argv[i], "--hsaco") && i + 1 < argc) hsaco = argv[++i];
    else if (!strcmp(argv[i], "--draft") && i + 1 < argc) draft = (uint32_t)atoi(argv[++i]);
    else if (!strcmp(argv[i], "--greedy")) greedy = 1;
    else path = argv[i];
  }
  if (!path) { fprintf(stderr, "usage: cache_check MODEL.gguf [--backend cpu|gpu] [--hsaco FILE] [--draft D] [--greedy]\n"); return 2; }
  const int gpu = !strcmp(be, "gpu");
  gguf_file g;
  gguf_open(&g, path);
  if (gpu) gpu_stream_init();
  model m;
  model_load(&m, &g);
  if (draft) {
    if (!model_load_mtp(&m, &g)) ie_die("no MTP head");
    model_make_draft_head(&m, &g);
  }
  const uint32_t chunk = gpu ? 64 : 1;
  graph gr;
  graph_build(&gr, &m, 1024, draft + 1 > chunk ? draft + 1 : chunk, draft > 0);
  plan_arena(&gr, 0);
  backend *b = gpu ? gpu_open(&m, &gr, hsaco) : cpu_open(&m, &gr);
  gen_ctx c;
  gen_init(&c, b, &m, &gr);
  gen_params p = {0};
  /* sampled (temperature 1, all tokens, one seed): a small change of the logits changes the answer */
  p.max_new = 24, p.draft = draft, p.pmin = 0.6f, p.chunk = chunk, p.cache = 1, p.emit = collect;
  p.temp = greedy ? 0.0f : 1.0f, p.top_p = 1.0f, p.top_k = 0, p.seed = 7;

  /* A: 150 tokens (a few chunks on the GPU); X: 7 tokens */
  static uint32_t pr[4096];
  uint32_t na = 150, n = 0;
  for (uint32_t i = 0; i < na; i++) pr[n++] = (i * 7919u + 13u) % (m.vocab < 5000 ? m.vocab : 5000);
  toks o1 = {0}, o2 = {0}, o3 = {0}, o4 = {0};
  gen_stats st;
  p.ctx = &o1;
  gen_run(&c, pr, n, &p, &st);
  show("1. A", &st, &o1);
  /* A + the answer (all but its last token: that is the one not run) + X */
  for (uint32_t i = 0; i + 1 < o1.n; i++) pr[n++] = o1.v[i];
  for (uint32_t i = 0; i < 7; i++) pr[n++] = (i * 31u + 5u) % (m.vocab < 5000 ? m.vocab : 5000);
  p.ctx = &o2;
  gen_run(&c, pr, n, &p, &st);
  show("2. A + answer + X (cache)", &st, &o2);
  const int hit2 = st.cache_hit;
  p.ctx = &o3;
  gen_run(&c, pr, n, &p, &st);
  show("3. the same again (cache)", &st, &o3);
  const int hit3 = st.cache_hit;
  p.ctx = &o4, p.cache = 0;
  gen_run(&c, pr, n, &p, &st);
  show("4. the same, no cache", &st, &o4);
  const int same2 = o2.n == o4.n && !memcmp(o2.v, o4.v, o2.n * 4), same3 = o3.n == o4.n && !memcmp(o3.v, o4.v, o3.n * 4);
  printf("snapshots used: %d (want 1), %d (want 0); answers 2 and 4 %s, 3 and 4 %s\n", hit2, hit3, same2 ? "same" : "DIFFERENT",
         same3 ? "same" : "DIFFERENT");
  const int ok = hit2 == 1 && hit3 == 0 && (gpu || (same2 && same3));
  printf("cache_check: %s\n", ok ? "pass" : "FAIL");
  gen_free(&c);
  b->close(b);
  return !ok;
}
