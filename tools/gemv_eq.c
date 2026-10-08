/* gemv_eq.c -- check that a multi-token GEMV kernel is bitwise equal to the
 * one-token kernel, on random MAT_Q4K (or KIND=q6k: MAT_Q6K; KIND=q8: Q8_0) data, and time both.
 *   build/gemv_eq build/ie_kernels.hsaco KERNEL ROWS_PER_WG T rows nb
 * KERNEL takes (qw, qs, xq, y, rows, nb, bias, res, T, xs, ys, rs); the
 * reference is ie_gemv_q4kq8 (ie_gemv_q6kq8, ie_gemv_q8q8) run once per token. Prints the number of
 * output values that differ (bitwise) and the times. */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int E;
static void *sym(void *l, const char *n) { void *p = dlsym(l, n); if (!p) { fprintf(stderr, "no %s\n", n); exit(1); } return p; }
static uint64_t rs_ = 88172645463325252ull;
static uint32_t rnd(void) { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 7; rs_ ^= rs_ << 17; return (uint32_t)rs_; }
static uint16_t f2h(float f) { /* normal positive f16 from a float in [2^-14, 1) (truncated) */
  uint32_t u; memcpy(&u, &f, 4);
  int e = (int)((u >> 23) & 255) - 127 + 15;
  return (uint16_t)((e << 10) | ((u >> 13) & 1023));
}
int main(int argc, char **argv) {
  if (argc < 7) { fprintf(stderr, "usage: gemv_eq hsaco KERNEL ROWS_PER_WG T rows nb\n"); return 1; }
  void *lib = dlopen("libamdhip64.so", RTLD_NOW); if (!lib) lib = dlopen("libamdhip64.so.7", RTLD_NOW);
  E (*Init)(unsigned) = sym(lib, "hipInit");
  E (*Malloc)(void **, size_t) = sym(lib, "hipMalloc");
  E (*Memcpy)(void *, const void *, size_t, int) = sym(lib, "hipMemcpy");
  E (*Load)(void **, const char *) = sym(lib, "hipModuleLoad");
  E (*Get)(void **, void *, const char *) = sym(lib, "hipModuleGetFunction");
  E (*Launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **) = sym(lib, "hipModuleLaunchKernel");
  E (*Sync)(void) = sym(lib, "hipDeviceSynchronize");
  E (*EvC)(void **) = sym(lib, "hipEventCreate");
  E (*EvR)(void *, void *) = sym(lib, "hipEventRecord");
  E (*EvS)(void *) = sym(lib, "hipEventSynchronize");
  E (*EvT)(float *, void *, void *) = sym(lib, "hipEventElapsedTime");
  Init(0);
  void *mod, *k1, *kt;
  if (Load(&mod, argv[1])) { fprintf(stderr, "load failed\n"); return 1; }
  const int q6 = getenv("KIND") && strcmp(getenv("KIND"), "q6k") == 0, q8 = getenv("KIND") && strcmp(getenv("KIND"), "q8") == 0;
  if (Get(&k1, mod, q8 ? "ie_gemv_q8q8" : q6 ? "ie_gemv_q6kq8" : "ie_gemv_q4kq8") || Get(&kt, mod, argv[2])) { fprintf(stderr, "no kernel\n"); return 1; }
  unsigned rpw = (unsigned)atoi(argv[3]), T = (unsigned)atoi(argv[4]), rows = (unsigned)atoi(argv[5]), nb = (unsigned)atoi(argv[6]);
  size_t nw = (size_t)rows * nb * (q8 ? 8 : q6 ? 6 : 4), ns = (size_t)rows * nb + (q8 ? 0 : (size_t)rows * (nb / 8) * 2);
  unsigned xs = (nb * 40 + 255) & ~255u;
  uint32_t *w = malloc(nw * 4); uint16_t *sc = malloc(ns * 2); uint8_t *x = calloc((size_t)xs * T, 1);
  for (size_t i = 0; i < nw; i++) w[i] = rnd();
  for (size_t i = 0; i < (size_t)rows * nb; i++) sc[i] = (uint16_t)((rnd() & 63) | ((rnd() & 63) << 8));
  for (size_t i = q8 ? 0 : (size_t)rows * nb; i < ns; i++) sc[i] = f2h(0.0001f + (rnd() % 1000) * 1e-5f);
  for (unsigned t = 0; t < T; t++) {
    uint8_t *xt = x + (size_t)t * xs;
    for (unsigned b = 0; b < nb; b++) {
      int s = 0;
      for (int j = 0; j < 32; j++) { int8_t v = (int8_t)(rnd() % 255 - 127); xt[b * 32 + j] = (uint8_t)v; s += v; }
      float d = 0.001f + (rnd() % 1000) * 1e-5f;
      memcpy(xt + 32 * nb + 4 * b, &d, 4);
      memcpy(xt + 36 * nb + 4 * b, &s, 4);
    }
  }
  void *dw, *ds, *dx, *y1, *yt;
  Malloc(&dw, nw * 4); Malloc(&ds, ns * 2); Malloc(&dx, (size_t)xs * T);
  Malloc(&y1, (size_t)rows * 4 * T); Malloc(&yt, (size_t)rows * 4 * T);
  Memcpy(dw, w, nw * 4, 1); Memcpy(ds, sc, ns * 2, 1); Memcpy(dx, x, (size_t)xs * T, 1);
  void *nul = NULL; float eps = 1e-6f;
  unsigned g1 = (rows + 7) / 8; if (g1 >= 253 && g1 <= 256) g1 = 257;
  unsigned gt = (rows + rpw - 1) / rpw; if (gt >= 253 && gt <= 256) gt = 257;
  unsigned ys = rows, rsz = 0;
  void *e0, *e1; EvC(&e0); EvC(&e1);
  const int it = 50;
  float ms1 = 0, mst = 0;
  for (int rep = -3; rep < it; rep++) {
    if (rep == 0) EvR(e0, NULL);
    for (unsigned t = 0; t < T; t++) {
      void *xp = (char *)dx + (size_t)t * xs, *yp = (char *)y1 + (size_t)t * rows * 4;
      void *a[] = {&dw, &ds, &xp, &yp, &rows, &nb, &nul, &nul, &nul, &nul, &nul, &eps, &nul};
      Launch(k1, g1, 1, 1, 256, 1, 1, 0, NULL, a, NULL);
    }
  }
  EvR(e1, NULL); EvS(e1); EvT(&ms1, e0, e1);
  for (int rep = -3; rep < it; rep++) {
    if (rep == 0) EvR(e0, NULL);
    void *a[] = {&dw, &ds, &dx, &yt, &rows, &nb, &nul, &nul, &T, &xs, &ys, &rsz};
    Launch(kt, gt, 1, 1, 256, 1, 1, 0, NULL, a, NULL);
  }
  EvR(e1, NULL); EvS(e1); EvT(&mst, e0, e1);
  Sync();
  uint32_t *h1 = malloc((size_t)rows * 4 * T), *ht = malloc((size_t)rows * 4 * T);
  Memcpy(h1, y1, (size_t)rows * 4 * T, 2); Memcpy(ht, yt, (size_t)rows * 4 * T, 2);
  size_t diff = 0, first = (size_t)-1;
  for (size_t i = 0; i < (size_t)rows * T; i++) if (h1[i] != ht[i]) { diff++; if (first == (size_t)-1) first = i; }
  printf("%s T=%u %ux%u: %zu of %zu values differ", argv[2], T, rows, nb, diff, (size_t)rows * T);
  if (diff) { float a, b; memcpy(&a, &h1[first], 4); memcpy(&b, &ht[first], 4); printf(" (first: token %zu row %zu: %.9g vs %.9g)", first / rows, first % rows, a, b); }
  double mb = (double)rows * nb * (q8 ? 34 : q6 ? 26.25 : 18) / 1e6;
  printf("; one-token x%u: %.1f us (%.0f GB/s per launch); %s: %.1f us (%.0f GB/s)\n", T, ms1 * 1e3 / it, mb * T / (ms1 / it) , argv[2], mst * 1e3 / it, mb / (mst / it));
  return diff != 0;
}
