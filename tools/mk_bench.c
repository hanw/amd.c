/* mk_bench.c -- megakernel probe: n MAT_Q4K GEMVs as n launches of
 * ie_gemv_q4kq8 against one persistent launch of ie_mk_gemv_chain (a grid
 * barrier between two GEMVs). Prints the time per pass of each and checks
 * that the outputs are bitwise equal.
 *   cc -O2 -o build/mk_bench tools/mk_bench.c -ldl
 *   build/mk_bench build/ie_kernels.hsaco [layers [grid ...]]
 * Each layer has 4 GEMVs with Qwen3.8-27B shapes (about 215 MB of weights
 * per layer, a different copy per layer, so the weights come from memory). */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int E;
static void *sym(void *l, const char *n) {
  void *p = dlsym(l, n);
  if (!p) { fprintf(stderr, "missing %s\n", n); exit(1); }
  return p;
}
#define CK(x) do { E e_ = (x); if (e_) { fprintf(stderr, "HIP error %d at line %d\n", e_, __LINE__); exit(1); } } while (0)

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: mk_bench KERNELS.hsaco [layers [grid ...]]\n"); return 2; }
  void *lib = dlopen("libamdhip64.so", RTLD_NOW);
  if (!lib) lib = dlopen("libamdhip64.so.7", RTLD_NOW);
  E (*Init)(unsigned) = sym(lib, "hipInit");
  E (*Malloc)(void **, size_t) = sym(lib, "hipMalloc");
  E (*Memset)(void *, int, size_t) = sym(lib, "hipMemset");
  E (*Cpy)(void *, const void *, size_t, int) = sym(lib, "hipMemcpy");
  E (*Load)(void **, const char *) = sym(lib, "hipModuleLoad");
  E (*Get)(void **, void *, const char *) = sym(lib, "hipModuleGetFunction");
  E (*Launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **) =
      sym(lib, "hipModuleLaunchKernel");
  E (*Coop)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **) =
      sym(lib, "hipModuleLaunchCooperativeKernel");
  E (*Sync)(void) = sym(lib, "hipDeviceSynchronize");
  E (*EvC)(void **) = sym(lib, "hipEventCreate");
  E (*EvR)(void *, void *) = sym(lib, "hipEventRecord");
  E (*EvS)(void *) = sym(lib, "hipEventSynchronize");
  E (*EvT)(float *, void *, void *) = sym(lib, "hipEventElapsedTime");
  CK(Init(0));
  void *mod, *kg, *kc;
  CK(Load(&mod, argv[1]));
  CK(Get(&kg, mod, "ie_gemv_q4kq8"));
  CK(Get(&kc, mod, "ie_mk_gemv_chain"));
  const unsigned L = argc > 2 ? (unsigned)atoi(argv[2]) : 8;
  static const unsigned shp[4][2] = {{16384, 160}, {5120, 192}, {34816, 160}, {5120, 544}};
  const unsigned n = 4 * L;
  uint32_t *d = calloc(8 * n, 4);
  void **qw = calloc(n, sizeof *qw), **qs = calloc(n, sizeof *qs), **ya = calloc(n, sizeof *ya), **yb = calloc(n, sizeof *yb);
  double mb = 0;
  unsigned maxrows = 0;
  for (unsigned i = 0; i < n; i++) {
    unsigned rows = shp[i % 4][0], nb = shp[i % 4][1];
    size_t wb = (size_t)rows * nb * 16, sb = (size_t)rows * nb * 3;
    CK(Malloc(&qw[i], wb)); CK(Malloc(&qs[i], sb));
    /* varied bytes: a different pattern per matrix */
    unsigned char *h = malloc(wb > sb ? wb : sb);
    for (size_t j = 0; j < wb; j++) h[j] = (unsigned char)(j * 2654435761u >> 13) ^ (unsigned char)i;
    CK(Cpy(qw[i], h, wb, 1));
    for (size_t j = 0; j < sb; j += 2) h[j] = (unsigned char)(j * 40503u >> 7), h[j + 1] = 0x11;
    CK(Cpy(qs[i], h, sb, 1));
    free(h);
    CK(Malloc(&ya[i], rows * 4)); CK(Malloc(&yb[i], rows * 4));
    uint64_t a = (uint64_t)qw[i], b = (uint64_t)qs[i], c = (uint64_t)yb[i];
    uint32_t e[8] = {(uint32_t)a, (uint32_t)(a >> 32), (uint32_t)b, (uint32_t)(b >> 32), (uint32_t)c, (uint32_t)(c >> 32), rows, nb};
    memcpy(d + 8 * i, e, sizeof e);
    mb += rows * nb * 18 / 1e6;
    if (rows > maxrows) maxrows = rows;
  }
  void *dd, *xq;
  CK(Malloc(&dd, 32 * n)); CK(Cpy(dd, d, 32 * n, 1));
  CK(Malloc(&xq, 544 * 40));
  { unsigned char *h = malloc(544 * 40); for (int j = 0; j < 544 * 40; j++) h[j] = (unsigned char)(j * 77 + 3) & 0x3f; CK(Cpy(xq, h, 544 * 40, 1)); free(h); }
  const int reps = 20;
  void *ws; CK(Malloc(&ws, 8 * (reps + 2) * 64)); CK(Memset(ws, 0, 8 * (reps + 2) * 64));
  void *e0, *e1; CK(EvC(&e0)); CK(EvC(&e1));
  void *nul = NULL; float eps = 1e-6f;
  /* A: n launches */
  float msa = 0;
  for (int r = -2; r < reps; r++) {
    if (r == 0) CK(EvR(e0, NULL));
    for (unsigned i = 0; i < n; i++) {
      unsigned rows = d[8 * i + 6], nb = d[8 * i + 7], g = (rows + 7) / 8;
      if (g >= 253 && g <= 256) g = 257;
      void *a[] = {&qw[i], &qs[i], &xq, &ya[i], &rows, &nb, &nul, &nul, &nul, &nul, &nul, &eps, &nul};
      CK(Launch(kg, g, 1, 1, 256, 1, 1, 0, NULL, a, NULL));
    }
  }
  CK(EvR(e1, NULL)); CK(EvS(e1)); CK(EvT(&msa, e0, e1));
  printf("%u GEMVs, %.0f MB per pass\n", n, mb);
  printf("launches      : %8.1f us per pass, %6.1f GB/s\n", msa * 1e3 / reps, mb * reps / msa);
  float *ha = malloc(maxrows * 4), *hb = malloc(maxrows * 4);
  unsigned grids[16] = {64, 128, 256}, ng = 3;
  if (argc > 3) { ng = 0; for (int i = 3; i < argc && ng < 16; i++) grids[ng++] = (unsigned)atoi(argv[i]); }
  for (unsigned gi = 0; gi < ng; gi++) {
    unsigned P = grids[gi];
    CK(Memset(ws, 0, 8 * (reps + 2) * 64));
    for (unsigned i = 0; i < n; i++) CK(Memset(yb[i], 0xff, d[8 * i + 6] * 4));
    CK(Sync());
    float ms = 0;
    for (int r = -2; r < reps; r++) {
      if (r == 0) CK(EvR(e0, NULL));
      void *w = (char *)ws + 256 * (r + 2);
      void *a[] = {&dd, (void *)&n, &xq, &w};
      E e = Coop(kc, P, 1, 1, 256, 1, 1, 0, NULL, a);
      if (e) { printf("grid %u: cooperative launch error %d\n", P, e); goto next; }
    }
    CK(EvR(e1, NULL)); CK(EvS(e1)); CK(EvT(&ms, e0, e1));
    {
      uint32_t ab[(reps + 2) * 64];
      CK(Cpy(ab, ws, sizeof ab, 2));
      int abort_ = 0;
      for (int r = 0; r < reps + 2; r++) abort_ |= ab[64 * r + 1];
      int same = 1;
      for (unsigned i = 0; i < n; i++) {
        unsigned rows = d[8 * i + 6];
        CK(Cpy(ha, ya[i], rows * 4, 2)); CK(Cpy(hb, yb[i], rows * 4, 2));
        if (memcmp(ha, hb, rows * 4)) same = 0;
      }
      printf("one launch %3u: %8.1f us per pass, %6.1f GB/s, %s%s\n", P, ms * 1e3 / reps, mb * reps / ms,
             same ? "SAME" : "DIFF", abort_ ? ", ABORT" : "");
    }
  next:;
  }
  return 0;
}
