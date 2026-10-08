/* gemm_bench.c -- time the prompt-chunk GEMM (ie_gemm_q8, or KERNEL=...) for
 * T tokens on several matrix shapes. Random int8 weights and activations.
 *   build/gemm_bench build/ie_kernels.hsaco [T [rows nb ...]]
 * Grid: ceil(rows / 128) * ceil(T / 64) workgroups (as src/hip.c). */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int E;
static void *sym(void *l, const char *n) { void *p = dlsym(l, n); if (!p) { fprintf(stderr, "no %s\n", n); exit(1); } return p; }
int main(int argc, char **argv) {
  void *lib = dlopen("libamdhip64.so", RTLD_NOW);
  if (!lib) lib = dlopen("libamdhip64.so.7", RTLD_NOW);
  E (*Init)(unsigned) = sym(lib, "hipInit");
  E (*Malloc)(void **, size_t) = sym(lib, "hipMalloc");
  E (*Free)(void *) = sym(lib, "hipFree");
  E (*Memset)(void *, int, size_t) = sym(lib, "hipMemset");
  E (*Load)(void **, const void *) = sym(lib, "hipModuleLoadData");
  E (*Get)(void **, void *, const char *) = sym(lib, "hipModuleGetFunction");
  E (*Launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **) = sym(lib, "hipModuleLaunchKernel");
  E (*EvC)(void **) = sym(lib, "hipEventCreate");
  E (*EvR)(void *, void *) = sym(lib, "hipEventRecord");
  E (*EvS)(void *) = sym(lib, "hipEventSynchronize");
  E (*EvT)(float *, void *, void *) = sym(lib, "hipEventElapsedTime");
  Init(0);
  FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  char *img = malloc(n); if (fread(img, 1, n, f) != (size_t)n) return 1; fclose(f);
  void *mod, *k; Load(&mod, img);
  const char *kn = getenv("KERNEL") ? getenv("KERNEL") : "ie_gemm_q8";
  if (Get(&k, mod, kn)) { fprintf(stderr, "no kernel %s\n", kn); return 1; }
  unsigned T = argc > 2 ? (unsigned)atoi(argv[2]) : 256;
  static const unsigned def[][2] = {{34816, 160}, {16480, 160}, {14336, 160}, {5120, 544}, {5120, 192}, {248320, 160}};
  unsigned ns = argc > 3 ? (unsigned)(argc - 3) / 2 : sizeof def / sizeof def[0];
  void *e0, *e1; EvC(&e0); EvC(&e1);
  printf("%8s %5s %5s %7s %9s %9s\n", "rows", "nb", "T", "wgs", "us", "TMAC/s");
  double tot = 0;
  for (unsigned s = 0; s < ns; s++) {
    unsigned rows = argc > 3 ? (unsigned)atoi(argv[3 + 2 * s]) : def[s][0], nb = argc > 3 ? (unsigned)atoi(argv[4 + 2 * s]) : def[s][1];
    size_t wb = (size_t)rows * nb * 32, sb = (size_t)rows * nb * 2, xs = ((size_t)nb * 40 + 255) & ~(size_t)255;
    void *qw, *qs, *xq, *y;
    Malloc(&qw, wb); Malloc(&qs, sb); Malloc(&xq, xs * T); Malloc(&y, (size_t)rows * 4 * T);
    Memset(qw, 0x5A, wb); Memset(qs, 0x11, sb); Memset(xq, 0x03, xs * T);
    unsigned xsu = (unsigned)xs, ys = rows, rs = 0;
    void *nul = NULL;
    void *args[] = {&qw, &qs, &xq, &y, &rows, &nb, &nul, &nul, &T, &xsu, &ys, &rs};
    unsigned g = ((rows + 127) / 128) * ((T + 63) / 64);
    const int it = 20;
    for (int i = -3; i < it; i++) {
      if (i == 0) EvR(e0, NULL);
      Launch(k, g, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
    }
    EvR(e1, NULL); EvS(e1);
    float ms; EvT(&ms, e0, e1);
    double us = ms * 1e3 / it, mac = (double)rows * nb * 32 * T;
    tot += us;
    printf("%8u %5u %5u %7u %9.1f %9.1f\n", rows, nb, T, g, us, mac / us / 1e6);
    Free(qw); Free(qs); Free(xq); Free(y);
  }
  printf("sum %.1f us\n", tot);
  return 0;
}
