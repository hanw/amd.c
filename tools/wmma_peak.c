/* wmma_peak.c -- time tools/wmma_peak_kernel.c: build/wmma_peak build/wmma_peak.hsaco */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
typedef int E;
static void *sym(void *l, const char *n) { void *p = dlsym(l, n); if (!p) { fprintf(stderr, "no %s\n", n); exit(1); } return p; }
int main(int argc, char **argv) {
  void *lib = dlopen("libamdhip64.so", RTLD_NOW);
  if (!lib) lib = dlopen("libamdhip64.so.7", RTLD_NOW);
  E (*Init)(unsigned) = sym(lib, "hipInit"); E (*Malloc)(void **, size_t) = sym(lib, "hipMalloc");
  E (*Load)(void **, const void *) = sym(lib, "hipModuleLoadData"); E (*Get)(void **, void *, const char *) = sym(lib, "hipModuleGetFunction");
  E (*Launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **) = sym(lib, "hipModuleLaunchKernel");
  E (*EvC)(void **) = sym(lib, "hipEventCreate"); E (*EvR)(void *, void *) = sym(lib, "hipEventRecord");
  E (*EvS)(void *) = sym(lib, "hipEventSynchronize"); E (*EvT)(float *, void *, void *) = sym(lib, "hipEventElapsedTime");
  Init(0);
  FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  char *img = malloc(n); if (fread(img, 1, n, f) != (size_t)n) return 1; fclose(f);
  void *mod, *k; Load(&mod, img); Get(&k, mod, getenv("KERNEL") ? getenv("KERNEL") : "wmma_peak");
  unsigned groups = 64 * 16, it = 4096; int seed = 1;
  void *out; Malloc(&out, (size_t)groups * 256 * 4);
  void *args[] = {&out, &it, &seed}, *e0, *e1; EvC(&e0); EvC(&e1);
  Launch(k, groups, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
  EvR(e0, NULL); Launch(k, groups, 1, 1, 256, 1, 1, 0, NULL, args, NULL); EvR(e1, NULL); EvS(e1);
  float ms; EvT(&ms, e0, e1);
  double wm = (double)groups * 8 /* waves */ * it * 8;
  printf("%.0f WMMA in %.2f ms: %.1f G WMMA/s = %.1f T MAC/s (int8, 16x16x16)\n", wm, ms, wm / ms / 1e6, wm * 4096 / ms / 1e9);
  return 0;
}
