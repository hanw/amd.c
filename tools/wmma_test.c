/* wmma_test.c -- check the WMMA lane layout (tools/wmma_test_kernel.c) on the GPU.
 *   build/wmma_test build/wmma_test.hsaco */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
typedef int E;
static void *sym(void *l, const char *n) { void *p = dlsym(l, n); if (!p) { fprintf(stderr, "no %s\n", n); exit(1); } return p; }
int main(int argc, char **argv) {
  void *lib = dlopen("libamdhip64.so", RTLD_NOW);
  if (!lib) lib = dlopen("libamdhip64.so.7", RTLD_NOW);
  E (*Init)(unsigned) = sym(lib, "hipInit");
  E (*Malloc)(void **, size_t) = sym(lib, "hipMalloc");
  E (*Memcpy)(void *, const void *, size_t, int) = sym(lib, "hipMemcpy");
  E (*Load)(void **, const void *) = sym(lib, "hipModuleLoadData");
  E (*Get)(void **, void *, const char *) = sym(lib, "hipModuleGetFunction");
  E (*Launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **) = sym(lib, "hipModuleLaunchKernel");
  E (*Sync)(void) = sym(lib, "hipDeviceSynchronize");
  Init(0);
  FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  char *img = malloc(n); if (fread(img, 1, n, f) != (size_t)n) return 1; fclose(f);
  void *mod, *k; Load(&mod, img); Get(&k, mod, "wmma_test");
  signed char A[256], Bt[256]; int D[256], R[256];
  srand(1);
  for (int i = 0; i < 256; i++) A[i] = (signed char)(rand() % 256 - 128), Bt[i] = (signed char)(rand() % 256 - 128);
  for (int m = 0; m < 16; m++) for (int nn = 0; nn < 16; nn++) { int s = 0; for (int kk = 0; kk < 16; kk++) s += A[m * 16 + kk] * Bt[nn * 16 + kk]; R[m * 16 + nn] = s; }
  void *dA, *dB, *dD; Malloc(&dA, 256); Malloc(&dB, 256); Malloc(&dD, 1024);
  Memcpy(dA, A, 256, 1); Memcpy(dB, Bt, 256, 1);
  void *args[] = {&dA, &dB, &dD};
  Launch(k, 1, 1, 1, 32, 1, 1, 0, NULL, args, NULL); Sync();
  Memcpy(D, dD, 1024, 2);
  int bad = 0; for (int i = 0; i < 256; i++) bad += D[i] != R[i];
  printf("wmma layout: %d of 256 wrong%s\n", bad, bad ? "" : " (layout ok)");
  return bad != 0;
}
