/* gemv_bench.c -- time the integer GEMV kernel (ie_gemv_q4q8) alone, for
 * several shapes, on the GPU. Random weights and activations (the values
 * do not matter for the time). Each shape: 20 warm-up launches, then 200
 * timed launches between two events; it prints the mean time per launch and
 * the bytes of weights per second.
 *   cc -O2 -o build/gemv_bench tools/gemv_bench.c -ldl
 *   build/gemv_bench build/ie_kernels.hsaco [rows nb ...]
 * With no shapes it runs a default list. Each launch reads a different copy
 * of the matrix (NCOPY copies, about 256 MB in all) so that the weights come
 * from memory, not from the 64 MB cache, as in a real model. */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int hipError_t;
typedef void *hipModule_t, *hipFunction_t, *hipStream_t, *hipEvent_t;
static struct {
  hipError_t (*Init)(unsigned);
  hipError_t (*Malloc)(void **, size_t);
  hipError_t (*Free)(void *);
  hipError_t (*Memset)(void *, int, size_t);
  hipError_t (*ModuleLoadData)(hipModule_t *, const void *);
  hipError_t (*ModuleGetFunction)(hipFunction_t *, hipModule_t, const char *);
  hipError_t (*ModuleLaunchKernel)(hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                   hipStream_t, void **, void **);
  hipError_t (*DeviceSynchronize)(void);
  hipError_t (*EventCreate)(hipEvent_t *);
  hipError_t (*EventRecord)(hipEvent_t, hipStream_t);
  hipError_t (*EventSynchronize)(hipEvent_t);
  hipError_t (*EventElapsedTime)(float *, hipEvent_t, hipEvent_t);
} H;

#define CK(x)                                                     \
  do {                                                            \
    hipError_t e_ = (x);                                          \
    if (e_) {                                                     \
      fprintf(stderr, "HIP error %d at line %d\n", e_, __LINE__); \
      exit(1);                                                    \
    }                                                             \
  } while (0)

static void *sym(void *lib, const char *n) {
  void *p = dlsym(lib, n);
  if (!p) {
    fprintf(stderr, "missing %s\n", n);
    exit(1);
  }
  return p;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: gemv_bench KERNELS.hsaco [rows nb ...]\n");
    return 2;
  }
  void *lib = dlopen("libamdhip64.so", RTLD_NOW);
  if (!lib) lib = dlopen("libamdhip64.so.7", RTLD_NOW);
  if (!lib) {
    fprintf(stderr, "cannot load libamdhip64.so\n");
    return 1;
  }
  *(void **)&H.Init = sym(lib, "hipInit");
  *(void **)&H.Malloc = sym(lib, "hipMalloc");
  *(void **)&H.Free = sym(lib, "hipFree");
  *(void **)&H.Memset = sym(lib, "hipMemset");
  *(void **)&H.ModuleLoadData = sym(lib, "hipModuleLoadData");
  *(void **)&H.ModuleGetFunction = sym(lib, "hipModuleGetFunction");
  *(void **)&H.ModuleLaunchKernel = sym(lib, "hipModuleLaunchKernel");
  *(void **)&H.DeviceSynchronize = sym(lib, "hipDeviceSynchronize");
  *(void **)&H.EventCreate = sym(lib, "hipEventCreate");
  *(void **)&H.EventRecord = sym(lib, "hipEventRecord");
  *(void **)&H.EventSynchronize = sym(lib, "hipEventSynchronize");
  *(void **)&H.EventElapsedTime = sym(lib, "hipEventElapsedTime");
  CK(H.Init(0));
  FILE *f = fopen(argv[1], "rb");
  if (!f) {
    perror(argv[1]);
    return 1;
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *img = malloc(n);
  if (fread(img, 1, n, f) != (size_t)n) return 1;
  fclose(f);
  hipModule_t mod;
  hipFunction_t k;
  CK(H.ModuleLoadData(&mod, img));
  const char *kn = getenv("KERNEL") ? getenv("KERNEL") : "ie_gemv_q4q8"; /* or ie_gemv_rows (prototype) */
  const unsigned rpw = getenv("ROWS_PER_WAVE") ? (unsigned)atoi(getenv("ROWS_PER_WAVE")) : 1u;
  CK(H.ModuleGetFunction(&k, mod, kn));

  static const unsigned def[][2] = {{896, 28},   {1152, 28},  {2048, 48},  {1536, 48},  {4096, 48},
                                    {8192, 48},  {17920, 48}, {1536, 280}, {2048, 64},  {4096, 64},
                                    {11008, 64}, {2048, 344}, {4096, 128}, {14336, 128}, {4096, 448}};
  unsigned ns = argc > 2 ? (unsigned)(argc - 2) / 2 : sizeof def / sizeof def[0];
  hipEvent_t e0, e1;
  CK(H.EventCreate(&e0));
  CK(H.EventCreate(&e1));
  printf("%8s %5s %6s %10s %9s %8s\n", "rows", "nb", "groups", "MB", "us", "GB/s");
  for (unsigned s = 0; s < ns; s++) {
    unsigned rows = argc > 2 ? (unsigned)atoi(argv[2 + 2 * s]) : def[s][0];
    unsigned nb = argc > 2 ? (unsigned)atoi(argv[3 + 2 * s]) : def[s][1];
    size_t wbytes = (size_t)rows * nb * 16, sbytes = (size_t)rows * nb * 2;
    size_t per = wbytes + sbytes;
    unsigned ncopy = (unsigned)(256u * 1024u * 1024u / per);
    if (ncopy < 2) ncopy = 2;
    if (ncopy > 64) ncopy = 64;
    void **qw = calloc(ncopy, sizeof(void *)), **qs = calloc(ncopy, sizeof(void *));
    for (unsigned c = 0; c < ncopy; c++) {
      CK(H.Malloc(&qw[c], wbytes));
      CK(H.Malloc(&qs[c], sbytes));
      CK(H.Memset(qw[c], 0x5A, wbytes));
      CK(H.Memset(qs[c], 0x11, sbytes)); /* f16 0x1111: a small number */
    }
    void *xq, *y;
    CK(H.Malloc(&xq, (size_t)nb * 40));
    CK(H.Malloc(&y, (size_t)rows * 4));
    CK(H.Memset(xq, 0x03, (size_t)nb * 40));
    void *nul = NULL;
    float eps = 1e-6f;
    unsigned groups = (rows + 8 * rpw - 1) / (8 * rpw);
    const char *pe = getenv("PAD_GROUPS"); /* launch at least this many workgroups (extra ones do nothing) */
    if (pe && (unsigned)atoi(pe) > groups) groups = (unsigned)atoi(pe);
    const int iters = 200;
    for (int it = -20; it < iters; it++) {
      if (it == 0) CK(H.EventRecord(e0, NULL));
      unsigned c = (unsigned)(it + 20) % ncopy;
      void *args[] = {&qw[c], &qs[c], &xq, &y, &rows, &nb, &nul, &nul, &nul, &nul, &nul, &eps, &nul};
      CK(H.ModuleLaunchKernel(k, groups, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
    }
    CK(H.EventRecord(e1, NULL));
    CK(H.EventSynchronize(e1));
    float ms = 0;
    CK(H.EventElapsedTime(&ms, e0, e1));
    double us = ms * 1e3 / iters, mb = (double)rows * nb * 18 / 1e6;
    printf("%8u %5u %6u %10.2f %9.2f %8.1f\n", rows, nb, groups, mb, us, 1000.0 * mb / us);
    for (unsigned c = 0; c < ncopy; c++) H.Free(qw[c]), H.Free(qs[c]);
    H.Free(xq), H.Free(y);
    free(qw), free(qs);
  }
  return 0;
}
