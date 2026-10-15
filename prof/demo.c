/* demo.c -- run prof/demo_kernel.c with the profiler; write demo_trace.json.
 *   demo KERNEL.hsaco [blocks [threads [iters]]] */
#include "sm_profiler.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

typedef int E;
static void *sym(void *l, const char *n) {
  void *p = dlsym(l, n);
  if (!p) fprintf(stderr, "missing %s\n", n), exit(1);
  return p;
}
int main(int argc, char **argv) {
  if (argc < 2) return fprintf(stderr, "usage: demo KERNEL.hsaco [blocks [threads [iters]]]\n"), 2;
  const unsigned nb = argc > 2 ? (unsigned)atoi(argv[2]) : 128, nt = argc > 3 ? (unsigned)atoi(argv[3]) : 256,
                 it = argc > 4 ? (unsigned)atoi(argv[4]) : 4;
  void *l = dlopen("libamdhip64.so", RTLD_NOW);
  if (!l) l = dlopen("libamdhip64.so.7", RTLD_NOW);
  if (!l) return fprintf(stderr, "no libamdhip64.so\n"), 1;
  E (*Init)(unsigned) = sym(l, "hipInit");
  E (*Malloc)(void **, size_t) = sym(l, "hipMalloc");
  E (*Memset)(void *, int, size_t) = sym(l, "hipMemset");
  E (*Memcpy)(void *, const void *, size_t, int) = sym(l, "hipMemcpy");
  E (*Load)(void **, const char *) = sym(l, "hipModuleLoad");
  E (*Get)(void **, void *, const char *) = sym(l, "hipModuleGetFunction");
  E (*Launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **) =
      sym(l, "hipModuleLaunchKernel");
  E (*Sync)(void) = sym(l, "hipDeviceSynchronize");
  Init(0);
  void *mod, *kd, *kh;
  if (Load(&mod, argv[1])) return fprintf(stderr, "cannot load %s\n", argv[1]), 1;
  Get(&kd, mod, "demo"), Get(&kh, mod, "hwid");

  /* the WGP decoding: distinct ids over the workgroups */
  unsigned *ids = calloc(2u * nb, 4), nd = 0;
  void *dids;
  Malloc(&dids, 8u * nb);
  void *ah[] = {&dids};
  Launch(kh, nb, 1, 1, nt, 1, 1, 0, NULL, ah, NULL), Sync();
  Memcpy(ids, dids, 8u * nb, 2);
  for (unsigned i = 0; i < nb; i++) {
    int seen = 0;
    for (unsigned j = 0; j < i; j++) seen |= (ids[2 * j + 1] & 0xFFFFu) == (ids[2 * i + 1] & 0xFFFFu);
    nd += !seen;
  }
  printf("hw_id1 of workgroup 0: 0x%08x; %u distinct WGP ids over %u workgroups\n", ids[0], nd, nb);

  const unsigned waves = (nt + 31u) / 32u, n = 1u << 24;
  smp_buffer_t p = smp_create(nb, waves, 8u * it + 8u, 1);
  if (!p) return 1;
  smp_register_event(p, 0, "compute");
  smp_register_event(p, 1, "memory");
  smp_register_event(p, 2, "sync");
  smp_register_event(p, 3, "barrier");
  void *data;
  Malloc(&data, (size_t)n * 4u), Memset(data, 0, (size_t)n * 4u);
  uint64_t *pb = smp_device_ptr(p);
  unsigned nn = n, ii = it;
  void *a[] = {&pb, &data, &nn, &ii};
  if (Launch(kd, nb, 1, 1, nt, 1, 1, 0, NULL, a, NULL) || Sync()) return fprintf(stderr, "launch failed\n"), 1;
  smp_print_stats(p, stdout);
  smp_export(p, "demo_trace.json");
  smp_destroy(p);
  return 0;
}
