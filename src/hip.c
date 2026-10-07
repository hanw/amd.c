/* hip.c -- the GPU backend: HIP runtime loaded with dlopen at run time.
 * No ROCm headers are needed to build: the few HIP C API functions used are
 * declared here. The kernels are in build/ie_kernels.hsaco (kernels/ie_kernels.c).
 * Not tested here (no AMD GPU in the build machine). */
#include <dlfcn.h>
#include <math.h>
#include <string.h>

#include "../core/ie_core.h"
#include "backend.h"
#include "common.h"

typedef int hipError_t; /* 0 = hipSuccess */
typedef void *hipModule_t, *hipFunction_t, *hipStream_t, *hipEvent_t;
enum { hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2 };

static struct {
  void *lib;
  hipError_t (*Init)(unsigned);
  hipError_t (*SetDevice)(int);
  hipError_t (*GetDeviceCount)(int *);
  hipError_t (*Malloc)(void **, size_t);
  hipError_t (*Free)(void *);
  hipError_t (*Memcpy)(void *, const void *, size_t, int);
  hipError_t (*Memset)(void *, int, size_t);
  hipError_t (*ModuleLoadData)(hipModule_t *, const void *);
  hipError_t (*ModuleGetFunction)(hipFunction_t *, hipModule_t, const char *);
  hipError_t (*ModuleLaunchKernel)(hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                   unsigned, hipStream_t, void **, void **);
  hipError_t (*DeviceSynchronize)(void);
  hipError_t (*EventCreate)(hipEvent_t *);
  hipError_t (*EventRecord)(hipEvent_t, hipStream_t);
  hipError_t (*EventSynchronize)(hipEvent_t);
  hipError_t (*EventElapsedTime)(float *, hipEvent_t, hipEvent_t);
  hipError_t (*EventDestroy)(hipEvent_t);
  const char *(*GetErrorString)(hipError_t);
} H;

#define HIP(call)                                                                                            \
  do {                                                                                                       \
    hipError_t e_ = (call);                                                                                  \
    if (e_ != 0) ie_die("HIP error %d (%s) in %s", e_, H.GetErrorString ? H.GetErrorString(e_) : "?", #call); \
  } while (0)

static void hip_load(void) {
  const char *names[] = {"libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.7", "/opt/rocm/lib/libamdhip64.so"};
  for (unsigned i = 0; i < sizeof names / sizeof names[0] && !H.lib; i++) H.lib = dlopen(names[i], RTLD_NOW);
  if (!H.lib)
    ie_die("GPU backend: cannot load libamdhip64.so (the ROCm HIP runtime). Install ROCm or set "
           "LD_LIBRARY_PATH=/opt/rocm/lib. (%s)", dlerror());
#define SYM(f, n)                                              \
  do {                                                         \
    *(void **)&H.f = dlsym(H.lib, n);                          \
    if (!H.f) ie_die("GPU backend: %s not found in libamdhip64", n); \
  } while (0)
  SYM(Init, "hipInit");
  SYM(SetDevice, "hipSetDevice");
  SYM(GetDeviceCount, "hipGetDeviceCount");
  SYM(Malloc, "hipMalloc");
  SYM(Free, "hipFree");
  SYM(Memcpy, "hipMemcpy");
  SYM(Memset, "hipMemset");
  SYM(ModuleLoadData, "hipModuleLoadData");
  SYM(ModuleGetFunction, "hipModuleGetFunction");
  SYM(ModuleLaunchKernel, "hipModuleLaunchKernel");
  SYM(DeviceSynchronize, "hipDeviceSynchronize");
  SYM(EventCreate, "hipEventCreate");
  SYM(EventRecord, "hipEventRecord");
  SYM(EventSynchronize, "hipEventSynchronize");
  SYM(EventElapsedTime, "hipEventElapsedTime");
  SYM(EventDestroy, "hipEventDestroy");
  SYM(GetErrorString, "hipGetErrorString");
#undef SYM
}

/* Kernels of ie_kernels.hsaco. */
enum { K_EMBED_Q4, K_EMBED_F32, K_RMSNORM, K_QUANT, K_GEMV_Q4, K_GEMV_Q8, K_GEMV_F32, K_BIAS, K_ADD, K_SWIGLU, K_ROPE, K_KV,
       K_ATTN, K_ARGMAX, K_ROPE_KV, K_ATTN_SPLIT, K_N };
static const char *kname[K_N] = {"ie_embed_q4", "ie_embed_f32", "ie_rmsnorm", "ie_quant_q8", "ie_gemv_q4q8", "ie_gemv_q8q8",
                                 "ie_gemv_f32", "ie_bias",      "ie_add",     "ie_swiglu",   "ie_rope",
                                 "ie_kv_store", "ie_attn",      "ie_argmax",   "ie_rope_kv", "ie_attn_split"};
/* The argmax kernel: workgroups, and its device scratch (partial results and
 * the counter of finished workgroups). */
enum { ARGMAX_GROUPS = 128 };

/* Device copies of the weights of one op. */
typedef struct {
  void *w0, *w1; /* Q4/Q8: words, scales; f32 matrix: floats; vec: floats */
  void *bias;    /* GEMV: the fused bias (o->v), or NULL */
} dop;

typedef struct {
  backend base;
  const model *m;
  const graph *g;
  hipModule_t mod;
  hipFunction_t k[K_N];
  char *arena; /* device */
  u32 *am_part, *am_count; /* argmax scratch (device) */
  u32 *at_count;           /* ie_attn_split: finished workgroups per head (device) */
  int attn_split;          /* use ie_attn_split (hd <= IE_ATT_MAX_HD, and not IE_ATTN=old) */
  /* IE_ATTN=check: after each ie_attn_split, also run ie_attn into chk and
   * compare the two outputs on the host (max |a - b| / max |a|). */
  int attn_check;
  float *chk, *chk_a, *chk_b;
  double chk_worst;
  uint32_t chk_n;
  float *kc, *vc, *rcos, *rsin;
  void *tok0, *tok1; /* token_embd */
  dop *d;
  /* host -> device map of uploaded arrays */
  const void **hk;
  void **hv;
  uint32_t nh, caph;
  hipEvent_t e0, e1;
  /* IE_PROFILE=1: one event after each op; the time between two events is
   * the GPU time of that op (with the gap before it). */
  int prof;
  hipEvent_t *pev;
  double *pms;
  uint32_t psteps;
} gpu_backend;

static void *upload(gpu_backend *b, const void *host, size_t bytes) {
  for (uint32_t i = 0; i < b->nh; i++)
    if (b->hk[i] == host) return b->hv[i]; /* tied output: one copy */
  void *dev;
  HIP(H.Malloc(&dev, bytes));
  HIP(H.Memcpy(dev, host, bytes, hipMemcpyHostToDevice));
  if (b->nh == b->caph) {
    b->caph = b->caph ? 2 * b->caph : 64;
    b->hk = realloc(b->hk, b->caph * sizeof(void *));
    b->hv = realloc(b->hv, b->caph * sizeof(void *));
  }
  b->hk[b->nh] = host;
  b->hv[b->nh++] = dev;
  return dev;
}

static void upload_mat(gpu_backend *b, const mat *w, void **w0, void **w1) {
  if (w->kind == MAT_Q4) {
    *w0 = upload(b, w->qw, (size_t)w->rows * w->nb * 16);
    *w1 = upload(b, w->qs, (size_t)w->rows * w->nb * 2);
  } else if (w->kind == MAT_Q8) {
    *w0 = upload(b, w->qw, (size_t)w->rows * w->nb * 32);
    *w1 = upload(b, w->qs, (size_t)w->rows * w->nb * 2);
  } else {
    *w0 = upload(b, w->f, (size_t)w->rows * w->cols * 4);
    *w1 = NULL;
  }
}

static uint32_t gpu_step(backend *bk, uint32_t tok, uint32_t pos, float *logits, double *ms);
static void gpu_close(backend *bk);

backend *gpu_open(const model *m, const graph *g, const char *hsaco) {
  hip_load();
  HIP(H.Init(0));
  int n = 0;
  HIP(H.GetDeviceCount(&n));
  if (n < 1) ie_die("GPU backend: no HIP device");
  HIP(H.SetDevice(0));

  FILE *f = fopen(hsaco, "rb");
  if (!f) ie_die("GPU backend: cannot open %s (build it with make)", hsaco);
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *img = malloc((size_t)sz);
  if (fread(img, 1, (size_t)sz, f) != (size_t)sz) ie_die("cannot read %s", hsaco);
  fclose(f);

  gpu_backend *b = calloc(1, sizeof *b);
  b->base.step = gpu_step;
  b->base.close = gpu_close;
  b->m = m;
  b->g = g;
  HIP(H.ModuleLoadData(&b->mod, img));
  free(img);
  for (int i = 0; i < K_N; i++) HIP(H.ModuleGetFunction(&b->k[i], b->mod, kname[i]));

  HIP(H.Malloc((void **)&b->arena, g->arena));
  HIP(H.Memset(b->arena, 0, g->arena));
  HIP(H.Malloc((void **)&b->am_part, ARGMAX_GROUPS * 8u));
  HIP(H.Malloc((void **)&b->am_count, 4u));
  HIP(H.Memset(b->am_count, 0, 4u)); /* the kernel sets it back to 0 */
  HIP(H.Malloc((void **)&b->at_count, m->n_head * 4u));
  HIP(H.Memset(b->at_count, 0, m->n_head * 4u)); /* the kernel sets it back to 0 */
  const char *ae = getenv("IE_ATTN");
  b->attn_split = m->hd <= IE_ATT_MAX_HD && !(ae && !strcmp(ae, "old"));
  b->attn_check = b->attn_split && ae && !strcmp(ae, "check");
  if (b->attn_check) {
    HIP(H.Malloc((void **)&b->chk, m->n_head * m->hd * 4u));
    b->chk_a = ie_alloc(m->n_head * m->hd * 4u);
    b->chk_b = ie_alloc(m->n_head * m->hd * 4u);
  }
  const size_t kv = (size_t)m->n_layer * g->n_ctx * m->n_kv * m->hd * 4;
  HIP(H.Malloc((void **)&b->kc, kv));
  HIP(H.Malloc((void **)&b->vc, kv));
  const size_t rt = (size_t)g->n_ctx * (m->hd / 2) * 4;
  b->rcos = upload(b, g->rope_cos, rt);
  b->rsin = upload(b, g->rope_sin, rt);
  upload_mat(b, &m->tok, &b->tok0, &b->tok1);

  b->d = calloc(g->n_ops, sizeof(dop));
  for (uint32_t i = 0; i < g->n_ops; i++) {
    const op *o = &g->ops[i];
    if (o->w) upload_mat(b, o->w, &b->d[i].w0, &b->d[i].w1);
    if (o->v && o->w) b->d[i].bias = upload(b, o->v->f, o->v->n * 4); /* GEMV bias */
    else if (o->v) b->d[i].w0 = upload(b, o->v->f, o->v->n * 4);
  }
  HIP(H.EventCreate(&b->e0));
  HIP(H.EventCreate(&b->e1));
  const char *pe = getenv("IE_PROFILE");
  b->prof = pe && pe[0] == '1';
  if (b->prof) {
    b->pev = calloc(b->g->n_ops, sizeof(hipEvent_t));
    b->pms = calloc(b->g->n_ops, sizeof(double));
    if (!b->pev || !b->pms) ie_die("out of memory");
    for (uint32_t i = 0; i < b->g->n_ops; i++) HIP(H.EventCreate(&b->pev[i]));
  }
  fprintf(stderr, "gpu: %u arrays uploaded, arena %u bytes, KV cache %zu bytes\n", b->nh, g->arena, 2 * kv);
  return &b->base;
}

static void launch(gpu_backend *b, int k, unsigned groups, void **args) {
  HIP(H.ModuleLaunchKernel(b->k[k], groups, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
}

static uint32_t gpu_step(backend *bk, uint32_t tok, uint32_t pos, float *logits, double *ms) {
  gpu_backend *b = (gpu_backend *)bk;
  const model *m = b->m;
  const graph *g = b->g;
  if (tok >= m->vocab) ie_die("token %u >= vocab %u", tok, m->vocab);
  if (pos >= g->n_ctx) ie_die("position %u >= context %u", pos, g->n_ctx);
  const u32 h2 = m->hd / 2, kvd = m->n_kv * m->hd, hd = m->hd, nhd = m->n_head, nkv = m->n_kv, nctx = g->n_ctx;
  HIP(H.EventRecord(b->e0, NULL));
  for (uint32_t i = 0; i < g->n_ops; i++) {
    const op *o = &g->ops[i];
    void *A = o->a >= 0 ? b->arena + g->bufs[o->a].off : NULL;
    void *B = o->b >= 0 ? b->arena + g->bufs[o->b].off : NULL;
    void *C = o->c >= 0 ? b->arena + g->bufs[o->c].off : NULL;
    void *w0 = b->d[i].w0, *w1 = b->d[i].w1, *bias = b->d[i].bias;
    void *Q = o->qo >= 0 ? b->arena + g->bufs[o->qo].off : NULL;
    void *R = o->res >= 0 ? b->arena + g->bufs[o->res].off : NULL;
    u32 n = o->n;
    const unsigned elem_groups = (n + 255u) / 256u;
    switch (o->kind) {
      case OP_EMBED:
        if (m->tok.kind == MAT_Q4) {
          u32 nb = m->tok.nb;
          void *args[] = {&A, &b->tok0, &b->tok1, &nb, &tok};
          launch(b, K_EMBED_Q4, 1, args);
        } else {
          u32 cols = m->tok.cols;
          void *args[] = {&A, &b->tok0, &cols, &tok};
          launch(b, K_EMBED_F32, 1, args);
        }
        break;
      case OP_RMSNORM: {
        float eps = m->eps;
        void *args[] = {&A, &w0, &B, &n, &eps, &Q};
        launch(b, K_RMSNORM, 1, args);
        break;
      }
      case OP_QUANT: {
        u32 nb = n / 32u;
        void *args[] = {&A, &B, &nb};
        launch(b, K_QUANT, (nb + 7u) / 8u, args);
        break;
      }
      case OP_GEMV_Q4: {
        u32 rows = o->w->rows, nb = o->w->nb;
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R};
        launch(b, K_GEMV_Q4, ie_gemv_ngroups(rows), args);
        break;
      }
      case OP_GEMV_Q8: {
        u32 rows = o->w->rows, nb = o->w->nb;
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R};
        launch(b, K_GEMV_Q8, ie_gemv_ngroups(rows), args);
        break;
      }
      case OP_GEMV_F32: {
        u32 rows = o->w->rows, cols = o->w->cols;
        void *args[] = {&w0, &A, &B, &rows, &cols, &bias, &R};
        launch(b, K_GEMV_F32, ie_gemv_ngroups(rows), args);
        break;
      }
      case OP_BIAS: {
        void *args[] = {&A, &w0, &n};
        launch(b, K_BIAS, elem_groups, args);
        break;
      }
      case OP_ADD: {
        void *args[] = {&A, &B, &C, &n};
        launch(b, K_ADD, elem_groups, args);
        break;
      }
      case OP_SWIGLU: {
        void *args[] = {&A, &B, &n, &Q};
        launch(b, K_SWIGLU, elem_groups, args);
        break;
      }
      case OP_ROPE: {
        float *cs = b->rcos + (size_t)pos * h2, *sn = b->rsin + (size_t)pos * h2;
        u32 nh = o->nh, neox = m->rope == ROPE_NEOX;
        void *args[] = {&A, &cs, &sn, &nh, (void *)&hd, &neox};
        launch(b, K_ROPE, (nh * h2 + 255u) / 256u, args);
        break;
      }
      case OP_ROPE_KV: {
        float *cs = b->rcos + (size_t)pos * h2, *sn = b->rsin + (size_t)pos * h2;
        const size_t base = ((size_t)o->layer * nctx + pos) * kvd;
        float *kc = b->kc + base, *vc = b->vc + base;
        u32 nq = o->nh, neox = m->rope == ROPE_NEOX;
        void *args[] = {&A, &B, &C, &kc, &vc, &cs, &sn, &nq, (void *)&nkv, (void *)&hd, &neox};
        launch(b, K_ROPE_KV, ((nq + nkv) * h2 + kvd + 255u) / 256u, args);
        break;
      }
      case OP_KV: {
        const size_t base = ((size_t)o->layer * nctx + pos) * kvd;
        float *kc = b->kc + base, *vc = b->vc + base;
        void *args[] = {&A, &B, &kc, &vc, &n};
        launch(b, K_KV, elem_groups, args);
        break;
      }
      case OP_ATTN: {
        const size_t base = (size_t)o->layer * nctx * kvd;
        float *kc = b->kc + base, *vc = b->vc + base;
        if (b->attn_split) {
          u32 ch = IE_ATT_CH, nsplit = (pos + ch) / ch; /* ceil((pos + 1) / ch) */
          void *args[] = {&A, &kc, &vc, &C, &B, &pos, (void *)&hd, (void *)&nhd, (void *)&nkv, &nsplit, &ch, &b->at_count, &Q};
          launch(b, K_ATTN_SPLIT, nhd * nsplit, args);
          if (b->attn_check) {
            void *nq = NULL;
            void *a2[] = {&A, &kc, &vc, &C, &b->chk, &pos, (void *)&nctx, (void *)&hd, (void *)&nhd, (void *)&nkv, &nq};
            launch(b, K_ATTN, nhd, a2);
            const size_t nb = (size_t)nhd * hd * 4u;
            HIP(H.Memcpy(b->chk_a, B, nb, hipMemcpyDeviceToHost));
            HIP(H.Memcpy(b->chk_b, b->chk, nb, hipMemcpyDeviceToHost));
            double mx = 0, d = 0;
            for (u32 j = 0; j < nhd * hd; j++) {
              const double x = b->chk_b[j], y = b->chk_a[j];
              if (fabs(x) > mx) mx = fabs(x);
              if (fabs(x - y) > d) d = fabs(x - y);
            }
            if (mx > 0 && d / mx > b->chk_worst) b->chk_worst = d / mx;
            b->chk_n++;
          }
        } else {
          void *args[] = {&A, &kc, &vc, &C, &B, &pos, (void *)&nctx, (void *)&hd, (void *)&nhd, (void *)&nkv, &Q};
          launch(b, K_ATTN, nhd, args);
        }
        break;
      }
      case OP_ARGMAX: {
        u32 ng = ARGMAX_GROUPS;
        void *args[] = {&A, &n, &B, &b->am_part, &b->am_count, &ng};
        launch(b, K_ARGMAX, ng, args);
        break;
      }
    }
    if (b->prof) HIP(H.EventRecord(b->pev[i], NULL));
  }
  HIP(H.EventRecord(b->e1, NULL));
  HIP(H.EventSynchronize(b->e1));
  float t = 0;
  HIP(H.EventElapsedTime(&t, b->e0, b->e1));
  if (ms) *ms = t;
  if (b->prof && pos > 0) { /* skip the first step (warm up) */
    for (uint32_t i = 0; i < g->n_ops; i++) {
      float d = 0;
      HIP(H.EventElapsedTime(&d, i ? b->pev[i - 1] : b->e0, b->pev[i]));
      b->pms[i] += d;
    }
    b->psteps++;
  }
  if (logits) HIP(H.Memcpy(logits, b->arena + g->bufs[g->logits].off, m->vocab * 4, hipMemcpyDeviceToHost));
  u32 r = 0;
  HIP(H.Memcpy(&r, b->arena + g->bufs[g->argmax].off, 4, hipMemcpyDeviceToHost));
  return r;
}

/* The kernel of op o (as gpu_step launches it). */
static int op_kernel(const gpu_backend *b, const op *o) {
  const model *m = b->m;
  switch (o->kind) {
    case OP_EMBED: return m->tok.kind == MAT_Q4 ? K_EMBED_Q4 : K_EMBED_F32;
    case OP_RMSNORM: return K_RMSNORM;
    case OP_QUANT: return K_QUANT;
    case OP_GEMV_Q4: return K_GEMV_Q4;
    case OP_GEMV_Q8: return K_GEMV_Q8;
    case OP_GEMV_F32: return K_GEMV_F32;
    case OP_BIAS: return K_BIAS;
    case OP_ADD: return K_ADD;
    case OP_SWIGLU: return K_SWIGLU;
    case OP_ROPE: return K_ROPE;
    case OP_KV: return K_KV;
    case OP_ROPE_KV: return K_ROPE_KV;
    case OP_ATTN: return b->attn_split ? K_ATTN_SPLIT : K_ATTN;
    default: return K_ARGMAX;
  }
}

/* Weight bytes that op o reads (0 if it reads no weight matrix). */
static double op_wbytes(const op *o) {
  if (o->kind == OP_GEMV_Q4) return (double)o->w->rows * o->w->nb * 18.0;
  if (o->kind == OP_GEMV_Q8) return (double)o->w->rows * o->w->nb * 34.0;
  if (o->kind == OP_GEMV_F32) return (double)o->w->rows * o->w->cols * 4.0;
  return 0;
}

/* Per kernel and matrix shape: ops per token, ms per token, share, GB/s. */
static void prof_report(gpu_backend *b) {
  const graph *g = b->g;
  if (!b->psteps) return;
  enum { MAXR = 64 };
  struct { int k; u32 rows, cols; double ms, bytes; u32 cnt; } r[MAXR];
  int nr = 0;
  double total = 0;
  for (uint32_t i = 0; i < g->n_ops; i++) {
    const op *o = &g->ops[i];
    int k = op_kernel(b, o);
    u32 rows = o->w && (o->kind == OP_GEMV_Q4 || o->kind == OP_GEMV_Q8 || o->kind == OP_GEMV_F32) ? o->w->rows : 0;
    u32 cols = rows ? o->w->cols : 0;
    int j = 0;
    while (j < nr && !(r[j].k == k && r[j].rows == rows && r[j].cols == cols)) j++;
    if (j == nr) {
      if (nr == MAXR) continue;
      r[nr].k = k, r[nr].rows = rows, r[nr].cols = cols, r[nr].ms = 0, r[nr].bytes = 0, r[nr].cnt = 0;
      nr++;
    }
    r[j].ms += b->pms[i] / b->psteps;
    r[j].bytes += op_wbytes(o);
    r[j].cnt++;
    total += b->pms[i] / b->psteps;
  }
  fprintf(stderr, "profile: %u steps, %.3f ms/token in the ops\n", b->psteps, total);
  fprintf(stderr, "  %-14s %13s %6s %9s %6s %10s %8s\n", "kernel", "rows x cols", "ops", "ms/token", "share",
          "MB/token", "GB/s");
  for (int pass = 0; pass < nr; pass++) { /* largest time first */
    int best = -1;
    for (int j = 0; j < nr; j++)
      if (r[j].cnt && (best < 0 || r[j].ms > r[best].ms)) best = j;
    if (best < 0) break;
    char shape[32] = "-";
    if (r[best].rows) snprintf(shape, sizeof shape, "%ux%u", r[best].rows, r[best].cols);
    fprintf(stderr, "  %-14s %13s %6u %9.4f %5.1f%%", kname[r[best].k], shape, r[best].cnt, r[best].ms,
            100.0 * r[best].ms / total);
    if (r[best].bytes > 0)
      fprintf(stderr, " %10.2f %8.1f\n", r[best].bytes / 1e6, r[best].bytes / (r[best].ms * 1e6));
    else
      fprintf(stderr, " %10s %8s\n", "-", "-");
    r[best].cnt = 0;
  }
}

/* Per layer: ms per token, ops, weight MB, GB/s, and the split into the
 * attention half (up to the first residual add) and the FFN half. Ops
 * outside the layers (embedding, output norm, output matrix, argmax) are
 * one row each. IE_PROFILE_CSV=FILE also writes every op as CSV. */
static void prof_layers(gpu_backend *b) {
  const graph *g = b->g;
  const model *m = b->m;
  const uint32_t nl = m->n_layer;
  double *ms = calloc(nl, sizeof(double)), *att = calloc(nl, sizeof(double)), *gemv = calloc(nl, sizeof(double));
  double *by = calloc(nl, sizeof(double));
  uint32_t *cnt = calloc(nl, sizeof(uint32_t));
  int *half = calloc(nl, sizeof(int));
  if (!ms || !att || !gemv || !by || !cnt || !half) ie_die("out of memory");
  double total = 0, in_layers = 0;
  for (uint32_t i = 0; i < g->n_ops; i++) total += b->pms[i] / b->psteps;
  fprintf(stderr, "profile by layer (ms/token):\n");
  fprintf(stderr, "  %-7s %9s %9s %9s %9s %6s %6s %9s %7s\n", "layer", "total", "attn", "ffn", "gemv", "share", "ops",
          "MB", "GB/s");
  for (uint32_t i = 0; i < g->n_ops; i++) {
    const op *o = &g->ops[i];
    double t = b->pms[i] / b->psteps;
    if (o->layer < 0) {
      double wb = op_wbytes(o);
      fprintf(stderr, "  %-7s %9.4f %9s %9s %9s %5.1f%% %6u", kname[op_kernel(b, o)] + 3, t, "-", "-", "-",
              100.0 * t / total, 1u);
      if (wb > 0) fprintf(stderr, " %9.2f %7.1f\n", wb / 1e6, wb / (t * 1e6));
      else fprintf(stderr, " %9s %7s\n", "-", "-");
      continue;
    }
    uint32_t l = (uint32_t)o->layer;
    ms[l] += t, cnt[l]++, by[l] += op_wbytes(o);
    if (o->kind == OP_GEMV_Q4 || o->kind == OP_GEMV_Q8 || o->kind == OP_GEMV_F32) gemv[l] += t;
    if (!half[l]) att[l] += t;
    if (o->kind == OP_ADD) half[l] = 1; /* the first add ends the attention half */
    in_layers += t;
  }
  for (uint32_t l = 0; l < nl; l++)
    fprintf(stderr, "  %-7u %9.4f %9.4f %9.4f %9.4f %5.1f%% %6u %9.2f %7.1f\n", l, ms[l], att[l], ms[l] - att[l],
            gemv[l], 100.0 * ms[l] / total, cnt[l], by[l] / 1e6, ms[l] > 0 ? by[l] / (ms[l] * 1e6) : 0.0);
  double sa = 0, sg = 0, sb = 0;
  uint32_t sc = 0;
  for (uint32_t l = 0; l < nl; l++) sa += att[l], sg += gemv[l], sb += by[l], sc += cnt[l];
  fprintf(stderr, "  %-7s %9.4f %9.4f %9.4f %9.4f %5.1f%% %6u %9.2f %7.1f\n", "layers", in_layers, sa, in_layers - sa,
          sg, 100.0 * in_layers / total, sc, sb / 1e6, sb / (in_layers * 1e6));
  fprintf(stderr, "  %-7s %9.4f\n", "all", total);

  const char *csv = getenv("IE_PROFILE_CSV");
  if (csv && csv[0]) {
    FILE *f = fopen(csv, "w");
    if (!f) ie_die("cannot write %s", csv);
    fprintf(f, "op,layer,kernel,rows,cols,ms_per_token,weight_bytes\n");
    for (uint32_t i = 0; i < g->n_ops; i++) {
      const op *o = &g->ops[i];
      int mat = o->kind == OP_GEMV_Q4 || o->kind == OP_GEMV_Q8 || o->kind == OP_GEMV_F32;
      fprintf(f, "%u,%d,%s,%u,%u,%.6f,%.0f\n", i, o->layer, kname[op_kernel(b, o)], mat ? o->w->rows : 0,
              mat ? o->w->cols : 0, b->pms[i] / b->psteps, op_wbytes(o));
    }
    fclose(f);
    fprintf(stderr, "profile: wrote %u ops to %s\n", g->n_ops, csv);
  }
  free(ms), free(att), free(gemv), free(by), free(cnt), free(half);
}

static void gpu_close(backend *bk) {
  gpu_backend *b = (gpu_backend *)bk;
  if (b->attn_check) {
    fprintf(stderr, "attn check: %u calls, max |split - old| / max |old| = %.2e\n", b->chk_n, b->chk_worst);
    H.Free(b->chk), free(b->chk_a), free(b->chk_b);
  }
  if (b->prof && b->psteps) prof_report(b), prof_layers(b);
  HIP(H.DeviceSynchronize());
  for (uint32_t i = 0; i < b->nh; i++) H.Free(b->hv[i]);
  H.Free(b->arena), H.Free(b->kc), H.Free(b->vc);
  H.Free(b->am_part), H.Free(b->am_count), H.Free(b->at_count);
  if (b->prof) {
    for (uint32_t i = 0; i < b->g->n_ops; i++) H.EventDestroy(b->pev[i]);
    free(b->pev), free(b->pms);
  }
  free(b->hk), free(b->hv), free(b->d), free(b);
}
