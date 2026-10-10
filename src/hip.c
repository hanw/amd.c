/* hip.c -- the GPU backend: HIP runtime loaded with dlopen at run time.
 * No ROCm headers are needed to build: the few HIP C API functions used are
 * declared here. The kernels are in build/ie_kernels.hsaco (kernels/ie_kernels.c).
 * Not tested here (no AMD GPU in the build machine). */
#include <dlfcn.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "../core/ie_core.h"
#include "backend.h"
#include "common.h"

typedef int hipError_t; /* 0 = hipSuccess */
typedef void *hipModule_t, *hipFunction_t, *hipStream_t, *hipEvent_t;
enum { hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2, hipMemcpyDeviceToDevice = 3 };

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
  hipError_t (*StreamCreate)(hipStream_t *);
  hipError_t (*StreamBeginCapture)(hipStream_t, int);
  hipError_t (*StreamEndCapture)(hipStream_t, void **);
  hipError_t (*GraphInstantiate)(void **, void *, void *, void *, size_t);
  hipError_t (*GraphLaunch)(void *, hipStream_t);
  hipError_t (*StreamSynchronize)(hipStream_t);
} H;
static hipStream_t cur_stream; /* the stream of all launches (NULL: the default stream) */

#define HIP(call)                                                                                            \
  do {                                                                                                       \
    hipError_t e_ = (call);                                                                                  \
    if (e_ != 0) ie_die("HIP error %d (%s) in %s", e_, H.GetErrorString ? H.GetErrorString(e_) : "?", #call); \
  } while (0)

static void hip_load(void) {
  if (H.lib) return;
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
  SYM(StreamCreate, "hipStreamCreate");
  SYM(StreamBeginCapture, "hipStreamBeginCapture");
  SYM(StreamEndCapture, "hipStreamEndCapture");
  SYM(GraphInstantiate, "hipGraphInstantiate");
  SYM(GraphLaunch, "hipGraphLaunch");
  SYM(StreamSynchronize, "hipStreamSynchronize");
#undef SYM
}

/* Kernels of ie_kernels.hsaco. */
enum { K_EMBED_Q4, K_EMBED_F32, K_RMSNORM, K_QUANT, K_GEMV_Q4, K_GEMV_Q8, K_GEMV_F32, K_BIAS, K_ADD, K_SWIGLU, K_ROPE, K_KV,
       K_ATTN, K_ARGMAX, K_ROPE_KV, K_ATTN_SPLIT, K_EMBED_Q8, K_QKN_ROPE_KV, K_GDN, K_GEMV_Q8_T, K_GEMV_Q8_TR, K_GEMM_Q8, K_RING_STORE, K_ATTN_PF, K_GEMM_H, K_GNORM, K_GDN1, K_RMSNORM_T, K_ATTN_PFG, K_GDN_PREP, K_GDN_WY, K_GDN_SEQ, K_GEMM_Q8R, K_GEMV_Q4K, K_GEMV_Q4K_TR, K_GEMM_Q4KR, K_GEMV_F32_T, K_GEMV_Q4K_TR8, K_GEMV_Q6K, K_GEMV_Q6K_TR, K_GEMV_Q6K_TR8, K_GEMM_Q6KR, K_GEMV_F32_TW, K_GEMV_Q4K_TS2, K_GEMV_Q6K_TS2, K_ATTN_SPLIT_T, K_TOPK_PART, K_TOPK_MERGE, K_ATTN_KV, K_ATTN_FA, K_ATTN_FA_Q8, K_ATTN_FA2, K_GEMM_F32_S, K_GEMM_Q4KR_SB, K_REQUANT_SB, K_N };
static const char *kname[K_N] = {"ie_embed_q4", "ie_embed_f32", "ie_rmsnorm", "ie_quant_q8", "ie_gemv_q4q8", "ie_gemv_q8q8",
                                 "ie_gemv_f32", "ie_bias",      "ie_add",     "ie_swiglu",   "ie_rope",
                                 "ie_kv_store", "ie_attn",      "ie_argmax",   "ie_rope_kv", "ie_attn_split",
                                 "ie_embed_q8", "ie_qkn_rope_kv", "ie_gdn", "ie_gemv_q8q8_t", "ie_gemv_q8q8_tr", "ie_gemm_q8", "ie_ring_store", "ie_attn_pf", "ie_gemm_h", "ie_gnorm", "ie_gdn1", "ie_rmsnorm_t", "ie_attn_pfg", "ie_gdn_prep", "ie_gdn_wy", "ie_gdn_seq", "ie_gemm_q8r", "ie_gemv_q4kq8", "ie_gemv_q4kq8_tr", "ie_gemm_q4kr", "ie_gemv_f32_t", "ie_gemv_q4kq8_tr8", "ie_gemv_q6kq8", "ie_gemv_q6kq8_tr", "ie_gemv_q6kq8_tr8", "ie_gemm_q6kr", "ie_gemv_f32_tw", "ie_gemv_q4kq8_ts2", "ie_gemv_q6kq8_ts2", "ie_attn_split_t", "ie_topk_part", "ie_topk_merge", "ie_attn_kv", "ie_attn_fa", "ie_attn_fa_q8", "ie_attn_fa2", "ie_gemm_f32_s", "ie_gemm_q4kr_sb", "ie_requant_sb"};
/* ie_gdn: conv input ring slots and state slots per linear layer */
/* ie_gdn: conv input ring slots (as in ie_kernels.c); the state slots per
 * linear layer are gpu_backend.ns: at least the tokens of a verify run */
enum { GDN_RING = 32 };
/* The argmax kernel: workgroups, and its device scratch (partial results and
 * the counter of finished workgroups). */
enum { ARGMAX_GROUPS = 128 };

/* Device copies of the weights of one op. */
typedef struct {
  void *w0, *w1; /* Q4/Q8: words, scales; f32 matrix: floats; vec: floats */
  void *bias;    /* GEMV: the fused bias (o->v), or NULL */
  void *nw;      /* GEMV: the weight of the fused RMSNorm (o->nv), or NULL */
} dop;

typedef struct {
  backend base;
  const model *m;
  const graph *g;
  hipModule_t mod;
  hipFunction_t k[K_N];
  char *arena; /* device */
  u32 *am_part, *am_count; /* argmax scratch (device) */
  u32 *at_count;           /* ie_attn_split: finished workgroups per head (device); ie_attn_split_t: per token too */
  float *at_part;          /* ie_attn_split_t: the partial results, at_pstride floats per token */
  float *tk_buf;           /* top K: candidates (TK_ROWS x TK_NG x TK_MAX values, then indices), results */
  u32 at_pstride, at_tmax;
  u32 *gm_count;           /* GEMV with a fused RMSNorm: finished workgroups (device) */
  u32 slot;                /* ie_gdn: the state slot that holds the current state */
  u32 ns;                  /* ie_gdn: state slots per linear layer (4, or 8 for more than 3 drafts) */
  int attn_split;          /* use ie_attn_split (hd <= IE_ATT_MAX_HD, and not IE_ATTN=old) */
  int attn_kv;             /* use ie_attn_kv for decode and verify (the K/V of a split read once; not IE_ATTN=split) */
  /* IE_ATTN=check: after each ie_attn_split, also run ie_attn into chk and
   * compare the two outputs on the host (max |a - b| / max |a|). */
  int attn_check;
  float *chk, *chk_a, *chk_b;
  double chk_worst;
  uint32_t chk_n;
  /* host time spent issuing the launches of a step (not waiting) */
  double host_ms;
  uint32_t host_n;
  void *rq;         /* ie_requant_sb output (prompt chunks): T x the largest Q4_K input row */
  size_t rq_bytes;
  void *kc, *vc;    /* KV cache: kvb bytes per element (2: fp16, the kernels have ie_kv_f16; 4: f32) */
  u32 kvb;
  float *rcos, *rsin;
  float *ring, *st; /* qwen35 linear attention state (see cpu.c) */
  float *snap_ring[2], *snap_st[2]; /* the prompt cache snapshots (gpu_snap), allocated at the first save */
  char *snap_h[2];
  /* the chunked linear attention of prompt chunks (ie_gdn_prep/wy/seq):
   * scratch for T tokens, or NULL (graph T <= 16 or no linear layers) */
  float *cq_qk, *cq_v, *cq_bg, *cq_u, *cq_w, *cq_m, *cq_g;
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
  if (w->d0) { /* streamed at load */
    *w0 = w->d0, *w1 = w->d1;
    return;
  }
  if (w->kind == MAT_Q4) {
    *w0 = upload(b, w->qw, (size_t)w->rows * w->nb * 16);
    *w1 = upload(b, w->qs, (size_t)w->rows * w->nb * 2);
  } else if (w->kind == MAT_Q4K) {
    *w0 = upload(b, w->qw, (size_t)w->rows * w->nb * 16);
    *w1 = upload(b, w->qs, (size_t)ie_q4k_qs_n(w->rows, w->nb) * 2);
  } else if (w->kind == MAT_Q6K) {
    *w0 = upload(b, w->qw, (size_t)ie_q6k_qw_n(w->rows, w->nb) * 4);
    *w1 = upload(b, w->qs, (size_t)ie_q6k_qs_n(w->rows, w->nb) * 2);
  } else if (w->kind == MAT_Q8) {
    *w0 = upload(b, w->qw, (size_t)w->rows * w->nb * 32);
    *w1 = upload(b, w->qs, (size_t)w->rows * w->nb * 2);
  } else {
    *w0 = upload(b, w->f, (size_t)w->rows * w->cols * 4);
    *w1 = NULL;
  }
}

static uint32_t gpu_step(backend *bk, uint32_t tok, uint32_t pos, float *logits, double *ms);
static void gpu_run(backend *bk, int sec, const uint32_t *toks, uint32_t T, uint32_t pos, int last_only, uint32_t *out,
                    float *prob);
static void gpu_accept(backend *bk, uint32_t k);
static void gpu_snap(backend *bk, int save, int k, int hbuf);
static void gpu_read_rows(backend *bk, int id, uint32_t r0, uint32_t n, float *dst);
static int gpu_topk(backend *bk, int id, uint32_t r0, uint32_t n, uint32_t nv, uint32_t K, float *val, uint32_t *idx);
static void gpu_copy_rows(backend *bk, int dst, uint32_t d0, int src, uint32_t r0, uint32_t n);
static void gpu_close(backend *bk);

/* Streaming load: the device arrays of the streamed matrices (freed at close). */
static void **sdev;
static uint32_t n_sdev, cap_sdev;
static size_t sdev_bytes;
static void *sdev_put(void *host, size_t bytes) {
  void *dev;
  HIP(H.Malloc(&dev, bytes));
  HIP(H.Memcpy(dev, host, bytes, hipMemcpyHostToDevice));
  if (n_sdev == cap_sdev) {
    cap_sdev = cap_sdev ? 2 * cap_sdev : 256;
    sdev = realloc(sdev, cap_sdev * sizeof(void *));
    if (!sdev) ie_die("out of memory");
  }
  sdev[n_sdev++] = dev;
  sdev_bytes += bytes;
  free(host);
  return dev;
}
static void gpu_sink(mat *w) {
  if (!w->rows || w->d0) return;
  if (w->kind == MAT_F32) {
    if (!w->f) return;
    w->d0 = sdev_put(w->f, (size_t)w->rows * w->cols * 4), w->f = NULL;
  } else {
    if (!w->qw) return;
    const size_t words = w->kind == MAT_Q4 || w->kind == MAT_Q4K ? 16 : w->kind == MAT_Q6K ? 24 : 32;
    const size_t qsn = w->kind == MAT_Q4K   ? (size_t)ie_q4k_qs_n(w->rows, w->nb)
                       : w->kind == MAT_Q6K ? (size_t)ie_q6k_qs_n(w->rows, w->nb)
                                            : (size_t)w->rows * w->nb;
    w->d0 = sdev_put(w->qw, (size_t)w->rows * w->nb * words), w->qw = NULL;
    w->d1 = sdev_put(w->qs, qsn * 2), w->qs = NULL;
  }
}
void gpu_stream_init(void) {
  hip_load();
  HIP(H.Init(0));
  HIP(H.SetDevice(0));
  ie_mat_sink = gpu_sink;
}

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
  b->base.run = gpu_run;
  b->base.copy_rows = gpu_copy_rows;
  b->base.accept = gpu_accept;
  b->base.snap = gpu_snap;
  b->base.read_rows = gpu_read_rows;
  b->base.topk = gpu_topk;
  b->base.close = gpu_close;
  b->m = m;
  b->g = g;
  HIP(H.ModuleLoadData(&b->mod, img));
  free(img);
  for (int i = 0; i < K_N; i++) HIP(H.ModuleGetFunction(&b->k[i], b->mod, kname[i]));
  {
    hipFunction_t f;
    b->kvb = H.ModuleGetFunction(&f, b->mod, "ie_kv_f16") == 0 ? 2u : 4u;
  }

  HIP(H.Malloc((void **)&b->arena, g->arena));
  HIP(H.Memset(b->arena, 0, g->arena));
  HIP(H.Malloc((void **)&b->am_part, ARGMAX_GROUPS * 12u));
  b->ns = g->T > 4u && g->mtp_h >= 0 ? 8u : 4u; /* a verify run of up to 8 tokens needs 8 states */
  HIP(H.Malloc((void **)&b->am_count, 4u));
  HIP(H.Memset(b->am_count, 0, 4u)); /* the kernel sets it back to 0 */
  HIP(H.Malloc((void **)&b->gm_count, 4u));
  HIP(H.Memset(b->gm_count, 0, 4u)); /* the kernel sets it back to 0 */
  b->at_tmax = 16u; /* ie_attn_split_t: up to 16 tokens (speculative verify) */
  HIP(H.Malloc((void **)&b->at_count, m->n_head * 4u * b->at_tmax));
  HIP(H.Memset(b->at_count, 0, m->n_head * 4u * b->at_tmax)); /* the kernel sets it back to 0 */
  b->at_pstride = m->n_head * ie_att_max_split(g->n_ctx) * (m->hd + 2u);
  b->at_part = NULL;
  if (g->T > 1u) HIP(H.Malloc((void **)&b->at_part, (size_t)b->at_pstride * 4u * b->at_tmax));
  const char *ae = getenv("IE_ATTN");
  b->attn_split = m->hd <= IE_ATT_MAX_HD && ie_att_max_split(g->n_ctx) <= IE_ATT_MAX_SPLIT && !(ae && !strcmp(ae, "old"));
  b->attn_check = b->attn_split && ae && !strcmp(ae, "check");
  b->attn_kv = b->attn_split && !(ae && (!strcmp(ae, "split") || !strcmp(ae, "check"))) && ie_att_ch(0) % 8u == 0u &&
               IE_ATT_MAX_SPLIT <= 4096u && m->hd % 4u == 0u; /* ie_attn_kv: the merge weights in its LDS buffer, float4 loads */
  if (b->attn_check) {
    HIP(H.Malloc((void **)&b->chk, m->n_head * m->hd * 4u));
    b->chk_a = ie_alloc(m->n_head * m->hd * 4u);
    b->chk_b = ie_alloc(m->n_head * m->hd * 4u);
  }
  if (g->T > 1u) { /* ie_requant_sb's buffer */
    size_t mx = 0;
    for (uint32_t i = 0; i < g->n_ops; i++)
      if (g->ops[i].kind == OP_GEMV_Q4K && (size_t)g->bufs[g->ops[i].a].stride > mx) mx = g->bufs[g->ops[i].a].stride;
    if (mx) {
      b->rq_bytes = mx * g->T;
      HIP(H.Malloc(&b->rq, b->rq_bytes));
    }
  }
  const size_t kv = (size_t)m->n_kvl * g->n_ctx * m->n_kv * m->hd * b->kvb + 4;
  HIP(H.Malloc((void **)&b->kc, kv));
  HIP(H.Malloc((void **)&b->vc, kv));
  const size_t rbytes = (size_t)m->n_rec * GDN_RING * m->conv_dim * 4 + 4,
               sbytes = (size_t)m->n_rec * b->ns * m->n_vh * m->sd * m->sd * 4 + 4;
  HIP(H.Malloc((void **)&b->ring, rbytes));
  HIP(H.Malloc((void **)&b->st, sbytes));
  if (m->n_rec && g->T > 16u && !(getenv("IE_GDN_CHUNK") && getenv("IE_GDN_CHUNK")[0] == '0')) {
    const size_t T = g->T, nb = (T + 31u) / 32u, nv = m->n_vh;
    HIP(H.Malloc((void **)&b->cq_qk, T * m->n_kh * 256u * 4u));
    HIP(H.Malloc((void **)&b->cq_v, T * nv * 128u * 4u));
    HIP(H.Malloc((void **)&b->cq_bg, T * nv * 2u * 4u));
    HIP(H.Malloc((void **)&b->cq_u, T * nv * 128u * 4u));
    HIP(H.Malloc((void **)&b->cq_w, T * nv * 128u * 4u));
    HIP(H.Malloc((void **)&b->cq_m, nb * nv * 32u * 32u * 4u));
    HIP(H.Malloc((void **)&b->cq_g, T * nv * 4u));
  }
  const size_t rt = (size_t)g->n_ctx * (m->n_rot / 2) * 4;
  b->rcos = upload(b, g->rope_cos, rt);
  b->rsin = upload(b, g->rope_sin, rt);
  upload_mat(b, &m->tok, &b->tok0, &b->tok1);

  b->d = calloc(g->n_ops, sizeof(dop));
  for (uint32_t i = 0; i < g->n_ops; i++) {
    const op *o = &g->ops[i];
    if (o->w) upload_mat(b, o->w, &b->d[i].w0, &b->d[i].w1);
    if (o->v && o->w) b->d[i].bias = upload(b, o->v->f, o->v->n * 4); /* GEMV bias */
    else if (o->v) b->d[i].w0 = upload(b, o->v->f, o->v->n * 4);
    if (o->nv) b->d[i].nw = upload(b, o->nv->f, o->nv->n * 4);
    if (o->kind == OP_GDN) { /* conv weights, dt bias, A, norm weight */
      const layer *L = &m->l[o->layer];
      b->d[i].w0 = upload(b, L->conv.f, L->conv.n * 4);
      b->d[i].w1 = upload(b, L->dt_bias.f, L->dt_bias.n * 4);
      b->d[i].bias = upload(b, L->ssm_a.f, L->ssm_a.n * 4);
      b->d[i].nw = upload(b, L->ssm_norm.f, L->ssm_norm.n * 4);
    }
  }
  HIP(H.EventCreate(&b->e0));
  HIP(H.EventCreate(&b->e1));
  const char *pe = getenv("IE_PROFILE");
  b->prof = pe && (pe[0] == '1' || pe[0] == '2') ? pe[0] - '0' : 0; /* 1: decode steps; 2: prompt chunks */
  if (b->prof) {
    b->pev = calloc(b->g->n_ops, sizeof(hipEvent_t));
    b->pms = calloc(b->g->n_ops, sizeof(double));
    if (!b->pev || !b->pms) ie_die("out of memory");
    for (uint32_t i = 0; i < b->g->n_ops; i++) HIP(H.EventCreate(&b->pev[i]));
  }
  if (n_sdev) fprintf(stderr, "gpu: %u arrays (%.2f GB) streamed at load\n", n_sdev, sdev_bytes / 1e9);
  fprintf(stderr, "gpu: %u arrays uploaded, arena %u bytes, KV cache %zu bytes, linear state %zu bytes\n", b->nh, g->arena,
          2 * kv, rbytes + sbytes);
  return &b->base;
}

static void launch_lds(gpu_backend *b, int k, unsigned groups, unsigned lds_bytes, void **args) {
  HIP(H.ModuleLaunchKernel(b->k[k], groups, 1, 1, 256, 1, 1, lds_bytes, cur_stream, args, NULL));
}
/* groups x T workgroups: grid dimension y is the token (kernels with byte strides) */
static void launch_t(gpu_backend *b, int k, unsigned groups, unsigned T, void **args) {
  HIP(H.ModuleLaunchKernel(b->k[k], groups, T, 1, 256, 1, 1, 0, cur_stream, args, NULL));
}
static void launch(gpu_backend *b, int k, unsigned groups, void **args) { launch_lds(b, k, groups, 0, args); }

/* GEMV grids: on the R9700 (gfx1201, 64 CUs) a GEMV with 253 .. 256
 * workgroups runs 2-3 times slower than with 252 or 257 (measured with
 * tools/gemv_bench.c: 2048 rows x 48 blocks, 17.0 us at 256 workgroups,
 * 6.5 us at 257). The cause is not known. A GEMV grid in that range gets
 * extra workgroups up to 257; they have no row (r >= rows) and do nothing. */
static unsigned gemv_groups(u32 rows) {
  const unsigned g = ie_gemv_ngroups(rows);
  return g >= 253u && g <= 256u ? 257u : g;
}

/* Launch op i for token t of a run: tok is its token id, pos its position.
 * For OP_GDN and, when T > 1, OP_GEMV_Q8, one launch does all T tokens (t
 * = 0). slot: the linear attention state slot to read (ie_gdn). */
static void launch_op(gpu_backend *b, uint32_t i, u32 t, u32 tok, u32 pos, u32 T, u32 slot, u32 wfrom) {
  const model *m = b->m;
  const graph *g = b->g;
  const u32 h2 = m->n_rot / 2, kvd = m->n_kv * m->hd, hd = m->hd, nhd = m->n_head, nkv = m->n_kv, nctx = g->n_ctx;
  (void)slot;
  {
    const op *o = &g->ops[i];
#define TP(id, o_) ((id) >= 0 ? (void *)(b->arena + g->bufs[id].off + (o_) + (size_t)t * g->bufs[id].stride) : NULL)
    void *A = TP(o->a, o->aoff);
    void *B = TP(o->b, o->boff);
    void *C = TP(o->c, o->coff);
    void *NW = b->d[i].nw;
    void *NY = TP(o->nout, 0);
    void *NQ = TP(o->nq, 0);
    float eps = m->eps;
    void *w0 = b->d[i].w0, *w1 = b->d[i].w1, *bias = b->d[i].bias;
    void *Q = TP(o->qo, 0);
    void *R = TP(o->res, 0);
#undef TP
    u32 n = o->n;
    const unsigned elem_groups = (n + 255u) / 256u;
    switch (o->kind) {
      case OP_EMBED:
        if (m->tok.kind == MAT_Q4 || m->tok.kind == MAT_Q8) {
          u32 nb = m->tok.nb;
          void *args[] = {&A, &b->tok0, &b->tok1, &nb, &tok};
          launch(b, m->tok.kind == MAT_Q4 ? K_EMBED_Q4 : K_EMBED_Q8, 1, args);
        } else {
          u32 cols = m->tok.cols;
          void *args[] = {&A, &b->tok0, &cols, &tok};
          launch(b, K_EMBED_F32, 1, args);
        }
        break;
      case OP_RMSNORM: {
        u32 xs = g->bufs[o->a].stride, ys = g->bufs[o->b].stride, qs = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
        void *args[] = {&A, &w0, &B, &n, &eps, &Q, &xs, &ys, &qs};
        if (T >= 8u) launch_t(b, K_RMSNORM_T, 1u, T, args); /* many tokens: one workgroup per token */
        else launch_t(b, K_RMSNORM, (n + 255u) / 256u, T, args); /* one block of 32 per wave */
        break;
      }
      case OP_QUANT: {
        u32 nb = n / 32u, xs = g->bufs[o->a].stride, qs = g->bufs[o->b].stride;
        void *args[] = {&A, &B, &nb, &xs, &qs};
        launch_t(b, K_QUANT, (nb + 7u) / 8u, T, args);
        break;
      }
      case OP_GEMV_Q4: {
        u32 rows = o->w->rows, nb = o->w->nb;
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &NW, &NY, &NQ, &eps, &b->gm_count};
        launch(b, K_GEMV_Q4, gemv_groups(rows), args);
        break;
      }
      case OP_GEMV_Q6K: {
        u32 rows = o->w->rows, nb = o->w->nb;
        if (T > 1) { /* all tokens: 8 at a time (8 rows per wave) or 16 at a time */
          if (NW) ie_die("a GEMV with a fused norm (IE_NORM_FUSE) cannot do several tokens");
          u32 xs = g->bufs[o->a].stride, ys = g->bufs[o->b].stride / 4u, rs = o->res >= 0 ? g->bufs[o->res].stride / 4u : 0u;
          static int gmin6 = -1;
          if (gmin6 < 0) gmin6 = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
          if ((int)T >= gmin6 && nb % 8u == 0u) { /* WMMA */
            void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &T, &xs, &ys, &rs};
            launch(b, K_GEMM_Q6KR, ((rows + 127u) / 128u) * ((T + 63u) / 64u), args); /* GK_R x GK_T */
            break;
          }
          const u32 step = T <= 8u ? 8u : 16u; /* <= 8 tokens (verify): 8 rows per wave */
          for (u32 t0 = 0; t0 < T; t0 += step) {
            u32 tn = T - t0 < step ? T - t0 : step;
            void *Ab = (char *)A + (size_t)t0 * xs, *Bb = (char *)B + (size_t)t0 * ys * 4u;
            const void *Rb = R ? (const char *)R + (size_t)t0 * rs * 4u : NULL;
            void *ta[] = {&w0, &w1, &Ab, &Bb, &rows, &nb, &bias, &Rb, &tn, &xs, &ys, &rs};
            const u32 g8 = step == 8u ? (rows + 63u) / 64u : (rows + 31u) / 32u;
            if (step == 8u && tn >= 6u) { /* the tokens split over 2 waves: more waves in flight */
              const u32 g2 = (rows + 31u) / 32u;
              launch(b, K_GEMV_Q6K_TS2, g2 >= 253u && g2 <= 256u ? 257u : g2, ta);
            } else
              launch(b, step == 8u ? K_GEMV_Q6K_TR8 : K_GEMV_Q6K_TR, g8 >= 253u && g8 <= 256u ? 257u : g8, ta);
          }
          break;
        }
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &NW, &NY, &NQ, &eps, &b->gm_count};
        launch(b, K_GEMV_Q6K, gemv_groups(rows), args);
        break;
      }
      case OP_GEMV_Q4K: {
        u32 rows = o->w->rows, nb = o->w->nb;
        if (T > 1) { /* all tokens, 16 at a time: the weights read once per 16 tokens */
          if (NW) ie_die("a GEMV with a fused norm (IE_NORM_FUSE) cannot do several tokens");
          u32 xs = g->bufs[o->a].stride, ys = g->bufs[o->b].stride / 4u, rs = o->res >= 0 ? g->bufs[o->res].stride / 4u : 0u;
          static int gmin = -1;
          if (gmin < 0) gmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
          if ((int)T >= gmin && nb % 8u == 0u) { /* WMMA */
            static int sb = -1; /* IE_Q4K_SB=1: activations requantized with one scale per 256 (ie_gemm_q4kr_sb) */
            if (sb < 0) sb = getenv("IE_Q4K_SB") && getenv("IE_Q4K_SB")[0] == '1';
            if (sb && b->rq && (size_t)T * xs <= b->rq_bytes) {
              void *Q = b->rq;
              void *ra[] = {&A, &Q, &nb, &xs, &xs};
              launch_t(b, K_REQUANT_SB, (nb / 8u + 7u) / 8u, T, ra);
              void *args[] = {&w0, &w1, &Q, &B, &rows, &nb, &bias, &R, &T, &xs, &ys, &rs};
              launch(b, K_GEMM_Q4KR_SB, ((rows + 127u) / 128u) * ((T + 63u) / 64u), args);
              break;
            }
            void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &T, &xs, &ys, &rs};
            launch(b, K_GEMM_Q4KR, ((rows + 127u) / 128u) * ((T + 63u) / 64u), args); /* GK_R x GK_T */
            break;
          }
          const u32 gr = (rows + 31u) / 32u;
          for (u32 t0 = 0; t0 < T; t0 += 16u) {
            u32 tn = T - t0 < 16u ? T - t0 : 16u;
            void *Ab = (char *)A + (size_t)t0 * xs, *Bb = (char *)B + (size_t)t0 * ys * 4u;
            const void *Rb = R ? (const char *)R + (size_t)t0 * rs * 4u : NULL;
            void *ta[] = {&w0, &w1, &Ab, &Bb, &rows, &nb, &bias, &Rb, &tn, &xs, &ys, &rs};
            if (tn >= 6u && tn <= 8u) { /* 8 rows per wave, the tokens split over 2 waves: more waves in flight */
              launch(b, K_GEMV_Q4K_TS2, gr >= 253u && gr <= 256u ? 257u : gr, ta);
            } else if (tn <= 8u) { /* 8 rows per wave: half the activation loads per weight byte */
              const u32 g8 = (rows + 63u) / 64u;
              launch(b, K_GEMV_Q4K_TR8, g8 >= 253u && g8 <= 256u ? 257u : g8, ta);
            } else {
              launch(b, K_GEMV_Q4K_TR, gr >= 253u && gr <= 256u ? 257u : gr, ta);
            }
          }
          break;
        }
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &NW, &NY, &NQ, &eps, &b->gm_count};
        launch(b, K_GEMV_Q4K, gemv_groups(rows), args);
        break;
      }
      case OP_GEMV_Q8: {
        u32 rows = o->w->rows, nb = o->w->nb;
        if (T > 1) { /* all tokens, the weights read once */
          if (NW) ie_die("a GEMV with a fused norm (IE_NORM_FUSE) cannot do several tokens");
          u32 xs = g->bufs[o->a].stride, ys = g->bufs[o->b].stride / 4u, rs = o->res >= 0 ? g->bufs[o->res].stride / 4u : 0u;
          void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &T, &xs, &ys, &rs};
          /* many tokens (prompt chunks): 4 rows per wave, 4 times less
           * activation traffic; few tokens (verify): one row per wave, more
           * workgroups. The same results either way. */
          static int trmin = -1, gmin = -1;
          if (trmin < 0) trmin = getenv("IE_TR_MIN") ? atoi(getenv("IE_TR_MIN")) : 2;
          if (gmin < 0) gmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
          /* ie_gemm_q8r (weights straight into registers, 256 rows per
           * workgroup) when nb % 4 == 0; else ie_gemm_q8 (128 rows).
           * IE_GEMM=old: always ie_gemm_q8; IE_GEMM=h: the fp16 WMMA kernel. */
          static int gsel = -1;
          if (gsel < 0) gsel = !getenv("IE_GEMM") ? 0 : getenv("IE_GEMM")[0] == 'h' ? 1 : 2;
          const int gk = gsel == 1 ? K_GEMM_H : gsel == 0 && nb % 4u == 0u ? K_GEMM_Q8R : K_GEMM_Q8;
          const u32 grows = gk == K_GEMM_Q8R ? 256u : 128u;
          if ((int)T >= gmin) { /* many tokens: the matrix instruction (WMMA) */
            launch(b, gk, ((rows + grows - 1u) / grows) * ((T + 63u) / 64u), args);
            if (getenv("IE_GEMM_CHECK")) { /* debug: compare with ie_gemv_q8q8_t, 16 tokens at a time, no bias/residual */
              static float *dev = NULL;
              static size_t cap = 0;
              const size_t need = (size_t)T * ys * 4;
              if (need > cap) { if (dev) H.Free(dev); HIP(H.Malloc((void **)&dev, need)); cap = need; }
              void *nul = NULL;
              u32 one = 1;
              (void)one;
              float *y1 = malloc(need), *y2 = malloc(need);
              void *ga[] = {&w0, &w1, &A, &dev, &rows, &nb, &nul, &nul, &T, &xs, &ys, &rs};
              launch(b, gk, ((rows + grows - 1u) / grows) * ((T + 63u) / 64u), ga);
              for (u32 t0 = 0; t0 < T; t0 += 16) {
                u32 tn = T - t0 < 16 ? T - t0 : 16;
                char *Ab = (char *)A + (size_t)t0 * xs;
                float *Bb = (float *)B + (size_t)t0 * ys;
                void *ta[] = {&w0, &w1, &Ab, &Bb, &rows, &nb, &nul, &nul, &tn, &xs, &ys, &rs};
                launch(b, K_GEMV_Q8_T, gemv_groups(rows), ta);
              }
              HIP(H.Memcpy(y1, dev, need, hipMemcpyDeviceToHost));
              HIP(H.Memcpy(y2, B, need, hipMemcpyDeviceToHost));
              double mx = 0, d = 0;
              for (u32 t = 0; t < T; t++)
                for (u32 r = 0; r < rows; r++) {
                  double u = y1[(size_t)t * ys + r], v = y2[(size_t)t * ys + r];
                  if (fabs(v) > mx) mx = fabs(v);
                  if (fabs(u - v) > d) d = fabs(u - v);
                }
              fprintf(stderr, "gemm check %ux%u T=%u: max |gemm - gemv| / max |gemv| = %.2e\n", rows, o->w->cols, T, mx > 0 ? d / mx : 0);
              free(y1), free(y2);
              if (bias || R) ie_die("IE_GEMM_CHECK: stop at the first GEMV with bias or residual (the check overwrote its output)");
            }
          } else if ((int)T >= trmin) {
            static unsigned rpw = 0; /* rows per wave of ie_gemv_q8q8_tr (GEMV_R) */
            if (!rpw) rpw = getenv("IE_TR_R") ? (unsigned)atoi(getenv("IE_TR_R")) : 4u;
            unsigned gr = (rows + 8u * rpw - 1u) / (8u * rpw);
            launch(b, K_GEMV_Q8_TR, gr >= 253u && gr <= 256u ? 257u : gr, args);
          } else {
            launch(b, K_GEMV_Q8_T, gemv_groups(rows), args);
          }
          break;
        }
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &NW, &NY, &NQ, &eps, &b->gm_count};
        launch(b, K_GEMV_Q8, gemv_groups(rows), args);
        break;
      }
      case OP_GEMV_F32: {
        u32 rows = o->w->rows, cols = o->w->cols;
        if (T > 1) { /* all tokens in one launch (grid y) */
          if (NW) ie_die("a GEMV with a fused norm (IE_NORM_FUSE) cannot do several tokens");
          u32 xs = g->bufs[o->a].stride, ys = g->bufs[o->b].stride, rs = o->res >= 0 ? g->bufs[o->res].stride : 0u;
          void *ta[] = {&w0, &A, &B, &rows, &cols, &bias, &R, &xs, &ys, &rs};
          static int gmin = -1;
          if (gmin < 0) gmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
          static int fs = -1; /* IE_GEMM_F32_S=0: off */
          if (fs < 0) fs = !(getenv("IE_GEMM_F32_S") && getenv("IE_GEMM_F32_S")[0] == '0');
          if ((int)T >= gmin && fs && rows <= 1024u) { /* small matrix: 8 rows x 8 tokens per workgroup */
            u32 TT = T;
            void *sa[] = {&w0, &A, &B, &rows, &cols, &bias, &R, &xs, &ys, &rs, &TT};
            launch(b, K_GEMM_F32_S, ((rows + 7u) / 8u) * ((T + 7u) / 8u), sa);
          } else if ((int)T >= gmin) launch_t(b, K_GEMV_F32_TW, gemv_groups(rows), T, ta); /* prompt chunk: one row per wave */
          else launch_t(b, K_GEMV_F32_T, rows >= 253u && rows <= 256u ? 257u : rows, T, ta); /* one row per workgroup */
          break;
        }
        void *args[] = {&w0, &A, &B, &rows, &cols, &bias, &R, &NW, &NY, &NQ, &eps, &b->gm_count};
        launch(b, K_GEMV_F32, rows >= 253u && rows <= 256u ? 257u : rows, args); /* one row per workgroup */
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
        u32 as = g->bufs[o->a].stride, qs = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
        void *args[] = {&A, &B, &n, &Q, &as, &qs};
        launch_t(b, K_SWIGLU, elem_groups, T, args);
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
        const size_t base = ((size_t)m->l[o->layer].kvi * nctx + pos) * kvd;
        void *kc = (char *)b->kc + base * b->kvb, *vc = (char *)b->vc + base * b->kvb;
        u32 nq = o->nh, neox = m->rope == ROPE_NEOX;
        void *args[] = {&A, &B, &C, &kc, &vc, &cs, &sn, &nq, (void *)&nkv, (void *)&hd, &neox};
        launch(b, K_ROPE_KV, ((nq + nkv) * h2 + kvd + 255u) / 256u, args);
        break;
      }
      case OP_QKN_ROPE_KV: {
        float *cs = b->rcos + (size_t)pos * h2, *sn = b->rsin + (size_t)pos * h2;
        const size_t base = ((size_t)m->l[o->layer].kvi * nctx + pos) * kvd;
        void *kc = (char *)b->kc + base * b->kvb, *vc = (char *)b->vc + base * b->kvb;
        u32 nq = o->nh, nrot = m->n_rot;
        u32 ins = g->bufs[o->a].stride, qos = g->bufs[o->b].stride;
        void *args[] = {&A, &B, &kc, &vc, &w0, &NW, &cs, &sn, &nq, (void *)&nkv, (void *)&hd, &nrot, &eps, &ins, &qos};
        launch_t(b, K_QKN_ROPE_KV, nq + 2u * nkv, T, args);
        break;
      }
      case OP_GDN: {
        const layer *L = &m->l[o->layer];
        float *ring = b->ring + (size_t)L->sti * GDN_RING * m->conv_dim;
        float *st = b->st + (size_t)L->sti * b->ns * m->n_vh * m->sd * m->sd;
        u32 ns = b->ns;
        void *cw = b->d[i].w0, *dtb = b->d[i].w1, *sa = b->d[i].bias, *nw = b->d[i].nw;
        u32 cd = m->conv_dim, nk = m->n_kh, nv = m->n_vh;
        u32 is = g->bufs[o->a].stride / 4u, os = g->bufs[o->b].stride / 4u, qs = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
        /* a prompt chunk (wfrom > 0) may have more tokens than the ring has
         * slots: ie_gdn does not write the ring, ie_ring_store does after it */
        u32 rw = wfrom == 0;
        if (T <= 8u) { /* decode and verify: one kernel, the norm fused */
          void *a1[] = {&A, &B, &Q, &ring, &st, &cw, &dtb, &sa, &nw, &pos, &cd, &nk, &nv, &eps, &T, &is, &os, &qs, &slot, &wfrom, &rw, &ns};
          launch(b, K_GDN1, nv, a1);
          if (!rw) {
            void *a2[] = {&A, &ring, &pos, &T, &cd, &is};
            launch(b, K_RING_STORE, (3u * cd + 255u) / 256u, a2);
          }
          break;
        }
        static int cmin = -1;
        if (cmin < 0) cmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
        if (b->cq_qk && wfrom == T - 1u && (int)T >= cmin && m->sd == 128u) {
          /* the chunked form: only the state after the last token is kept */
          static int gchk = -1;
          if (gchk < 0) gchk = getenv("IE_GDN_CHECK") != NULL;
          float *ref = NULL, *sref = NULL;
          const size_t ob = (size_t)T * os * 4u, sb = (size_t)nv * 16384u * 4u;
          float *sw = st + (size_t)((slot + T - 1u) % ns) * nv * 16384u;
          /* debug: the sequential kernel first, its output and state kept (not
           * when the state after the chunk goes to the slot it is read from) */
          if (gchk && (slot + T - 1u) % ns == slot) gchk = 2;
          else if (gchk == 2) gchk = 1;
          if (gchk == 1) {
            void *args[] = {&A, &B, &ring, &st, &cw, &dtb, &sa, &pos, &cd, &nk, &nv, &eps, &T, &is, &os, &slot, &wfrom, &rw, &ns};
            launch(b, K_GDN, nv * 4u, args);
            ref = malloc(ob), sref = malloc(sb);
            HIP(H.Memcpy(ref, B, ob, hipMemcpyDeviceToHost));
            HIP(H.Memcpy(sref, sw, sb, hipMemcpyDeviceToHost));
          }
          void *p1[] = {&A, &ring, &cw, &dtb, &sa, &pos, &cd, &nk, &nv, &eps, &is, &b->cq_qk, &b->cq_v, &b->cq_bg};
          launch_t(b, K_GDN_PREP, 1, T, p1);
          void *p2[] = {&b->cq_qk, &b->cq_v, &b->cq_bg, &b->cq_u, &b->cq_w, &b->cq_m, &b->cq_g, &T, &nk, &nv};
          launch(b, K_GDN_WY, nv * ((T + 31u) / 32u), p2);
          void *p3[] = {&b->cq_qk, &b->cq_u, &b->cq_w, &b->cq_m, &b->cq_g, &B, &os, &st, &slot, &ns, &pos, &T, &nk, &nv};
          launch(b, K_GDN_SEQ, nv * 4u, p3); /* 4 workgroups per value head (GD_SPLIT) */
          if (gchk == 1) {
            float *x = malloc(ob), *y = malloc(sb);
            HIP(H.Memcpy(x, B, ob, hipMemcpyDeviceToHost));
            HIP(H.Memcpy(y, sw, sb, hipMemcpyDeviceToHost));
            double eo = 0, mo = 0, es = 0, ms = 0;
            for (u32 t = 0; t < T; t++)
              for (u32 k = 0; k < nv * 128u; k++) {
                const double a = ref[(size_t)t * os + k], d = fabs(a - x[(size_t)t * os + k]);
                if (d > eo) eo = d;
                if (fabs(a) > mo) mo = fabs(a);
              }
            for (size_t k = 0; k < sb / 4u; k++) {
              const double d = fabs(sref[k] - y[k]);
              if (d > es) es = d;
              if (fabs(sref[k]) > ms) ms = fabs(sref[k]);
            }
            fprintf(stderr, "gdn check layer %u T %u pos %u: out %.3g / %.3g, state %.3g / %.3g\n", o->layer, T, pos, eo, mo, es, ms);
            free(x), free(y), free(ref), free(sref);
          }
        } else {
          void *args[] = {&A, &B, &ring, &st, &cw, &dtb, &sa, &pos, &cd, &nk, &nv, &eps, &T, &is, &os, &slot, &wfrom, &rw, &ns};
          launch(b, K_GDN, nv * 4u, args); /* prompt chunks: 4 workgroups per value head (GD_SPLIT) */
        }
        if (!rw) {
          void *a2[] = {&A, &ring, &pos, &T, &cd, &is};
          launch(b, K_RING_STORE, (3u * cd + 255u) / 256u, a2);
        }
        u32 isb = is * 4u, osb = os * 4u; /* the gated norm, all tokens: grid y */
        void *a3[] = {&B, &A, &nw, &Q, &nv, &cd, &eps, &isb, &osb, &qs};
        launch_t(b, K_GNORM, (nv + 7u) / 8u, T, a3);
        break;
      }
      case OP_KV: {
        const size_t base = ((size_t)m->l[o->layer].kvi * nctx + pos) * kvd;
        void *kc = (char *)b->kc + base * b->kvb, *vc = (char *)b->vc + base * b->kvb;
        void *args[] = {&A, &B, &kc, &vc, &n};
        launch(b, K_KV, elem_groups, args);
        break;
      }
      case OP_ATTN: {
        const size_t base = (size_t)m->l[o->layer].kvi * nctx * kvd;
        void *kc = (char *)b->kc + base * b->kvb, *vc = (char *)b->vc + base * b->kvb;
        void *GT = o->gt >= 0 ? b->arena + g->bufs[o->gt].off + o->gtoff + (size_t)t * g->bufs[o->gt].stride : NULL;
        u32 gs = o->gstride;
        static int agmin = -1;
        if (agmin < 0) agmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
        if (T > 1 && (int)T >= agmin) { /* a prompt chunk on the fast path: all tokens in one launch */
          GT = o->gt >= 0 ? b->arena + g->bufs[o->gt].off + o->gtoff : NULL;
          u32 qst = g->bufs[o->a].stride, ost = g->bufs[o->b].stride, oqst = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
          u32 gst = o->gt >= 0 ? g->bufs[o->gt].stride : 0u;
          /* ie_attn_fa (fp16 WMMA flash attention; IE_ATTN_FA=0: off): K/V read once per 16 tokens and
           * group of query heads, Q K^T and P V on the matrix units */
          static int fa = -1;
          if (fa < 0) fa = !(getenv("IE_ATTN_FA") && getenv("IE_ATTN_FA")[0] == '0');
          /* from IE_ATTN_FA_MIN positions (default 1024): a short first chunk is faster with
           * ie_attn_pfg (512 tokens: 356 vs 379 ms for the whole prefill, 27B Q4_K_M) */
          static int fa_min = -1;
          if (fa_min < 0) fa_min = getenv("IE_ATTN_FA_MIN") ? atoi(getenv("IE_ATTN_FA_MIN")) : 1024;
          if (fa && pos + T > (u32)fa_min && nhd / nkv <= 8u && hd % 32u == 0u && hd <= 256u) {
            u32 TT = T;
            const u32 ntile = (T + 15u) / 16u, nparts = (hd + 127u) / 128u;
            void *a1[] = {&A, &kc, &vc, &B, &pos, (void *)&hd, (void *)&nhd, (void *)&nkv, &GT, &gs, &qst, &ost, &gst, &TT};
            /* ie_attn_fa2 (IE_ATTN_FA2=0: off): S once per block for all columns, the output tiles on all 8 waves */
            static int fa2 = -1;
            if (fa2 < 0) fa2 = !(getenv("IE_ATTN_FA2") && getenv("IE_ATTN_FA2")[0] == '0');
            if (fa2 && hd == 256u && nhd / nkv <= 6u) launch(b, K_ATTN_FA2, nkv * ntile, a1);
            else launch(b, K_ATTN_FA, nkv * nparts * ntile, a1);
            if (Q) {
              u32 nbq = nhd * hd / 32u;
              void *a2[] = {&B, &Q, &nbq, &ost, &oqst};
              launch_t(b, K_ATTN_FA_Q8, (nbq + 7u) / 8u, T, a2);
            }
            break;
          }
          void *args[] = {&A, &kc, &vc, &B, &pos, (void *)&hd, (void *)&nhd, (void *)&nkv, &Q, &GT, &gs, &qst, &ost, &oqst, &gst};
          if (nhd / nkv <= 8u) launch_t(b, K_ATTN_PFG, nkv, T, args); /* K/V read once per group of query heads */
          else launch_t(b, K_ATTN_PF, nhd, T, args);
          break;
        }
        /* decode and verify: ie_attn_kv (bitwise equal to ie_attn_split per
         * token) from IE_ATTN_KV_MIN positions (default 768). Measured on the
         * R9700 (27B, one token, ms/token in attention, ie_attn_kv vs
         * ie_attn_split): 100 positions 0.67 vs 0.39, 600: 0.91 vs 0.88,
         * 900: 1.09 vs 1.09, 7300: 3.44 vs 7.43. */
        static int kv_min = -1;
        if (kv_min < 0) kv_min = getenv("IE_ATTN_KV_MIN") ? atoi(getenv("IE_ATTN_KV_MIN")) : 768;
        if (b->attn_kv && T <= b->at_tmax && pos + T > (u32)kv_min) {
          u32 ch = ie_att_ch(pos), smax = ie_att_nsplit(pos + T - 1u);
          if (ie_att_ch(pos + T - 1u) != ch) ie_die("ie_attn_kv: the chunk depends on the position");
          if (T == 1u) GT = o->gt >= 0 ? b->arena + g->bufs[o->gt].off + o->gtoff + (size_t)t * g->bufs[o->gt].stride : NULL;
          else GT = o->gt >= 0 ? b->arena + g->bufs[o->gt].off + o->gtoff : NULL;
          u32 qst = g->bufs[o->a].stride, ost = g->bufs[o->b].stride, oqst = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
          u32 gst = o->gt >= 0 ? g->bufs[o->gt].stride : 0u;
          /* one token: the part area is C (as ie_attn_split); more: at_part, at_pstride floats per token */
          void *P = T == 1u ? C : (void *)b->at_part;
          u32 pst = T == 1u ? 0u : b->at_pstride, TT = T, nb = (nhd / nkv * T + 7u) / 8u;
          /* splits per workgroup: about 128 workgroups (2 per CU). Measured on the
           * R9700 (27B, one token): at 7300 positions 8 splits per workgroup
           * (116 workgroups) 3.44 ms/token in attention, 1 split 4.42; at 900
           * positions 1 split (116) 1.09, 8 splits 2.21. It does not change the result. */
          static int spw_env = -1;
          if (spw_env < 0) spw_env = getenv("IE_ATTN_SPW") ? atoi(getenv("IE_ATTN_SPW")) : 0;
          u32 spw = spw_env > 0 ? (u32)spw_env : (u32)(smax * nkv * nb / 128u);
          if (spw < 1u) spw = 1u;
          if (spw > 16u) spw = 16u;
          const u32 ngs = (smax + spw - 1u) / spw;
          void *args[] = {&A, &kc, &vc, &P, &B, &pos, (void *)&hd, (void *)&nhd, (void *)&nkv, &smax, &ch,
                          &b->at_count, &Q, &GT, &gs, &qst, &ost, &oqst, &gst, &pst, &TT, &nb, &spw};
          launch(b, K_ATTN_KV, nkv * ngs * nb, args);
          break;
        }
        if (b->attn_split && T > 1) { /* verify: all tokens in one launch (ie_attn_split_t: bitwise equal per token) */
          u32 ch = ie_att_ch(pos), smax = ie_att_nsplit(pos + T - 1u);
          if (ie_att_ch(pos + T - 1u) != ch) ie_die("ie_attn_split_t: the chunk depends on the position");
          if (T > b->at_tmax) ie_die("ie_attn_split_t: %u tokens, room for %u", T, b->at_tmax);
          u32 qst = g->bufs[o->a].stride, ost = g->bufs[o->b].stride, oqst = o->qo >= 0 ? g->bufs[o->qo].stride : 0u;
          u32 gst = o->gt >= 0 ? g->bufs[o->gt].stride : 0u, pst = b->at_pstride;
          GT = o->gt >= 0 ? b->arena + g->bufs[o->gt].off + o->gtoff : NULL;
          void *args[] = {&A, &kc, &vc, &b->at_part, &B, &pos, (void *)&hd, (void *)&nhd, (void *)&nkv, &smax, &ch,
                          &b->at_count, &Q, &GT, &gs, &qst, &ost, &oqst, &gst, &pst};
          launch_t(b, K_ATTN_SPLIT_T, nhd * smax, T, args);
          break;
        }
        if (b->attn_split) {
          u32 ch = ie_att_ch(pos), nsplit = ie_att_nsplit(pos);
          void *args[] = {&A, &kc, &vc, &C, &B, &pos, (void *)&hd, (void *)&nhd, (void *)&nkv, &nsplit, &ch, &b->at_count, &Q, &GT, &gs};
          launch(b, K_ATTN_SPLIT, nhd * nsplit, args);
          if (b->attn_check) {
            void *nq = NULL;
            void *a2[] = {&A, &kc, &vc, &C, &b->chk, &pos, (void *)&nctx, (void *)&hd, (void *)&nhd, (void *)&nkv, &nq, &GT, &gs};
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
          void *args[] = {&A, &kc, &vc, &C, &B, &pos, (void *)&nctx, (void *)&hd, (void *)&nhd, (void *)&nkv, &Q, &GT, &gs};
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
  }
}

/* Run the ops [i0, i1) for T tokens toks at positions pos .. pos + T - 1. */
static void run_ops(gpu_backend *b, uint32_t i0, uint32_t i1, const u32 *toks, u32 T, u32 pos, u32 wfrom) {
  const u32 slot = b->slot;
  /* a prompt chunk (wfrom > 0, more than one token): the model head only for
   * the last token, the MTP head not at all (its drafts are not needed) */
  const int chunk = wfrom > 0 && !b->base.all_logits;
  const graph *g = b->g;
  for (u32 t = 0; t < T; t++) {
    if (toks[t] >= b->m->vocab) ie_die("token %u >= vocab %u", toks[t], b->m->vocab);
    if (pos + t >= g->n_ctx) ie_die("position %u >= context %u", pos + t, g->n_ctx);
  }
  if (T > g->T) ie_die("%u tokens in one run; the graph has room for %u", T, g->T);
  for (uint32_t i = i0; i < i1; i++) {
    const int k = g->ops[i].kind;
    if (chunk && i >= g->i_head && i < g->n_main) { /* last token only (its row T - 1) */
      launch_op(b, i, T - 1, toks[T - 1], pos + T - 1, 1, slot, wfrom);
      continue;
    }
    if (chunk && g->mtp_logits >= 0 && i >= g->i_mtp_head) continue;
    if (T > 1 && g->mtp_logits >= 0 && i >= g->i_mtp_head) { /* MTP catch-up: only the last token's draft is used */
      launch_op(b, i, T - 1, toks[T - 1], pos + T - 1, 1, slot, wfrom);
      continue;
    }
    static int nobatch = -1;
    if (nobatch < 0) nobatch = getenv("IE_GEMV_NOBATCH") != NULL; /* debug: one GEMV launch per token */
    /* one launch for all tokens: GDN (sequential inside), Q8 GEMV (weights
     * read once), and the per-token kernels with a token grid dimension */
    static int gmin = -1;
    if (gmin < 0) gmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
    const int all = k == OP_GDN || ((k == OP_GEMV_Q8 || k == OP_GEMV_Q4K || k == OP_GEMV_Q6K || k == OP_GEMV_F32) && !nobatch) || k == OP_RMSNORM || k == OP_QUANT || k == OP_SWIGLU ||
                    k == OP_QKN_ROPE_KV || (k == OP_ATTN && (int)T >= gmin) /* the fast (not bitwise) prefill path */
                    || (k == OP_ATTN && b->attn_split && T <= b->at_tmax && b->at_part); /* verify: ie_attn_split_t (bitwise) */
    if (all) launch_op(b, i, 0, toks[0], pos, T, slot, wfrom);
    else
      for (u32 t = 0; t < T; t++) launch_op(b, i, t, toks[t], pos + t, 1, slot, wfrom);
    if (b->prof && i < g->n_main) HIP(H.EventRecord(b->pev[i], NULL));
  }
}

static uint32_t gpu_step(backend *bk, uint32_t tok, uint32_t pos, float *logits, double *ms) {
  gpu_backend *b = (gpu_backend *)bk;
  const model *m = b->m;
  const graph *g = b->g;
  static int gb = -1;
  if (gb < 0) gb = getenv("IE_GRAPH_BENCH") ? atoi(getenv("IE_GRAPH_BENCH")) : 0;
  if (gb > 0 && pos > 0) { /* debug: one step as a HIP graph, replayed gb times, against the stream */
    hipStream_t s;
    HIP(H.StreamCreate(&s));
    cur_stream = s;
    HIP(H.DeviceSynchronize());
    HIP(H.EventRecord(b->e0, s));
    for (int r = 0; r < gb; r++) run_ops(b, 0, g->n_main, &tok, 1, pos, 0);
    HIP(H.EventRecord(b->e1, s));
    HIP(H.EventSynchronize(b->e1));
    float ts = 0, tg = 0;
    HIP(H.EventElapsedTime(&ts, b->e0, b->e1));
    void *gr, *ge;
    HIP(H.StreamBeginCapture(s, 2));
    run_ops(b, 0, g->n_main, &tok, 1, pos, 0);
    HIP(H.StreamEndCapture(s, &gr));
    HIP(H.GraphInstantiate(&ge, gr, NULL, NULL, 0));
    HIP(H.GraphLaunch(ge, s));
    HIP(H.StreamSynchronize(s));
    HIP(H.EventRecord(b->e0, s));
    for (int r = 0; r < gb; r++) HIP(H.GraphLaunch(ge, s));
    HIP(H.EventRecord(b->e1, s));
    HIP(H.EventSynchronize(b->e1));
    HIP(H.EventElapsedTime(&tg, b->e0, b->e1));
    fprintf(stderr, "graph bench: stream %.3f ms/step, graph %.3f ms/step\n", ts / gb, tg / gb);
    exit(0);
  }
  HIP(H.EventRecord(b->e0, NULL));
  struct timespec h0, h1;
  clock_gettime(CLOCK_MONOTONIC, &h0);
  run_ops(b, 0, g->n_main, &tok, 1, pos, 0); /* state: slot -> slot */
  clock_gettime(CLOCK_MONOTONIC, &h1);
  if (pos > 0) b->host_ms += (h1.tv_sec - h0.tv_sec) * 1e3 + (h1.tv_nsec - h0.tv_nsec) / 1e6, b->host_n++;
  HIP(H.EventRecord(b->e1, NULL));
  HIP(H.EventSynchronize(b->e1));
  float t = 0;
  HIP(H.EventElapsedTime(&t, b->e0, b->e1));
  if (ms) *ms = t;
  if (b->prof == 1 && pos > 0) { /* skip the first step (warm up) */
    for (uint32_t i = 0; i < g->n_main; i++) {
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

static void gpu_run(backend *bk, int sec, const uint32_t *toks, uint32_t T, uint32_t pos, int last_only, uint32_t *out,
                    float *prob) {
  gpu_backend *b = (gpu_backend *)bk;
  const graph *g = b->g;
  if (sec == 1 && g->mtp_argmax < 0) ie_die("the graph has no MTP ops");
  /* IE_PROFILE=2: the ops of the prompt chunks (ms per chunk in the report) */
  const int pc = b->prof == 2 && !sec && T > 1u;
  if (pc) HIP(H.EventRecord(b->e0, NULL));
  run_ops(b, sec ? g->n_main : 0, sec ? g->n_ops : g->n_main, toks, T, pos, last_only ? T - 1 : 0);
  if (pc) {
    HIP(H.DeviceSynchronize()); /* the events are on the NULL stream: run with IE_STREAM=0 for exact times */
    hipEvent_t prev = b->e0;
    for (uint32_t i = 0; i < g->n_main; i++) { /* an op a chunk skips has no event: time 0 */
      float d = 0;
      if (H.EventElapsedTime(&d, prev, b->pev[i]) == 0) b->pms[i] += d, prev = b->pev[i];
    }
    b->psteps++;
  }
  const buf *am = &g->bufs[sec ? g->mtp_argmax : g->argmax];
  u32 *tmp = malloc((size_t)T * am->stride);
  if (!tmp) ie_die("out of memory");
  HIP(H.Memcpy(tmp, b->arena + am->off, (size_t)T * am->stride, hipMemcpyDeviceToHost));
  for (u32 t = 0; t < T; t++) {
    out[t] = tmp[t * am->stride / 4u];
    if (prob) memcpy(&prob[t], &tmp[t * am->stride / 4u + 1u], 4);
  }
  free(tmp);
}

/* The prompt cache snapshot: the current state slot of each linear layer,
 * the whole conv ring (it is indexed by position), and row 0 of hbuf. */
static void gpu_snap(backend *bk, int save, int k, int hbuf) {
  gpu_backend *b = (gpu_backend *)bk;
  const model *m = b->m;
  const size_t per = (size_t)m->n_vh * m->sd * m->sd * 4, rbytes = (size_t)m->n_rec * GDN_RING * m->conv_dim * 4;
  const size_t hb = hbuf >= 0 ? b->g->bufs[hbuf].stride : 0;
  if (!b->snap_ring[k]) {
    if (!save) ie_die("gpu_snap: no snapshot %d to restore", k);
    HIP(H.Malloc((void **)&b->snap_ring[k], rbytes + 4));
    HIP(H.Malloc((void **)&b->snap_st[k], (size_t)m->n_rec * per + 4));
    HIP(H.Malloc((void **)&b->snap_h[k], b->g->bufs[b->g->h_out].stride + 4)); /* the rows of h buffers have one size */
  }
  float *snap_ring = b->snap_ring[k], *snap_st = b->snap_st[k];
  char *snap_h = b->snap_h[k];
  for (uint32_t l = 0; l < m->n_rec; l++) {
    char *live = (char *)b->st + ((size_t)l * b->ns + b->slot) * per, *keep = (char *)snap_st + (size_t)l * per;
    HIP(H.Memcpy(save ? keep : live, save ? live : keep, per, hipMemcpyDeviceToDevice));
  }
  if (rbytes) HIP(H.Memcpy(save ? (void *)snap_ring : (void *)b->ring, save ? (void *)b->ring : (void *)snap_ring, rbytes, hipMemcpyDeviceToDevice));
  if (hb) {
    char *row = b->arena + b->g->bufs[hbuf].off;
    HIP(H.Memcpy(save ? snap_h : row, save ? row : snap_h, hb, hipMemcpyDeviceToDevice));
  }
}

static void gpu_accept(backend *bk, uint32_t k) {
  gpu_backend *b = (gpu_backend *)bk;
  b->slot = (b->slot + k) % b->ns;
}

static void gpu_read_rows(backend *bk, int id, uint32_t r0, uint32_t n, float *dst) {
  gpu_backend *b = (gpu_backend *)bk;
  const buf *B = &b->g->bufs[id];
  HIP(H.Memcpy(dst, b->arena + B->off + (size_t)r0 * B->stride, (size_t)n * B->stride, hipMemcpyDeviceToHost));
}

/* top K on the GPU (ie_topk_part, ie_topk_merge): see ie_kernels.c */
enum { TK_MAX = 64, TK_CS = 4096, TK_ROWS = 16, TK_NG = 64 };
static int gpu_topk(backend *bk, int id, uint32_t r0, uint32_t n, uint32_t nv, uint32_t K, float *val, uint32_t *idx) {
  gpu_backend *b = (gpu_backend *)bk;
  const u32 ng = (nv + TK_CS - 1u) / TK_CS;
  if (K == 0u || K > TK_MAX || n > TK_ROWS || ng > TK_NG) return 0;
  const buf *B = &b->g->bufs[id];
  if (!b->tk_buf) HIP(H.Malloc((void **)&b->tk_buf, (size_t)TK_ROWS * (TK_NG + 1u) * TK_MAX * 8u));
  float *cv = b->tk_buf, *ov = cv + (size_t)TK_ROWS * TK_NG * TK_MAX * 2u;
  u32 *ci = (u32 *)(cv + (size_t)TK_ROWS * TK_NG * TK_MAX), *oi = (u32 *)(ov + (size_t)TK_ROWS * TK_MAX);
  void *x = b->arena + B->off + (size_t)r0 * B->stride;
  u32 xs = B->stride / 4u, cs = (nv + ng - 1u) / ng, nc = ng * K;
  void *a1[] = {&x, &nv, &xs, &K, &cs, (void *)&ng, &cv, &ci};
  launch_t(b, K_TOPK_PART, ng, n, a1);
  void *a2[] = {&cv, &ci, &nc, &K, &ov, &oi};
  launch(b, K_TOPK_MERGE, n, a2);
  static float tmp[TK_ROWS * TK_MAX * 2]; /* values, then indices: one copy */
  HIP(H.Memcpy(tmp, ov, sizeof tmp, hipMemcpyDeviceToHost));
  memcpy(val, tmp, (size_t)n * K * 4u);
  memcpy(idx, tmp + TK_ROWS * TK_MAX, (size_t)n * K * 4u);
  return 1;
}

static void gpu_copy_rows(backend *bk, int dst, uint32_t d0, int src, uint32_t r0, uint32_t n) {
  gpu_backend *b = (gpu_backend *)bk;
  const graph *g = b->g;
  const buf *D = &g->bufs[dst];
  char *dp = b->arena + D->off + (size_t)d0 * D->stride;
  if (src < 0) {
    HIP(H.Memset(dp, 0, (size_t)n * D->stride));
    return;
  }
  const buf *S = &g->bufs[src];
  if (S->stride != D->stride) ie_die("copy_rows: strides differ");
  HIP(H.Memcpy(dp, b->arena + S->off + (size_t)r0 * S->stride, (size_t)n * D->stride, 3 /* device to device */));
}

/* The kernel of op o (as gpu_step launches it). */
static int op_kernel(const gpu_backend *b, const op *o) {
  const model *m = b->m;
  switch (o->kind) {
    case OP_EMBED: return m->tok.kind == MAT_Q4 ? K_EMBED_Q4 : m->tok.kind == MAT_Q8 ? K_EMBED_Q8 : K_EMBED_F32;
    case OP_QKN_ROPE_KV: return K_QKN_ROPE_KV;
    case OP_GDN: return K_GDN;
    case OP_RMSNORM: return K_RMSNORM;
    case OP_QUANT: return K_QUANT;
    case OP_GEMV_Q4: return K_GEMV_Q4;
    case OP_GEMV_Q4K: return K_GEMV_Q4K;
    case OP_GEMV_Q6K: return K_GEMV_Q6K;
    case OP_GEMV_Q8: return K_GEMV_Q8;
    case OP_GEMV_F32: return K_GEMV_F32;
    case OP_BIAS: return K_BIAS;
    case OP_ADD: return K_ADD;
    case OP_SWIGLU: return K_SWIGLU;
    case OP_ROPE: return K_ROPE;
    case OP_KV: return K_KV;
    case OP_ROPE_KV: return K_ROPE_KV;
    case OP_ATTN: return b->attn_kv ? K_ATTN_KV : b->attn_split ? K_ATTN_SPLIT : K_ATTN;
    default: return K_ARGMAX;
  }
}

/* Weight bytes that op o reads (0 if it reads no weight matrix). */
static double op_wbytes(const op *o) {
  if (o->kind == OP_GEMV_Q4 || o->kind == OP_GEMV_Q4K) return (double)o->w->rows * o->w->nb * 18.0;
  if (o->kind == OP_GEMV_Q6K) return (double)o->w->rows * o->w->nb * 26.25;
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
  for (uint32_t i = 0; i < g->n_main; i++) {
    const op *o = &g->ops[i];
    int k = op_kernel(b, o);
    u32 rows = o->w && (o->kind == OP_GEMV_Q4 || o->kind == OP_GEMV_Q4K || o->kind == OP_GEMV_Q6K || o->kind == OP_GEMV_Q8 || o->kind == OP_GEMV_F32) ? o->w->rows : 0;
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
  for (uint32_t i = 0; i < g->n_main; i++) total += b->pms[i] / b->psteps;
  fprintf(stderr, "profile by layer (ms/token):\n");
  fprintf(stderr, "  %-7s %9s %9s %9s %9s %6s %6s %9s %7s\n", "layer", "total", "attn", "ffn", "gemv", "share", "ops",
          "MB", "GB/s");
  for (uint32_t i = 0; i < g->n_main; i++) {
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
    if (o->kind == OP_GEMV_Q4 || o->kind == OP_GEMV_Q4K || o->kind == OP_GEMV_Q6K || o->kind == OP_GEMV_Q8 || o->kind == OP_GEMV_F32) gemv[l] += t;
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
    for (uint32_t i = 0; i < g->n_main; i++) {
      const op *o = &g->ops[i];
      int mat = o->kind == OP_GEMV_Q4 || o->kind == OP_GEMV_Q4K || o->kind == OP_GEMV_Q6K || o->kind == OP_GEMV_Q8 || o->kind == OP_GEMV_F32;
      fprintf(f, "%u,%d,%s,%u,%u,%.6f,%.0f\n", i, o->layer, kname[op_kernel(b, o)], mat ? o->w->rows : 0,
              mat ? o->w->cols : 0, b->pms[i] / b->psteps, op_wbytes(o));
    }
    fclose(f);
    fprintf(stderr, "profile: wrote %u ops to %s\n", g->n_main, csv);
  }
  free(ms), free(att), free(gemv), free(by), free(cnt), free(half);
}

static void gpu_close(backend *bk) {
  gpu_backend *b = (gpu_backend *)bk;
  if (b->host_n)
    fprintf(stderr, "gpu: host issue time %.3f ms/token (%u launches per token)\n", b->host_ms / b->host_n, b->g->n_ops);
  if (b->attn_check) {
    fprintf(stderr, "attn check: %u calls, max |split - old| / max |old| = %.2e\n", b->chk_n, b->chk_worst);
    H.Free(b->chk), free(b->chk_a), free(b->chk_b);
  }
  if (b->prof && b->psteps) prof_report(b), prof_layers(b);
  HIP(H.DeviceSynchronize());
  for (uint32_t i = 0; i < b->nh; i++) H.Free(b->hv[i]);
  for (uint32_t i = 0; i < n_sdev; i++) H.Free(sdev[i]);
  free(sdev), sdev = NULL, n_sdev = cap_sdev = 0;
  H.Free(b->arena), H.Free(b->kc), H.Free(b->vc), H.Free(b->ring), H.Free(b->st);
  for (int k = 0; k < 2; k++)
    if (b->snap_ring[k]) H.Free(b->snap_ring[k]), H.Free(b->snap_st[k]), H.Free(b->snap_h[k]);
  H.Free(b->am_part), H.Free(b->am_count), H.Free(b->at_count), H.Free(b->at_part), H.Free(b->tk_buf), H.Free(b->gm_count);
  if (b->prof) {
    for (uint32_t i = 0; i < b->g->n_ops; i++) H.EventDestroy(b->pev[i]);
    free(b->pev), free(b->pms);
  }
  free(b->hk), free(b->hv), free(b->d), free(b);
}
