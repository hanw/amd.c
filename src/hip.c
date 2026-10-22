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
uint32_t spec_verify_max(void); /* src/gen.c */

typedef int hipError_t; /* 0 = hipSuccess */
typedef void *hipModule_t, *hipFunction_t, *hipStream_t, *hipEvent_t;
enum { hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2, hipMemcpyDeviceToDevice = 3 };

static struct {
  void *lib;
  hipError_t (*Init)(unsigned);
  hipError_t (*SetDevice)(int);
  hipError_t (*GetDeviceCount)(int *);
  hipError_t (*Malloc)(void **, size_t);
  hipError_t (*MemGetInfo)(size_t *, size_t *);
  hipError_t (*Free)(void *);
  hipError_t (*Memcpy)(void *, const void *, size_t, int);
  hipError_t (*MemcpyAsync)(void *, const void *, size_t, int, hipStream_t);
  hipError_t (*HostMalloc)(void **, size_t, unsigned);
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
  hipError_t (*ModuleLaunchCooperativeKernel)(hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned,
                                              unsigned, unsigned, hipStream_t, void **);
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
  SYM(MemGetInfo, "hipMemGetInfo");
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
  SYM(ModuleLaunchCooperativeKernel, "hipModuleLaunchCooperativeKernel");
  SYM(MemcpyAsync, "hipMemcpyAsync");
  SYM(HostMalloc, "hipHostMalloc");
#undef SYM
}

/* Kernels of ie_kernels.hsaco. */
enum { K_EMBED_Q4, K_EMBED_F32, K_RMSNORM, K_QUANT, K_GEMV_Q4, K_GEMV_Q8, K_GEMV_F32, K_BIAS, K_ADD, K_SWIGLU, K_ROPE, K_KV,
       K_ATTN, K_ARGMAX, K_ROPE_KV, K_ATTN_SPLIT, K_EMBED_Q8, K_QKN_ROPE_KV, K_GDN, K_GEMV_Q8_T, K_GEMV_Q8_TR, K_GEMM_Q8, K_RING_STORE, K_ATTN_PF, K_GEMM_H, K_GNORM, K_GDN1, K_RMSNORM_T, K_ATTN_PFG, K_GDN_PREP, K_GDN_WY, K_GDN_SEQ, K_GEMM_Q8R, K_GEMV_Q4K, K_GEMV_Q4K_TR, K_GEMM_Q4KR, K_GEMV_F32_T, K_GEMV_Q4K_TR8, K_GEMV_Q6K, K_GEMV_Q6K_TR, K_GEMV_Q6K_TR8, K_GEMM_Q6KR, K_GEMV_F32_TW, K_GEMV_Q4K_TS2, K_GEMV_Q6K_TS2, K_ATTN_SPLIT_T, K_TOPK_PART, K_TOPK_MERGE, K_ATTN_KV, K_ATTN_FA, K_ATTN_FA_Q8, K_ATTN_FA2, K_GEMM_F32_S, K_GEMM_Q4KR_SB, K_REQUANT_SB, K_Q8_F16, K_GEMM_Q4K_H, K_GEMM_Q4K_H256, K_COPY_ROWS, K_DF_CONV, K_DF_QKV, K_DF_ATTN, K_GEMV_Q5K, K_GEMV_IQ4XS, K_GEXP, K_MOE_ROUTE, K_MOE_SUM, K_QKN_ROPE_KV2, K_ATTN_SW, K_ATTN_GQ, K_N };
static const char *kname[K_N] = {"ie_embed_q4", "ie_embed_f32", "ie_rmsnorm", "ie_quant_q8", "ie_gemv_q4q8", "ie_gemv_q8q8",
                                 "ie_gemv_f32", "ie_bias",      "ie_add",     "ie_swiglu",   "ie_rope",
                                 "ie_kv_store", "ie_attn",      "ie_argmax",   "ie_rope_kv", "ie_attn_split",
                                 "ie_embed_q8", "ie_qkn_rope_kv", "ie_gdn", "ie_gemv_q8q8_t", "ie_gemv_q8q8_tr", "ie_gemm_q8", "ie_ring_store", "ie_attn_pf", "ie_gemm_h", "ie_gnorm", "ie_gdn1", "ie_rmsnorm_t", "ie_attn_pfg", "ie_gdn_prep", "ie_gdn_wy", "ie_gdn_seq", "ie_gemm_q8r", "ie_gemv_q4kq8", "ie_gemv_q4kq8_tr", "ie_gemm_q4kr", "ie_gemv_f32_t", "ie_gemv_q4kq8_tr8", "ie_gemv_q6kq8", "ie_gemv_q6kq8_tr", "ie_gemv_q6kq8_tr8", "ie_gemm_q6kr", "ie_gemv_f32_tw", "ie_gemv_q4kq8_ts2", "ie_gemv_q6kq8_ts2", "ie_attn_split_t", "ie_topk_part", "ie_topk_merge", "ie_attn_kv", "ie_attn_fa", "ie_attn_fa_q8", "ie_attn_fa2", "ie_gemm_f32_s", "ie_gemm_q4kr_sb", "ie_requant_sb", "ie_q8_f16", "ie_gemm_q4k_h", "ie_gemm_q4k_h256", "ie_copy_rows", "ie_df_conv", "ie_df_qkv", "ie_df_attn", "ie_gemv_q5kq8", "ie_gemv_iq4xsq8", "ie_gexp", "ie_moe_route", "ie_moe_sum", "ie_qkn_rope_kv2", "ie_attn_sw", "ie_attn_gq"};
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
  float *dfk, *dfv;        /* DFlash: the drafter KV cache [layer][df_ring][n_kv * hd] (f32), row = position % df_ring */
  u32 df_ring;
  u32 ns;                  /* ie_gdn: state slots per linear layer (4, or 8 for more than 3 drafts) */
  int attn_split;          /* use ie_attn_split (hd <= IE_ATT_MAX_HD, and not IE_ATTN=old) */
  int attn_gq;             /* flint: ie_attn_gq for decode (IE_ATTN_SW=1: ie_attn_sw) */
  uint32_t gq_ch;          /* its chunk: 256 positions (IE_ATTN_GQ_CH: 1 .. 256, for tests) */
  uint32_t gq_min;         /* from this position count on (IE_ATTN_GQ_MIN) */
  int flint;               /* the model is flint (its megakernel builds) */
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
  void *xh;         /* ie_gemm_q4k_h: the fp16 activations of a chunk (T x the largest Q4_K K) */
  size_t xh_bytes;
  void *rq;         /* ie_requant_sb output (prompt chunks): T x the largest Q4_K input row */
  size_t rq_bytes;
  void *kc, *vc;    /* KV cache: kvb bytes per element (2: fp16, the kernels have ie_kv_f16; 4: f32) */
  u32 kvb;
  float *rcos, *rsin;
  float *ring, *st; /* qwen35 linear attention state (see cpu.c) */
  float *snap_ring[2], *snap_st[2]; /* the prompt cache snapshots (gpu_snap), allocated at the first save */
  char *snap_h[2];
  char *snap_kv[2];  /* the KV rings of the sliding window layers (flint), at the first save */
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

/* flint: ie_attn_gq (the q heads of a kv head together) for a token at
 * position pos when pos + 1 >= gq_min, else ie_attn_sw; the whole step the
 * same (the megakernel build). Default 1: ie_attn_gq at every position
 * (UD-Q4_K_XL, ctx 4096: 5.79 against 5.85 ms per token). */
#ifndef IE_GQ_MIN
#define IE_GQ_MIN 1u
#endif
static int gq_use(const gpu_backend *b, uint32_t pos) { return b->attn_gq && pos + 1u >= b->gq_min; }

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
  const size_t ne = w->n_exp > 1 ? w->n_exp : 1; /* experts: n_exp matrices one after the other */
  if (w->kind != MAT_F32) {
    *w0 = upload(b, w->qw, (size_t)mat_qw_words(w->kind, w->rows, w->cols) * 4 * ne);
    *w1 = upload(b, w->qs, (size_t)mat_qs_n(w->kind, w->rows, w->cols) * 2 * ne);
  } else {
    *w0 = upload(b, w->f, (size_t)w->rows * w->cols * 4 * ne);
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
static double op_wbytes(const op *o);

/* IE_TRACE=FILE: one CSV row per kernel launch, in launch order (all
 * launches go to one stream, so this is also the GPU order). The rows carry
 * what a profiler does not know: the run (call, phase, first position, T),
 * the op and its layer. tools/ie_viz.py joins them with a rocprofv3 kernel
 * trace (the n-th ie_ dispatch is row n) and draws the timeline. */
static FILE *tr_f;
static const char *tr_phase = "-"; /* decode, prefill, verify, mtp, sample */
static int tr_op = -1, tr_layer = -1, tr_first;
static uint32_t tr_call, tr_n, tr_pos, tr_T;
static double tr_wb;
static void tr_launch(int k, unsigned groups, unsigned T) {
  if (!tr_f) return;
  /* the weight bytes of an op go on its first launch only */
  fprintf(tr_f, "%u,%u,%u,%u,%s,%d,%d,%s,%u,%u,%.0f\n", tr_call, tr_pos, tr_T, tr_n++, tr_phase, tr_op, tr_layer, kname[k],
          groups, T, tr_first ? tr_wb : 0.0);
  tr_first = 0;
}
/* Start a run of the trace: phase, first position, tokens. */
static void tr_run(const char *phase, uint32_t pos, uint32_t T) {
  if (!tr_f) return;
  tr_phase = phase, tr_pos = pos, tr_T = T, tr_call++;
  tr_op = tr_layer = -1, tr_first = 0;
}

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
  const size_t ne = w->n_exp > 1 ? w->n_exp : 1; /* experts: n_exp matrices one after the other */
  if (w->kind == MAT_F32) {
    if (!w->f) return;
    w->d0 = sdev_put(w->f, (size_t)w->rows * w->cols * 4 * ne), w->f = NULL;
  } else {
    if (!w->qw) return;
    w->d0 = sdev_put(w->qw, (size_t)mat_qw_words(w->kind, w->rows, w->cols) * 4 * ne), w->qw = NULL;
    w->d1 = sdev_put(w->qs, (size_t)mat_qs_n(w->kind, w->rows, w->cols) * 2 * ne), w->qs = NULL;
  }
}
void gpu_stream_init(void) {
  hip_load();
  HIP(H.Init(0));
  HIP(H.SetDevice(0));
  ie_mat_sink = gpu_sink;
}

/* The megakernel (IE_MK=1, kernels/ie_mk.c): while mk.rec is set, the
 * launches of the kernels of the current build (mk.set: decode or MTP) go
 * to a command list (256 bytes each: {id, groups x, groups y, join}, then
 * the arguments packed as a launch packs them); mk_flush copies the list to
 * the device (asynchronously, from a ring of pinned host buffers) and runs
 * it as one cooperative launch of mk.np[set] workgroups. A launch of
 * another kernel flushes the list first, so the order is always the order
 * of the launches. */
#include "ie_mk_gen.h"
#include "../prof/sm_profiler.h"
enum { MK_CMD = 256, MK_CAP = 4096, MK_ARGEND = 232, MK_RING = 4 };
static struct {
  int on, rec, set;
  hipModule_t mod[MK_NSETS];
  hipFunction_t f[MK_NSETS];
  u32 np[MK_NSETS];
  int id[K_N];       /* kernel -> MK_ id, or -1 */
  unsigned char *cmd; /* host list: mk.ring[mk.slot] */
  unsigned char *ring[MK_RING]; /* pinned */
  hipEvent_t ev[MK_RING];       /* the copy of a ring buffer is done */
  int slot, mtp, light, free;
  int off[MK_NSETS]; /* a run of this kind had a kernel the build does not have */
  void *dring[MK_RING]; /* device lists (one per ring buffer: a launch reads its own) */
  u32 *ws;           /* device: barrier counter, abort flag */
  volatile u32 *hflag; /* pinned: the abort flag after the last launch */
  u32 n, base, pf, pfall, join;
  uint64_t joined;
  uint64_t launches, cmds;
  /* IE_MK_PROF=1: per command the clock at its end; per kernel the sum */
  unsigned long *dprof, *hprof;
  int kof[MK_CAP];
  double pus[K_N];
  uint32_t pcnt[K_N], psteps;
  /* IE_MK_TRACE=FILE: the timeline of one launch (IE_MK_TRACE_AT: which,
   * default 10) per workgroup, to FILE (Chrome trace, prof/sm_profiler.h) */
  smp_buffer_t trace;
  const char *trace_file;
  uint64_t trace_at;
} mk;

/* The prefetch of the barrier after command i: the first bytes (up to
 * mk.pf, IE_MK_PF MB, default 16) of the weights of the next GEMV command j;
 * all the barriers before j share the claim cursor of j. */
static void mk_prefetch(void) {
  for (u32 i = 0; i + 1u < mk.n; i++) {
    const int ki = mk.kof[i];
    if (mk.pfall == 0 && (ki == K_GEMV_Q4K || ki == K_GEMV_Q6K || ki == K_GEMV_Q8)) continue; /* the tail of a GEMV: memory is busy */
    u32 j = i + 1u;
    for (; j < mk.n; j++) {
      const int k = mk.kof[j];
      if (k == K_GEMV_Q4K || k == K_GEMV_Q6K || k == K_GEMV_Q8) break;
    }
    if (j == mk.n || !mk.pf) continue;
    unsigned char *c = mk.cmd + (size_t)j * MK_CMD;
    uint64_t qw;
    u32 rows, nb;
    memcpy(&qw, c + 16, 8), memcpy(&rows, c + 48, 4), memcpy(&nb, c + 52, 4);
    uint64_t bytes = (uint64_t)rows * nb * (mk.kof[j] == K_GEMV_Q8 ? 32u : 16u);
    /* an s_prefetch_data reaches past its address: stay 64 KB inside the matrix */
    bytes = bytes > 65536u ? bytes - 65536u : 0u;
    if (bytes > mk.pf) bytes = mk.pf;
    unsigned char *d = mk.cmd + (size_t)i * MK_CMD;
    const u32 w[4] = {(u32)qw, (u32)(qw >> 32), (u32)bytes, j};
    memcpy(d + 232, w, sizeof w);
  }
}

/* A strided data range: n pieces of len bytes, piece t at base + t stride
 * (n = 1: one interval). */
typedef struct { uint64_t base, len, stride, n; } mk_sr;
typedef struct { mk_sr r, w, b, s; } mk_rng; /* activations read, result written, bias, residual */
static mk_sr mk_sr1(uint64_t base, uint64_t len, uint64_t stride, uint64_t n) {
  mk_sr x = {base, base ? len : 0u, stride, n ? n : 1u};
  return x;
}
static int mk_ivl(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1) { return a0 < b1 && b0 < a1; }
/* 1: the ranges may overlap. Exact for one interval each and for two ranges
 * with one stride whose pieces fit in it; else the hull intervals. */
static int mk_ovl(const mk_sr *x, const mk_sr *y) {
  if (!x->len || !y->len) return 0;
  const uint64_t x1 = x->base + (x->n - 1u) * x->stride + x->len, y1 = y->base + (y->n - 1u) * y->stride + y->len;
  if (!mk_ivl(x->base, x1, y->base, y1)) return 0;
  if (x->n > 1u && x->stride == y->stride && x->len <= x->stride && y->len <= y->stride) {
    const uint64_t S = x->stride, d = (y->base >= x->base ? y->base - x->base : x->base - y->base) % S;
    const uint64_t lo = y->base >= x->base ? x->len : y->len, hi = y->base >= x->base ? y->len : x->len;
    /* the later piece starts d after the earlier one, in every frame */
    return !(lo <= d && d + hi <= S);
  }
  return 1;
}
/* The data ranges of a GEMV command (MAT_Q4K / Q6K / Q8 or F32, one token
 * or several). 0: not a GEMV of these, or with the fused RMSNorm (its counter). */
static int mk_gemv_rng(u32 i, mk_rng *g) {
  const int k = mk.kof[i];
  const unsigned char *c = mk.cmd + (size_t)i * MK_CMD + 16;
  uint64_t p[8];
  u32 rows, n2, t[4];
  memset(g, 0, sizeof *g);
  if (k == K_GEMV_Q4K || k == K_GEMV_Q6K || k == K_GEMV_Q8 || k == K_GEMV_Q5K || k == K_GEMV_IQ4XS) {
    /* qw qs xq y rows nb bias res nw ... */
    memcpy(p, c, 32), memcpy(&rows, c + 32, 4), memcpy(&n2, c + 36, 4), memcpy(p + 4, c + 40, 24);
    if (p[6]) return 0;
    g->r = mk_sr1(p[2], (uint64_t)n2 * 40u, 0, 1), g->w = mk_sr1(p[3], (uint64_t)rows * 4u, 0, 1);
    g->s = mk_sr1(p[5], (uint64_t)rows * 4u, 0, 1);
  } else if (k == K_GEMV_F32) {
    /* w x y rows cols bias res nw ... */
    memcpy(p, c, 24), memcpy(&rows, c + 24, 4), memcpy(&n2, c + 28, 4), memcpy(p + 4, c + 32, 24);
    if (p[6]) return 0;
    g->r = mk_sr1(p[1], (uint64_t)n2 * 4u, 0, 1), g->w = mk_sr1(p[2], (uint64_t)rows * 4u, 0, 1);
    g->s = mk_sr1(p[5], (uint64_t)rows * 4u, 0, 1);
  } else if (k == K_GEMV_Q4K_TR || k == K_GEMV_Q6K_TR || k == K_GEMV_Q8_TR) {
    /* qw qs xq y rows nb bias res T xs ys rs: T tokens; xs in bytes, ys and rs in floats */
    memcpy(p, c, 32), memcpy(&rows, c + 32, 4), memcpy(&n2, c + 36, 4), memcpy(p + 4, c + 40, 16), memcpy(t, c + 56, 16);
    g->r = mk_sr1(p[2], (uint64_t)n2 * 40u, t[1], t[0]), g->w = mk_sr1(p[3], (uint64_t)rows * 4u, t[2] * 4ull, t[0]);
    g->s = mk_sr1(p[5], (uint64_t)rows * 4u, t[3] * 4ull, t[0]);
  } else if (k == K_GEXP) {
    /* qw qs xq y rows nb route qwn qsn astr bstr kind; grid y: the K slots (xq: astr apart, or one input; y: bstr apart) */
    u32 gy, as, bs;
    memcpy(p, c, 32), memcpy(&rows, c + 32, 4), memcpy(&n2, c + 36, 4), memcpy(&as, c + 56, 4), memcpy(&bs, c + 60, 4);
    memcpy(&gy, c - 8, 4);
    g->r = mk_sr1(p[2], (uint64_t)n2 * 40u, as, as ? gy : 1u), g->w = mk_sr1(p[3], (uint64_t)rows * 4u, bs, gy);
    return 1; /* no bias, no residual; the route is read (a GEMV phase does not write it) */
  } else if (k == K_GEMV_F32_T) {
    /* w x y rows cols bias res xs ys rs (bytes); grid y: the tokens */
    u32 gy;
    memcpy(p, c, 24), memcpy(&rows, c + 24, 4), memcpy(&n2, c + 28, 4), memcpy(p + 4, c + 32, 16), memcpy(t, c + 48, 12);
    memcpy(&gy, c - 8, 4);
    g->r = mk_sr1(p[1], (uint64_t)n2 * 4u, t[0], gy), g->w = mk_sr1(p[2], (uint64_t)rows * 4u, t[1], gy);
    g->s = mk_sr1(p[5], (uint64_t)rows * 4u, t[2], gy);
  } else
    return 0;
  g->b = mk_sr1(p[4], (uint64_t)rows * 4u, 0, 1);
  return 1;
}
/* Command i joins the phase of the commands ph .. i - 1 (no barrier before
 * it: IE_MK_JOIN, default on) when all are GEMVs and no range i writes is
 * read or written by the phase and no range i reads is written by it. */
static void mk_join(void) {
  u32 ph = 0;
  for (u32 i = 1; i < mk.n; i++) {
    unsigned char *c = mk.cmd + (size_t)i * MK_CMD;
    mk_rng gi, gj;
    int ok = mk.join && mk_gemv_rng(i, &gi);
    for (u32 j = ph; ok && j < i; j++) {
      if (!mk_gemv_rng(j, &gj)) ok = 0;
      else if (mk_ovl(&gi.w, &gj.r) || mk_ovl(&gi.w, &gj.w) || mk_ovl(&gi.w, &gj.b) || mk_ovl(&gi.w, &gj.s) ||
               mk_ovl(&gj.w, &gi.r) || mk_ovl(&gj.w, &gi.b) || mk_ovl(&gj.w, &gi.s))
        ok = 0;
    }
    static int jd = -1;
    if (jd < 0) jd = getenv("IE_MK_JDEBUG") != NULL;
    if (jd && mk.launches == 2) {
      mk_rng gx;
      fprintf(stderr, "join %u %s: %s (gemv %d, phase from %u)\n", i, kname[mk.kof[i]], ok ? "joined" : "barrier", mk_gemv_rng(i, &gx), ph);
    }
    u32 f;
    memcpy(&f, c + 12, 4);
    f = (f & ~1u) | (ok ? 1u : 0u);
    memcpy(c + 12, &f, 4);
    if (!ok) ph = i;
    else mk.joined++;
  }
}

static void mk_flush(void) {
  if (!mk.n) return;
  mk_join();
  mk_prefetch();
  const u32 np = mk.np[mk.set];
  void *dc = mk.dring[mk.slot];
  HIP(H.MemcpyAsync(dc, mk.cmd, (size_t)mk.n * MK_CMD, hipMemcpyHostToDevice, cur_stream));
  uint64_t *tp = mk.trace && mk.launches == mk.trace_at ? smp_device_ptr(mk.trace) : NULL;
  void *a[] = {&dc, &mk.n, &mk.ws, &mk.base, &mk.dprof, &tp};
  HIP(H.ModuleLaunchCooperativeKernel(mk.f[mk.set], np, 1, 1, 256, 1, 1, 0, cur_stream, a));
  HIP(H.EventRecord(mk.ev[mk.slot], cur_stream)); /* the copy and the launch are done */
  HIP(H.MemcpyAsync((void *)mk.hflag, mk.ws + 1, 4, hipMemcpyDeviceToHost, cur_stream));
  u32 nbar = 0;
  for (u32 i = 1; i < mk.n; i++) {
    u32 f;
    memcpy(&f, mk.cmd + (size_t)i * MK_CMD + 12, 4);
    nbar += (f & 1u) == 0u;
  }
  mk.base += np * nbar;
  mk.launches++, mk.cmds += mk.n;
  if (tp) {
    smp_print_stats(mk.trace, stderr);
    smp_export(mk.trace, mk.trace_file);
    smp_destroy(mk.trace), mk.trace = NULL;
  }
  if (mk.dprof) { /* the first command: from its predecessor's end is not known; skip it */
    HIP(H.Memcpy(mk.hprof, mk.dprof, (size_t)mk.n * 8u, hipMemcpyDeviceToHost));
    for (u32 i = 1; i + 1u < mk.n; i++) mk.pus[mk.kof[i]] += (mk.hprof[i] - mk.hprof[i - 1]) * 0.01, mk.pcnt[mk.kof[i]]++;
    mk.psteps++;
  }
  mk.n = 0;
  /* the next list goes to the next ring buffers, once their last launch is done */
  mk.slot = (mk.slot + 1) % MK_RING;
  HIP(H.EventSynchronize(mk.ev[mk.slot]));
  mk.cmd = mk.ring[mk.slot];
}
/* After a run (its results read with a blocking copy: every launch is done):
 * stop if a grid barrier of the megakernel timed out. */
static void mk_check(u32 pos) {
  if (mk.launches && *mk.hflag) ie_die("megakernel: a grid barrier timed out (run at position %u)", pos);
}
/* The MTP build of the megakernel has the multi-token GEMVs with 4 rows per
 * wave (ie_gemv_q4kq8_tr, ie_gemv_q6kq8_tr; bitwise equal to the others):
 * the 8-row ones (ts2, tr8) need about 192 VGPRs alone, and with them the
 * megakernel spills (IE_MK_LIGHT=0: keep them, launched apart). */
/* the builds (the sets of tools/mk_gen.py, in order), next to the kernels */
static const char *const mk_file[MK_NSETS] = {"ie_mk.hsaco", "ie_mk_mtp.hsaco", "ie_mk_flint.hsaco", "ie_mk_flintl.hsaco"};
static const char *const mk_setname[MK_NSETS] = {"decode", "verify", "flint decode", "flint decode (long)"};
static int mk_light(void) { return mk.rec && mk.set == MKSET_MTP && mk.light; }
/* 1: the launch went to the list */
static int mk_put(int k, unsigned gx, unsigned gy, unsigned lds_bytes, void **args) {
  if (!mk.rec) return 0;
  if (lds_bytes || mk.id[k] < 0 || !mk_in_set[mk.set][mk.id[k]]) {
    /* a kernel the build does not have: the runs of this kind go back to
     * launches (a list cut at every such kernel is slower than launches) */
    if (!mk.off[mk.set])
      fprintf(stderr, "gpu: megakernel off for %s runs (%s is not in it)\n", mk_setname[mk.set],
              kname[k]);
    mk.off[mk.set] = 1;
    mk_flush();
    return 0;
  }
  if (mk.n == MK_CAP) mk_flush();
  const int id = mk.id[k];
  mk.kof[mk.n] = k;
  unsigned char *c = mk.cmd + (size_t)mk.n * MK_CMD;
  memset(c, 0, MK_CMD);
  /* bit 1: the waves need not meet between workgroup bodies (no LDS, no
   * workgroup barrier): the GEMVs without the fused norm, the multi-token
   * GEMVs, the element-wise kernels */
  const int fk = k == K_GEMV_Q4K || k == K_GEMV_Q6K || k == K_GEMV_Q8 || k == K_GEMV_Q4 || k == K_GEMV_Q5K || k == K_GEMV_IQ4XS;
  const int fr = (fk && !*(void **)args[8]) || k == K_GEMV_Q4K_TR || k == K_GEMV_Q6K_TR || k == K_GEMV_Q8_TR ||
                 k == K_SWIGLU || k == K_EMBED_Q8 || k == K_EMBED_Q4 || k == K_EMBED_F32 || k == K_ROPE_KV || k == K_QUANT ||
                 k == K_GEXP || k == K_MOE_SUM || k == K_MOE_ROUTE; /* flint: no LDS, no workgroup barrier */
  const u32 h[4] = {(u32)id, gx, gy, fr && mk.free ? 2u : 0u};
  memcpy(c, h, sizeof h);
  size_t off = 16;
  for (int i = 0; i < mk_nargs[id]; i++) {
    const size_t sz = mk_argsz[id][i];
    off = (off + sz - 1) / sz * sz;
    if (off + sz > MK_ARGEND) ie_die("megakernel: the arguments of %s do not fit", kname[k]);
    memcpy(c + off, args[i], sz);
    off += sz;
  }
  mk.n++;
  return 1;
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
  /* The megakernel (default; IE_MK=0: off): decode steps (ie_mk.hsaco) and
   * verify runs (ie_mk_mtp.hsaco) through it; the files next to the kernels. IE_MK_NP,
   * IE_MK_NP_MTP: the workgroups, default 128: two per CU, the most that are
   * resident with 192 VGPRs and 14 KB of LDS (the hardware places a
   * workgroup in one CU: the occupancy query, per WGP, says 4) */
  /* on by default; IE_MK=0: off. Without the two files: off, with a note. */
  const char *mke = getenv("IE_MK");
  int mkon = !(mke && mke[0] == '0');
  if (mkon) {
    const char *sl = strrchr(hsaco, '/');
    const size_t dl = sl ? (size_t)(sl - hsaco + 1) : 0;
    for (int st = 0; st < MK_NSETS && mkon; st++) {
      char *path = malloc(dl + 32);
      memcpy(path, hsaco, dl);
      strcpy(path + dl, mk_file[st]);
      FILE *mf = fopen(path, "rb");
      if (!mf) fprintf(stderr, "gpu: megakernel off (no %s; IE_MK=0 to turn it off quietly)\n", path), mkon = 0;
      else fclose(mf);
      free(path);
    }
  }
  if (mkon) {
    const char *sl = strrchr(hsaco, '/');
    const size_t dl = sl ? (size_t)(sl - hsaco + 1) : 0;
    for (int st = 0; st < MK_NSETS; st++) {
      char *path = malloc(dl + 32);
      memcpy(path, hsaco, dl);
      strcpy(path + dl, mk_file[st]);
      FILE *mf = fopen(path, "rb");
      if (!mf) ie_die("IE_MK: cannot open %s (make %s)", path, path);
      fseek(mf, 0, SEEK_END);
      const long ms = ftell(mf);
      fseek(mf, 0, SEEK_SET);
      char *mi = malloc((size_t)ms);
      if (fread(mi, 1, (size_t)ms, mf) != (size_t)ms) ie_die("cannot read %s", path);
      fclose(mf);
      HIP(H.ModuleLoadData(&mk.mod[st], mi));
      free(mi), free(path);
      HIP(H.ModuleGetFunction(&mk.f[st], mk.mod[st], "ie_mk"));
    }
    const char *skip = getenv("IE_MK_SKIP"); /* debug: kernels launched as usual (comma list) */
    for (int i = 0; i < K_N; i++) {
      mk.id[i] = -1;
      for (int j = 0; j < MK_N; j++)
        if (!strcmp(kname[i], mk_names[j])) mk.id[i] = j;
      if (skip) {
        const char *q = strstr(skip, kname[i]);
        const size_t kl = strlen(kname[i]);
        if (q && (q[kl] == ',' || q[kl] == 0) && (q == skip || q[-1] == ',')) mk.id[i] = -1;
      }
    }
    mk.np[MKSET_DECODE] = getenv("IE_MK_NP") ? (u32)atoi(getenv("IE_MK_NP")) : 128u;
    mk.np[MKSET_MTP] = getenv("IE_MK_NP_MTP") ? (u32)atoi(getenv("IE_MK_NP_MTP")) : 128u;
    mk.np[MKSET_FLINT] = mk.np[MKSET_FLINTL] = mk.np[MKSET_DECODE];
    /* IE_MK_MTP: 3 (default) verify runs only, 1 verify runs and drafts, 2 drafts only, 0 neither
     * (measured, MTP code 7 drafts: launches 94.2, 3: 99.2, 1: 98.4 tokens/s; a draft has
     * few launches, and a cooperative launch costs more than an ordinary one) */
    mk.mtp = getenv("IE_MK_MTP") ? atoi(getenv("IE_MK_MTP")) : 3;
    mk.light = !getenv("IE_MK_LIGHT") || atoi(getenv("IE_MK_LIGHT")) != 0;
    mk.free = !getenv("IE_MK_FREE") || atoi(getenv("IE_MK_FREE")) != 0; /* 0: the waves meet before every body */
    mk.pfall = getenv("IE_MK_PFALL") != NULL;
    mk.join = !getenv("IE_MK_JOIN") || atoi(getenv("IE_MK_JOIN")) != 0;
    /* IE_MK_PF (MB): in the barriers after small kernels the waiting
     * workgroups prefetch the first bytes of the next GEMV's weights;
     * default 16 (27B decode 29.13 -> 28.98 ms/token; 0: off) */
    mk.pf = (getenv("IE_MK_PF") ? (u32)atoi(getenv("IE_MK_PF")) : 16u) << 20;
    for (int r = 0; r < MK_RING; r++) {
      HIP(H.HostMalloc((void **)&mk.ring[r], (size_t)MK_CAP * MK_CMD, 0));
      HIP(H.EventCreate(&mk.ev[r]));
    }
    mk.cmd = mk.ring[0];
    HIP(H.HostMalloc((void **)&mk.hflag, 64, 0));
    *mk.hflag = 0;
    for (int r = 0; r < MK_RING; r++) HIP(H.Malloc(&mk.dring[r], (size_t)MK_CAP * MK_CMD));
    HIP(H.Malloc((void **)&mk.ws, 64));
    HIP(H.Memset(mk.ws, 0, 64));
    if (getenv("IE_MK_PROF")) {
      HIP(H.Malloc((void **)&mk.dprof, (size_t)MK_CAP * 8u));
      mk.hprof = malloc((size_t)MK_CAP * 8u);
    }
    /* the cooperative launch fails (720) when the workgroups cannot all be
     * resident: try an empty list; halve np down to 32, else off */
    for (int st = 0; st < MK_NSETS; st++) {
      for (;;) {
        u32 zero = 0;
        void *nul = NULL;
        void *a0[] = {&mk.dring[0], &zero, &mk.ws, &mk.base, &mk.dprof, &nul};
        if (H.ModuleLaunchCooperativeKernel(mk.f[st], mk.np[st], 1, 1, 256, 1, 1, 0, cur_stream, a0) == 0) break;
        if (mk.np[st] <= 32u) {
          fprintf(stderr, "gpu: megakernel off for %s runs (no cooperative launch)\n", mk_setname[st]);
          mk.off[st] = 1;
          break;
        }
        mk.np[st] /= 2u;
      }
    }
    HIP(H.DeviceSynchronize());
    if ((mk.trace_file = getenv("IE_MK_TRACE")) && mk.trace_file[0]) {
      u32 nmax = 0;
      for (int st = 0; st < MK_NSETS; st++) nmax = mk.np[st] > nmax ? mk.np[st] : nmax;
      mk.trace_at = getenv("IE_MK_TRACE_AT") ? (uint64_t)atoll(getenv("IE_MK_TRACE_AT")) : 10u;
      if ((mk.trace = smp_create(nmax, 1, 16384, 1))) {
        for (int j = 0; j < MK_N && j < 30; j++) smp_register_event(mk.trace, (uint32_t)j, mk_names[j] + 3);
        if (MK_N > 30) smp_register_event(mk.trace, 30, "other");
        smp_register_event(mk.trace, 31, "grid barrier");
      }
    }
    mk.on = 1;
    static const char *const mkm[4] = {"off", "verify runs and drafts", "drafts", "verify runs"};
    fprintf(stderr, "gpu: megakernel on (decode: %u workgroups; MTP: %s, %u workgroups; IE_MK=0: off)\n",
            mk.np[MKSET_DECODE], mk.mtp >= 0 && mk.mtp <= 3 ? mkm[mk.mtp] : "?", mk.np[MKSET_MTP]);
  }

  HIP(H.Malloc((void **)&b->arena, g->arena));
  HIP(H.Memset(b->arena, 0, g->arena));
  HIP(H.Malloc((void **)&b->am_part, ARGMAX_GROUPS * 12u));
  /* a verify run of up to 8 tokens (7 MTP drafts) needs 8 states; with the
   * prompt lookup (src/gen.c, up to 15 drafts) 16 */
  b->ns = g->T > 4u && g->mtp_h >= 0 ? (g->T >= 16u && spec_verify_max() > 8u ? 16u : 8u) : 4u;
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
  b->attn_gq = !(getenv("IE_ATTN_SW") && atoi(getenv("IE_ATTN_SW")));
  b->flint = m->arch == ARCH_FLINT;
  b->gq_min = getenv("IE_ATTN_GQ_MIN") ? (uint32_t)atoi(getenv("IE_ATTN_GQ_MIN")) : IE_GQ_MIN;
  b->gq_ch = getenv("IE_ATTN_GQ_CH") ? (uint32_t)atoi(getenv("IE_ATTN_GQ_CH")) : 256u;
  if (b->gq_ch < 1u || b->gq_ch > 256u) ie_die("IE_ATTN_GQ_CH: 1 .. 256");
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
    size_t mk = 0; /* the largest Q4_K K (fp16 bytes per token) */
    for (uint32_t i = 0; i < g->n_ops; i++)
      if (g->ops[i].kind == OP_GEMV_Q4K && (size_t)g->ops[i].w->nb * 64u > mk) mk = (size_t)g->ops[i].w->nb * 64u;
    if (mk) {
      b->xh_bytes = mk * g->T;
      HIP(H.Malloc(&b->xh, b->xh_bytes));
    }
  }
  const size_t kv = (size_t)g->kv_total * m->n_kv * m->hd * b->kvb + 4;
  HIP(H.Malloc((void **)&b->kc, kv));
  HIP(H.Malloc((void **)&b->vc, kv));
  /* 16 linear attention states (verify runs of up to 16 tokens) only if they
   * fit with 512 MB to spare (a long context, for example 131072 with
   * Qwen3.8-27B, leaves room for 8 only); else 8 (7 drafts) */
  if (b->ns == 16u) {
    size_t fr = 0, tot = 0;
    const size_t s16 = (size_t)m->n_rec * 16u * m->n_vh * m->sd * m->sd * 4;
    if (H.MemGetInfo(&fr, &tot) == 0 && fr < s16 + ((size_t)512 << 20)) {
      b->ns = 8u;
      fprintf(stderr, "gpu: %.0f MB free: 8 linear attention states (verify runs of up to 8 tokens)\n", fr / 1e6);
    }
  }
  b->base.verify_max = b->ns > 4u ? b->ns : 1u;
  const size_t rbytes = (size_t)m->n_rec * GDN_RING * m->conv_dim * 4 + 4,
               sbytes = (size_t)m->n_rec * b->ns * m->n_vh * m->sd * m->sd * 4 + 4;
  HIP(H.Malloc((void **)&b->ring, rbytes));
  if (g->df_feat >= 0) { /* a ring of window + 32 positions per layer (the block: up to 16) */
    b->df_ring = m->df->window + 32u < g->n_ctx + 16u ? m->df->window + 32u : g->n_ctx + 16u;
    const size_t dkv = (size_t)m->df->n_layer * b->df_ring * m->df->n_kv * m->df->hd * 4u;
    HIP(H.Malloc((void **)&b->dfk, dkv));
    HIP(H.Malloc((void **)&b->dfv, dkv));
    fprintf(stderr, "gpu: dflash KV cache %.1f MB\n", 2.0 * dkv / 1e6);
  }
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
    if (o->kind == OP_ATTN && o->sk) b->d[i].w1 = upload(b, o->sk->f, o->sk->n * 4); /* flint: the sinks */
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
  const char *te = getenv("IE_TRACE");
  if (te && te[0]) {
    if (!(tr_f = fopen(te, "w"))) ie_die("cannot write %s", te);
    fprintf(tr_f, "call,pos,T_run,launch,phase,op,layer,kernel,groups,T,weight_bytes\n");
  }
  if (n_sdev) fprintf(stderr, "gpu: %u arrays (%.2f GB) streamed at load\n", n_sdev, sdev_bytes / 1e9);
  fprintf(stderr, "gpu: %u arrays uploaded, arena %u bytes, KV cache %zu bytes, linear state %zu bytes\n", b->nh, g->arena,
          2 * kv, rbytes + sbytes);
  return &b->base;
}

static void launch_lds(gpu_backend *b, int k, unsigned groups, unsigned lds_bytes, void **args) {
  tr_launch(k, groups, 1);
  if (mk_put(k, groups, 1, lds_bytes, args)) return;
  HIP(H.ModuleLaunchKernel(b->k[k], groups, 1, 1, 256, 1, 1, lds_bytes, cur_stream, args, NULL));
}
/* groups x T workgroups: grid dimension y is the token (kernels with byte strides) */
static void launch_t(gpu_backend *b, int k, unsigned groups, unsigned T, void **args) {
  tr_launch(k, groups, T);
  if (mk_put(k, groups, T, 0, args)) return;
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
/* The row of position p in the KV cache: entry kv_off of attention layer kvi, slot p % kv_len. */
#define KVROW(kvi, p) ((size_t)g->kv_off[(kvi)] + (p) % g->kv_len[(kvi)])
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
          const u32 step = T <= 8u && !mk_light() ? 8u : 16u; /* <= 8 tokens (verify): 8 rows per wave */
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
            /* IE_GEMM_H=1: ie_gemm_q4k_h (fp16 activations, the weights dequantized to fp16). Off by default:
             * not faster than ie_gemm_q4kr (512 tokens: 34816 x 5120 1.982 vs 1.957 ms; 5120 x 17408 1.269 vs 1.006) */
            static int gh = -1;
            if (gh < 0) gh = getenv("IE_GEMM_H") && getenv("IE_GEMM_H")[0] == '1';
            if (gh && b->xh && rows % 64u == 0u && (size_t)T * nb * 64u <= b->xh_bytes) {
              void *X = b->xh;
              u32 hsb = nb * 64u, hs = nb * 32u; /* fp16 row: bytes, halves */
              void *ca[] = {&A, &X, &nb, &xs, &hsb};
              launch_t(b, K_Q8_F16, (nb * 32u + 255u) / 256u, T, ca);
              void *ga[] = {&w0, &w1, &X, &B, &rows, &nb, &bias, &R, &T, &hs, &ys, &rs};
              static int ht = -1; /* IE_GEMM_HT=256|512: tokens per workgroup (default: by the shape) */
              if (ht < 0) ht = getenv("IE_GEMM_HT") ? atoi(getenv("IE_GEMM_HT")) : 0;
              const int t256 = ht ? ht == 256 : rows < 16384u;
              if (t256) launch(b, K_GEMM_Q4K_H256, (rows / 64u) * ((T + 255u) / 256u), ga);
              else launch(b, K_GEMM_Q4K_H, (rows / 64u) * ((T + 511u) / 512u), ga);
              break;
            }
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
            if (mk_light()) { /* the megakernel: 4 rows per wave (fewer registers) */
              launch(b, K_GEMV_Q4K_TR, gr >= 253u && gr <= 256u ? 257u : gr, ta);
            } else if (tn >= 6u && tn <= 8u) { /* 8 rows per wave, the tokens split over 2 waves: more waves in flight */
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
      case OP_GEMV_Q5K:
      case OP_GEMV_IQ4XS: {
        if (T > 1u) ie_die("Q5_K / IQ4_XS GEMV: one token per launch");
        u32 rows = o->w->rows, nb = o->w->nb;
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &bias, &R, &NW, &NY, &NQ, &eps, &b->gm_count};
        launch(b, o->kind == OP_GEMV_Q5K ? K_GEMV_Q5K : K_GEMV_IQ4XS, gemv_groups(rows), args);
        break;
      }
      case OP_MOE_ROUTE: {
        u32 K = o->nh;
        float sc = m->escale;
        void *EB = b->d[i].w0; /* the router bias (o->v), or NULL */
        void *args[] = {&A, &n, &K, &EB, &sc, &B};
        launch(b, K_MOE_ROUTE, 1, args);
        break;
      }
      case OP_GEMV_EXP: {
        const mat *w = o->w;
        if (w->kind != MAT_Q8 && w->kind != MAT_Q4K && w->kind != MAT_Q6K && w->kind != MAT_Q5K && w->kind != MAT_IQ4XS)
          ie_die("expert GEMV: kind %d has no GPU kernel", w->kind);
        u32 rows = w->rows, nb = w->nb, as = o->astr, bs = o->bstr, kind = (u32)w->kind;
        u32 qwn = (u32)mat_qw_words(w->kind, w->rows, w->cols), qsn = (u32)mat_qs_n(w->kind, w->rows, w->cols);
        /* IE_GEXP_TWO=1: the gate/up experts (nb 64) two rows per wave too. Off by
         * default: measured slower (flint UD-Q4_K_XL: 1.45 vs 1.26 ms per token in ie_gexp) */
        static int two_env = -1;
        if (two_env < 0) two_env = getenv("IE_GEXP_TWO") && getenv("IE_GEXP_TWO")[0] == '1';
        u32 two_off = !two_env;
        void *args[] = {&w0, &w1, &A, &B, &rows, &nb, &C, &qwn, &qsn, &as, &bs, &kind, &two_off};
        /* nb <= 64: two rows per wave, 16 per workgroup (ie_gexp) */
        const unsigned gx = nb <= 16u || (nb <= 64u && !two_off) ? (rows + 15u) / 16u : gemv_groups(rows);
        launch_t(b, K_GEXP, gx, o->nh, args);
        break;
      }
      case OP_MOE_SWIGLU: { /* per slot: ie_swiglu with the slot strides (grid y = slot) */
        void *Bu = (char *)A + (size_t)n * 4u;
        u32 as = o->astr, qs = (u32)q8_bytes(n / 32u);
        void *args[] = {&A, &Bu, &n, &Q, &as, &qs};
        launch_t(b, K_SWIGLU, elem_groups, o->nh, args);
        break;
      }
      case OP_MOE_SUM: {
        u32 K = o->nh;
        void *args[] = {&A, &B, &C, &R, &n, &K};
        launch(b, K_MOE_SUM, elem_groups, args);
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
      case OP_COPY: {
        u32 as = g->bufs[o->a].stride / 4u, bs = g->bufs[o->b].stride / 4u;
        void *args[] = {&A, &B, &n, &as, &bs};
        launch_t(b, K_COPY_ROWS, elem_groups, T, args);
        break;
      }
      case OP_DF_CONV: {
        const dflash *d = m->df;
        u32 which = o->nh, gs = d->group, as = g->bufs[o->a].stride / 4u, bs = g->bufs[o->b].stride / 4u,
            ds = g->bufs[o->c].stride / 4u, rs = o->res >= 0 ? g->bufs[o->res].stride / 4u : 0u;
        void *args[] = {&A, &B, &C, &w0, &R, &n, &which, &gs, &as, &bs, &ds, &rs};
        launch_t(b, K_DF_CONV, elem_groups, T, args);
        break;
      }
      case OP_DF_QKV: {
        const dflash *d = m->df;
        const size_t lofs = (size_t)o->layer * b->df_ring * d->n_kv * d->hd;
        float *kc = b->dfk + lofs, *vc = b->dfv + lofs;
        u32 xs = g->bufs[o->a].stride / 4u, nh = o->nh, nk = d->n_kv, dh = d->hd, ring = b->df_ring;
        float de = d->eps, rb = d->rope_base;
        void *qw = w0;
        void *args[] = {&A, &xs, &qw, &NW, &kc, &vc, &nh, &nk, &dh, &pos, &de, &rb, &ring};
        launch_t(b, K_DF_QKV, nh + 2u * nk, T, args);
        break;
      }
      case OP_DF_ATTN: {
        const dflash *d = m->df;
        const size_t lofs = (size_t)o->layer * b->df_ring * d->n_kv * d->hd;
        float *kc = b->dfk + lofs, *vc = b->dfv + lofs;
        u32 qs = g->bufs[o->a].stride / 4u, os = g->bufs[o->b].stride / 4u, nh = o->nh, nk = d->n_kv, dh = d->hd, win = d->window,
            ring = b->df_ring;
        void *args[] = {&A, &qs, &B, &os, &kc, &vc, &pos, &T, &nh, &nk, &dh, &win, &ring};
        launch_t(b, K_DF_ATTN, nh, T, args);
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
        const size_t base = KVROW(m->l[o->layer].kvi, pos) * kvd;
        void *kc = (char *)b->kc + base * b->kvb, *vc = (char *)b->vc + base * b->kvb;
        u32 nq = o->nh, neox = m->rope == ROPE_NEOX;
        void *args[] = {&A, &B, &C, &kc, &vc, &cs, &sn, &nq, (void *)&nkv, (void *)&hd, &neox};
        launch(b, K_ROPE_KV, ((nq + nkv) * h2 + kvd + 255u) / 256u, args);
        break;
      }
      case OP_QKN_ROPE_KV: {
        float *cs = b->rcos + (size_t)pos * h2, *sn = b->rsin + (size_t)pos * h2;
        const size_t base = KVROW(m->l[o->layer].kvi, pos) * kvd;
        void *kc = (char *)b->kc + base * b->kvb, *vc = (char *)b->vc + base * b->kvb;
        if (m->arch == ARCH_FLINT) { /* one token: q heads of qstr floats, no RoPE for a norope layer */
          if (T > 1u) ie_die("flint: QK norm of one token per launch");
          u32 nq = o->nh, nrot = o->norope ? 0u : m->n_rot, qstr = o->qstr;
          void *args[] = {&A, &B, &kc, &vc, &w0, &NW, &cs, &sn, &nq, (void *)&nkv, (void *)&hd, &nrot, &eps, &qstr};
          launch(b, K_QKN_ROPE_KV2, nq + 2u * nkv, args);
          break;
        }
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
        if (T <= 8u || (wfrom == 0u && T <= ns)) { /* decode and verify: one kernel, the norm fused */
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
        const size_t base = KVROW(m->l[o->layer].kvi, pos) * kvd;
        void *kc = (char *)b->kc + base * b->kvb, *vc = (char *)b->vc + base * b->kvb;
        void *args[] = {&A, &B, &kc, &vc, &n};
        launch(b, K_KV, elem_groups, args);
        break;
      }
      case OP_ATTN: {
        const size_t base = (size_t)g->kv_off[m->l[o->layer].kvi] * kvd;
        void *kc = (char *)b->kc + base * b->kvb, *vc = (char *)b->vc + base * b->kvb;
        if (m->arch == ARCH_FLINT) { /* one token: window, ring, sinks (ie_attn_sw) */
          if (T > 1u) ie_die("flint: attention of one token per launch");
          u32 len = g->kv_len[m->l[o->layer].kvi], p0 = o->win && pos + 1u > o->win ? pos + 1u - o->win : 0u, np = pos + 1u - p0;
          /* ie_attn_gq (the q heads of a kv head together, chunks of 256
           * positions) when it can; IE_ATTN_SW=1: ie_attn_sw */
          const int gq = gq_use(b, pos) && nhd / nkv <= 8u && hd <= 128u && hd % 8u == 0u;
          u32 ch = gq ? b->gq_ch : ie_att_ch(pos), nsplit = (np + ch - 1u) / ch;
          void *SK = b->d[i].w1; /* the sinks (gpu_open), or NULL */
          void *args[] = {&A, &kc, &vc, &C, &B, &p0, &np, &len, (void *)&hd, (void *)&nhd, (void *)&nkv, &nsplit, &ch,
                          &b->at_count, &Q, &SK};
          launch(b, gq ? K_ATTN_GQ : K_ATTN_SW, (gq ? nkv : nhd) * nsplit, args);
          break;
        }
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
    if (tr_f) tr_op = (int)i, tr_layer = g->ops[i].layer, tr_wb = op_wbytes(&g->ops[i]), tr_first = 1;
    if (chunk && i >= g->i_head && i < g->n_main) { /* last token only (its row T - 1) */
      launch_op(b, i, T - 1, toks[T - 1], pos + T - 1, 1, slot, wfrom);
      continue;
    }
    if (chunk && g->mtp_logits >= 0 && i >= g->i_mtp_head && i < g->n_mtp) continue;
    if (T > 1 && g->mtp_logits >= 0 && i >= g->i_mtp_head && i < g->n_mtp) { /* MTP catch-up: only the last token's draft is used */
      launch_op(b, i, T - 1, toks[T - 1], pos + T - 1, 1, slot, wfrom);
      continue;
    }
    static int nobatch = -1;
    if (nobatch < 0) nobatch = getenv("IE_GEMV_NOBATCH") != NULL; /* debug: one GEMV launch per token */
    /* one launch for all tokens: GDN (sequential inside), Q8 GEMV (weights
     * read once), and the per-token kernels with a token grid dimension */
    static int gmin = -1;
    if (gmin < 0) gmin = getenv("IE_GEMM_MIN") ? atoi(getenv("IE_GEMM_MIN")) : 17;
    const int all = b->m->arch == ARCH_FLINT ? 0 : /* flint: every op one token at a time (for now) */
                    k == OP_GDN || ((k == OP_GEMV_Q8 || k == OP_GEMV_Q4K || k == OP_GEMV_Q6K || k == OP_GEMV_F32) && !nobatch) || k == OP_RMSNORM || k == OP_QUANT || k == OP_SWIGLU ||
                    k == OP_QKN_ROPE_KV || k == OP_COPY || k == OP_DF_CONV || k == OP_DF_QKV || k == OP_DF_ATTN || (k == OP_ATTN && (int)T >= gmin) /* the fast (not bitwise) prefill path */
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
  tr_run("decode", pos, 1);
  mk.set = b->flint ? (gq_use(b, pos) ? MKSET_FLINTL : MKSET_FLINT) : MKSET_DECODE;
  mk.rec = mk.on && !mk.off[mk.set];
  run_ops(b, 0, g->n_main, &tok, 1, pos, 0); /* state: slot -> slot */
  mk.rec = 0;
  mk_flush();
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
  mk_check(pos);
  return r;
}

static void gpu_run(backend *bk, int sec, const uint32_t *toks, uint32_t T, uint32_t pos, int last_only, uint32_t *out,
                    float *prob) {
  gpu_backend *b = (gpu_backend *)bk;
  const graph *g = b->g;
  if (sec == 1 && g->mtp_argmax < 0) ie_die("the graph has no MTP ops");
  if (sec == 4) { /* the MTP layer only (its KV cache), not its output head */
    tr_run("mtp_kv", pos, T);
    run_ops(b, g->n_main, g->i_mtp_head, toks, T, pos, 0);
    (void)last_only, (void)out, (void)prob;
    return;
  }
  if (sec >= 2) { /* DFlash: 2 the context ops, 3 the block ops (launches; the host reads df_logits / df_sel) */
    if (g->df_feat < 0) ie_die("the graph has no DFlash ops");
    tr_run(sec == 2 ? "df_ctx" : "df_block", pos, T);
    run_ops(b, sec == 2 ? g->i_dfc : g->i_dfb, sec == 2 ? g->i_dfb : g->n_ops, toks, T, pos, 0);
    (void)last_only, (void)out, (void)prob;
    return;
  }
  /* IE_PROFILE=2: the ops of the prompt chunks (ms per chunk in the report) */
  const int pc = b->prof == 2 && !sec && T > 1u;
  if (pc) HIP(H.EventRecord(b->e0, NULL));
  tr_run(sec ? "mtp" : last_only && T > 1u ? "prefill" : "verify", pos, T);
  /* verify runs and MTP drafts (not prompt chunks): the MTP build of the
   * megakernel (IE_MK_MTP=0: off) */
  mk.rec = mk.on && !mk.off[MKSET_MTP] && mk.mtp && !(last_only && T > 1u) && (mk.mtp == 1 || (mk.mtp == 2) == (sec != 0)),
  mk.set = MKSET_MTP;
  run_ops(b, sec ? g->n_main : 0, sec ? g->n_mtp : g->n_main, toks, T, pos, last_only ? T - 1 : 0);
  mk.rec = 0;
  mk_flush();
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
  mk_check(pos);
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
  /* a ring slot is reused by a later position: keep the rings (K and V) */
  const graph *g = b->g;
  const size_t kvb = (size_t)m->n_kv * m->hd * b->kvb;
  size_t nr = 0;
  for (uint32_t i = 0; i < m->n_kvl; i++)
    if (g->kv_len[i] < g->n_ctx) nr += g->kv_len[i];
  if (!nr) return;
  if (!b->snap_kv[k]) HIP(H.Malloc((void **)&b->snap_kv[k], 2 * nr * kvb));
  char *sp = b->snap_kv[k];
  for (uint32_t i = 0; i < m->n_kvl; i++) {
    if (g->kv_len[i] >= g->n_ctx) continue;
    const size_t n = g->kv_len[i] * kvb, o = g->kv_off[i] * kvb;
    HIP(H.Memcpy(save ? (void *)sp : (char *)b->kc + o, save ? (char *)b->kc + o : (void *)sp, n, hipMemcpyDeviceToDevice));
    sp += n;
    HIP(H.Memcpy(save ? (void *)sp : (char *)b->vc + o, save ? (char *)b->vc + o : (void *)sp, n, hipMemcpyDeviceToDevice));
    sp += n;
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
  tr_run("sample", 0, n);
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
    case OP_ATTN: return m->arch == ARCH_FLINT ? (b->attn_gq && b->gq_min <= 1u ? K_ATTN_GQ : K_ATTN_SW) : b->attn_kv ? K_ATTN_KV : b->attn_split ? K_ATTN_SPLIT : K_ATTN;
    case OP_GEMV_Q5K: return K_GEMV_Q5K;
    case OP_GEMV_IQ4XS: return K_GEMV_IQ4XS;
    case OP_MOE_ROUTE: return K_MOE_ROUTE;
    case OP_MOE_SUM: return K_MOE_SUM;
    case OP_MOE_SWIGLU: return K_SWIGLU;
    case OP_GEMV_EXP: return K_GEXP;
    default: return K_ARGMAX;
  }
}

/* Weight bytes that op o reads (0 if it reads no weight matrix). */
static double op_wbytes(const op *o) {
  if (o->kind == OP_GEMV_Q4 || o->kind == OP_GEMV_Q4K) return (double)o->w->rows * o->w->nb * 18.0;
  if (o->kind == OP_GEMV_Q6K) return (double)o->w->rows * o->w->nb * 26.25;
  if (o->kind == OP_GEMV_Q8) return (double)o->w->rows * o->w->nb * 34.0;
  if (o->kind == OP_GEMV_F32) return (double)o->w->rows * o->w->cols * 4.0;
  if (o->kind == OP_GEMV_Q5K) return (double)o->w->rows * o->w->nb * 22.0;
  if (o->kind == OP_GEMV_IQ4XS) return (double)o->w->rows * o->w->nb * 17.0;
  if (o->kind == OP_GEMV_EXP) {
    const int k = o->w->kind;
    const double per = k == MAT_Q8 ? 34.0 : k == MAT_Q5K ? 22.0 : k == MAT_IQ4XS ? 17.0 : k == MAT_Q6K ? 26.25 : 18.0;
    return (double)o->w->rows * o->w->nb * per * o->nh;
  }
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
  if (mk.on && mk.psteps) {
    double tot = 0;
    for (int k = 0; k < K_N; k++) tot += mk.pus[k];
    fprintf(stderr, "megakernel: %.1f commands joined per launch\n", (double)mk.joined / mk.launches);
    fprintf(stderr, "megakernel profile: %u launches, %.3f ms per launch (commands 2 .. n-1, barrier included)\n",
            mk.psteps, tot / 1e3 / mk.psteps);
    for (int k = 0; k < K_N; k++)
      if (mk.pcnt[k])
        fprintf(stderr, "  %-18s %5.1f per launch %9.2f us each %8.3f ms per launch\n", kname[k],
                (double)mk.pcnt[k] / mk.psteps, mk.pus[k] / mk.pcnt[k], mk.pus[k] / 1e3 / mk.psteps);
  }
  if (tr_f) fprintf(stderr, "trace: wrote %u launches in %u runs\n", tr_n, tr_call), fclose(tr_f), tr_f = NULL;
  HIP(H.DeviceSynchronize());
  for (uint32_t i = 0; i < b->nh; i++) H.Free(b->hv[i]);
  for (uint32_t i = 0; i < n_sdev; i++) H.Free(sdev[i]);
  free(sdev), sdev = NULL, n_sdev = cap_sdev = 0;
  H.Free(b->arena), H.Free(b->kc), H.Free(b->vc), H.Free(b->ring), H.Free(b->st);
  for (int k = 0; k < 2; k++) {
    if (b->snap_ring[k]) H.Free(b->snap_ring[k]), H.Free(b->snap_st[k]), H.Free(b->snap_h[k]);
    if (b->snap_kv[k]) H.Free(b->snap_kv[k]);
  }
  H.Free(b->am_part), H.Free(b->am_count), H.Free(b->at_count), H.Free(b->at_part), H.Free(b->tk_buf), H.Free(b->gm_count);
  if (b->prof) {
    for (uint32_t i = 0; i < b->g->n_ops; i++) H.EventDestroy(b->pev[i]);
    free(b->pev), free(b->pms);
  }
  free(b->hk), free(b->hv), free(b->d), free(b);
}
