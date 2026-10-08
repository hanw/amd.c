/* backend.h -- the two backends run the op list of graph.h per token. */
#ifndef IE_BACKEND_H
#define IE_BACKEND_H

#include "model.h"

typedef struct backend backend;
struct backend {
  /* One decode step: token tok at position pos. Writes vocab logits (if
   * not NULL) and returns the argmax token. ms: time of the step. */
  uint32_t (*step)(backend *b, uint32_t tok, uint32_t pos, float *logits, double *ms);
  void (*close)(backend *b);
  /* GPU only (NULL on the CPU). Run section sec of the op list (0: the
   * model, 1: the MTP head) for T tokens toks at positions pos .. pos + T -
   * 1. The linear attention state after each token t is kept (or, if
   * last_only, after the last token only) until accept. out: the T argmax
   * tokens. */
  void (*run)(backend *b, int sec, const uint32_t *toks, uint32_t T, uint32_t pos, int last_only, uint32_t *out,
              float *prob); /* prob (or NULL): the softmax probability of each argmax token */
  /* The tokens 0 .. k of the last model run are final: later runs continue
   * from the linear attention state after token k. (A one-token step needs
   * no accept.) */
  void (*accept)(backend *b, uint32_t k);
  /* Copy n token rows of buffer id (from row r0) to the host: dst gets n *
   * buf.stride bytes (rows buf.stride apart). */
  void (*read_rows)(backend *b, int id, uint32_t r0, uint32_t n, float *dst);
  /* Copy n token rows of buffer src (from row r0) to buffer dst (from row
   * d0); src < 0: write zeros. */
  void (*copy_rows)(backend *b, int dst, uint32_t d0, int src, uint32_t r0, uint32_t n);
  /* 1: prompt chunks compute the logits of every token (perplexity); 0
   * (default): only of the last token */
  int all_logits;
};

/* CPU: the same algorithm and the same work split as the GPU kernels. */
backend *cpu_open(const model *m, const graph *g);
/* GPU (HIP, gfx1201): kernels from the hsaco file; exits with a message
 * if libamdhip64.so or a device is missing. */
backend *gpu_open(const model *m, const graph *g, const char *hsaco);
/* Before model_load: start HIP and set ie_mat_sink, so that each matrix
 * goes to the GPU as soon as it is repacked (host memory: one layer). */
void gpu_stream_init(void);


/* The CPU kernels, also used by tests. */
void cpu_quant_q8(const float *x, uint8_t *q, uint32_t nb);
void cpu_gemv_q4(const mat *w, const uint8_t *xq, float *y);
void cpu_gemv_q8(const mat *w, const uint8_t *xq, float *y);

#endif
