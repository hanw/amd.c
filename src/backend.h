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
