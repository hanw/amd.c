/* demo_kernel.c -- prof/sm_profiler.h in a C kernel (clang, amdgcn):
 * per wave a compute phase, a memory phase, a workgroup barrier and an
 * instant event, as the CUDA demo of leepoly/sm-profiler. */
#include "sm_profiler.h"

#define G __attribute__((address_space(1)))
#define EVT_COMPUTE 0u
#define EVT_MEMORY 1u
#define EVT_SYNC 2u
#define EVT_BARRIER 3u

__attribute__((amdgpu_kernel, visibility("protected"))) void demo(G uint64_t *prof, G float *data, uint32_t n,
                                                                    uint32_t iters) {
  const int l0 = smp_lane() == 0u;
  const uint32_t t = __builtin_amdgcn_workitem_id_x(), g = __builtin_amdgcn_workgroup_id_x();
  float sum = 0.0f;
  for (uint32_t it = 0; it < iters; it++) {
    smp_start(prof, EVT_COMPUTE, l0);
    for (uint32_t i = 0; i < 2000u + 500u * (g % 4u); i++) sum = sum * 0.999f + (float)(t + i) * 1e-3f;
    smp_end(prof, EVT_COMPUTE, l0);
    smp_start(prof, EVT_MEMORY, l0);
    for (uint32_t i = g * 256u + t; i < n; i += __builtin_amdgcn_grid_size_x()) sum += data[i];
    smp_end(prof, EVT_MEMORY, l0);
    smp_start(prof, EVT_BARRIER, l0);
    __builtin_amdgcn_s_barrier();
    smp_end(prof, EVT_BARRIER, l0);
    smp_instant(prof, EVT_SYNC, l0);
  }
  if (sum == -1.0f) data[0] = sum;
}

/* the hardware id of each workgroup (wave 0): to check the WGP decoding */
__attribute__((amdgpu_kernel, visibility("protected"))) void hwid(G uint32_t *out) {
  if (__builtin_amdgcn_workitem_id_x() == 0u) {
    out[2u * __builtin_amdgcn_workgroup_id_x()] = smp_hw_id();
    out[2u * __builtin_amdgcn_workgroup_id_x() + 1u] = smp_sm_id();
  }
}
