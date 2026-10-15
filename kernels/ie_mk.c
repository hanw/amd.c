/* ie_mk.c -- the megakernel: one persistent launch runs a list of kernel
 * launches (the command list of a decode step, src/hip.c), in order.
 *
 * The kernels are those of ie_kernels.c, compiled here as functions
 * (KERNEL is a static inline function, IE_MEGA: wgid(), tokid() and
 * ngroups_x() read the current workgroup of the list from LDS). Command i:
 * 256 bytes, {kernel id, groups x, groups y, 0} then the kernel arguments in
 * launch layout (tools/mk_gen.py). Workgroup p of the np launched ones runs
 * the workgroups p, p + np, ... of command i (x fastest), then waits at a
 * grid barrier: command i + 1 starts when every workgroup of command i is
 * done, as with one launch after the other on a stream. So every kernel
 * runs the same code on the same data: the output is bitwise equal.
 *
 * The launch must be cooperative (all np workgroups resident: the grid
 * barrier needs them). ws: ws[0] the barrier counter (never reset; base:
 * its value at launch), ws[1] the abort flag (set when a wait times out:
 * then every workgroup leaves, and the host stops). prof (or NULL): per
 * command, the clock after its grid barrier (workgroup 0). trace (or
 * NULL): the per-workgroup timeline of prof/sm_profiler.h.
 * Bytes 232 .. 255 of a command: the prefetch of its barrier (the first
 * bytes of the weights of the next GEMV; src/hip.c mk_flush) and the claim
 * cursor (word 63, zero in the list the host writes). */
#define IE_MEGA 1
#pragma clang diagnostic ignored "-Wunused-function" /* the kernels a decode step does not use */
#define KERNEL static inline __attribute__((always_inline)) void /* (noinline: 192 VGPRs, but slower, and still 64 workgroups at most) */
#include "ie_kernels.c"
#include "ie_mk_gen.h"
#include "../prof/sm_profiler.h"
_Static_assert(MK_N < 31, "the profiler has 32 event types: the kernels, 31 the grid barrier");
#define MK_EV_BARRIER 31u

#define MK_CMD 256u

__attribute__((amdgpu_kernel, visibility("protected"))) void ie_mk(G u8 *cmds, u32 n, G u32 *ws, u32 base, G unsigned long *prof, G uint64_t *trace) {
  const u32 np = __builtin_amdgcn_grid_size_x() / NT, p = __builtin_amdgcn_workgroup_id_x();
  /* IE_MK_TRACE: thread 0 records the timeline of this workgroup */
  const int t0 = trace && tid() == 0u;
  smp_rec tr = {0};
  if (t0) {
    const smp_info ti = smp_parse(trace);
    tr = smp_rec_open(&ti, p, 0u);
  }
  unsigned long ts = 0;
  u32 nbar = 0, off = 0; /* off: the workgroups of the phase before command i, mod np */
  for (u32 i = 0; i < n; i++) {
    const G u32 *h = (const G u32 *)(cmds + (unsigned long)i * MK_CMD);
    const u32 kid = h[0], gx = h[1], tot = gx * h[2];
    const G u8 *a = cmds + (unsigned long)i * MK_CMD + 16u;
    /* the workgroups of a phase are dealt out in one sequence: workgroup p
     * takes the phase indices p, p + np, ... */
    /* word 3, bit 1: the kernel uses no LDS and no workgroup barrier
     * (src/hip.c mk_free): its waves go on to the next workgroup body
     * without meeting; else the workgroup meets before each body (the
     * previous body is done with its LDS) */
    const u32 fr = h[3] & 2u, w = tid() >> 5;
    for (u32 v = (p + np - off) % np; v < tot; v += np) {
      if (!fr) barrier();
      if ((tid() & 31u) == 0u) mk_vid[w][0] = v % gx, mk_vid[w][1] = v / gx, mk_vid[w][2] = gx;
      /* IE_MK_TRACE: one range per workgroup body, on the timeline of
       * workgroup p (prof/sm_profiler.h; thread 0 records) */
      if (t0) ts = smp_time_ns();
      mk_dispatch(kid, a);
      if (t0) smp_rec_range(&tr, kid, ts, smp_time_ns());
    }
    /* word 3 of the next command: 1 = it joins this phase (independent
     * GEMVs, src/hip.c mk_join): no barrier */
    const u32 join = i + 1u < n ? ((const G u32 *)(cmds + (unsigned long)(i + 1u) * MK_CMD))[3] & 1u : 0u;
    if (i + 1u < n && !join) {
      nbar++, off = 0;
      /* the prefetch of this barrier: words 58..59 the address, 60 the bytes,
       * 61 the command whose word 63 is the claim cursor */
      const unsigned long pa = ((unsigned long)h[59] << 32) | h[58];
      G u32 *cur = (G u32 *)(cmds + (unsigned long)h[61] * MK_CMD) + 63;
      if (t0) ts = smp_time_ns();
      if (!mk_grid_sync_pf(ws, base + np * nbar, cur, (const G u8 *)pa, h[60])) {
        if (t0) smp_rec_close(&tr);
        return;
      }
      if (t0) smp_rec_range(&tr, MK_EV_BARRIER, ts, smp_time_ns());
    }
    if (join) off = (off + tot) % np;
    /* IE_MK_PROF: the time (100 MHz clock) when command i is done everywhere */
    if (prof && p == 0u && tid() == 0u) prof[i] = __builtin_amdgcn_s_sendmsg_rtnl(0x83);
  }
  if (t0) smp_rec_close(&tr);
}
