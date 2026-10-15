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
 * command, the clock after its grid barrier (workgroup 0). */
#define IE_MEGA 1
#define KERNEL static inline __attribute__((always_inline)) void
#include "ie_kernels.c"
#include "ie_mk_gen.h"

#define MK_CMD 256u

__attribute__((amdgpu_kernel, visibility("protected"))) void ie_mk(const G u8 *cmds, u32 n, G u32 *ws, u32 base, G unsigned long *prof) {
  const u32 np = __builtin_amdgcn_grid_size_x() / NT, p = __builtin_amdgcn_workgroup_id_x();
  for (u32 i = 0; i < n; i++) {
    const G u32 *h = (const G u32 *)(cmds + (unsigned long)i * MK_CMD);
    const u32 kid = h[0], gx = h[1], tot = gx * h[2];
    const G u8 *a = cmds + (unsigned long)i * MK_CMD + 16u;
    for (u32 v = p; v < tot; v += np) {
      barrier(); /* the previous workgroup body is done with mk_vid and its LDS */
      if (tid() == 0u) mk_vid[0] = v % gx, mk_vid[1] = v / gx, mk_vid[2] = gx;
      barrier();
      MK_DISPATCH(kid, a);
    }
    if (i + 1u < n && !mk_grid_sync(ws, base + np * (i + 1u))) return;
    /* IE_MK_PROF: the time (100 MHz clock) when command i is done everywhere */
    if (prof && p == 0u && tid() == 0u) prof[i] = __builtin_amdgcn_s_sendmsg_rtnl(0x83);
  }
}
