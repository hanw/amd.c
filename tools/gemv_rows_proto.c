/* gemv_rows_proto.c -- prototype for tools/gemv_bench.c: a Q4 x Q8 GEMV in
 * which each wave computes R rows (rows (g * 8 + v) * R + k, k < R). A lane
 * loads each activation block once and uses it for the R rows. Not used by
 * the engine. Build with -DR=1, 2, 4 (see the commands in the report). */
#include "../core/ie_core.h"
#define G __attribute__((address_space(1)))
#define KERNEL __attribute__((amdgpu_kernel, visibility("protected"))) void
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef _Float16 f16;
typedef unsigned char u8;
static inline u32 tid(void) { return __builtin_amdgcn_workitem_id_x(); }
static inline u32 wgid(void) { return __builtin_amdgcn_workgroup_id_x(); }
static inline int xrow_i(int v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false, false); }
#define F2I(x) __builtin_bit_cast(int, (x))
#define I2F(x) __builtin_bit_cast(float, (x))
static inline float wave_tree_f(float v) {
  v += I2F(xrow_i(F2I(v)));
  v += I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0x108, 0xF, 0xF, true));
  v += I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0x104, 0xF, 0xF, true));
  v += I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0x102, 0xF, 0xF, true));
  v += I2F(__builtin_amdgcn_update_dpp(0, F2I(v), 0x101, 0xF, 0xF, true));
  return v;
}
#ifndef R
#define R 2u
#endif
KERNEL ie_gemv_rows(const G u32 *qw, const G f16 *qs, const G u8 *xq, G float *y, u32 rows, u32 nb) {
  const u32 r0 = (wgid() * 8u + (tid() >> 5)) * R, l = tid() & 31u;
  if (r0 >= rows) return;
  const G u32 *aw = (const G u32 *)xq;
  const G float *da = (const G float *)(xq + 32u * nb);
  const G u32 *asum = (const G u32 *)(xq + 36u * nb);
  float acc[R];
  for (u32 k = 0; k < R; k++) acc[k] = 0.0f;
  for (u32 t = 0; ie_lane_blk(l, t) < nb; t++) {
    const u32 b = ie_lane_blk(l, t);
    const u32x4 a0 = *(const G u32x4 *)(aw + b * 8u), a1 = *(const G u32x4 *)(aw + b * 8u + 4u);
    const u32 av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float d = da[b];
    const u32 s = asum[b];
    u32x4 w4[R];
    float sc[R];
    for (u32 k = 0; k < R; k++) {
      const u32 r = r0 + k < rows ? r0 + k : r0;
      w4[k] = *(const G u32x4 *)(qw + ie_q4_dst_word(r, b, 0u, nb));
      sc[k] = (float)qs[ie_q4_dst_scale(r, b, nb)];
    }
    for (u32 k = 0; k < R; k++) {
      const u32 wq[4] = {w4[k].x, w4[k].y, w4[k].z, w4[k].w};
      const Mem qm = {wq}, am = {av};
      acc[k] += (sc[k] * d) * (float)(int)ie_q4q8_block(qm, 0u, am, 0u, s);
    }
  }
  for (u32 k = 0; k < R; k++) {
    const float v = wave_tree_f(acc[k]);
    if (l == 0u && r0 + k < rows) y[r0 + k] = v;
  }
}
