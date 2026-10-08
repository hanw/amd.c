/* wmma_peak_kernel.c -- raw v_wmma_i32_16x16x16_iu8 throughput: each wave
 * runs n iterations of 8 independent WMMA on registers (no memory). */
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define G __attribute__((address_space(1)))
__attribute__((amdgpu_kernel, visibility("protected"))) void wmma_peak(G int *out, unsigned n, int seed) {
  const unsigned l = __builtin_amdgcn_workitem_id_x();
  v2i a = {seed + (int)l, seed * 3}, b = {seed ^ (int)l, seed + 7};
  v8i c[8];
  for (int i = 0; i < 8; i++) c[i] = (v8i){i, 0, 0, 0, 0, 0, 0, 0};
  for (unsigned it = 0; it < n; it++)
    for (int i = 0; i < 8; i++) c[i] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a, 1, b, c[i], 0);
  int s = 0;
  for (int i = 0; i < 8; i++) s += c[i][0] + c[i][7];
  out[__builtin_amdgcn_workgroup_id_x() * 256 + l] = s;
}

/* The same with the operands read from LDS before each pair of WMMA (2 x 2
 * tiles per wave, as ie_gemm_q8): the cost of feeding WMMA from LDS. */
#define LDS __attribute__((address_space(3)))
static LDS unsigned char lds_buf[16384] __attribute__((loader_uninitialized));
__attribute__((amdgpu_kernel, visibility("protected"))) void wmma_peak_lds(G int *out, unsigned n, int seed) {
  const unsigned t = __builtin_amdgcn_workitem_id_x(), l = t & 31u;
  for (unsigned i = t; i < 16384u / 4u; i += 256u) ((LDS int *)lds_buf)[i] = seed * (int)i;
  __builtin_amdgcn_s_barrier();
  v8i c[4];
  for (int i = 0; i < 4; i++) c[i] = (v8i){i, 0, 0, 0, 0, 0, 0, 0};
  for (unsigned it = 0; it < n; it++) {
    const unsigned o = ((it * 64u) & 8191u) + (l & 15u) * 144u % 4096u + 8u * (l >> 4);
    v2i a0 = *(LDS v2i *)(lds_buf + o), a1 = *(LDS v2i *)(lds_buf + o + 16u);
    v2i b0 = *(LDS v2i *)(lds_buf + 8192u + o), b1 = *(LDS v2i *)(lds_buf + 8192u + o + 16u);
    v2i a2 = *(LDS v2i *)(lds_buf + o + 2304u), a3 = *(LDS v2i *)(lds_buf + o + 2320u);
    v2i b2 = *(LDS v2i *)(lds_buf + 8192u + o + 2304u), b3 = *(LDS v2i *)(lds_buf + 8192u + o + 2320u);
    c[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b0, c[0], 0);
    c[1] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a0, 1, b2, c[1], 0);
    c[2] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a2, 1, b0, c[2], 0);
    c[3] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a2, 1, b2, c[3], 0);
    c[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b1, c[0], 0);
    c[1] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a1, 1, b3, c[1], 0);
    c[2] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a3, 1, b1, c[2], 0);
    c[3] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a3, 1, b3, c[3], 0);
  }
  int s = 0;
  for (int i = 0; i < 4; i++) s += c[i][0] + c[i][7];
  out[__builtin_amdgcn_workgroup_id_x() * 256 + t] = s;
}
