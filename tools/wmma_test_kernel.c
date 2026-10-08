/* wmma_test_kernel.c -- one wave computes D = A B (16x16x16, int8 -> int32)
 * with v_wmma_i32_16x16x16_iu8, loading A, B and storing D with the lane
 * layout that ie_gemm_q8 assumes (gfx12, wave32):
 *   A (M x K, row major bytes): lane l holds row l % 16, K = 8 (l / 16) .. + 7
 *   B (K x N, given as N x K bytes, i.e. B^T row major): lane l holds column
 *     l % 16, K = 8 (l / 16) .. + 7
 *   D (M x N): lane l, element v holds row 8 (l / 16) + v, column l % 16. */
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
#define G __attribute__((address_space(1)))
__attribute__((amdgpu_kernel, visibility("protected"))) void wmma_test(const G signed char *A, const G signed char *Bt, G int *D) {
  const unsigned l = __builtin_amdgcn_workitem_id_x();
  if (l >= 32) return;
  const v2i a = *(const G v2i *)(A + (l % 16) * 16 + 8 * (l / 16));
  const v2i b = *(const G v2i *)(Bt + (l % 16) * 16 + 8 * (l / 16));
  const v8i z = {0, 0, 0, 0, 0, 0, 0, 0};
  const v8i d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(1, a, 1, b, z, 0);
  for (int v = 0; v < 8; v++) D[(8 * (l / 16) + v) * 16 + (l % 16)] = d[v];
}
