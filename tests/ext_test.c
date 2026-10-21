/* ext_test.c -- the helpers of core/ie_ext.h (not proved, so tested here):
 *  - ie_spread4 for all 16 inputs;
 *  - the GPU form of ie_iq4_map (two byte permutes and a select, with
 *    v_perm_b32 modelled here) against the table, for all 65536 inputs;
 *  - the CPU GEMV of MAT_Q5K and MAT_IQ4XS against plain loops over the
 *    GGUF bytes (the formulas of ggml's dequantize_row_q5_K / _iq4_xs),
 *    for several shapes (lanes with 1 and 2+ steps, rows not a multiple of 8). */
#include <math.h>
#include <string.h>

#include "../core/ie_core.h"
#include "../core/ie_ext.h"
#include "../src/backend.h"
#include "../src/common.h"

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) {
  rs ^= rs << 13, rs ^= rs >> 7, rs ^= rs << 17;
  return (uint32_t)rs;
}
static float frnd(void) { return (float)rnd() / 4294967296.0f * 2.0f - 1.0f; }

/* v_perm_b32(a, b, sel) for selectors 0..7: byte i = byte sel_i of {a: 7..4, b: 3..0} */
static uint32_t perm(uint32_t a, uint32_t b, uint32_t sel) {
  const uint64_t ab = ((uint64_t)a << 32) | b;
  uint32_t r = 0;
  for (int i = 0; i < 4; i++) r |= (uint32_t)((ab >> (8 * ((sel >> (8 * i)) & 7))) & 255) << (8 * i);
  return r;
}
static uint32_t iq4_map_gpu(uint32_t n4) { /* the __AMDGCN__ branch of ie_iq4_map */
  const uint32_t sel = n4 & 0x07070707u;
  const uint32_t lo = perm(IE_KV1, IE_KV0, sel), hi = perm(IE_KV3, IE_KV2, sel);
  const uint32_t m = ((n4 >> 3u) & 0x01010101u) * 255u;
  return (hi & m) | (lo & ~m);
}
static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static int scale_min(int j, const uint8_t *q, int *m) { /* ggml's get_scale_min_k4 */
  if (j < 4) return *m = q[j + 4] & 63, q[j] & 63;
  *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
}
/* weight j (< 256) of one GGUF block */
static double w_q5k(const uint8_t *b, int j) {
  uint16_t h0, h1;
  memcpy(&h0, b, 2), memcpy(&h1, b + 2, 2);
  const int is = j / 32, l = j % 32, grp = j / 64;
  int mn, sc = scale_min(is, b + 4, &mn);
  const uint8_t ql = b[48 + 32 * grp + l];
  const int q = ((is % 2) ? ql >> 4 : ql & 15) + (((b[16 + l] >> is) & 1) << 4);
  return (double)ie_f16_to_f32(h0) * sc * q - (double)ie_f16_to_f32(h1) * mn;
}
static double w_iq4xs(const uint8_t *b, int j) {
  uint16_t h, sh;
  memcpy(&h, b, 2), memcpy(&sh, b + 2, 2);
  const int ib = j / 32, l = j % 32;
  const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
  const uint8_t q = b[8 + 16 * ib + (l % 16)];
  return (double)ie_f16_to_f32(h) * (ls - 32) * kv[l < 16 ? q & 15 : q >> 4];
}

int main(void) {
  int fails = 0;
  for (uint32_t x = 0; x < 16; x++)
    for (int e = 0; e < 4; e++)
      if (((ie_spread4(x) >> (8 * e)) & 255) != ((x >> e) & 1)) fails++;
  for (uint32_t n = 0; n < 65536; n++) {
    const uint32_t n4 = (n & 15) | ((n >> 4) & 15) << 8 | ((n >> 8) & 15) << 16 | ((n >> 12) & 15) << 24;
    const uint32_t want = (uint8_t)kv[n & 15] | (uint32_t)(uint8_t)kv[(n >> 4) & 15] << 8 | (uint32_t)(uint8_t)kv[(n >> 8) & 15] << 16 |
                          (uint32_t)(uint8_t)kv[(n >> 12) & 15] << 24;
    if (ie_iq4_map(n4) != want || iq4_map_gpu(n4) != want) fails++;
  }
  printf("ext_test: ie_spread4, ie_iq4_map (table and GPU form, 65536 inputs): %s\n", fails ? "FAIL" : "pass");

  const uint32_t shapes[][2] = {{1, 256}, {7, 256}, {9, 2048}, {13, 4352}, {3, 8192}};
  const uint32_t types[2] = {GGML_Q5_K, GGML_IQ4_XS}, bsz[2] = {176, 136};
  for (int ti = 0; ti < 2; ti++) {
    int tf = 0;
    double worst = 0;
    for (unsigned s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
      const uint32_t rows = shapes[s][0], cols = shapes[s][1], nb = cols / 32, ns = cols / 256;
      uint8_t *src = malloc((size_t)rows * ns * bsz[ti]);
      for (size_t i = 0; i < (size_t)rows * ns * bsz[ti]; i++) src[i] = (uint8_t)rnd();
      for (uint32_t i = 0; i < rows * ns; i++) { /* small finite d (and dmin) */
        const uint16_t d = ie_f32_to_f16(0.0005f + 0.001f * (float)(rnd() % 100) / 100.0f);
        memcpy(src + (size_t)i * bsz[ti], &d, 2);
        if (ti == 0) memcpy(src + (size_t)i * bsz[ti] + 2, &d, 2);
      }
      mat w;
      model_mat_from_gguf(&w, types[ti], src, rows, cols);
      if (w.kind != (ti == 0 ? MAT_Q5K : MAT_IQ4XS)) tf++;
      float *x = malloc(cols * 4), *y = malloc(rows * 4);
      for (uint32_t i = 0; i < cols; i++) x[i] = frnd() * (i % 7 == 0 ? 3.0f : 1.0f);
      uint8_t *xq = malloc(q8_bytes(nb));
      cpu_quant_q8(x, xq, nb);
      if (ti == 0) cpu_gemv_q5k(&w, xq, y);
      else cpu_gemv_iq4xs(&w, xq, y);
      const int8_t *qa = (const int8_t *)xq;
      const float *da = (const float *)(xq + 32u * nb);
      for (uint32_t r = 0; r < rows; r++) {
        double ref = 0, mag = 0;
        for (uint32_t j = 0; j < cols; j++) {
          const uint8_t *blk = src + ((size_t)r * ns + j / 256) * bsz[ti];
          const double t = (ti == 0 ? w_q5k(blk, (int)(j % 256)) : w_iq4xs(blk, (int)(j % 256))) * qa[j] * da[j / 32];
          ref += t, mag += fabs(t);
        }
        const double err = fabs(ref - y[r]) / (mag > 0 ? mag : 1);
        if (err > worst) worst = err;
        if (err > 1e-5) tf++;
      }
      free(src), free(x), free(y), free(xq), free(w.qw), free(w.qs);
    }
    printf("ext_test: %s GEMV, %u shapes, max rel err (vs sum |terms|) %.2e, %s\n", ti == 0 ? "Q5_K" : "IQ4_XS",
           (unsigned)(sizeof shapes / sizeof shapes[0]), worst, tf ? "FAIL" : "pass");
    fails += tf;
  }
  return fails != 0;
}
