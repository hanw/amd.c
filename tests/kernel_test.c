/* kernel_test.c -- the CPU GEMV (same work split as the GPU kernel) against
 * a plain loop over the GGUF Q4_0 bytes, for many shapes: nb < 32, nb = 32,
 * nb > 32 (lanes do several steps), nb = 511 (the ie_sizes_ok limit), rows
 * not a multiple of 8. Also: the repacked nibbles are the GGUF nibbles, and
 * the Q8 sum stored by the quantizer is ie_q8_sum. */
#include <math.h>
#include <string.h>

#include "../core/ie_core.h"
#include "../src/backend.h"
#include "../src/common.h"

static uint64_t rs = 88172645463325252ull;
static uint32_t rnd(void) {
  rs ^= rs << 13, rs ^= rs >> 7, rs ^= rs << 17;
  return (uint32_t)rs;
}
static float frnd(void) { return (float)rnd() / 4294967296.0f * 2.0f - 1.0f; }

static uint16_t f32_to_f16_simple(float f) { /* normal range only */
  uint32_t b;
  memcpy(&b, &f, 4);
  uint32_t s = (b >> 16) & 0x8000u, e = (b >> 23) & 0xFFu, m = (b >> 13) & 0x3FFu;
  return (uint16_t)(s | ((e - 127 + 15) << 10) | m);
}

int main(void) {
  const uint32_t shapes[][2] = {{1, 1}, {7, 3}, {9, 11}, {8, 32}, {13, 33}, {17, 36}, {5, 64}, {3, 100}, {2, 511}};
  int fails = 0;
  double worst = 0;
  for (unsigned s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
    const uint32_t rows = shapes[s][0], nb = shapes[s][1], cols = nb * 32;
    uint8_t *src = malloc((size_t)rows * nb * 18);
    for (uint32_t i = 0; i < rows * nb; i++) {
      uint16_t d = f32_to_f16_simple(0.01f + 0.02f * (float)(rnd() % 100) / 100.0f);
      memcpy(src + i * 18, &d, 2);
      for (int k = 0; k < 16; k++) src[i * 18 + 2 + k] = (uint8_t)rnd();
    }
    mat w = {.kind = MAT_Q4, .rows = rows, .cols = cols, .nb = nb};
    repack_q4(&w, src);
    /* nibbles: GPU layout == GGUF */
    for (uint32_t r = 0; r < rows; r++)
      for (uint32_t b = 0; b < nb; b++)
        for (uint32_t j = 0; j < 32; j++) {
          uint32_t qb = src[ie_q4_src_qbyte(r, b, j & 15u, nb)];
          uint32_t want = j < 16 ? (qb & 15u) : (qb >> 4);
          uint32_t got = ie_q4_nib(w.qw[ie_q4_dst_word(r, b, ie_q4_word_of(j), nb)], j);
          if (got != want) fails++;
        }
    float *x = malloc(cols * 4), *y = malloc(rows * 4);
    for (uint32_t i = 0; i < cols; i++) x[i] = frnd() * (i % 7 == 0 ? 3.0f : 1.0f);
    uint8_t *xq = malloc(q8_bytes(nb));
    cpu_quant_q8(x, xq, nb);
    for (uint32_t b = 0; b < nb; b++) {
      Mem aw = {(const u32 *)xq};
      if (ie_q8_sum(aw, b * 8u) != ((const u32 *)(xq + 36u * nb))[b]) fails++;
    }
    cpu_gemv_q4(&w, xq, y);
    /* plain reference: double, element by element from the GGUF bytes */
    const int8_t *qa = (const int8_t *)xq;
    const float *da = (const float *)(xq + 32u * nb);
    for (uint32_t r = 0; r < rows; r++) {
      double ref = 0, mag = 0;
      for (uint32_t b = 0; b < nb; b++) {
        uint16_t h;
        memcpy(&h, src + ie_q4_src_blk(r, b, nb), 2);
        long dot = 0;
        for (uint32_t j = 0; j < 32; j++) {
          uint32_t qb = src[ie_q4_src_qbyte(r, b, j & 15u, nb)];
          int q = (int)(j < 16 ? (qb & 15u) : (qb >> 4)) - 8;
          dot += (long)q * qa[b * 32 + j];
        }
        double t = (double)ie_f16_to_f32(h) * da[b] * (double)dot;
        ref += t;
        mag += fabs(t);
      }
      double err = fabs(ref - y[r]) / (mag > 0 ? mag : 1);
      if (err > worst) worst = err;
      if (err > 1e-5) fails++;
    }
    free(src), free(x), free(y), free(xq), free(w.qw), free(w.qs);
  }
  printf("kernel_test: Q4_0, %u shapes, max rel err (vs sum |terms|) %.2e, %s\n",
         (unsigned)(sizeof shapes / sizeof shapes[0]), worst, fails ? "FAIL" : "pass");

  /* Q8_0: the same checks for the Q8_0 path (nb up to 255, the
   * ie_q8_sizes_ok limit). */
  const uint32_t shapes8[][2] = {{1, 1}, {7, 3}, {9, 11}, {8, 32}, {13, 33}, {17, 36}, {5, 64}, {3, 100}, {2, 255}};
  int fails8 = 0;
  double worst8 = 0;
  for (unsigned s = 0; s < sizeof shapes8 / sizeof shapes8[0]; s++) {
    const uint32_t rows = shapes8[s][0], nb = shapes8[s][1], cols = nb * 32;
    uint8_t *src = malloc((size_t)rows * nb * 34);
    for (uint32_t i = 0; i < rows * nb; i++) {
      uint16_t d = f32_to_f16_simple(0.001f + 0.002f * (float)(rnd() % 100) / 100.0f);
      memcpy(src + i * 34, &d, 2);
      for (int k = 0; k < 32; k++) src[i * 34 + 2 + k] = (uint8_t)rnd();
    }
    mat w = {.kind = MAT_Q8, .rows = rows, .cols = cols, .nb = nb};
    repack_q8(&w, src);
    /* int8 weights: GPU layout == GGUF */
    for (uint32_t r = 0; r < rows; r++)
      for (uint32_t b = 0; b < nb; b++)
        for (uint32_t j = 0; j < 32; j++)
          if (ie_byte(w.qw[ie_q8_dst_word(r, b, j >> 2, nb)], j & 3u) != src[ie_q8_src_qbyte(r, b, j, nb)]) fails8++;
    float *x = malloc(cols * 4), *y = malloc(rows * 4);
    for (uint32_t i = 0; i < cols; i++) x[i] = frnd() * (i % 7 == 0 ? 3.0f : 1.0f);
    uint8_t *xq = malloc(q8_bytes(nb));
    cpu_quant_q8(x, xq, nb);
    cpu_gemv_q8(&w, xq, y);
    const int8_t *qa = (const int8_t *)xq;
    const float *da = (const float *)(xq + 32u * nb);
    for (uint32_t r = 0; r < rows; r++) {
      double ref = 0, mag = 0;
      for (uint32_t b = 0; b < nb; b++) {
        uint16_t h;
        memcpy(&h, src + ie_q8_src_blk(r, b, nb), 2);
        long dot = 0;
        for (uint32_t j = 0; j < 32; j++) dot += (long)(int8_t)src[ie_q8_src_qbyte(r, b, j, nb)] * qa[b * 32 + j];
        double t = (double)ie_f16_to_f32(h) * da[b] * (double)dot;
        ref += t;
        mag += fabs(t);
      }
      double err = fabs(ref - y[r]) / (mag > 0 ? mag : 1);
      if (err > worst8) worst8 = err;
      if (err > 1e-5) fails8++;
    }
    free(src), free(x), free(y), free(xq), free(w.qw), free(w.qs);
  }
  printf("kernel_test: Q8_0, %u shapes, max rel err (vs sum |terms|) %.2e, %s\n",
         (unsigned)(sizeof shapes8 / sizeof shapes8[0]), worst8, fails8 ? "FAIL" : "pass");
  fails += fails8;
  return fails != 0;
}
