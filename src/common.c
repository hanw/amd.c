#include "common.h"

#include <stdarg.h>
#include <string.h>

void ie_die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "ie: ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(1);
}

void *ie_alloc(size_t n) {
  void *p = NULL;
  size_t m = (n + 255u) & ~(size_t)255u;
  if (m == 0) m = 256;
  if (posix_memalign(&p, 256, m) != 0) ie_die("out of memory (%zu bytes)", n);
  memset(p, 0, m);
  return p;
}

float ie_f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else { /* subnormal: normalize */
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        exp--;
      }
      bits = sign | (exp << 23) | ((man & 0x3FFu) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

uint16_t ie_f32_to_f16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u, ax = x & 0x7FFFFFFFu;
  if (ax >= 0x7F800000u) return (uint16_t)(sign | 0x7C00u | (ax > 0x7F800000u ? 0x200u : 0u)); /* inf, nan */
  if (ax >= 0x477FF000u) return (uint16_t)(sign | 0x7C00u); /* rounds to >= 65520: inf */
  if (ax < 0x38800000u) { /* below the smallest normal half: subnormal or 0 */
    if (ax < 0x33000000u) return (uint16_t)sign; /* < 2^-25: rounds to 0 */
    const uint32_t m = (ax & 0x7FFFFFu) | 0x800000u;
    const int shift = 126 - (int)(ax >> 23); /* 14 .. 24 */
    uint32_t h = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1u))) h++;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((ax >> 13) - (112u << 10)); /* rebias exponent 127 -> 15 */
  const uint32_t rem = ax & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
  return (uint16_t)(sign | h);
}
