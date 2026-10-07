/* common.h -- small helpers shared by the host code. */
#ifndef IE_COMMON_H
#define IE_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* GGML tensor types that the engine reads. */
enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q8_0 = 8, GGML_Q6_K = 14 };

/* Print a message and exit(1). */
void ie_die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));

/* Zeroed memory, 256-byte aligned; exits when out of memory. */
void *ie_alloc(size_t n);

/* IEEE half -> float (exact). */
float ie_f16_to_f32(uint16_t h);
/* float -> IEEE half, round to nearest even (as numpy and ggml's F16C path). */
uint16_t ie_f32_to_f16(float f);

/* Q8 activation buffer of nb blocks (one buffer in the arena):
 *   bytes [0, 32 nb)        : the int8 values, 8 u32 words per block
 *   bytes [32 nb, 36 nb)    : the float scale d of each block
 *   bytes [36 nb, 40 nb)    : the u32 sum of the 32 int8 values (asum)   */
static inline size_t q8_bytes(uint32_t nb) { return (size_t)nb * 40u; }

#endif
