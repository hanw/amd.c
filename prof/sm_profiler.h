/* sm_profiler.h -- per-wave event timeline inside GPU kernels, for AMD GPUs
 * (RDNA / CDNA, HIP runtime), in plain C.
 *
 * The API and the buffer layout follow leepoly/sm-profiler (CUDA): a kernel
 * records range and instant events per (workgroup, wave) with no atomics;
 * the host copies the buffer back and writes a Chrome trace (JSON) that
 * ui.perfetto.dev or chrome://tracing opens. One process per WGP (or CU),
 * one thread per (workgroup, wave), as the CUDA version has one per SM.
 *
 * The AMD parts:
 *   - time: s_sendmsg_rtn MSG_RTN_GET_REALTIME (gfx11+, the 100 MHz
 *     constant clock, the same on every CU): 10 ns per tick; gfx9/gfx10:
 *     s_memrealtime. Stored in ns, as the CUDA %globaltimer.
 *   - the "SM": s_getreg HW_REG_HW_ID1 (gfx10+): shader engine, shader
 *     array and WGP id give one linear id per WGP (the raw register is in
 *     the upper half of the sm_id word: see SMP_HWID_*).
 *   - lane and wave: mbcnt and the wave size (32 or 64) of the build.
 *
 * Device side: C for clang --target=amdgcn-amd-amdhsa (the kernels of this
 * repository), or HIP C++ (define SMP_HIP before the include). The kernel
 * gets the buffer pointer (smp_device_ptr) as an argument. The workgroup
 * index and wave index come from the hardware ids by default; a persistent
 * kernel that runs the work of many logical workgroups passes its own with
 * the *_at functions.
 *
 * Host side: sm_profiler.c (loads libamdhip64.so with dlopen: no ROCm
 * headers needed to build). */
#ifndef SM_PROFILER_H
#define SM_PROFILER_H

#if defined(__AMDGCN__) && !defined(SMP_HIP) && !defined(UINT32_MAX)
typedef __UINT32_TYPE__ uint32_t; /* a GPU build without the C library */
typedef __UINT64_TYPE__ uint64_t;
#else
#include <stdint.h>
#endif

#define SMP_MAX_EVENT_TYPES 32
#define SMP_MAX_EVENT_NAME_LEN 128
#define SMP_EVENT_RANGE 0u
#define SMP_EVENT_INSTANT 1u

/* One event: 40 bytes (the layout of the CUDA version). sm_id: bits 0..15
 * the linear WGP id, bits 16..17 the SIMD of the wave (see smp_sm_id). */
typedef struct __attribute__((packed)) {
  uint32_t event_id;  /* index within its (workgroup, wave) */
  uint32_t event_no;  /* the event type (0 .. 31) */
  uint32_t block_id;  /* linear workgroup index */
  uint32_t group_idx; /* wave index within the workgroup */
  uint32_t sm_id;     /* linear WGP id (low 16 bits); SIMD id << 16 */
  uint32_t type;      /* SMP_EVENT_RANGE or SMP_EVENT_INSTANT */
  uint64_t st_ns;
  uint64_t en_ns;
} smp_event;

/* Buffer layout (64-bit words): [0] num_blocks << 32 | num_groups,
 * [1] max_events << 32 | enabled, then the counters (u32 per (block,
 * group)), the active table (u32 per (block, group, event type)), the
 * events ([block][group][max_events]). Each part starts at a 64-bit word. */
static inline uint64_t smp_words(uint64_t bytes) { return (bytes + 7u) / 8u; }

/* ------------------------------------------------------------------ host */
#if !defined(__AMDGCN__) || defined(SMP_HOST_ONLY)
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct smp_buffer *smp_buffer_t;
/* NULL on error (no HIP, or no memory). enabled = 0: a header only. */
smp_buffer_t smp_create(uint32_t num_blocks, uint32_t num_groups, uint32_t max_events_per_group, int enabled);
void smp_destroy(smp_buffer_t b);
/* the device pointer for the kernel argument */
uint64_t *smp_device_ptr(smp_buffer_t b);
/* clear counters and events (the names stay); call before each launch */
int smp_init(smp_buffer_t b);
int smp_register_event(smp_buffer_t b, uint32_t event_no, const char *name);
/* copy back and write the Chrome trace; 0 on success */
int smp_export(smp_buffer_t b, const char *filename);
/* copy back and print per event type: count, total, mean, min, max (us),
 * and the events that did not fit (buffer full) */
int smp_print_stats(smp_buffer_t b, FILE *out);
#ifdef __cplusplus
}
#endif
#endif

/* ---------------------------------------------------------------- device */
#if defined(__AMDGCN__) && !defined(SMP_HOST_ONLY)
#ifdef SMP_HIP
#define SMP_FN __device__ static inline __attribute__((always_inline))
#define SMP_G
#else
#define SMP_FN static inline __attribute__((always_inline))
#define SMP_G __attribute__((address_space(1)))
#endif
#define SMP_RD(p) (*(const volatile SMP_G uint32_t *)(p))

#if __has_builtin(__builtin_amdgcn_wavefrontsize) /* clang 19+ (the macro is deprecated there) */
#define SMP_WAVE ((uint32_t)__builtin_amdgcn_wavefrontsize())
#else
#define SMP_WAVE ((uint32_t)__AMDGCN_WAVEFRONT_SIZE)
#endif
SMP_FN uint64_t smp_time_ns(void) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
  return __builtin_amdgcn_s_sendmsg_rtnl(0x83) * 10u; /* MSG_RTN_GET_REALTIME: 100 MHz */
#else
  return __builtin_amdgcn_s_memrealtime() * 10u; /* gfx9 / gfx10: 100 MHz */
#endif
}
SMP_FN uint32_t smp_lane(void) {
  const uint32_t lo = __builtin_amdgcn_mbcnt_lo(~0u, 0u);
  return SMP_WAVE == 64u ? __builtin_amdgcn_mbcnt_hi(~0u, lo) : lo;
}
SMP_FN uint32_t smp_wave(void) {
  const uint32_t t = (__builtin_amdgcn_workitem_id_z() * __builtin_amdgcn_workgroup_size_y() +
                      __builtin_amdgcn_workitem_id_y()) * __builtin_amdgcn_workgroup_size_x() +
                     __builtin_amdgcn_workitem_id_x();
  return t / SMP_WAVE;
}
SMP_FN uint32_t smp_block(void) {
  const uint32_t gx = __builtin_amdgcn_grid_size_x() / __builtin_amdgcn_workgroup_size_x();
  const uint32_t gy = __builtin_amdgcn_grid_size_y() / __builtin_amdgcn_workgroup_size_y();
  return (__builtin_amdgcn_workgroup_id_z() * gy + __builtin_amdgcn_workgroup_id_y()) * gx +
         __builtin_amdgcn_workgroup_id_x();
}
/* HW_REG_HW_ID1 (gfx10+): WAVE_ID [4:0], SIMD_ID [9:8], WGP_ID [13:10],
 * SA_ID [16], SE_ID [20:18]. The linear id: SE, SA, WGP (gaps where an SA
 * has fewer WGPs). */
#define SMP_HWID1 ((31 << 11) | (0 << 6) | 23)
SMP_FN uint32_t smp_hw_id(void) { return __builtin_amdgcn_s_getreg(SMP_HWID1); }
SMP_FN uint32_t smp_sm_id(void) {
  const uint32_t h = smp_hw_id();
  const uint32_t wgp = (h >> 10) & 15u, sa = (h >> 16) & 1u, se = (h >> 18) & 7u, simd = (h >> 8) & 3u;
  return ((se << 5) | (sa << 4) | wgp) | (simd << 16);
}

typedef struct {
  uint32_t nb, ng, ne, en;
  SMP_G uint32_t *cnt, *act;
  SMP_G smp_event *ev;
} smp_info;
SMP_FN smp_info smp_parse(SMP_G uint64_t *buf) {
  smp_info i;
  const uint64_t h0 = buf[0], h1 = buf[1];
  i.nb = (uint32_t)(h0 >> 32), i.ng = (uint32_t)h0, i.ne = (uint32_t)(h1 >> 32), i.en = (uint32_t)h1;
  const uint64_t ca = 2u + smp_words((uint64_t)i.nb * i.ng * 4u);
  const uint64_t ev = ca + smp_words((uint64_t)i.nb * i.ng * SMP_MAX_EVENT_TYPES * 4u);
  i.cnt = (SMP_G uint32_t *)(buf + 2), i.act = (SMP_G uint32_t *)(buf + ca), i.ev = (SMP_G smp_event *)(buf + ev);
  return i;
}
/* One lane per wave calls these (pred: for example smp_lane() == 0). */
SMP_FN void smp_start_at(SMP_G uint64_t *buf, uint32_t no, uint32_t blk, uint32_t grp, int pred) {
  if (!pred || no >= SMP_MAX_EVENT_TYPES) return;
  smp_info i = smp_parse(buf);
  if (!i.en || blk >= i.nb || grp >= i.ng) return;
  SMP_G uint32_t *a = i.act + ((uint64_t)blk * i.ng + grp) * SMP_MAX_EVENT_TYPES + no;
  if (*a) return; /* already open */
  SMP_G uint32_t *c = i.cnt + (uint64_t)blk * i.ng + grp;
  const uint32_t id = *c;
  *c = id + 1u;
  if (id >= i.ne) return; /* full: counted, not stored */
  SMP_G smp_event *e = i.ev + ((uint64_t)blk * i.ng + grp) * i.ne + id;
  e->event_id = id, e->event_no = no, e->block_id = blk, e->group_idx = grp;
  e->sm_id = smp_sm_id(), e->type = SMP_EVENT_RANGE, e->en_ns = 0;
  __asm__ volatile("" ::: "memory");
  e->st_ns = smp_time_ns();
  *a = id + 1u;
}
SMP_FN void smp_end_at(SMP_G uint64_t *buf, uint32_t no, uint32_t blk, uint32_t grp, int pred) {
  if (!pred || no >= SMP_MAX_EVENT_TYPES) return;
  const uint64_t t = smp_time_ns();
  __asm__ volatile("" ::: "memory");
  smp_info i = smp_parse(buf);
  if (!i.en || blk >= i.nb || grp >= i.ng) return;
  SMP_G uint32_t *a = i.act + ((uint64_t)blk * i.ng + grp) * SMP_MAX_EVENT_TYPES + no;
  const uint32_t id1 = *a;
  if (!id1) return;
  i.ev[((uint64_t)blk * i.ng + grp) * i.ne + id1 - 1u].en_ns = t;
  *a = 0u;
}
SMP_FN void smp_instant_at(SMP_G uint64_t *buf, uint32_t no, uint32_t blk, uint32_t grp, int pred) {
  if (!pred || no >= SMP_MAX_EVENT_TYPES) return;
  const uint64_t t = smp_time_ns();
  smp_info i = smp_parse(buf);
  if (!i.en || blk >= i.nb || grp >= i.ng) return;
  SMP_G uint32_t *c = i.cnt + (uint64_t)blk * i.ng + grp;
  const uint32_t id = *c;
  *c = id + 1u;
  if (id >= i.ne) return;
  SMP_G smp_event *e = i.ev + ((uint64_t)blk * i.ng + grp) * i.ne + id;
  e->event_id = id, e->event_no = no, e->block_id = blk, e->group_idx = grp;
  e->sm_id = smp_sm_id(), e->type = SMP_EVENT_INSTANT, e->st_ns = t, e->en_ns = 0;
}
/* The hardware workgroup and wave */
SMP_FN void smp_start(SMP_G uint64_t *buf, uint32_t no, int pred) { smp_start_at(buf, no, smp_block(), smp_wave(), pred); }
SMP_FN void smp_end(SMP_G uint64_t *buf, uint32_t no, int pred) { smp_end_at(buf, no, smp_block(), smp_wave(), pred); }
SMP_FN void smp_instant(SMP_G uint64_t *buf, uint32_t no, int pred) { smp_instant_at(buf, no, smp_block(), smp_wave(), pred); }
#endif
#endif
