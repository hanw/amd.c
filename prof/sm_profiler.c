/* sm_profiler.c -- host side of prof/sm_profiler.h: the buffer, the copy
 * back, the Chrome trace and the statistics. HIP comes from libamdhip64.so
 * by dlopen (as src/hip.c), so no ROCm headers are needed to build. */
#include "sm_profiler.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

static struct {
  int ok;
  int (*Malloc)(void **, size_t);
  int (*Free)(void *);
  int (*Memcpy)(void *, const void *, size_t, int);
  int (*Sync)(void);
} H;

static int hip_load(void) {
  if (H.ok) return 0;
  void *l = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
  if (!l) l = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_GLOBAL);
  if (!l) l = dlopen("libamdhip64.so.6", RTLD_NOW | RTLD_GLOBAL);
  if (!l) return -1;
  *(void **)&H.Malloc = dlsym(l, "hipMalloc");
  *(void **)&H.Free = dlsym(l, "hipFree");
  *(void **)&H.Memcpy = dlsym(l, "hipMemcpy");
  *(void **)&H.Sync = dlsym(l, "hipDeviceSynchronize");
  if (!H.Malloc || !H.Free || !H.Memcpy || !H.Sync) return -1;
  H.ok = 1;
  return 0;
}

struct smp_buffer {
  uint32_t nb, ng, ne;
  int enabled;
  uint64_t *dev, *host;
  size_t bytes;
  uint64_t ca, ev; /* word offsets: active table, events */
  char names[SMP_MAX_EVENT_TYPES][SMP_MAX_EVENT_NAME_LEN];
};

typedef char smp_event_is_40_bytes[sizeof(smp_event) == 40 ? 1 : -1];

smp_buffer_t smp_create(uint32_t nb, uint32_t ng, uint32_t ne, int enabled) {
  if (!nb || !ng || !ne) return NULL;
  if (hip_load()) {
    fprintf(stderr, "sm_profiler: cannot load libamdhip64.so\n");
    return NULL;
  }
  smp_buffer_t b = calloc(1, sizeof *b);
  if (!b) return NULL;
  b->nb = nb, b->ng = ng, b->ne = ne, b->enabled = enabled;
  b->ca = 2u + smp_words((uint64_t)nb * ng * 4u);
  b->ev = b->ca + smp_words((uint64_t)nb * ng * SMP_MAX_EVENT_TYPES * 4u);
  b->bytes = enabled ? (size_t)(b->ev + smp_words((uint64_t)nb * ng * ne * sizeof(smp_event))) * 8u : 16u;
  b->host = calloc(1, b->bytes);
  if (!b->host || H.Malloc((void **)&b->dev, b->bytes)) {
    fprintf(stderr, "sm_profiler: no memory for %zu bytes\n", b->bytes);
    free(b->host), free(b);
    return NULL;
  }
  if (smp_init(b)) {
    smp_destroy(b);
    return NULL;
  }
  return b;
}

void smp_destroy(smp_buffer_t b) {
  if (!b) return;
  if (b->dev) H.Free(b->dev);
  free(b->host), free(b);
}

uint64_t *smp_device_ptr(smp_buffer_t b) { return b ? b->dev : NULL; }

int smp_init(smp_buffer_t b) {
  if (!b) return -1;
  memset(b->host, 0, b->bytes);
  b->host[0] = (uint64_t)b->nb << 32 | b->ng;
  b->host[1] = (uint64_t)b->ne << 32 | (b->enabled ? 1u : 0u);
  if (H.Memcpy(b->dev, b->host, b->bytes, 1 /* host to device */)) {
    fprintf(stderr, "sm_profiler: copy to the device failed\n");
    return -1;
  }
  return 0;
}

int smp_register_event(smp_buffer_t b, uint32_t no, const char *name) {
  if (!b || !name || no >= SMP_MAX_EVENT_TYPES) return -1;
  strncpy(b->names[no], name, SMP_MAX_EVENT_NAME_LEN - 1);
  b->names[no][SMP_MAX_EVENT_NAME_LEN - 1] = 0;
  return 0;
}

/* Copy back; the events in one array (sorted by start); *lost: the events
 * that did not fit. */
static smp_event *collect(smp_buffer_t b, size_t *n, uint64_t *lost) {
  *n = 0, *lost = 0;
  if (!b || !b->enabled) return NULL;
  if (H.Sync() || H.Memcpy(b->host, b->dev, b->bytes, 2 /* device to host */)) {
    fprintf(stderr, "sm_profiler: copy from the device failed\n");
    return NULL;
  }
  const uint32_t *cnt = (const uint32_t *)(b->host + 2);
  const smp_event *ev = (const smp_event *)(b->host + b->ev);
  size_t tot = 0;
  for (size_t i = 0; i < (size_t)b->nb * b->ng; i++) {
    tot += cnt[i] < b->ne ? cnt[i] : b->ne;
    if (cnt[i] > b->ne) *lost += cnt[i] - b->ne;
  }
  smp_event *out = malloc((tot ? tot : 1) * sizeof *out);
  if (!out) return NULL;
  for (size_t i = 0; i < (size_t)b->nb * b->ng; i++)
    for (uint32_t e = 0; e < cnt[i] && e < b->ne; e++) out[(*n)++] = ev[i * b->ne + e];
  return out;
}

static int by_start(const void *x, const void *y) {
  const smp_event *a = x, *c = y;
  return a->st_ns < c->st_ns ? -1 : a->st_ns > c->st_ns;
}

int smp_export(smp_buffer_t b, const char *filename) {
  size_t n;
  uint64_t lost;
  smp_event *ev = collect(b, &n, &lost);
  if (!ev) return -1;
  qsort(ev, n, sizeof *ev, by_start);
  FILE *f = fopen(filename, "w");
  if (!f) {
    fprintf(stderr, "sm_profiler: cannot write %s\n", filename);
    free(ev);
    return -1;
  }
  const uint64_t t0 = n ? ev[0].st_ns : 0;
  fprintf(f, "{\"traceEvents\": [\n");
  /* process names: one per WGP */
  int first = 1;
  for (size_t i = 0; i < n; i++) {
    const smp_event *e = &ev[i];
    const char *nm = b->names[e->event_no][0] ? b->names[e->event_no] : "unknown";
    const unsigned pid = e->sm_id & 0xFFFFu;
    const double ts = (double)(e->st_ns - t0) / 1000.0;
    if (!first) fprintf(f, ",\n");
    first = 0;
    if (e->type == SMP_EVENT_RANGE && e->en_ns >= e->st_ns)
      fprintf(f, "{\"name\": \"%s\", \"ph\": \"X\", \"ts\": %.3f, \"dur\": %.3f, \"pid\": %u, \"tid\": \"block_%u_wave_%u\", "
                 "\"cat\": \"gpu\", \"args\": {\"id\": %u, \"simd\": %u}}",
              nm, ts, (double)(e->en_ns - e->st_ns) / 1000.0, pid, e->block_id, e->group_idx, e->event_id, e->sm_id >> 16);
    else /* an instant event, or a range that was not closed */
      fprintf(f, "{\"name\": \"%s\", \"ph\": \"i\", \"s\": \"t\", \"ts\": %.3f, \"pid\": %u, \"tid\": \"block_%u_wave_%u\", "
                 "\"cat\": \"gpu\", \"args\": {\"id\": %u, \"open\": %d}}",
              nm, ts, pid, e->block_id, e->group_idx, e->event_id, e->type == SMP_EVENT_RANGE);
  }
  /* name the processes */
  unsigned char seen[65536 / 8];
  memset(seen, 0, sizeof seen);
  for (size_t i = 0; i < n; i++) {
    const unsigned pid = ev[i].sm_id & 0xFFFFu;
    if (seen[pid >> 3] & (1u << (pid & 7u))) continue;
    seen[pid >> 3] |= (unsigned char)(1u << (pid & 7u));
    fprintf(f, ",\n{\"name\": \"process_name\", \"ph\": \"M\", \"pid\": %u, \"args\": {\"name\": \"SE%u SA%u WGP%u\"}}", pid,
            pid >> 5, (pid >> 4) & 1u, pid & 15u);
  }
  fprintf(f, "\n], \"displayTimeUnit\": \"ns\", \"metadata\": {\"num_blocks\": %u, \"num_groups\": %u, "
             "\"max_events_per_group\": %u, \"total_events\": %zu, \"lost_events\": %llu, \"clock\": \"100 MHz realtime\"}}\n",
          b->nb, b->ng, b->ne, n, (unsigned long long)lost);
  fclose(f);
  fprintf(stderr, "sm_profiler: %zu events to %s%s\n", n, filename, lost ? " (some lost: buffer full)" : "");
  free(ev);
  return 0;
}

int smp_print_stats(smp_buffer_t b, FILE *out) {
  size_t n;
  uint64_t lost;
  smp_event *ev = collect(b, &n, &lost);
  if (!ev) return -1;
  double tot[SMP_MAX_EVENT_TYPES] = {0}, mn[SMP_MAX_EVENT_TYPES], mx[SMP_MAX_EVENT_TYPES] = {0};
  uint64_t cnt[SMP_MAX_EVENT_TYPES] = {0}, inst[SMP_MAX_EVENT_TYPES] = {0};
  for (int k = 0; k < SMP_MAX_EVENT_TYPES; k++) mn[k] = 1e300;
  uint64_t t0 = ~0ull, t1 = 0;
  for (size_t i = 0; i < n; i++) {
    const smp_event *e = &ev[i];
    if (e->st_ns < t0) t0 = e->st_ns;
    if (e->type == SMP_EVENT_INSTANT || e->en_ns < e->st_ns) {
      inst[e->event_no]++;
      continue;
    }
    if (e->en_ns > t1) t1 = e->en_ns;
    const double d = (double)(e->en_ns - e->st_ns) / 1000.0;
    tot[e->event_no] += d, cnt[e->event_no]++;
    if (d < mn[e->event_no]) mn[e->event_no] = d;
    if (d > mx[e->event_no]) mx[e->event_no] = d;
  }
  fprintf(out, "sm_profiler: %zu events over %.3f us%s\n", n, t1 > t0 ? (t1 - t0) / 1000.0 : 0.0,
          lost ? " (some lost: buffer full)" : "");
  fprintf(out, "  %-24s %8s %12s %10s %10s %10s %8s\n", "event", "ranges", "total us", "mean us", "min us", "max us", "instant");
  for (int k = 0; k < SMP_MAX_EVENT_TYPES; k++)
    if (cnt[k] || inst[k])
      fprintf(out, "  %-24s %8llu %12.3f %10.3f %10.3f %10.3f %8llu\n", b->names[k][0] ? b->names[k] : "unknown",
              (unsigned long long)cnt[k], tot[k], cnt[k] ? tot[k] / cnt[k] : 0.0, cnt[k] ? mn[k] : 0.0, mx[k],
              (unsigned long long)inst[k]);
  free(ev);
  return 0;
}
