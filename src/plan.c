/* plan.c -- static plan of the activation arena.
 *
 * Greedy: take the buffers from the largest to the smallest; put each at the
 * lowest 256-aligned offset where it does not overlap any already placed
 * buffer whose lifetime overlaps its own. The planner is NOT verified; its
 * output is checked by the verified ie_plan_check (core/ie_core.h), and the
 * engine stops if the check fails. The laws plan_fits, plan_aligned and
 * plan_no_clobber say what an accepted plan guarantees. */
#include <string.h>

#include "../core/ie_core.h"
#include "common.h"
#include "model.h"

static const graph *sort_g;
static int by_size(const void *a, const void *b) {
  const buf *x = &sort_g->bufs[*(const int *)a], *y = &sort_g->bufs[*(const int *)b];
  if (x->size != y->size) return x->size < y->size ? 1 : -1;
  return *(const int *)a - *(const int *)b;
}

static int live_overlap(const buf *a, const buf *b) { return a->first <= b->last && b->first <= a->last; }

void plan_arena(graph *g, int verbose) {
  uint32_t n = g->n_bufs;
  int *order = malloc(n * sizeof(int));
  char *placed = calloc(n, 1);
  for (uint32_t i = 0; i < n; i++) order[i] = (int)i;
  sort_g = g;
  qsort(order, n, sizeof(int), by_size);

  uint64_t top = 0;
  for (uint32_t k = 0; k < n; k++) {
    buf *b = &g->bufs[order[k]];
    uint64_t off = 0;
    for (int moved = 1; moved;) { /* first fit: move past every conflict */
      moved = 0;
      for (uint32_t j = 0; j < n; j++) {
        const buf *c = &g->bufs[j];
        if (!placed[j] || !live_overlap(b, c)) continue;
        if (off < (uint64_t)c->off + c->size && (uint64_t)c->off < off + b->size) {
          off = ((uint64_t)c->off + c->size + 255u) & ~(uint64_t)255u;
          moved = 1;
        }
      }
    }
    if (off + b->size > 0xFFFFFF00u) ie_die("activation arena larger than 4 GiB");
    b->off = (uint32_t)off;
    placed[order[k]] = 1;
    if (off + b->size > top) top = off + b->size;
  }
  g->arena = (uint32_t)((top + 255u) & ~(uint64_t)255u);

  /* The verified check. */
  u32 *off = malloc(n * 4), *size = malloc(n * 4), *first = malloc(n * 4), *last = malloc(n * 4);
  for (uint32_t i = 0; i < n; i++) off[i] = g->bufs[i].off, size[i] = g->bufs[i].size, first[i] = g->bufs[i].first, last[i] = g->bufs[i].last;
  Plan p = {{off}, {size}, {first}, {last}, n, g->arena};
  int ok = ie_plan_check(p);
  uint64_t sum = 0;
  for (uint32_t i = 0; i < n; i++) sum += g->bufs[i].size;
  if (verbose)
    fprintf(stderr, "plan: %u buffers over %u ops, arena %u bytes (sum of sizes %llu), ie_plan_check: %s\n", n,
            g->n_ops, g->arena, (unsigned long long)sum, ok ? "ok" : "FAILED");
  free(off), free(size), free(first), free(last), free(order), free(placed);
  if (!ok) ie_die("ie_plan_check rejected the activation plan");
}
