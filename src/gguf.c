/* gguf.c -- GGUF v3 parser. Every read is bounds-checked against the file. */
#include "gguf.h"

#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "common.h"

typedef struct {
  const uint8_t *p;
  uint64_t pos, size;
  const char *path;
} rd;

static void need(rd *r, uint64_t n) {
  if (n > r->size || r->pos > r->size - n) ie_die("%s: truncated GGUF (at byte %llu)", r->path, (unsigned long long)r->pos);
}
static uint64_t rd_n(rd *r, int n) {
  need(r, (uint64_t)n);
  uint64_t v = 0;
  memcpy(&v, r->p + r->pos, (size_t)n); /* little-endian host */
  r->pos += (uint64_t)n;
  return v;
}
static gguf_str rd_str(rd *r) {
  gguf_str s;
  s.n = rd_n(r, 8);
  need(r, s.n);
  s.p = (const char *)r->p + r->pos;
  r->pos += s.n;
  return s;
}
static char *dup_str(gguf_str s) {
  char *c = malloc(s.n + 1);
  if (!c) ie_die("out of memory");
  memcpy(c, s.p, s.n);
  c[s.n] = 0;
  return c;
}

static int scalar_size(uint32_t t) {
  switch (t) {
    case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
    case GGUF_U16: case GGUF_I16: return 2;
    case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
    case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
    default: return 0;
  }
}

/* Read one scalar of type t into kv->v. */
static void rd_scalar(rd *r, uint32_t t, gguf_kv *kv) {
  uint64_t raw = rd_n(r, scalar_size(t));
  switch (t) {
    case GGUF_U8: case GGUF_U16: case GGUF_U32: case GGUF_U64: case GGUF_BOOL: kv->v.u = raw; break;
    case GGUF_I8: kv->v.i = (int8_t)raw; break;
    case GGUF_I16: kv->v.i = (int16_t)raw; break;
    case GGUF_I32: kv->v.i = (int32_t)raw; break;
    case GGUF_I64: kv->v.i = (int64_t)raw; break;
    case GGUF_F32: { float f; uint32_t b = (uint32_t)raw; memcpy(&f, &b, 4); kv->v.f = f; break; }
    case GGUF_F64: { double d; memcpy(&d, &raw, 8); kv->v.f = d; break; }
  }
}

static void rd_value(rd *r, uint32_t t, gguf_kv *kv, int depth) {
  if (t == GGUF_STR) {
    kv->v.s = rd_str(r);
  } else if (t == GGUF_ARR) {
    uint32_t at = (uint32_t)rd_n(r, 4);
    uint64_t n = rd_n(r, 8);
    kv->arr_type = at;
    kv->arr_n = n;
    kv->arr_data = r->p + r->pos;
    if (at == GGUF_STR) {
      if (n > r->size / 8) ie_die("%s: bad string array length", r->path);
      kv->arr_str = malloc((n ? n : 1) * sizeof(gguf_str));
      if (!kv->arr_str) ie_die("out of memory");
      for (uint64_t i = 0; i < n; i++) kv->arr_str[i] = rd_str(r);
    } else if (at == GGUF_ARR) {
      /* nested arrays: skip them (no model key the engine needs uses them) */
      if (depth > 4) ie_die("%s: arrays nested too deep", r->path);
      for (uint64_t i = 0; i < n; i++) {
        gguf_kv tmp = {0};
        rd_value(r, GGUF_ARR, &tmp, depth + 1);
        free(tmp.arr_str);
      }
    } else {
      int s = scalar_size(at);
      if (!s) ie_die("%s: bad array type %u", r->path, at);
      if (n > r->size / (uint64_t)s) ie_die("%s: bad array length", r->path);
      need(r, n * (uint64_t)s);
      r->pos += n * (uint64_t)s;
    }
  } else {
    if (!scalar_size(t)) ie_die("%s: bad value type %u", r->path, t);
    rd_scalar(r, t, kv);
  }
}

uint64_t ggml_nbytes(uint32_t type, uint64_t n) {
  switch (type) {
    case GGML_F32: return n * 4;
    case GGML_F16: return n * 2;
    case GGML_Q4_0: return n % 32 ? 0 : n / 32 * 18;
    case GGML_Q8_0: return n % 32 ? 0 : n / 32 * 34;
    case GGML_Q4_K: return n % 256 ? 0 : n / 256 * 144;
    case GGML_Q5_K: return n % 256 ? 0 : n / 256 * 176;
    case GGML_Q6_K: return n % 256 ? 0 : n / 256 * 210;
    default: return 0;
  }
}

const char *ggml_type_name(uint32_t type) {
  switch (type) {
    case GGML_F32: return "F32";
    case GGML_F16: return "F16";
    case GGML_Q4_0: return "Q4_0";
    case GGML_Q8_0: return "Q8_0";
    case GGML_Q4_K: return "Q4_K";
    case GGML_Q5_K: return "Q5_K";
    case GGML_Q6_K: return "Q6_K";
    default: return "?";
  }
}

void gguf_open(gguf_file *g, const char *path) {
  memset(g, 0, sizeof *g);
  int fd = open(path, O_RDONLY);
  if (fd < 0) ie_die("cannot open %s", path);
  struct stat st;
  if (fstat(fd, &st) != 0) ie_die("cannot stat %s", path);
  g->size = (uint64_t)st.st_size;
  g->map = mmap(NULL, g->size ? g->size : 1, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (g->map == MAP_FAILED) ie_die("cannot mmap %s", path);

  rd r = {g->map, 0, g->size, path};
  if (rd_n(&r, 4) != 0x46554747u) ie_die("%s: not a GGUF file", path); /* "GGUF" */
  g->version = (uint32_t)rd_n(&r, 4);
  if (g->version != 3) ie_die("%s: GGUF version %u (only v3 is supported)", path, g->version);
  g->n_tensors = rd_n(&r, 8);
  g->n_kv = rd_n(&r, 8);
  if (g->n_kv > g->size / 8 || g->n_tensors > g->size / 8) ie_die("%s: bad counts", path);

  g->kv = calloc(g->n_kv ? g->n_kv : 1, sizeof(gguf_kv));
  g->t = calloc(g->n_tensors ? g->n_tensors : 1, sizeof(gguf_tensor));
  if (!g->kv || !g->t) ie_die("out of memory");
  g->alignment = 32;
  for (uint64_t i = 0; i < g->n_kv; i++) {
    gguf_kv *kv = &g->kv[i];
    kv->key = dup_str(rd_str(&r));
    kv->type = (uint32_t)rd_n(&r, 4);
    rd_value(&r, kv->type, kv, 0);
    if (strcmp(kv->key, "general.alignment") == 0 && kv->type == GGUF_U32) g->alignment = (uint32_t)kv->v.u;
  }
  if (g->alignment == 0 || (g->alignment & (g->alignment - 1))) ie_die("%s: bad alignment %u", path, g->alignment);

  for (uint64_t i = 0; i < g->n_tensors; i++) {
    gguf_tensor *t = &g->t[i];
    t->name = dup_str(rd_str(&r));
    t->n_dims = (uint32_t)rd_n(&r, 4);
    if (t->n_dims < 1 || t->n_dims > 4) ie_die("%s: tensor %s has %u dims", path, t->name, t->n_dims);
    uint64_t n = 1;
    for (int d = 0; d < 4; d++) t->ne[d] = 1;
    for (uint32_t d = 0; d < t->n_dims; d++) {
      t->ne[d] = rd_n(&r, 8);
      if (t->ne[d] && n > UINT64_MAX / t->ne[d]) ie_die("%s: tensor %s too large", path, t->name);
      n *= t->ne[d];
    }
    t->type = (uint32_t)rd_n(&r, 4);
    t->offset = rd_n(&r, 8);
    t->nbytes = ggml_nbytes(t->type, n);
  }
  g->data_off = (r.pos + g->alignment - 1) / g->alignment * g->alignment;
  for (uint64_t i = 0; i < g->n_tensors; i++) {
    gguf_tensor *t = &g->t[i];
    if (t->nbytes == 0) continue; /* unsupported type: checked when used */
    if (t->offset % g->alignment) ie_die("%s: tensor %s is not aligned", path, t->name);
    if (g->data_off > g->size || t->offset > g->size - g->data_off || t->nbytes > g->size - g->data_off - t->offset)
      ie_die("%s: tensor %s is outside the file", path, t->name);
    t->data = (const uint8_t *)g->map + g->data_off + t->offset;
  }
}

void gguf_close(gguf_file *g) {
  for (uint64_t i = 0; i < g->n_kv; i++) {
    free(g->kv[i].key);
    free(g->kv[i].arr_str);
  }
  for (uint64_t i = 0; i < g->n_tensors; i++) free(g->t[i].name);
  free(g->kv);
  free(g->t);
  munmap(g->map, g->size ? g->size : 1);
  memset(g, 0, sizeof *g);
}

const gguf_kv *gguf_find(const gguf_file *g, const char *key) {
  for (uint64_t i = 0; i < g->n_kv; i++)
    if (strcmp(g->kv[i].key, key) == 0) return &g->kv[i];
  return NULL;
}

int gguf_get_int(const gguf_file *g, const char *key, int64_t *out) {
  const gguf_kv *kv = gguf_find(g, key);
  if (!kv) return 0;
  switch (kv->type) {
    case GGUF_U8: case GGUF_U16: case GGUF_U32: case GGUF_U64: case GGUF_BOOL: *out = (int64_t)kv->v.u; return 1;
    case GGUF_I8: case GGUF_I16: case GGUF_I32: case GGUF_I64: *out = kv->v.i; return 1;
    default: return 0;
  }
}

int gguf_get_float(const gguf_file *g, const char *key, double *out) {
  const gguf_kv *kv = gguf_find(g, key);
  int64_t i;
  if (!kv) return 0;
  if (kv->type == GGUF_F32 || kv->type == GGUF_F64) {
    *out = kv->v.f;
    return 1;
  }
  if (gguf_get_int(g, key, &i)) {
    *out = (double)i;
    return 1;
  }
  return 0;
}

char *gguf_get_str(const gguf_file *g, const char *key) {
  const gguf_kv *kv = gguf_find(g, key);
  return kv && kv->type == GGUF_STR ? dup_str(kv->v.s) : NULL;
}

const gguf_tensor *gguf_tensor_find(const gguf_file *g, const char *name) {
  for (uint64_t i = 0; i < g->n_tensors; i++)
    if (strcmp(g->t[i].name, name) == 0) return &g->t[i];
  return NULL;
}

void gguf_print(const gguf_file *g, FILE *f) {
  fprintf(f, "GGUF v%u: %llu keys, %llu tensors, alignment %u\n", g->version, (unsigned long long)g->n_kv,
          (unsigned long long)g->n_tensors, g->alignment);
  for (uint64_t i = 0; i < g->n_kv; i++) {
    const gguf_kv *kv = &g->kv[i];
    fprintf(f, "  %s = ", kv->key);
    if (kv->type == GGUF_STR) fprintf(f, "\"%.*s\"", (int)(kv->v.s.n > 60 ? 60 : kv->v.s.n), kv->v.s.p);
    else if (kv->type == GGUF_ARR) fprintf(f, "[array type %u, %llu items]", kv->arr_type, (unsigned long long)kv->arr_n);
    else if (kv->type == GGUF_F32 || kv->type == GGUF_F64) fprintf(f, "%g", kv->v.f);
    else if (kv->type == GGUF_I8 || kv->type == GGUF_I16 || kv->type == GGUF_I32 || kv->type == GGUF_I64) fprintf(f, "%lld", (long long)kv->v.i);
    else fprintf(f, "%llu", (unsigned long long)kv->v.u);
    fprintf(f, "\n");
  }
}
