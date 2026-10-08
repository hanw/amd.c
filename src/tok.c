/* tok.c -- text to token ids (see tok.h).
 *
 * Steps for one text:
 *   1. Split on special tokens (longest match first).
 *   2. Split each other part with the pre-tokenizer regex (hand-written
 *      matcher, below), on code points.
 *   3. Map each piece's bytes to the GPT2 byte-level characters and apply
 *      the BPE merges (lowest rank first, leftmost first on equal rank).
 *   4. Look up each final symbol; a symbol not in the vocabulary becomes one
 *      token per byte (as llama.cpp).
 * Checked against llama.cpp's test vocabularies (models/ggml-vocab-qwen2.gguf
 * and ggml-vocab-qwen35.gguf with their .inp/.out files): tools/tok_check.c. */
#include <string.h>

#include "common.h"
#include "tok.h"
#include "tok_unicode.h"

/* ---- hash map: byte string -> u32 ---- */
typedef struct {
  const char *s; /* NULL: empty slot */
  uint32_t n, v;
} hent;
typedef struct {
  hent *t;
  uint32_t cap; /* power of 2 */
} hmap;

static uint64_t fnv1a(const char *s, uint32_t n) {
  uint64_t h = 1469598103934665603ull;
  for (uint32_t i = 0; i < n; i++) h = (h ^ (unsigned char)s[i]) * 1099511628211ull;
  return h;
}
static void hmap_init(hmap *h, uint32_t n) {
  h->cap = 16;
  while (h->cap < 2 * n + 16) h->cap *= 2;
  h->t = calloc(h->cap, sizeof(hent));
  if (!h->t) ie_die("tok: out of memory");
}
/* s must live as long as the map. The first value for a key is kept. */
static void hmap_put(hmap *h, const char *s, uint32_t n, uint32_t v) {
  for (uint64_t i = fnv1a(s, n) & (h->cap - 1);; i = (i + 1) & (h->cap - 1)) {
    hent *e = &h->t[i];
    if (!e->s) { e->s = s, e->n = n, e->v = v; return; }
    if (e->n == n && !memcmp(e->s, s, n)) return;
  }
}
static int64_t hmap_get(const hmap *h, const char *s, uint32_t n) {
  for (uint64_t i = fnv1a(s, n) & (h->cap - 1);; i = (i + 1) & (h->cap - 1)) {
    const hent *e = &h->t[i];
    if (!e->s) return -1;
    if (e->n == n && !memcmp(e->s, s, n)) return e->v;
  }
}

/* ---- the tokenizer ---- */
enum { PRE_QWEN2 = 0, PRE_QWEN35 = 1 };
typedef struct {
  const char *s;
  uint32_t n, id;
  int control; /* 1: type 3 (control), 0: type 4 (user-defined) */
} special;

struct tokenizer {
  int pre;
  char *pre_name;
  hmap vocab;  /* token string -> id (strings: the model's) */
  hmap merges; /* "left right" -> rank */
  char *mbuf;  /* the merge strings */
  special *sp; /* longest first */
  uint32_t n_sp;
  uint8_t sp_first[256]; /* 1: some special token starts with this byte */
  uint32_t byte_id[256]; /* the token of each single byte (UINT32_MAX: none) */
  char bchar[256][3];    /* GPT2 byte -> character (UTF-8, 1 or 2 bytes) */
  uint8_t blen[256];
};

void tok_ids_push(tok_ids *a, uint32_t id) {
  if (a->n == a->cap) {
    a->cap = a->cap ? 2 * a->cap : 64;
    a->v = realloc(a->v, (size_t)a->cap * 4);
    if (!a->v) ie_die("tok: out of memory");
  }
  a->v[a->n++] = id;
}
void tok_ids_free(tok_ids *a) {
  free(a->v);
  memset(a, 0, sizeof *a);
}

/* GPT2 bytes_to_unicode(): printable bytes map to themselves, the other 68 to 256, 257, ... */
static void byte_chars(tokenizer *t) {
  uint32_t n = 0;
  for (uint32_t b = 0; b < 256; b++) {
    const int keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    const uint32_t cp = keep ? b : 256 + n++;
    if (cp < 0x80) t->bchar[b][0] = (char)cp, t->blen[b] = 1;
    else t->bchar[b][0] = (char)(0xC0 | (cp >> 6)), t->bchar[b][1] = (char)(0x80 | (cp & 0x3F)), t->blen[b] = 2;
  }
}

static int cmp_special(const void *a, const void *b) {
  const special *x = a, *y = b;
  return x->n != y->n ? (x->n > y->n ? -1 : 1) : (x->id < y->id ? -1 : x->id > y->id);
}

tokenizer *tok_open(const model *m, const gguf_file *g, char *err, size_t errn) {
  if (!m->tok_gpt2 || !m->n_tokens) { snprintf(err, errn, "the vocabulary is not GPT2 byte-level BPE"); return NULL; }
  char *pre = gguf_get_str(g, "tokenizer.ggml.pre");
  int pt;
  if (pre && !strcmp(pre, "qwen35")) pt = PRE_QWEN35;
  else if (pre && (!strcmp(pre, "qwen2") || !strcmp(pre, "deepseek-r1-qwen"))) pt = PRE_QWEN2;
  else {
    snprintf(err, errn, "pre-tokenizer \"%s\" is not supported (qwen2, qwen35)", pre ? pre : "(none)");
    free(pre);
    return NULL;
  }
  const gguf_kv *mk = gguf_find(g, "tokenizer.ggml.merges");
  if (!mk || mk->type != GGUF_ARR || mk->arr_type != GGUF_STR) {
    snprintf(err, errn, "no tokenizer.ggml.merges");
    free(pre);
    return NULL;
  }
  tokenizer *t = calloc(1, sizeof *t);
  if (!t) ie_die("tok: out of memory");
  t->pre = pt, t->pre_name = pre;
  byte_chars(t);

  hmap_init(&t->vocab, m->n_tokens);
  for (uint32_t i = 0; i < m->n_tokens; i++) hmap_put(&t->vocab, m->tokens[i], (uint32_t)strlen(m->tokens[i]), i);

  /* merges: one buffer for all strings */
  size_t tot = 0;
  for (uint64_t i = 0; i < mk->arr_n; i++) tot += mk->arr_str[i].n;
  t->mbuf = malloc(tot ? tot : 1);
  if (!t->mbuf) ie_die("tok: out of memory");
  hmap_init(&t->merges, (uint32_t)mk->arr_n);
  size_t o = 0;
  for (uint64_t i = 0; i < mk->arr_n; i++) {
    memcpy(t->mbuf + o, mk->arr_str[i].p, mk->arr_str[i].n);
    hmap_put(&t->merges, t->mbuf + o, (uint32_t)mk->arr_str[i].n, (uint32_t)i);
    o += mk->arr_str[i].n;
  }

  /* special tokens: type 3 (control) and 4 (user-defined) */
  const gguf_kv *tt = gguf_find(g, "tokenizer.ggml.token_type");
  if (tt && tt->type == GGUF_ARR && tt->arr_type == GGUF_I32) {
    t->sp = calloc(tt->arr_n + 1, sizeof(special));
    for (uint64_t i = 0; i < tt->arr_n && i < m->n_tokens; i++) {
      int32_t ty;
      memcpy(&ty, tt->arr_data + i * 4, 4); /* the file data may be unaligned */
      const uint32_t n = (uint32_t)strlen(m->tokens[i]);
      if ((ty == 3 || ty == 4) && n) {
        t->sp[t->n_sp++] = (special){m->tokens[i], n, (uint32_t)i, ty == 3};
        t->sp_first[(unsigned char)m->tokens[i][0]] = 1;
      }
    }
    qsort(t->sp, t->n_sp, sizeof(special), cmp_special);
  }

  for (uint32_t b = 0; b < 256; b++) {
    const int64_t id = hmap_get(&t->vocab, t->bchar[b], t->blen[b]);
    t->byte_id[b] = id < 0 ? UINT32_MAX : (uint32_t)id;
  }
  return t;
}

void tok_close(tokenizer *t) {
  if (!t) return;
  free(t->vocab.t), free(t->merges.t), free(t->mbuf), free(t->sp), free(t->pre_name);
  free(t);
}

int64_t tok_find(const tokenizer *t, const char *s) { return hmap_get(&t->vocab, s, (uint32_t)strlen(s)); }
const char *tok_pre(const tokenizer *t) { return t->pre_name; }

/* ---- code point classes ---- */
static int uc_flags(uint32_t cp) {
  uint32_t lo = 0, hi = UC_NRANGES;
  while (lo < hi) {
    const uint32_t mid = (lo + hi) / 2;
    if (cp < uc_ranges[mid][0]) hi = mid;
    else if (cp > uc_ranges[mid][1]) lo = mid + 1;
    else return (int)uc_ranges[mid][2];
  }
  return 0;
}
/* \s: the Unicode White_Space property */
static int uc_space(uint32_t cp) {
  return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
         cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

/* A part of the text as code points: cp[i] starts at byte off[i]. An
 * invalid byte becomes one code point of class "other" (cp 0xFFFFFFFF). */
typedef struct {
  uint32_t *cp, *off;
  uint8_t *cl; /* bits: UC_L, UC_M, UC_N, 8 = space, 16 = CR or LF */
  uint32_t n;
} cps;

static void decode(const char *s, size_t len, cps *c) {
  c->cp = malloc((len + 1) * 4), c->off = malloc((len + 1) * 4), c->cl = malloc(len + 1);
  if (!c->cp || !c->off || !c->cl) ie_die("tok: out of memory");
  uint32_t n = 0;
  const unsigned char *p = (const unsigned char *)s;
  size_t i = 0;
  while (i < len) {
    const uint32_t b0 = p[i];
    uint32_t cp, need, k = 1;
    if (b0 < 0x80) cp = b0, need = 0;
    else if ((b0 & 0xE0) == 0xC0) cp = b0 & 31, need = 1;
    else if ((b0 & 0xF0) == 0xE0) cp = b0 & 15, need = 2;
    else if ((b0 & 0xF8) == 0xF0) cp = b0 & 7, need = 3;
    else cp = 0xFFFFFFFFu, need = 0; /* a continuation byte alone, or 0xF8 .. 0xFF */
    if (need && i + need >= len) cp = 0xFFFFFFFFu, need = 0; /* cut off at the end */
    for (uint32_t j = 1; j <= need; j++) {
      if ((p[i + j] & 0xC0) != 0x80) { cp = 0xFFFFFFFFu; break; }
      cp = (cp << 6) | (p[i + j] & 0x3F), k++;
    }
    if (cp == 0xFFFFFFFFu) k = 1;
    c->cp[n] = cp, c->off[n] = (uint32_t)i;
    uint8_t cl = cp == 0xFFFFFFFFu ? 0 : (uint8_t)uc_flags(cp);
    if (cp != 0xFFFFFFFFu && uc_space(cp)) cl |= 8;
    if (cp == '\r' || cp == '\n') cl |= 16;
    c->cl[n++] = cl;
    i += k;
  }
  c->off[n] = (uint32_t)len;
  c->n = n;
}

static uint32_t lower_ascii(uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

/* The end of the regex match at i (> i). Qwen2:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * Qwen3.5: \p{L} -> [\p{L}\p{M}] in the second alternative, and \p{M} also
 * excluded in the fourth. */
static uint32_t match(const cps *c, uint32_t i, int pre) {
  const uint32_t n = c->n;
  const uint8_t *cl = c->cl;
  const uint8_t LET = pre == PRE_QWEN35 ? (UC_L | UC_M) : UC_L, OTH = LET | UC_N | 8;
  /* 1: contractions */
  if (c->cp[i] == '\'' && i + 1 < n) {
    const uint32_t a = lower_ascii(c->cp[i + 1]);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
    if (i + 2 < n) {
      const uint32_t b = lower_ascii(c->cp[i + 2]);
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
    }
  }
  /* 2: one optional char that is not CR, LF, L or N; then letters */
  {
    uint32_t j = i;
    if (!(cl[j] & (LET | 16 | UC_L | UC_N)) && j + 1 < n && (cl[j + 1] & LET)) j++;
    if (cl[j] & LET) {
      while (j < n && (cl[j] & LET)) j++;
      return j;
    }
  }
  /* 3: one number */
  if (cl[i] & UC_N) return i + 1;
  /* 4: optional space, chars that are not space, L or N, then CR and LF */
  {
    uint32_t j = i + (c->cp[i] == ' ' ? 1 : 0), k = j;
    while (k < n && !(cl[k] & OTH)) k++;
    if (k > j) {
      while (k < n && (cl[k] & 16)) k++;
      return k;
    }
  }
  if (cl[i] & 8) {
    uint32_t j = i;
    while (j < n && (cl[j] & 8)) j++;
    /* 5: \s*[\r\n]+ ends after the last CR or LF of the space run */
    for (uint32_t k = j; k > i; k--)
      if (cl[k - 1] & 16) return k;
    /* 6: \s+(?!\S) */
    if (j == n) return j;
    if (j - 1 > i) return j - 1;
    /* 7: \s+ */
    return j;
  }
  return i + 1;
}

/* BPE on one piece (bytes s[0 .. len)). */
static void bpe(const tokenizer *t, const char *s, uint32_t len, tok_ids *out) {
  /* the GPT2 characters of the bytes */
  char *buf = malloc((size_t)len * 2 + 1);
  uint32_t *off = malloc((size_t)(len + 1) * 4), *ln = malloc((size_t)(len + 1) * 4), *rk = malloc((size_t)(len + 1) * 4);
  if (!buf || !off || !ln || !rk) ie_die("tok: out of memory");
  uint32_t bn = 0, ns = 0;
  for (uint32_t i = 0; i < len; i++) {
    const unsigned char b = (unsigned char)s[i];
    off[ns] = bn, ln[ns] = t->blen[b], ns++;
    memcpy(buf + bn, t->bchar[b], t->blen[b]);
    bn += t->blen[b];
  }
  char *key = malloc((size_t)bn + 2);
  if (!key) ie_die("tok: out of memory");
#define RANK(k)                                                                       \
  ((k) + 1 < ns ? (memcpy(key, buf + off[k], ln[k]), key[ln[k]] = ' ',               \
                   memcpy(key + ln[k] + 1, buf + off[(k) + 1], ln[(k) + 1]),          \
                   (uint32_t)hmap_get(&t->merges, key, ln[k] + 1 + ln[(k) + 1]))      \
                : UINT32_MAX)
  for (uint32_t k = 0; k < ns; k++) rk[k] = RANK(k); /* -1 (no merge) is UINT32_MAX */
  for (;;) {
    uint32_t best = UINT32_MAX, bk = 0;
    for (uint32_t k = 0; k + 1 < ns; k++)
      if (rk[k] < best) best = rk[k], bk = k;
    if (best == UINT32_MAX) break;
    ln[bk] += ln[bk + 1]; /* the symbols are adjacent in buf */
    memmove(off + bk + 1, off + bk + 2, (ns - bk - 2) * 4);
    memmove(ln + bk + 1, ln + bk + 2, (ns - bk - 2) * 4);
    memmove(rk + bk + 1, rk + bk + 2, (ns - bk - 2) * 4);
    ns--;
    rk[bk] = RANK(bk);
    if (bk) rk[bk - 1] = RANK(bk - 1);
  }
#undef RANK
  for (uint32_t k = 0; k < ns; k++) {
    const int64_t id = hmap_get(&t->vocab, buf + off[k], ln[k]);
    if (id >= 0) { tok_ids_push(out, (uint32_t)id); continue; }
    /* not in the vocabulary: one token per byte (find the bytes of this symbol) */
    for (uint32_t p = off[k]; p < off[k] + ln[k];) {
      const unsigned char c0 = (unsigned char)buf[p];
      const uint32_t cl = c0 < 0x80 ? 1 : 2;
      for (uint32_t b = 0; b < 256; b++)
        if (t->blen[b] == cl && !memcmp(t->bchar[b], buf + p, cl)) {
          if (t->byte_id[b] != UINT32_MAX) tok_ids_push(out, t->byte_id[b]);
          break;
        }
      p += cl;
    }
  }
  free(buf), free(off), free(ln), free(rk), free(key);
}

/* A part without special tokens. */
static void encode_plain(const tokenizer *t, const char *s, size_t len, tok_ids *out) {
  if (!len) return;
  cps c;
  decode(s, len, &c);
  for (uint32_t i = 0; i < c.n;) {
    const uint32_t j = match(&c, i, t->pre);
    bpe(t, s + c.off[i], c.off[j] - c.off[i], out);
    i = j;
  }
  free(c.cp), free(c.off), free(c.cl);
}

void tok_encode(const tokenizer *t, const char *text, size_t len, int parse_control, tok_ids *out) {
  size_t a = 0; /* start of the current plain part */
  for (size_t i = 0; i < len;) {
    const special *hit = NULL;
    if (t->sp_first[(unsigned char)text[i]])
      for (uint32_t k = 0; k < t->n_sp; k++) { /* longest first */
        const special *sp = &t->sp[k];
        if ((parse_control || !sp->control) && sp->n <= len - i && !memcmp(text + i, sp->s, sp->n)) { hit = sp; break; }
      }
    if (!hit) { i++; continue; }
    encode_plain(t, text + a, i - a, out);
    tok_ids_push(out, hit->id);
    i += hit->n, a = i;
  }
  encode_plain(t, text + a, len - a, out);
}
