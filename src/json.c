/* json.c -- see json.h. */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

/* ---- string buffer ---- */
static void sb_grow(sbuf *b, size_t extra) {
  if (b->n + extra + 1 <= b->cap) return;
  size_t c = b->cap ? b->cap : 256;
  while (c < b->n + extra + 1) c *= 2;
  char *p = realloc(b->p, c);
  if (!p) {
    fprintf(stderr, "sbuf: out of memory\n");
    exit(1);
  }
  b->p = p, b->cap = c;
}
void sb_add(sbuf *b, const char *s, size_t n) {
  sb_grow(b, n);
  memcpy(b->p + b->n, s, n);
  b->n += n;
  b->p[b->n] = 0;
}
void sb_puts(sbuf *b, const char *s) { sb_add(b, s, strlen(s)); }
void sb_printf(sbuf *b, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int k = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  if (k < 0) return;
  sb_grow(b, (size_t)k);
  va_start(ap, fmt);
  vsnprintf(b->p + b->n, (size_t)k + 1, fmt, ap);
  va_end(ap);
  b->n += (size_t)k;
}
void sb_free(sbuf *b) {
  free(b->p);
  b->p = NULL, b->n = b->cap = 0;
}

/* The length of the valid UTF-8 sequence at s (n bytes left), or 0. */
static size_t utf8_seq(const unsigned char *s, size_t n) {
  size_t k;
  if (s[0] < 0x80) return 1;
  if (s[0] >= 0xC2 && s[0] <= 0xDF) k = 2;
  else if (s[0] >= 0xE0 && s[0] <= 0xEF) k = 3;
  else if (s[0] >= 0xF0 && s[0] <= 0xF4) k = 4;
  else return 0;
  if (n < k) return 0;
  for (size_t i = 1; i < k; i++)
    if ((s[i] & 0xC0) != 0x80) return 0;
  if (k == 3 && s[0] == 0xE0 && s[1] < 0xA0) return 0; /* overlong */
  if (k == 3 && s[0] == 0xED && s[1] >= 0xA0) return 0; /* surrogates */
  if (k == 4 && s[0] == 0xF0 && s[1] < 0x90) return 0;
  if (k == 4 && s[0] == 0xF4 && s[1] >= 0x90) return 0;
  return k;
}

size_t utf8_whole(const char *s, size_t n) {
  const unsigned char *p = (const unsigned char *)s;
  /* look back at most 3 bytes for the start of a cut sequence */
  for (size_t back = 1; back <= 3 && back <= n; back++) {
    const unsigned char c = p[n - back];
    if ((c & 0xC0) == 0x80) continue; /* a continuation byte */
    size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    return need > back ? n - back : n;
  }
  return n;
}

void sb_json_str(sbuf *b, const char *s, size_t n) {
  const unsigned char *p = (const unsigned char *)s;
  sb_add(b, "\"", 1);
  size_t i = 0;
  while (i < n) {
    const unsigned char c = p[i];
    if (c == '"') sb_add(b, "\\\"", 2), i++;
    else if (c == '\\') sb_add(b, "\\\\", 2), i++;
    else if (c == '\n') sb_add(b, "\\n", 2), i++;
    else if (c == '\r') sb_add(b, "\\r", 2), i++;
    else if (c == '\t') sb_add(b, "\\t", 2), i++;
    else if (c < 0x20 || c == 0x7F) sb_printf(b, "\\u%04x", c), i++;
    else if (c < 0x80) sb_add(b, (const char *)p + i, 1), i++;
    else {
      const size_t k = utf8_seq(p + i, n - i);
      if (k) sb_add(b, (const char *)p + i, k), i += k;
      else sb_add(b, "\xEF\xBF\xBD", 3), i++; /* U+FFFD */
    }
  }
  sb_add(b, "\"", 1);
}

/* ---- parser ---- */
typedef struct {
  const char *s, *e;
  char *err;
  size_t errn;
  int depth;
} jp;

static void ws(jp *p) {
  while (p->s < p->e && (*p->s == ' ' || *p->s == '\t' || *p->s == '\n' || *p->s == '\r')) p->s++;
}
static jval *fail(jp *p, const char *msg) {
  if (p->err && !p->err[0]) snprintf(p->err, p->errn, "json: %s", msg);
  return NULL;
}
static jval *mk(jtype t) {
  jval *v = calloc(1, sizeof *v);
  if (!v) {
    fprintf(stderr, "json: out of memory\n");
    exit(1);
  }
  v->t = t;
  return v;
}
static int hex4(const char *s, unsigned *out) {
  unsigned v = 0;
  for (int i = 0; i < 4; i++) {
    const char c = s[i];
    v <<= 4;
    if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
    else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
    else return 0;
  }
  *out = v;
  return 1;
}
static void put_utf8(sbuf *b, unsigned cp) {
  char o[4];
  size_t k;
  if (cp < 0x80) o[0] = (char)cp, k = 1;
  else if (cp < 0x800) o[0] = (char)(0xC0 | (cp >> 6)), o[1] = (char)(0x80 | (cp & 0x3F)), k = 2;
  else if (cp < 0x10000)
    o[0] = (char)(0xE0 | (cp >> 12)), o[1] = (char)(0x80 | ((cp >> 6) & 0x3F)), o[2] = (char)(0x80 | (cp & 0x3F)), k = 3;
  else
    o[0] = (char)(0xF0 | (cp >> 18)), o[1] = (char)(0x80 | ((cp >> 12) & 0x3F)), o[2] = (char)(0x80 | ((cp >> 6) & 0x3F)),
    o[3] = (char)(0x80 | (cp & 0x3F)), k = 4;
  sb_add(b, o, k);
}
/* at the opening quote; returns a malloc'ed string */
static char *pstr(jp *p, size_t *len) {
  p->s++;
  sbuf b = {0};
  sb_grow(&b, 16);
  while (p->s < p->e && *p->s != '"') {
    const char c = *p->s++;
    if ((unsigned char)c < 0x20) { sb_free(&b); fail(p, "control character in a string"); return NULL; }
    if (c != '\\') { sb_add(&b, &c, 1); continue; }
    if (p->s >= p->e) break;
    const char x = *p->s++;
    switch (x) {
    case '"': sb_add(&b, "\"", 1); break;
    case '\\': sb_add(&b, "\\", 1); break;
    case '/': sb_add(&b, "/", 1); break;
    case 'b': sb_add(&b, "\b", 1); break;
    case 'f': sb_add(&b, "\f", 1); break;
    case 'n': sb_add(&b, "\n", 1); break;
    case 'r': sb_add(&b, "\r", 1); break;
    case 't': sb_add(&b, "\t", 1); break;
    case 'u': {
      unsigned cp, lo;
      if (p->e - p->s < 4 || !hex4(p->s, &cp)) { sb_free(&b); fail(p, "bad \\u escape"); return NULL; }
      p->s += 4;
      if (cp >= 0xD800 && cp <= 0xDBFF) { /* a surrogate pair */
        if (p->e - p->s >= 6 && p->s[0] == '\\' && p->s[1] == 'u' && hex4(p->s + 2, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
          cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          p->s += 6;
        } else cp = 0xFFFD;
      } else if (cp >= 0xDC00 && cp <= 0xDFFF) cp = 0xFFFD;
      put_utf8(&b, cp);
      break;
    }
    default: sb_free(&b); fail(p, "bad escape"); return NULL;
    }
  }
  if (p->s >= p->e) { sb_free(&b); fail(p, "unterminated string"); return NULL; }
  p->s++;
  *len = b.n;
  return b.p;
}
static jval *pval(jp *p);
static jval *pval(jp *p) {
  ws(p);
  if (p->s >= p->e) return fail(p, "unexpected end");
  const char c = *p->s;
  if (c == '{' || c == '[') {
    if (++p->depth > 64) return fail(p, "too deep");
    const int obj = c == '{';
    jval *v = mk(obj ? J_OBJ : J_ARR), **tail = &v->kid;
    p->s++;
    ws(p);
    if (p->s < p->e && *p->s == (obj ? '}' : ']')) { p->s++, p->depth--; return v; }
    for (;;) {
      char *key = NULL;
      if (obj) {
        ws(p);
        size_t kl;
        if (p->s >= p->e || *p->s != '"' || !(key = pstr(p, &kl))) { json_free(v); return fail(p, "expected a key"); }
        ws(p);
        if (p->s >= p->e || *p->s != ':') { free(key); json_free(v); return fail(p, "expected ':'"); }
        p->s++;
      }
      jval *k = pval(p);
      if (!k) { free(key); json_free(v); return NULL; }
      k->key = key;
      *tail = k, tail = &k->next;
      ws(p);
      if (p->s < p->e && *p->s == ',') { p->s++; continue; }
      if (p->s < p->e && *p->s == (obj ? '}' : ']')) { p->s++; break; }
      json_free(v);
      return fail(p, obj ? "expected ',' or '}'" : "expected ',' or ']'");
    }
    p->depth--;
    return v;
  }
  if (c == '"') {
    jval *v = mk(J_STR);
    if (!(v->str = pstr(p, &v->slen))) { free(v); return NULL; }
    return v;
  }
  if (p->e - p->s >= 4 && !memcmp(p->s, "true", 4)) { p->s += 4; return mk(J_TRUE); }
  if (p->e - p->s >= 5 && !memcmp(p->s, "false", 5)) { p->s += 5; return mk(J_FALSE); }
  if (p->e - p->s >= 4 && !memcmp(p->s, "null", 4)) { p->s += 4; return mk(J_NULL); }
  if (c == '-' || (c >= '0' && c <= '9')) {
    char tmp[64];
    size_t k = 0;
    while (p->s < p->e && k < sizeof tmp - 1 && strchr("+-0123456789.eE", *p->s)) tmp[k++] = *p->s++;
    tmp[k] = 0;
    char *end;
    jval *v = mk(J_NUM);
    v->num = strtod(tmp, &end);
    if (*end) { free(v); return fail(p, "bad number"); }
    return v;
  }
  return fail(p, "unexpected character");
}

jval *json_parse(const char *s, size_t n, char *err, size_t errn) {
  if (err && errn) err[0] = 0;
  jp p = {s, s + n, err, errn, 0};
  jval *v = pval(&p);
  if (!v) return NULL;
  ws(&p);
  if (p.s != p.e) { json_free(v); return fail(&p, "text after the value"); }
  return v;
}

void json_free(jval *v) {
  while (v) {
    jval *nx = v->next;
    json_free(v->kid);
    free(v->str), free(v->key), free(v);
    v = nx;
  }
}

const jval *json_get(const jval *o, const char *key) {
  if (!o || o->t != J_OBJ) return NULL;
  for (const jval *k = o->kid; k; k = k->next)
    if (k->key && !strcmp(k->key, key)) return k;
  return NULL;
}
