/* json.h -- a small JSON parser (for HTTP request bodies) and a string
 * buffer that writes JSON. */
#ifndef IE_JSON_H
#define IE_JSON_H

#include <stddef.h>

typedef enum { J_NULL, J_FALSE, J_TRUE, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jval {
  jtype t;
  double num;
  char *str;    /* J_STR: decoded UTF-8, 0-terminated */
  size_t slen;  /* J_STR: bytes (the string may hold 0 bytes) */
  char *key;    /* a member of an object: its key */
  struct jval *kid, *next; /* J_ARR, J_OBJ: the first element; the next sibling */
} jval;

/* NULL on error (a message in err). Depth at most 64. */
jval *json_parse(const char *s, size_t n, char *err, size_t errn);
void json_free(jval *v);
/* The member key of object o, or NULL. */
const jval *json_get(const jval *o, const char *key);

typedef struct {
  char *p; /* 0-terminated */
  size_t n, cap;
} sbuf;

void sb_add(sbuf *b, const char *s, size_t n);
void sb_puts(sbuf *b, const char *s);
void sb_printf(sbuf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
/* s[0 .. n) as a JSON string (with the quotes). Invalid UTF-8 bytes become U+FFFD. */
void sb_json_str(sbuf *b, const char *s, size_t n);
void sb_free(sbuf *b);

/* The length of the longest prefix of s[0 .. n) that does not end inside a
 * UTF-8 sequence (a cut sequence of at most 3 bytes at the end is left out). */
size_t utf8_whole(const char *s, size_t n);

#endif
