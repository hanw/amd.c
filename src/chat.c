/* chat.c -- see chat.h. The structure follows the template's Jinja code
 * (ie-serve --show-template); the checks are in SERVE.md section 6. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat.h"

/* The literal texts of the template's tools system message. */
static const char TOOLS_HEAD[] =
"# Tools\n"
"\n"
"You have access to the following functions:\n"
"\n"
"<tools>"
;
static const char TOOLS_INSTR[] =
"\n"
"\n"
"If you choose to call a function ONLY reply in the following format with NO suffix:\n"
"\n"
"<tool_call>\n"
"<function=example_function_name>\n"
"<parameter=example_parameter_1>\n"
"value_1\n"
"</parameter>\n"
"<parameter=example_parameter_2>\n"
"This is the value for the second parameter\n"
"that can span\n"
"multiple lines\n"
"</parameter>\n"
"</function>\n"
"</tool_call>\n"
"\n"
"<IMPORTANT>\n"
"Reminder:\n"
"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
"- Required parameters MUST be specified\n"
"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
"</IMPORTANT>"
;

/* ---- helpers ---- */

/* Python's str.isspace() for one code point */
static int py_space(unsigned cp) {
  return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
/* the code point at s (n > 0 bytes left) and its length; an invalid byte is itself */
static unsigned cp_at(const unsigned char *s, size_t n, size_t *len) {
  unsigned c = s[0], k = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
  if (k > n) k = 1;
  if (k == 1) { *len = 1; return c; }
  unsigned cp = c & (0x3F >> (k - 1));
  for (unsigned i = 1; i < k; i++) {
    if ((s[i] & 0xC0) != 0x80) { *len = 1; return c; }
    cp = (cp << 6) | (s[i] & 0x3F);
  }
  *len = k;
  return cp;
}
/* Jinja's |trim (Python's str.strip()) */
static void py_trim(const char **s, size_t *n) {
  const unsigned char *p = (const unsigned char *)*s;
  size_t a = 0, e = *n, len;
  while (a < e && py_space(cp_at(p + a, e - a, &len))) a += len;
  while (e > a) { /* back up to the start of the last code point */
    size_t b = e - 1;
    while (b > a && (p[b] & 0xC0) == 0x80 && e - b < 4) b--;
    if (!py_space(cp_at(p + b, e - b, &len)) || b + len != e) break;
    e = b;
  }
  *s += a, *n = e - a;
}

static void seg_push(chat_prompt *p, int kind, size_t off, size_t n) {
  if (kind == SEG_TEXT && !n) return;
  if (kind == SEG_TEXT && p->n_seg && p->seg[p->n_seg - 1].kind == SEG_TEXT &&
      p->seg[p->n_seg - 1].off + p->seg[p->n_seg - 1].n == off) { /* join adjacent text */
    p->seg[p->n_seg - 1].n += n;
    return;
  }
  if (p->n_seg == p->cap) {
    p->cap = p->cap ? 2 * p->cap : 32;
    p->seg = realloc(p->seg, p->cap * sizeof(chat_seg));
    if (!p->seg) abort();
  }
  p->seg[p->n_seg++] = (chat_seg){kind, off, n};
}
static void T(chat_prompt *p, const char *s, size_t n) {
  const size_t off = p->text.n;
  sb_add(&p->text, s, n);
  seg_push(p, SEG_TEXT, off, n);
}
static void TS(chat_prompt *p, const char *s) { T(p, s, strlen(s)); }
static void START(chat_prompt *p) {
  sb_puts(&p->text, "<|im_start|>");
  seg_push(p, SEG_IM_START, 0, 0);
}
static void END(chat_prompt *p) {
  sb_puts(&p->text, "<|im_end|>");
  seg_push(p, SEG_IM_END, 0, 0);
}

void chat_prompt_free(chat_prompt *p) {
  sb_free(&p->text);
  free(p->seg);
  memset(p, 0, sizeof *p);
}

/* ---- Python json.dumps ---- */
static void py_str(sbuf *b, const char *s, size_t n) {
  sb_add(b, "\"", 1);
  for (size_t i = 0; i < n; i++) {
    const unsigned char c = (unsigned char)s[i];
    if (c == '"') sb_add(b, "\\\"", 2);
    else if (c == '\\') sb_add(b, "\\\\", 2);
    else if (c == '\n') sb_add(b, "\\n", 2);
    else if (c == '\r') sb_add(b, "\\r", 2);
    else if (c == '\t') sb_add(b, "\\t", 2);
    else if (c == '\b') sb_add(b, "\\b", 2);
    else if (c == '\f') sb_add(b, "\\f", 2);
    else if (c < 0x20) sb_printf(b, "\\u%04x", c);
    else sb_add(b, (const char *)s + i, 1);
  }
  sb_add(b, "\"", 1);
}
void py_json(sbuf *b, const jval *v) {
  switch (v->t) {
  case J_NULL: sb_puts(b, "null"); break;
  case J_TRUE: sb_puts(b, "true"); break;
  case J_FALSE: sb_puts(b, "false"); break;
  case J_NUM: sb_puts(b, v->raw ? v->raw : "0"); break;
  case J_STR: py_str(b, v->str, v->slen); break;
  case J_ARR:
  case J_OBJ:
    sb_add(b, v->t == J_ARR ? "[" : "{", 1);
    for (const jval *k = v->kid; k; k = k->next) {
      if (k != v->kid) sb_add(b, ", ", 2);
      if (v->t == J_OBJ) py_str(b, k->key, strlen(k->key)), sb_add(b, ": ", 2);
      py_json(b, k);
    }
    sb_add(b, v->t == J_ARR ? "]" : "}", 1);
    break;
  }
}

/* ---- the template ---- */

/* render_content: a string, null, or parts with "text" (no images) */
static int content_of(const jval *c, sbuf *out) {
  if (!c || c->t == J_NULL) return 1;
  if (c->t == J_STR) { sb_add(out, c->str, c->slen); return 1; }
  if (c->t != J_ARR) return 0;
  for (const jval *p = c->kid; p; p = p->next) {
    const jval *tx = json_get(p, "text");
    if (json_get(p, "image") || json_get(p, "image_url") || json_get(p, "video") || !tx || tx->t != J_STR) return 0;
    sb_add(out, tx->str, tx->slen);
  }
  return 1;
}

static const char *effort_text(const char *e) {
  if (!strcmp(e, "low") || !strcmp(e, "minimal"))
    return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
           "unnecessary elaboration.";
  if (!strcmp(e, "medium")) return "";
  return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
         "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
}

static const char *role_of(const jval *m) {
  const jval *r = json_get(m, "role");
  if (!r || r->t != J_STR) return NULL;
  return !strcmp(r->str, "developer") ? "system" : r->str;
}

/* An assistant tool call: <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n...</function>\n</tool_call> */
static int render_call(chat_prompt *p, const jval *tc, const char *lead, char *err, size_t errn) {
  const jval *f = json_get(tc, "function");
  if (f) tc = f;
  const jval *name = json_get(tc, "name"), *args = json_get(tc, "arguments");
  if (!name || name->t != J_STR) { snprintf(err, errn, "a tool call has no function name"); return 0; }
  TS(p, lead), TS(p, "<tool_call>\n<function="), T(p, name->str, name->slen), TS(p, ">\n");
  jval *parsed = NULL;
  if (args && args->t == J_STR && args->slen) { /* OpenAI sends the arguments as a JSON string */
    char e2[128];
    parsed = json_parse(args->str, args->slen, e2, sizeof e2);
    if (!parsed || parsed->t != J_OBJ) {
      json_free(parsed);
      snprintf(err, errn, "the arguments of a tool call are not a JSON object");
      return 0;
    }
    args = parsed;
  }
  if (args && args->t == J_OBJ)
    for (const jval *a = args->kid; a; a = a->next) {
      TS(p, "<parameter="), TS(p, a->key), TS(p, ">\n");
      if (a->t == J_STR) T(p, a->str, a->slen);
      else {
        sbuf j = {0};
        py_json(&j, a);
        T(p, j.p, j.n);
        sb_free(&j);
      }
      TS(p, "\n</parameter>\n");
    }
  TS(p, "</function>\n</tool_call>");
  json_free(parsed);
  return 1;
}

int chat_render(const jval *msgs, const jval *tools, int think, const char *effort, chat_prompt *p, char *err, size_t errn) {
  memset(p, 0, sizeof *p);
  if (!msgs || msgs->t != J_ARR || !msgs->kid) { snprintf(err, errn, "\"messages\" must be a non-empty array"); return 0; }
  if (tools && (tools->t != J_ARR || !tools->kid)) tools = NULL;
  const char *ri = think ? effort_text(effort) : "";
  size_t n_msg = 0;
  for (const jval *m = msgs->kid; m; m = m->next) n_msg++;
  const jval **ms = calloc(n_msg, sizeof *ms);
  sbuf *cs = calloc(n_msg, sizeof *cs); /* the trimmed contents */
  const char **rs = calloc(n_msg, sizeof *rs);
  size_t i = 0, *cn = calloc(n_msg, sizeof *cn);
  int ok = 0, user_query = 0;
  for (const jval *m = msgs->kid; m; m = m->next, i++) {
    ms[i] = m;
    const char *r = rs[i] = role_of(m);
    if (!r || (strcmp(r, "system") && strcmp(r, "user") && strcmp(r, "assistant") && strcmp(r, "tool"))) {
      snprintf(err, errn, "a message's role must be system, user, assistant or tool");
      goto done;
    }
    if (!strcmp(r, "system") && i) { snprintf(err, errn, "a system message must be the first message"); goto done; }
    if (!content_of(json_get(m, "content"), &cs[i])) { snprintf(err, errn, "only text content is supported"); goto done; }
    const char *c = cs[i].p ? cs[i].p : "";
    size_t n = cs[i].n;
    py_trim(&c, &n);
    if (cs[i].p) { /* keep the trimmed text at the start */
      memmove(cs[i].p, c, n);
      cs[i].p[n] = 0;
    }
    cn[i] = n;
    if (!strcmp(r, "user")) { /* the template needs a user message that is not only a tool response */
      const char *t = cs[i].p ? cs[i].p : "";
      if (!(n >= 15 + 16 && !memcmp(t, "<tool_response>", 15) && !memcmp(t + n - 16, "</tool_response>", 16))) user_query = 1;
    }
  }
  if (!user_query) { snprintf(err, errn, "no user message"); goto done; }

  /* the system message */
  const int sys = !strcmp(rs[0], "system");
  if (tools) {
    START(p), TS(p, "system\n");
    if (ri[0]) TS(p, ri), TS(p, "\n\n");
    TS(p, TOOLS_HEAD);
    for (const jval *t = tools->kid; t; t = t->next) {
      sbuf j = {0};
      py_json(&j, t);
      TS(p, "\n"), T(p, j.p, j.n);
      sb_free(&j);
    }
    TS(p, "\n</tools>"), TS(p, TOOLS_INSTR);
    if (sys && cn[0]) TS(p, "\n\n"), T(p, cs[0].p, cn[0]);
    END(p), TS(p, "\n");
  } else if ((sys && cn[0]) || ri[0]) {
    START(p), TS(p, "system\n");
    if (ri[0]) TS(p, ri), TS(p, sys && cn[0] ? "\n\n" : "");
    if (sys && cn[0]) T(p, cs[0].p, cn[0]);
    END(p), TS(p, "\n");
  }

  for (i = 0; i < n_msg; i++) {
    const char *r = rs[i], *c = cs[i].p ? cs[i].p : "";
    size_t n = cn[i];
    if (!strcmp(r, "user")) {
      START(p), TS(p, "user\n"), T(p, c, n), END(p), TS(p, "\n");
    } else if (!strcmp(r, "assistant")) {
      /* reasoning: reasoning_content, or (Open WebUI) the text before </think> */
      const char *rt = "";
      size_t rn = 0;
      const jval *rc = json_get(ms[i], "reasoning_content");
      const char *e = NULL;
      for (const char *q = c; (q = strstr(q, "</think>")) && q < c + n; q++) e = q;
      if (e) {
        rt = c, rn = (size_t)(e - c);
        py_trim(&rt, &rn);
        if (rn >= 7 && !memcmp(rt, "<think>", 7)) rt += 7, rn -= 7;
        n -= (size_t)(e + 8 - c), c = e + 8;
        py_trim(&c, &n);
      } else if (rc && rc->t == J_STR) rt = rc->str, rn = rc->slen;
      py_trim(&rt, &rn);
      START(p), TS(p, "assistant\n<think>\n"), T(p, rt, rn), TS(p, "\n</think>\n\n"), T(p, c, n);
      const jval *tcs = json_get(ms[i], "tool_calls");
      if (tcs && tcs->t == J_ARR)
        for (const jval *tc = tcs->kid; tc; tc = tc->next)
          if (!render_call(p, tc, tc != tcs->kid ? "\n" : n ? "\n\n" : "", err, errn)) goto done;
      END(p), TS(p, "\n");
    } else if (!strcmp(r, "tool")) {
      if (i && strcmp(rs[i - 1], "tool")) START(p), TS(p, "user");
      TS(p, "\n<tool_response>\n"), T(p, c, n), TS(p, "\n</tool_response>");
      if (i + 1 == n_msg || strcmp(rs[i + 1], "tool")) END(p), TS(p, "\n");
    }
  }
  START(p), TS(p, think ? "assistant\n<think>\n" : "assistant\n<think>\n\n</think>\n\n");
  ok = 1;
done:
  for (i = 0; i < n_msg; i++) sb_free(&cs[i]);
  free(cs), free(cn), free(rs), free(ms);
  if (!ok) chat_prompt_free(p);
  return ok;
}

/* ---- the model's tool calls ---- */

static const char *find(const char *s, const char *e, const char *pat) {
  const size_t k = strlen(pat);
  for (const char *q = s; q + k <= e; q++)
    if (!memcmp(q, pat, k)) return q;
  return NULL;
}

/* the schema type of parameter key of tool name ("" if not known) */
static const char *param_type(const jval *tools, const char *name, size_t nn, const char *key, size_t kn) {
  if (!tools) return "";
  for (const jval *t = tools->kid; t; t = t->next) {
    const jval *f = json_get(t, "function");
    if (!f) f = t;
    const jval *fn = json_get(f, "name");
    if (!fn || fn->t != J_STR || fn->slen != nn || memcmp(fn->str, name, nn)) continue;
    const jval *props = json_get(json_get(f, "parameters"), "properties");
    for (const jval *pr = props ? props->kid : NULL; pr; pr = pr->next)
      if (strlen(pr->key) == kn && !memcmp(pr->key, key, kn)) {
        const jval *ty = json_get(pr, "type");
        if (ty && ty->t == J_STR) return ty->str;
        if (ty && ty->t == J_ARR && ty->kid && ty->kid->t == J_STR) return ty->kid->str; /* ["integer","null"] */
        return "";
      }
  }
  return "";
}

int chat_tool_calls(const char *s, size_t n, const jval *tools, const char *id_prefix, sbuf *out) {
  const char *e = s + n, *q = s;
  sbuf arr = {0};
  int k = 0;
  sb_add(&arr, "[", 1);
  while ((q = find(q, e, "<tool_call>"))) {
    const char *f = find(q, e, "<function="), *fe;
    if (!f || !(fe = find(f, e, ">"))) goto bad;
    const char *name = f + 10;
    size_t nn = (size_t)(fe - name);
    const char *end = find(fe, e, "</function>");
    if (!end) goto bad;
    if (k) sb_add(&arr, ",", 1);
    sb_printf(&arr, "{\"id\":\"%s%d\",\"type\":\"function\",\"function\":{\"name\":", id_prefix, k);
    sb_json_str(&arr, name, nn);
    sbuf args = {0};
    sb_add(&args, "{", 1);
    int np = 0;
    for (const char *pp = fe + 1; (pp = find(pp, end, "<parameter=")) != NULL;) {
      const char *ke = find(pp, end, ">");
      if (!ke) { sb_free(&args); goto bad; }
      const char *key = pp + 11, *v = ke + 1, *ve = find(v, end, "</parameter>");
      if (!ve) ve = end; /* the model may leave out the last </parameter> */
      const size_t kn = (size_t)(ke - key);
      const char *vs = v;
      size_t vn = (size_t)(ve - v);
      if (vn && vs[0] == '\n') vs++, vn--; /* the template writes <parameter=K>\nV\n</parameter> */
      if (vn && vs[vn - 1] == '\n') vn--;
      if (np++) sb_add(&args, ", ", 2);
      sb_json_str(&args, key, kn);
      sb_add(&args, ": ", 2);
      const char *ty = param_type(tools, name, nn, key, kn);
      jval *jv = NULL;
      if (strcmp(ty, "string") && *ty) { /* a number, boolean, object or array: the value as JSON if it is valid */
        char e2[64];
        jv = json_parse(vs, vn, e2, sizeof e2);
      }
      if (jv) py_json(&args, jv);
      else sb_json_str(&args, vs, vn);
      json_free(jv);
      pp = ve == end ? end : ve + 12;
    }
    sb_add(&args, "}", 1);
    sb_puts(&arr, ",\"arguments\":");
    sb_json_str(&arr, args.p, args.n);
    sb_puts(&arr, "}}");
    sb_free(&args);
    k++;
    q = end + 11;
  }
  sb_add(&arr, "]", 1);
  if (!k) goto bad;
  sb_add(out, arr.p, arr.n);
  sb_free(&arr);
  return k;
bad:
  sb_free(&arr);
  return 0;
}
