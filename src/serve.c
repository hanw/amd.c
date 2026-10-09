/* serve.c -- ie-serve: an OpenAI-compatible HTTP server for one model.
 *   ie-serve MODEL.gguf [--backend gpu|cpu] [--host 127.0.0.1] [--port 8000]
 *            [--api-key KEY] [--ctx 8192] [--draft D] ...   (--help: all options)
 * Endpoints:
 *   GET  /health                  {"status":"ok"}
 *   GET  /v1/models               the one model
 *   POST /v1/chat/completions     messages -> the Qwen chat template -> tokens;
 *                                 "stream": true gives server-sent events (SSE)
 * Threads: one engine thread owns the model and the backend (all HIP calls
 * are on it) and runs one request at a time from a queue; one short thread
 * per connection reads and checks the request, makes the prompt tokens and
 * puts the request in the queue. The engine thread writes the response.
 * A request whose client is gone (closed connection) stops at the next
 * check (every 8 tokens) or at the next failed write. */
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "backend.h"
#include "common.h"
#include "gen.h"
#include "gguf.h"
#include "chat.h"
#include "json.h"
#include "model.h"
#include "tok.h"

/* ---- options ---- */
typedef struct {
  const char *path, *backend, *hsaco, *host, *api_key, *mtp, *name, *effort;
  int port, think, queue_max, cache;
  uint32_t ctx, draft, chunk, top_k, max_body;
  float pmin, temp, top_p;
} options;

static options O = {
    .effort = "xhigh", .backend = "gpu", .hsaco = "build/ie_kernels.hsaco", .host = "127.0.0.1", .port = 8000, .think = 1, .cache = 1, .queue_max = 8,
    .ctx = 8192, .draft = 0, .chunk = 256, .top_k = 20, .max_body = 4u << 20, .pmin = 0.6f, .temp = 0.7f, .top_p = 0.8f,
};

static void usage(void) {
  fprintf(stderr,
          "usage: ie-serve MODEL.gguf [options]\n"
          "  --backend gpu|cpu   (default gpu; cpu: for tests with small models)\n"
          "  --hsaco FILE        GPU kernels (default build/ie_kernels.hsaco)\n"
          "  --host ADDR         listen address (default 127.0.0.1; a Tailscale address, or 0.0.0.0)\n"
          "  --port N            (default 8000)\n"
          "  --api-key KEY       require \"Authorization: Bearer KEY\" (or the environment variable IE_API_KEY)\n"
          "  --name ID           the model id in /v1/models (default: general.name of the file; spaces become -)\n"
          "  --ctx N             context: prompt + output tokens (default 8192)\n"
          "  --draft D           MTP drafts per step, 1 to 7 (GPU; default 0: off)\n"
          "  --draft-pmin P      (default 0.6)   --mtp FILE   the MTP head from FILE\n"
          "  --chunk N           prompt chunk, 1 to 512 (GPU; default 256)\n"
          "  --temp T --top-k K --top-p P   defaults when a request does not set them (0.7, 20, 0.8)\n"
          "  --no-think          end the prompt with an empty <think></think> block (no reasoning text)\n"
          "  --effort E          reasoning effort when thinking: xhigh (default), medium, low\n"
          "  --no-cache          compute every prompt from the start (no prompt cache)\n"
          "  --queue N           the most waiting requests (default 8; more: HTTP 503)\n"
          "  --show-template     print the chat template of the model file and exit\n");
  exit(2);
}

/* ---- the engine (only the engine thread uses these after the start) ---- */
static gguf_file G, G2;
static model M;
static graph GR;
static backend *B;
static gen_ctx GC;
static tokenizer *TK;          /* read-only after the start: the connection threads use it too */
static int64_t STOP_IDS[4];    /* eos, <|im_end|>, <|endoftext|>; -1: none */
static int64_t IM_START, IM_END;
static char MODEL_ID[256];
static time_t T_START;

/* ---- the queue ---- */
typedef struct job {
  int fd;
  unsigned long rid;
  int stream, usage;
  tok_ids prompt;
  gen_params gp;
  char id[48];
  long created;
  jval *req;           /* the request (the tools' schemas are used for the answer) */
  const jval *tools;   /* its "tools" array, or NULL */
  struct job *next;
  /* while running */
  sbuf all;            /* the whole answer text (with "<think>\n" first when thinking) */
  size_t sent;         /* streaming: bytes of all sent; else the end of the content */
  size_t scan;         /* all is searched for </think> and <tool_call> from about here */
  long think_end;      /* the end of "</think>" in all (0 without thinking; -1: not yet) */
  long tc_at;          /* the start of the first "<tool_call>" (-1: none) */
  uint32_t n_tok;      /* output tokens, without the stop token */
  int finish;          /* 0: length, 1: stop, 2: client gone */
  int think;           /* the prompt ends with <think>\n: the answer text starts with it too */
  int dead;            /* a write failed */
} job;

static pthread_mutex_t QM = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t QC = PTHREAD_COND_INITIALIZER, READY = PTHREAD_COND_INITIALIZER;
static job *QH, *QT;
static int QN, BUSY, IS_READY;
static unsigned long NEXT_RID;

/* ---- socket helpers ---- */
static int send_all(int fd, const char *p, size_t n) {
  while (n) {
    ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return -1;
    p += k, n -= (size_t)k;
  }
  return 0;
}
static const char *status_text(int code) {
  switch (code) {
  case 200: return "OK";
  case 204: return "No Content";
  case 400: return "Bad Request";
  case 401: return "Unauthorized";
  case 404: return "Not Found";
  case 405: return "Method Not Allowed";
  case 411: return "Length Required";
  case 413: return "Payload Too Large";
  case 503: return "Service Unavailable";
  default: return "Error";
  }
}
static void http_reply(int fd, int code, const char *ctype, const char *body, size_t n) {
  sbuf h = {0};
  sb_printf(&h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", code, status_text(code),
            ctype, n);
  send_all(fd, h.p, h.n);
  if (n) send_all(fd, body, n);
  sb_free(&h);
}
static void http_error(int fd, int code, const char *type, const char *fmt, ...) __attribute__((format(printf, 4, 5)));
static void http_error(int fd, int code, const char *type, const char *fmt, ...) {
  char msg[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof msg, fmt, ap);
  va_end(ap);
  sbuf b = {0};
  sb_puts(&b, "{\"error\":{\"message\":");
  sb_json_str(&b, msg, strlen(msg));
  sb_printf(&b, ",\"type\":\"%s\",\"code\":%d}}", type, code);
  http_reply(fd, code, "application/json", b.p, b.n);
  sb_free(&b);
}
/* 1 if the client closed its side (a check that does not wait) */
static int peer_gone(int fd) {
  struct pollfd p = {fd, POLLIN | POLLRDHUP, 0};
  if (poll(&p, 1, 0) <= 0) return 0;
  if (p.revents & (POLLRDHUP | POLLHUP | POLLERR)) return 1;
  char c;
  return recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT) == 0;
}

/* ---- the response, written by the engine thread ---- */
static int is_stop(uint32_t t) {
  for (int i = 0; i < 4; i++)
    if (STOP_IDS[i] >= 0 && (uint32_t)STOP_IDS[i] == t) return 1;
  return 0;
}
static void sse_begin(sbuf *b, const job *j) {
  sb_printf(b, "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", j->id, j->created);
  sb_json_str(b, MODEL_ID, strlen(MODEL_ID));
}
static void sse_send(job *j, sbuf *b) {
  if (send_all(j->fd, b->p, b->n)) j->dead = 1;
  sb_free(b);
}
/* text (NULL: the role chunk) or finish */
static void sse_chunk(job *j, const char *text, size_t n, const char *finish) {
  sbuf b = {0};
  sse_begin(&b, j);
  sb_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{");
  if (!text) sb_puts(&b, "\"role\":\"assistant\",\"content\":\"\"");
  else if (n) sb_puts(&b, "\"content\":"), sb_json_str(&b, text, n);
  sb_puts(&b, "},\"finish_reason\":");
  if (finish) sb_printf(&b, "\"%s\"", finish);
  else sb_puts(&b, "null");
  sb_puts(&b, "}]}\n\n");
  sse_send(j, &b);
}
/* the tool calls (a JSON array from chat_tool_calls) as one delta, each with its index */
static void sse_tool_calls(job *j, const char *arr, size_t n) {
  char err[64];
  jval *a = json_parse(arr, n, err, sizeof err);
  sbuf b = {0};
  sse_begin(&b, j);
  sb_puts(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[");
  int k = 0;
  for (const jval *c = a ? a->kid : NULL; c; c = c->next, k++) {
    sbuf one = {0};
    py_json(&one, c);
    sb_printf(&b, "%s{\"index\":%d, ", k ? "," : "", k);
    sb_add(&b, one.p + 1, one.n - 1); /* the object without its "{" */
    sb_free(&one);
  }
  sb_puts(&b, "]},\"finish_reason\":null}]}\n\n");
  sse_send(j, &b);
  json_free(a);
}

static const char *find_in(const char *s, size_t from, size_t n, const char *pat) {
  const size_t k = strlen(pat);
  for (size_t i = from; i + k <= n; i++)
    if (!memcmp(s + i, pat, k)) return s + i;
  return NULL;
}
/* How much of all can go to the client now. With tools, the text from the
 * first <tool_call> after the reasoning is held back (it becomes tool_calls),
 * and so is an end of the text that could be the start of "<tool_call>". */
static size_t safe_end(job *j, int final) {
  const char *s = j->all.p;
  const size_t n = j->all.n;
  if (!s) return 0;
  if (j->tools && j->tc_at < 0) {
    const size_t from = j->scan > 16 ? j->scan - 16 : 0;
    if (j->think_end < 0) {
      const char *e = find_in(s, from, n, "</think>");
      if (e) j->think_end = (long)(e - s) + 8;
    }
    if (j->think_end >= 0) {
      const size_t f2 = from > (size_t)j->think_end ? from : (size_t)j->think_end;
      const char *c = find_in(s, f2, n, "<tool_call>");
      if (c) j->tc_at = (long)(c - s);
    }
    j->scan = n;
  }
  if (j->tc_at >= 0) return (size_t)j->tc_at;
  if (final) return n;
  size_t lim = n;
  if (j->tools && j->think_end >= 0) /* hold back a possible start of <tool_call> */
    for (size_t k = 10; k >= 1; k--)
      if (n >= (size_t)j->think_end + k && !memcmp(s + n - k, "<tool_call>", k)) { lim = n - k; break; }
  return utf8_whole(s, lim);
}
static void flush_text(job *j, int final) {
  const size_t e = safe_end(j, final);
  if (e <= j->sent) return;
  if (j->stream) sse_chunk(j, j->all.p + j->sent, e - j->sent, NULL);
  j->sent = e;
}
static int on_tokens(void *ctx, const uint32_t *t, uint32_t n) {
  job *j = ctx;
  for (uint32_t i = 0; i < n; i++) {
    if (is_stop(t[i])) {
      j->finish = 1;
      return 1;
    }
    uint32_t len;
    char *s = model_detok(&M, t[i], &len);
    sb_add(&j->all, s, len);
    free(s);
    j->n_tok++;
  }
  flush_text(j, 0);
  if (j->dead || (j->n_tok % 8 == 0 && peer_gone(j->fd))) {
    j->finish = 2;
    return 1;
  }
  return 0;
}

static void run_job(job *j) {
  if (peer_gone(j->fd)) { /* gone while it waited in the queue */
    fprintf(stderr, "[%lu] client gone before the start\n", j->rid);
    return;
  }
  if (j->stream) {
    const char *h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if (send_all(j->fd, h, strlen(h))) return;
    sse_chunk(j, NULL, 0, NULL); /* the role */
  }
  j->think_end = j->think ? -1 : 0, j->tc_at = -1;
  sb_add(&j->all, "", 0);
  if (j->think) sb_add(&j->all, "<think>\n", 8); /* the prompt opened <think>: the client sees the whole block (Open WebUI folds it) */
  flush_text(j, 0);
  j->gp.emit = on_tokens, j->gp.ctx = j;
  gen_stats st;
  gen_run(&GC, j->prompt.v, j->prompt.n, &j->gp, &st);

  /* tool calls: the text from the first <tool_call> */
  sbuf calls = {0};
  char idp[64];
  snprintf(idp, sizeof idp, "call_%lx%lu_", (unsigned long)T_START, j->rid);
  int n_calls = 0;
  if (j->tc_at >= 0 && j->finish != 2) {
    n_calls = chat_tool_calls(j->all.p + j->tc_at, j->all.n - (size_t)j->tc_at, j->tools, idp, &calls);
    if (!n_calls) j->tc_at = -1; /* not valid: send it as text */
  }
  flush_text(j, 1); /* the rest of the text (an invalid UTF-8 sequence becomes U+FFFD) */
  const char *fin = n_calls ? "tool_calls" : j->finish == 1 ? "stop" : "length";

  const uint32_t n_gen = j->n_tok;
  /* usage: the OpenAI counts, then the speed (Open WebUI shows all of usage
   * in the (i) tooltip under an answer) */
  sbuf us = {0};
  {
    const double gs = st.gen_ms / 1e3, ps = st.prefill_ms / 1e3, nc = (double)(st.n_prompt - st.cached);
    sb_printf(&us, "{\"prompt_tokens\":%u,\"completion_tokens\":%u,\"total_tokens\":%u", j->prompt.n, n_gen, j->prompt.n + n_gen);
    sb_printf(&us, ",\"prompt_tokens_details\":{\"cached_tokens\":%u}", st.cached);
    sb_printf(&us, ",\"response_token/s\":%.1f,\"prompt_token/s\":%.0f", gs > 0 ? n_gen / gs : 0.0, ps > 0 ? nc / ps : 0.0);
    sb_printf(&us, ",\"prompt_ms\":%.0f,\"response_ms\":%.0f", st.prefill_ms, st.gen_ms);
    if (st.steps) sb_printf(&us, ",\"mtp_tokens_per_step\":%.2f", (double)st.n_out / st.steps);
    sb_puts(&us, "}");
  }
  if (j->stream && j->finish != 2 && !j->dead) {
    if (n_calls) sse_tool_calls(j, calls.p, calls.n);
    { /* the last chunk carries usage too (as llama.cpp does), so clients that do not ask still see it */
      sbuf b = {0};
      sse_begin(&b, j);
      sb_printf(&b, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}],\"usage\":%s}\n\n", fin, us.p);
      sse_send(j, &b);
    }
    if (j->usage) {
      sbuf b = {0};
      sse_begin(&b, j);
      sb_printf(&b, ",\"choices\":[],\"usage\":%s}\n\n", us.p);
      sse_send(j, &b);
    }
    send_all(j->fd, "data: [DONE]\n\n", 14);
  } else if (!j->stream && j->finish != 2) {
    size_t cn = j->sent; /* the content: the text before the tool calls, without the white space before them */
    if (n_calls)
      while (cn && strchr(" \t\r\n", j->all.p[cn - 1])) cn--;
    sbuf b = {0};
    sb_printf(&b, "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", j->id, j->created);
    sb_json_str(&b, MODEL_ID, strlen(MODEL_ID));
    sb_puts(&b, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    sb_json_str(&b, j->all.p ? j->all.p : "", cn);
    if (n_calls) sb_puts(&b, ",\"tool_calls\":"), sb_add(&b, calls.p, calls.n);
    sb_printf(&b, "},\"finish_reason\":\"%s\"}],\"usage\":%s}", fin, us.p);
    http_reply(j->fd, 200, "application/json", b.p, b.n);
    sb_free(&b);
  }
  sb_free(&calls), sb_free(&us);
  const double gs = st.gen_ms / 1e3;
  fprintf(stderr, "[%lu] prompt %u tokens (%u cached%s, %.0f ms), output %u tokens in %.2f s (%.1f tokens/s)", j->rid, st.n_prompt,
          st.cached, st.cache_hit == 1 ? " with the last answer" : "", st.prefill_ms, n_gen, gs, gs > 0 ? n_gen / gs : 0.0);
  if (st.steps) fprintf(stderr, ", mtp %.2f tokens per step", (double)st.n_out / st.steps);
  if (n_calls) fprintf(stderr, ", %d tool calls", n_calls);
  fprintf(stderr, ", %s\n", j->finish == 2 ? "client gone" : fin);
}

static void job_free(job *j) {
  if (j->fd >= 0) close(j->fd);
  tok_ids_free(&j->prompt);
  sb_free(&j->all);
  json_free(j->req);
  free(j);
}

/* ---- engine thread: load, then run the queue ---- */
static void *engine_main(void *arg) {
  (void)arg;
  const int gpu = !strcmp(O.backend, "gpu");
  gguf_open(&G, O.path);
  const char *se = getenv("IE_STREAM");
  if (gpu && !(se && se[0] == '0')) gpu_stream_init();
  model_load(&M, &G);
  if (O.draft) {
    if (O.mtp) gguf_open(&G2, O.mtp);
    if (!model_load_mtp(&M, O.mtp ? &G2 : &G)) ie_die("no MTP head (blk.N.nextn.*) in %s", O.mtp ? O.mtp : O.path);
    const char *dh = getenv("IE_DRAFT_HEAD");
    if (!(dh && !strcmp(dh, "q8"))) model_make_draft_head(&M, &G);
  }
  char err[256];
  if (!(TK = tok_open(&M, &G, err, sizeof err))) ie_die("tokenizer: %s", err);
  IM_START = tok_find(TK, "<|im_start|>"), IM_END = tok_find(TK, "<|im_end|>");
  if (IM_START < 0 || IM_END < 0) ie_die("the vocabulary has no <|im_start|> or <|im_end|> (ChatML)");
  int64_t eos = -1;
  if (!gguf_get_int(&G, "tokenizer.ggml.eos_token_id", &eos)) eos = -1;
  STOP_IDS[0] = eos, STOP_IDS[1] = IM_END, STOP_IDS[2] = tok_find(TK, "<|endoftext|>"), STOP_IDS[3] = -1;
  const char *xs = getenv("IE_EXTRA_STOP"); /* tests: one more stop token id */
  if (xs) STOP_IDS[3] = atoll(xs);
  if (!O.name) {
    char *gn = gguf_get_str(&G, "general.name");
    snprintf(MODEL_ID, sizeof MODEL_ID, "%s", gn ? gn : "amd-infer");
    free(gn);
  } else snprintf(MODEL_ID, sizeof MODEL_ID, "%s", O.name);
  /* no white space in the id (Open WebUI refuses a model id with spaces): "Qwen3.8 27B" -> "Qwen3.8-27B" */
  for (char *c = MODEL_ID; *c; c++)
    if (*c == ' ' || *c == '\t' || *c == '\n' || *c == '\r') *c = '-';

  const uint32_t chunk = gpu ? O.chunk : 1;
  graph_build(&GR, &M, O.ctx, O.draft + 1 > chunk ? O.draft + 1 : chunk, O.draft > 0);
  plan_arena(&GR, 1);
  B = gpu ? gpu_open(&M, &GR, O.hsaco) : cpu_open(&M, &GR);
  gen_init(&GC, B, &M, &GR);
  fprintf(stderr, "model \"%s\": vocab %u, pre-tokenizer %s, context %u, drafts %u, stop tokens %lld %lld %lld\n", MODEL_ID, M.vocab,
          tok_pre(TK), O.ctx, O.draft, (long long)STOP_IDS[0], (long long)STOP_IDS[1], (long long)STOP_IDS[2]);

  pthread_mutex_lock(&QM);
  IS_READY = 1;
  pthread_cond_broadcast(&READY);
  pthread_mutex_unlock(&QM);
  for (;;) {
    pthread_mutex_lock(&QM);
    while (!QH) pthread_cond_wait(&QC, &QM);
    job *j = QH;
    QH = j->next;
    if (!QH) QT = NULL;
    QN--, BUSY = 1;
    pthread_mutex_unlock(&QM);
    run_job(j);
    job_free(j);
    pthread_mutex_lock(&QM);
    BUSY = 0;
    pthread_mutex_unlock(&QM);
  }
  return NULL;
}

/* ---- the request: HTTP ---- */
typedef struct {
  char method[16], path[256];
  char *body;
  size_t body_n;
  int has_len, chunked, expect100;
  size_t len;
  char auth[512];
} request;

/* Read the head and the body. 0: ok; otherwise the HTTP code to send. */
static int read_request(int fd, request *r) {
  memset(r, 0, sizeof *r);
  char *buf = malloc(65536 + 1);
  size_t n = 0;
  char *he = NULL;
  while (!he) {
    if (n == 65536) { free(buf); return 413; }
    ssize_t k = recv(fd, buf + n, 65536 - n, 0);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) { free(buf); return -1; }
    n += (size_t)k;
    buf[n] = 0;
    he = strstr(buf, "\r\n\r\n");
  }
  *he = 0;
  const size_t head_n = (size_t)(he - buf) + 4;
  if (sscanf(buf, "%15s %255s", r->method, r->path) != 2) { free(buf); return 400; }
  char *q = strchr(r->path, '?');
  if (q) *q = 0;
  for (char *line = strstr(buf, "\r\n"); line; line = strstr(line, "\r\n")) {
    line += 2;
    char *colon = strchr(line, ':'), *eol = strstr(line, "\r\n");
    if (!colon || (eol && colon > eol)) continue;
    char *v = colon + 1;
    while (*v == ' ' || *v == '\t') v++;
    const size_t vn = eol ? (size_t)(eol - v) : strlen(v);
    const size_t kn = (size_t)(colon - line);
    if (kn == 14 && !strncasecmp(line, "Content-Length", 14)) r->len = strtoul(v, NULL, 10), r->has_len = 1;
    else if (kn == 17 && !strncasecmp(line, "Transfer-Encoding", 17) && vn >= 7 && !strncasecmp(v, "chunked", 7)) r->chunked = 1;
    else if (kn == 6 && !strncasecmp(line, "Expect", 6) && vn >= 12 && !strncasecmp(v, "100-continue", 12)) r->expect100 = 1;
    else if (kn == 13 && !strncasecmp(line, "Authorization", 13)) snprintf(r->auth, sizeof r->auth, "%.*s", (int)vn, v);
  }
  if (!strcmp(r->method, "POST")) {
    if (r->chunked) { free(buf); return 411; }
    if (!r->has_len) { free(buf); return 411; }
    if (r->len > O.max_body) { free(buf); return 413; }
    if (r->expect100) send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n", 25);
    r->body = malloc(r->len + 1);
    size_t have = n - head_n < r->len ? n - head_n : r->len;
    memcpy(r->body, buf + head_n, have);
    while (have < r->len) {
      ssize_t k = recv(fd, r->body + have, r->len - have, 0);
      if (k < 0 && errno == EINTR) continue;
      if (k <= 0) { free(buf); free(r->body); r->body = NULL; return -1; }
      have += (size_t)k;
    }
    r->body[r->len] = 0;
    r->body_n = r->len;
  }
  free(buf);
  return 0;
}

/* The compare time does not depend on where the strings differ. */
static int same_secret(const char *a, const char *b) {
  const size_t na = strlen(a), nb = strlen(b);
  unsigned char d = na != nb;
  for (size_t i = 0; i < na; i++) d |= (unsigned char)(a[i] ^ b[i % (nb ? nb : 1)]);
  return d == 0;
}

/* ---- the request: chat ---- */
/* The prompt tokens: control tokens as ids, text without control tokens
 * (a user cannot write <|im_start|> into the prompt). */
static void prompt_tokens(const chat_prompt *p, tok_ids *out) {
  for (size_t i = 0; i < p->n_seg; i++) {
    const chat_seg *s = &p->seg[i];
    if (s->kind == SEG_IM_START) tok_ids_push(out, (uint32_t)IM_START);
    else if (s->kind == SEG_IM_END) tok_ids_push(out, (uint32_t)IM_END);
    else tok_encode(TK, p->text.p + s->off, s->n, 0, out);
  }
}

static int num_field(const jval *o, const char *k, double *v) {
  const jval *x = json_get(o, k);
  if (!x || x->t != J_NUM) return 0;
  *v = x->num;
  return 1;
}

static void handle_chat(int fd, const request *r) {
  char err[512];
  jval *root = json_parse(r->body, r->body_n, err, sizeof err);
  if (!root || root->t != J_OBJ) {
    if (!root) http_error(fd, 400, "invalid_request_error", "%s", err);
    else http_error(fd, 400, "invalid_request_error", "the body must be a JSON object");
    json_free(root);
    close(fd);
    return;
  }
  job *j = calloc(1, sizeof *j);
  j->fd = fd;
  j->req = root; /* the job frees it */
  double v;
  /* sampling: the request's values, else the server's defaults */
  j->gp.temp = O.temp, j->gp.top_p = O.top_p, j->gp.top_k = O.top_k;
  if (num_field(root, "temperature", &v)) j->gp.temp = (float)v;
  if (num_field(root, "top_p", &v)) j->gp.top_p = (float)v;
  if (num_field(root, "top_k", &v)) j->gp.top_k = v < 0 ? 0 : (uint32_t)v;
  j->gp.seed = num_field(root, "seed", &v) ? (uint64_t)(int64_t)v : (uint64_t)time(NULL) ^ ((uint64_t)fd << 32);
  j->gp.draft = O.draft, j->gp.pmin = O.pmin, j->gp.chunk = O.chunk, j->gp.cache = O.cache;
  j->gp.stops = STOP_IDS, j->gp.n_stops = 4;
  if (num_field(root, "n", &v) && v != 1) {
    http_error(fd, 400, "invalid_request_error", "only n = 1 is supported");
    goto bad;
  }
  if (j->gp.top_p <= 0.0f || j->gp.top_p > 1.0f) j->gp.top_p = 1.0f;
  const jval *st = json_get(root, "stream");
  j->stream = st && st->t == J_TRUE;
  const jval *so = json_get(root, "stream_options"), *iu = json_get(so, "include_usage");
  j->usage = iu && iu->t == J_TRUE;
  int think = O.think;
  const jval *kw = json_get(root, "chat_template_kwargs"), *et = json_get(kw, "enable_thinking");
  if (et && (et->t == J_TRUE || et->t == J_FALSE)) think = et->t == J_TRUE;
  const char *effort = O.effort;
  const jval *ef = json_get(root, "reasoning_effort");
  if (!ef || ef->t != J_STR) ef = json_get(kw, "reasoning_effort");
  if (ef && ef->t == J_STR) effort = ef->str;
  j->think = think;

  /* tools: rendered into the prompt, and the answer's <tool_call> blocks become tool_calls */
  const jval *tools = json_get(root, "tools"), *tc = json_get(root, "tool_choice");
  if (!tools || tools->t != J_ARR || !tools->kid || (tc && tc->t == J_STR && !strcmp(tc->str, "none"))) tools = NULL;
  j->tools = tools;
  chat_prompt cp;
  if (!chat_render(json_get(root, "messages"), tools, think, effort, &cp, err, sizeof err)) {
    http_error(fd, 400, "invalid_request_error", "%s", err);
    goto bad;
  }
  prompt_tokens(&cp, &j->prompt);
  chat_prompt_free(&cp);
  /* the room in the context: n_prompt + max_new + draft <= ctx */
  if (getenv("IE_SERVE_DEBUG")) { /* the prompt ids, to compare with llama-tokenize */
    sbuf b = {0};
    for (uint32_t i = 0; i < j->prompt.n; i++) sb_printf(&b, "%s%u", i ? " " : "", j->prompt.v[i]);
    fprintf(stderr, "prompt ids: %s\n", b.p ? b.p : "");
    sb_free(&b);
  }
  const uint32_t used = j->prompt.n + O.draft;
  if (used >= O.ctx) {
    http_error(fd, 400, "invalid_request_error", "the prompt has %u tokens; the context is %u (--ctx)", j->prompt.n, O.ctx);
    goto bad;
  }
  uint32_t room = O.ctx - used;
  if ((num_field(root, "max_completion_tokens", &v) || num_field(root, "max_tokens", &v)) && v >= 1 && v < room) room = (uint32_t)v;
  j->gp.max_new = room;

  pthread_mutex_lock(&QM);
  if (QN >= O.queue_max) {
    pthread_mutex_unlock(&QM);
    http_error(fd, 503, "server_busy", "%d requests are waiting; try again later", O.queue_max);
    j->fd = -1;
    job_free(j);
    close(fd);
    return;
  }
  j->rid = ++NEXT_RID;
  snprintf(j->id, sizeof j->id, "chatcmpl-%lx%lu", (unsigned long)T_START, j->rid);
  j->created = (long)time(NULL);
  if (QT) QT->next = j;
  else QH = j;
  QT = j, QN++;
  const int waiting = QN - 1 + BUSY;
  pthread_cond_signal(&QC);
  pthread_mutex_unlock(&QM);
  fprintf(stderr, "[%lu] queued: %u prompt tokens, max %u output tokens, %s%s, %d before it\n", j->rid, j->prompt.n, j->gp.max_new,
          j->stream ? "stream" : "no stream", j->tools ? ", tools" : "", waiting);
  return; /* the engine thread writes the response and closes fd */
bad:
  j->fd = -1;
  job_free(j);
  close(fd);
}

static void *conn_main(void *arg) {
  const int fd = (int)(intptr_t)arg;
  request r;
  const int rc = read_request(fd, &r);
  if (rc) {
    if (rc > 0) http_error(fd, rc, "invalid_request_error", "%s", status_text(rc));
    close(fd);
    return NULL;
  }
  const int is_v1 = !strncmp(r.path, "/v1/", 4);
  const char *p = is_v1 ? r.path + 3 : r.path; /* "/models" etc. */
  if (!strcmp(r.method, "OPTIONS")) {
    http_reply(fd, 204, "text/plain", "", 0);
  } else if (!strcmp(p, "/health")) {
    const char *b = "{\"status\":\"ok\"}";
    http_reply(fd, 200, "application/json", b, strlen(b));
  } else if (O.api_key && (strncmp(r.auth, "Bearer ", 7) || !same_secret(r.auth + 7, O.api_key))) {
    http_error(fd, 401, "authentication_error", "missing or wrong API key (Authorization: Bearer KEY)");
  } else if (!strcmp(p, "/models")) {
    if (strcmp(r.method, "GET")) http_error(fd, 405, "invalid_request_error", "use GET");
    else {
      sbuf b = {0};
      sb_puts(&b, "{\"object\":\"list\",\"data\":[{\"id\":");
      sb_json_str(&b, MODEL_ID, strlen(MODEL_ID));
      sb_printf(&b, ",\"object\":\"model\",\"created\":%ld,\"owned_by\":\"amd-infer\",\"context_length\":%u}]}", (long)T_START, O.ctx);
      http_reply(fd, 200, "application/json", b.p, b.n);
      sb_free(&b);
    }
  } else if (!strcmp(p, "/chat/completions")) {
    if (strcmp(r.method, "POST")) http_error(fd, 405, "invalid_request_error", "use POST");
    else {
      handle_chat(fd, &r); /* owns fd from here */
      free(r.body);
      return NULL;
    }
  } else {
    http_error(fd, 404, "invalid_request_error", "no endpoint %s %s", r.method, r.path);
  }
  free(r.body);
  close(fd);
  return NULL;
}

int main(int argc, char **argv) {
  int show_template = 0;
  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
#define ARG (i + 1 < argc ? argv[++i] : (usage(), ""))
    if (!strcmp(a, "--backend")) O.backend = ARG;
    else if (!strcmp(a, "--hsaco")) O.hsaco = ARG;
    else if (!strcmp(a, "--host")) O.host = ARG;
    else if (!strcmp(a, "--port")) O.port = atoi(ARG);
    else if (!strcmp(a, "--api-key")) O.api_key = ARG;
    else if (!strcmp(a, "--name")) O.name = ARG;
    else if (!strcmp(a, "--ctx")) O.ctx = (uint32_t)atoi(ARG);
    else if (!strcmp(a, "--draft")) O.draft = (uint32_t)atoi(ARG);
    else if (!strcmp(a, "--draft-pmin")) O.pmin = (float)atof(ARG);
    else if (!strcmp(a, "--mtp")) O.mtp = ARG;
    else if (!strcmp(a, "--chunk")) O.chunk = (uint32_t)atoi(ARG);
    else if (!strcmp(a, "--temp")) O.temp = (float)atof(ARG);
    else if (!strcmp(a, "--top-k")) O.top_k = (uint32_t)atoi(ARG);
    else if (!strcmp(a, "--top-p")) O.top_p = (float)atof(ARG);
    else if (!strcmp(a, "--no-think")) O.think = 0;
    else if (!strcmp(a, "--effort")) O.effort = ARG;
    else if (!strcmp(a, "--no-cache")) O.cache = 0;
    else if (!strcmp(a, "--queue")) O.queue_max = atoi(ARG);
    else if (!strcmp(a, "--show-template")) show_template = 1;
    else if (a[0] != '-' && !O.path) O.path = a;
    else usage();
#undef ARG
  }
  if (!O.path) usage();
  if (!O.api_key) O.api_key = getenv("IE_API_KEY");
  if (O.api_key && !O.api_key[0]) O.api_key = NULL;
  if (show_template) {
    gguf_open(&G, O.path);
    char *t = gguf_get_str(&G, "tokenizer.chat_template");
    printf("%s\n", t ? t : "(no tokenizer.chat_template)");
    return 0;
  }
  const int gpu = !strcmp(O.backend, "gpu");
  if (!gpu && strcmp(O.backend, "cpu")) usage();
  if (O.draft > 7 || (O.draft && !gpu)) ie_die("--draft: 1 to 7, GPU only");
  if (O.chunk < 1 || O.chunk > 512) ie_die("--chunk: 1 to 512");
  if (O.ctx < 16) ie_die("--ctx: at least 16");
  if (O.queue_max < 1) O.queue_max = 1;
  signal(SIGPIPE, SIG_IGN);
  T_START = time(NULL);

  /* bind first: a busy port is an error before the long model load */
  struct sockaddr_in sa = {0};
  sa.sin_family = AF_INET;
  sa.sin_port = htons((uint16_t)O.port);
  if (inet_pton(AF_INET, O.host, &sa.sin_addr) != 1) ie_die("--host: not an IPv4 address: %s", O.host);
  const int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  if (bind(ls, (struct sockaddr *)&sa, sizeof sa)) ie_die("bind %s:%d: %s", O.host, O.port, strerror(errno));
  if (!O.api_key && strcmp(O.host, "127.0.0.1"))
    fprintf(stderr, "warning: listening on %s without --api-key: every device that can reach this address can use the model\n", O.host);

  pthread_t et;
  pthread_create(&et, NULL, engine_main, NULL);
  pthread_mutex_lock(&QM);
  while (!IS_READY) pthread_cond_wait(&READY, &QM);
  pthread_mutex_unlock(&QM);
  if (listen(ls, 64)) ie_die("listen: %s", strerror(errno));
  fprintf(stderr, "ie-serve: http://%s:%d/v1 (%s)\n", O.host, O.port, O.api_key ? "API key required" : "no API key");

  for (;;) {
    const int fd = accept(ls, NULL, NULL);
    if (fd < 0) {
      if (errno != EINTR) perror("accept");
      continue;
    }
    struct timeval tv = {30, 0}; /* a client that stops reading or writing for 30 s is dropped */
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    pthread_t t;
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&t, &at, conn_main, (void *)(intptr_t)fd)) close(fd);
    pthread_attr_destroy(&at);
  }
}
