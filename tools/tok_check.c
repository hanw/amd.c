/* tok_check.c -- check src/tok.c against llama.cpp's tokenizer tests.
 *   tok_check VOCAB.gguf TESTS.inp TESTS.out
 * VOCAB.gguf: a vocabulary file (for example llama.cpp's
 * models/ggml-vocab-qwen35.gguf; a full model file also works). TESTS.inp:
 * texts separated by "\n__ggml_vocab_test__\n"; TESTS.out: the expected ids,
 * one line per text. Also:
 *   tok_check VOCAB.gguf --text "some text"   prints the ids */
#include <string.h>

#include "../src/common.h"
#include "../src/gguf.h"
#include "../src/tok.h"

/* Only the vocabulary of a model: what tok_open reads. */
static void load_vocab(model *m, const gguf_file *g) {
  memset(m, 0, sizeof *m);
  char *tm = gguf_get_str(g, "tokenizer.ggml.model");
  m->tok_gpt2 = tm && !strcmp(tm, "gpt2");
  free(tm);
  const gguf_kv *tk = gguf_find(g, "tokenizer.ggml.tokens");
  if (!tk || tk->type != GGUF_ARR || tk->arr_type != GGUF_STR) ie_die("no tokenizer.ggml.tokens");
  m->n_tokens = (uint32_t)tk->arr_n;
  m->tokens = calloc(m->n_tokens, sizeof(char *));
  for (uint32_t i = 0; i < m->n_tokens; i++) {
    m->tokens[i] = malloc(tk->arr_str[i].n + 1);
    memcpy(m->tokens[i], tk->arr_str[i].p, tk->arr_str[i].n);
    m->tokens[i][tk->arr_str[i].n] = 0;
  }
}

static char *slurp(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) ie_die("cannot read %s", path);
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *b = malloc((size_t)len + 1);
  if (fread(b, 1, (size_t)len, f) != (size_t)len) ie_die("cannot read %s", path);
  b[len] = 0;
  fclose(f);
  *n = (size_t)len;
  return b;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: tok_check VOCAB.gguf TESTS.inp TESTS.out | tok_check VOCAB.gguf --text TEXT\n");
    return 2;
  }
  gguf_file g;
  gguf_open(&g, argv[1]);
  model m;
  load_vocab(&m, &g);
  char err[256];
  tokenizer *t = tok_open(&m, &g, err, sizeof err);
  if (!t) ie_die("%s", err);
  if (!strcmp(argv[2], "--text")) {
    tok_ids ids = {0};
    tok_encode(t, argv[3], strlen(argv[3]), 1, &ids);
    for (uint32_t i = 0; i < ids.n; i++) printf("%s%u", i ? " " : "", ids.v[i]);
    printf("\n");
    return 0;
  }
  size_t ni, no;
  char *inp = slurp(argv[2], &ni), *outp = slurp(argv[3], &no);
  const char *sep = "\n__ggml_vocab_test__\n";
  int n_test = 0, n_fail = 0;
  char *ip = inp, *op = outp;
  for (;;) {
    char *e = strstr(ip, sep);
    if (!e) break;
    char *ol = op, *oe = strchr(op, '\n');
    if (!oe) ie_die("%s: fewer lines than tests", argv[3]);
    *oe = 0, op = oe + 1;
    tok_ids ids = {0};
    tok_encode(t, ip, (size_t)(e - ip), 0, &ids);
    char got[16384];
    size_t k = 0;
    got[0] = 0;
    for (uint32_t i = 0; i < ids.n && k + 16 < sizeof got; i++) k += (size_t)snprintf(got + k, sizeof got - k, "%s%u", i ? " " : "", ids.v[i]);
    /* the expected line may have spaces at the start or the end */
    size_t ol_n = strlen(ol);
    while (ol_n && (ol[ol_n - 1] == ' ' || ol[ol_n - 1] == '\r')) ol[--ol_n] = 0;
    while (*ol == ' ') ol++;
    n_test++;
    if (strcmp(got, ol)) {
      n_fail++;
      fprintf(stderr, "FAIL test %d: \"%.*s\"\n  want %s\n  got  %s\n", n_test, (int)(e - ip), ip, ol, got);
    }
    tok_ids_free(&ids);
    ip = e + strlen(sep);
  }
  printf("tok_check %s (pre %s): %d tests, %d failed\n", argv[1], tok_pre(t), n_test, n_fail);
  tok_close(t);
  return n_fail != 0;
}
