/* chat_render.c -- print the prompt that ie-serve makes for a request.
 *   chat_render [--flint] REQUEST.json   (a /v1/chat/completions body)
 * For checks against the model's Jinja template (tools/chat_check.py). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/chat.h"

int main(int argc, char **argv) {
  const int fl = argc == 3 && !strcmp(argv[1], "--flint");
  if (argc != 2 && !fl) { fprintf(stderr, "usage: chat_render [--flint] REQUEST.json\n"); return 2; }
  const char *path = argv[argc - 1];
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); return 1; }
  sbuf b = {0};
  char buf[65536];
  size_t k;
  while ((k = fread(buf, 1, sizeof buf, f))) sb_add(&b, buf, k);
  fclose(f);
  char err[256];
  jval *req = json_parse(b.p, b.n, err, sizeof err);
  if (!req) { fprintf(stderr, "%s\n", err); return 1; }
  int think = 1;
  const jval *kw = json_get(req, "chat_template_kwargs"), *et = json_get(kw, "enable_thinking");
  if (et && (et->t == J_TRUE || et->t == J_FALSE)) think = et->t == J_TRUE;
  const char *effort = "xhigh";
  const jval *ef = json_get(req, "reasoning_effort");
  if (ef && ef->t == J_STR) effort = ef->str;
  chat_prompt p;
  if (fl ? !chat_render_flint(json_get(req, "messages"), json_get(req, "tools"), &p, err, sizeof err)
         : !chat_render(json_get(req, "messages"), json_get(req, "tools"), think, effort, &p, err, sizeof err)) {
    fprintf(stderr, "error: %s\n", err);
    return 1;
  }
  fwrite(p.text.p, 1, p.text.n, stdout);
  chat_prompt_free(&p);
  json_free(req);
  sb_free(&b);
  return 0;
}
