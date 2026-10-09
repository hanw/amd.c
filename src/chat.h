/* chat.h -- the Qwen3.8 chat template (as the GGUF's tokenizer.chat_template
 * renders it with Hugging Face transformers), without images, and the
 * parser of the model's tool calls (<tool_call><function=..>...). */
#ifndef IE_CHAT_H
#define IE_CHAT_H

#include <stddef.h>

#include "json.h"

/* The rendered prompt: the whole text, and its parts. A part is plain text
 * (tokenized without control tokens) or one control token. */
enum { SEG_TEXT = 0, SEG_IM_START = 1, SEG_IM_END = 2 };
typedef struct {
  int kind;
  size_t off, n; /* SEG_TEXT: text.p[off .. off + n) */
} chat_seg;

typedef struct {
  sbuf text; /* the whole prompt, control tokens written as "<|im_start|>" and "<|im_end|>" */
  chat_seg *seg;
  size_t n_seg, cap;
} chat_prompt;

/* messages: the request's array; tools: its "tools" array or NULL. think:
 * enable_thinking; effort: reasoning_effort (xhigh, medium, low). 0 and a
 * message in err if the request is not valid. */
int chat_render(const jval *messages, const jval *tools, int think, const char *effort, chat_prompt *p, char *err, size_t errn);
void chat_prompt_free(chat_prompt *p);

/* v as Python's json.dumps(v, ensure_ascii=False) writes it (the template's
 * tojson): separators ", " and ": ", keys in the given order, numbers as
 * written in the request. */
void py_json(sbuf *b, const jval *v);

/* Parse the tool calls in s[0 .. n) (the text from the first "<tool_call>").
 * out gets a JSON array of OpenAI tool calls {"id","type","function":{
 * "name","arguments"}}; id_prefix makes the ids. Parameter values become
 * JSON by the tool's schema (tools: the request's array; NULL: strings).
 * Returns the number of calls (0: not valid, out unchanged). */
int chat_tool_calls(const char *s, size_t n, const jval *tools, const char *id_prefix, sbuf *out);

#endif
