/* tok.h -- text to token ids for GPT2 byte-level BPE vocabularies (Qwen2,
 * Qwen3.5 / 3.8): special tokens, the Qwen pre-tokenizer regex, BPE merges.
 * The decoder (ids to text) is model_detok in model.c. */
#ifndef IE_TOK_H
#define IE_TOK_H

#include <stddef.h>
#include <stdint.h>

#include "gguf.h"
#include "model.h"

typedef struct tokenizer tokenizer;

typedef struct {
  uint32_t *v;
  uint32_t n, cap;
} tok_ids;

void tok_ids_push(tok_ids *a, uint32_t id);
void tok_ids_free(tok_ids *a);

/* NULL (and a message in err) if the vocabulary is not GPT2 byte-level or
 * the pre-tokenizer (tokenizer.ggml.pre) is not known. Reads the token
 * strings of m and the merges and token types of g; keeps no pointers into
 * g. */
tokenizer *tok_open(const model *m, const gguf_file *g, char *err, size_t errn);
void tok_close(tokenizer *t);
/* The id of the token whose string is s (for example "<|im_end|>"), or -1. */
int64_t tok_find(const tokenizer *t, const char *s);
/* tokenizer.ggml.pre */
const char *tok_pre(const tokenizer *t);
/* Append the ids of text[0 .. len) to out. User-defined special tokens
 * (token type 4, for example <think>) are always matched as one token, as
 * llama.cpp does; control tokens (type 3, for example <|im_start|>) only if
 * parse_control. Thread-safe (no shared scratch). */
void tok_encode(const tokenizer *t, const char *text, size_t len, int parse_control, tok_ids *out);

#endif
