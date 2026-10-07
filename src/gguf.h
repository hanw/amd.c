/* gguf.h -- read a GGUF v3 file (mmap, no copy of the tensor data). */
#ifndef IE_GGUF_H
#define IE_GGUF_H

#include <stdint.h>
#include <stdio.h>

enum {
  GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32, GGUF_F32,
  GGUF_BOOL, GGUF_STR, GGUF_ARR, GGUF_U64, GGUF_I64, GGUF_F64
};

typedef struct {
  const char *p; /* not 0-terminated: points into the file */
  uint64_t n;
} gguf_str;

typedef struct {
  char *key;
  uint32_t type;
  union {
    uint64_t u;
    int64_t i;
    double f;
    gguf_str s;
  } v; /* scalars and strings */
  /* arrays (type == GGUF_ARR) */
  uint32_t arr_type;
  uint64_t arr_n;
  const uint8_t *arr_data; /* first element in the file */
  gguf_str *arr_str;       /* when arr_type == GGUF_STR: the strings */
} gguf_kv;

typedef struct {
  char *name;
  uint32_t n_dims;
  uint64_t ne[4]; /* ne[0] is the inner (contiguous) dimension */
  uint32_t type;
  uint64_t offset; /* from the start of the data section */
  uint64_t nbytes;
  const void *data;
} gguf_tensor;

typedef struct {
  void *map;
  uint64_t size;
  uint32_t version;
  uint64_t n_kv, n_tensors;
  gguf_kv *kv;
  gguf_tensor *t;
  uint32_t alignment;
  uint64_t data_off;
} gguf_file;

void gguf_open(gguf_file *g, const char *path); /* exits on error */
void gguf_close(gguf_file *g);

const gguf_kv *gguf_find(const gguf_file *g, const char *key);
/* 1 if found and of a numeric type (integer, bool or float). */
int gguf_get_int(const gguf_file *g, const char *key, int64_t *out);
int gguf_get_float(const gguf_file *g, const char *key, double *out);
/* A malloc'ed 0-terminated copy, or NULL. */
char *gguf_get_str(const gguf_file *g, const char *key);
const gguf_tensor *gguf_tensor_find(const gguf_file *g, const char *name);
/* Bytes of n elements of a GGML type; 0 if the type is unknown. */
uint64_t ggml_nbytes(uint32_t type, uint64_t n);
const char *ggml_type_name(uint32_t type);
void gguf_print(const gguf_file *g, FILE *f);

#endif
