#!/usr/bin/env python3
"""Write a tiny random Qwen2 model that has the tokenizer of another GGUF
file (for example llama.cpp's models/ggml-vocab-qwen35.gguf), to test
ie-serve on the CPU without a real model. The output text is random.
usage: make_tiny_chat_gguf.py VOCAB.gguf OUT.gguf    (needs: pip install gguf numpy)"""
import os
import sys

import numpy as np
from gguf import GGUFReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tiny_gguf import (ALIGN, GGML_F32, GGML_Q4_0, T_ARR, T_F32, T_I32, T_STR, T_U32, f32, q4_0,  # noqa: E402
                            write_gguf)


def field(r, key):
    f = r.fields.get(key)
    if f is None:
        return None
    if len(f.data) == 1 and f.types[0] != 9:  # a scalar or a string
        v = f.parts[f.data[0]]
        return bytes(v).decode("utf-8") if f.types[0] == 8 else v[0].item()
    if f.types[-1] == 8:  # an array of strings
        return [bytes(f.parts[i]).decode("utf-8") for i in f.data]
    return [f.parts[i][0].item() for i in f.data]


def main():
    src, out = sys.argv[1], sys.argv[2]
    r = GGUFReader(src)
    toks = field(r, "tokenizer.ggml.tokens")
    types = field(r, "tokenizer.ggml.token_type")
    merges = field(r, "tokenizer.ggml.merges")
    pre = field(r, "tokenizer.ggml.pre")
    eos = field(r, "tokenizer.ggml.eos_token_id")
    vocab = len(toks)
    vocab_pad = (vocab + 31) // 32 * 32  # rows of the tied Q4_0 embedding: a multiple of 32 is not needed, but keeps it simple
    toks = toks + ["[PAD%d]" % i for i in range(vocab_pad - vocab)]
    types = types + [1] * (vocab_pad - vocab)
    arch, dim, ffn, n_head, n_kv, n_layer = "qwen2", 64, 128, 4, 2, 1
    hd = dim // n_head
    kvd = n_kv * hd
    rng = np.random.default_rng(7)
    kvs = [
        ("general.architecture", T_STR, arch),
        ("general.name", T_STR, "tiny-chat"),
        ("general.alignment", T_U32, ALIGN),
        (arch + ".block_count", T_U32, n_layer),
        (arch + ".context_length", T_U32, 4096),
        (arch + ".embedding_length", T_U32, dim),
        (arch + ".feed_forward_length", T_U32, ffn),
        (arch + ".attention.head_count", T_U32, n_head),
        (arch + ".attention.head_count_kv", T_U32, n_kv),
        (arch + ".attention.layer_norm_rms_epsilon", T_F32, 1e-6),
        (arch + ".rope.freq_base", T_F32, 1000000.0),
        ("tokenizer.ggml.model", T_STR, "gpt2"),
        ("tokenizer.ggml.pre", T_STR, pre),
        ("tokenizer.ggml.tokens", T_ARR, (T_STR, toks)),
        ("tokenizer.ggml.token_type", T_ARR, (T_I32, types)),
        ("tokenizer.ggml.merges", T_ARR, (T_STR, merges)),
    ]
    if eos is not None:
        kvs.append(("tokenizer.ggml.eos_token_id", T_U32, int(eos)))
    sc = 0.2 / np.sqrt(dim)
    T = [("token_embd.weight", [dim, vocab_pad], GGML_Q4_0, q4_0(rng, vocab_pad, dim, 0.15))]
    for l in range(n_layer):
        p = "blk.%d." % l
        T.append((p + "attn_norm.weight", [dim], GGML_F32, f32(np.ones(dim))))
        T.append((p + "attn_q.weight", [dim, dim], GGML_Q4_0, q4_0(rng, dim, dim, sc)))
        T.append((p + "attn_k.weight", [dim, kvd], GGML_Q4_0, q4_0(rng, kvd, dim, sc)))
        T.append((p + "attn_v.weight", [dim, kvd], GGML_Q4_0, q4_0(rng, kvd, dim, sc)))
        T.append((p + "attn_output.weight", [dim, dim], GGML_Q4_0, q4_0(rng, dim, dim, sc)))
        T.append((p + "ffn_norm.weight", [dim], GGML_F32, f32(np.ones(dim))))
        T.append((p + "ffn_gate.weight", [dim, ffn], GGML_Q4_0, q4_0(rng, ffn, dim, sc)))
        T.append((p + "ffn_up.weight", [dim, ffn], GGML_Q4_0, q4_0(rng, ffn, dim, sc)))
        T.append((p + "ffn_down.weight", [ffn, dim], GGML_Q4_0, q4_0(rng, dim, ffn, 0.2 / np.sqrt(ffn))))
    T.append(("output_norm.weight", [dim], GGML_F32, f32(np.ones(dim))))
    write_gguf(out, kvs, T)
    print("wrote %s: vocab %d (%d with padding), pre %s, eos %s" % (out, vocab, vocab_pad, pre, eos))


if __name__ == "__main__":
    main()
