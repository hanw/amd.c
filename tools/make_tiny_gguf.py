#!/usr/bin/env python3
"""Write small random Llama / Qwen2 models as GGUF v3 files (own writer,
numpy only). The shapes exercise the edge cases of the GEMV work split:
  qwen2: dim 96 (3 blocks: fewer blocks than lanes), ffn 352 (11 blocks),
         q/k/v biases, tied output (no output.weight), Q4_0 token_embd,
         vocab 300 (not a multiple of 8 rows), GPT2 byte-level tokens.
  llama: dim 1152 (36 blocks > 32: lanes do 2 steps), ffn 1376 (43 blocks),
         F16 token_embd, separate Q4_0 output, one Q8_0 matrix (blk.1.attn_v),
         SentencePiece tokens with <0xXX> byte tokens.
usage: make_tiny_gguf.py OUT_DIR
"""
import struct
import sys

import numpy as np

GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q8_0 = 0, 1, 2, 8
T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_F64 = 4, 5, 6, 7, 8, 9, 10, 12
ALIGN = 32


def s_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def s_val(t, v):
    if t == T_U32:
        return struct.pack("<I", v)
    if t == T_I32:
        return struct.pack("<i", v)
    if t == T_F32:
        return struct.pack("<f", v)
    if t == T_F64:
        return struct.pack("<d", v)
    if t == T_U64:
        return struct.pack("<Q", v)
    if t == T_BOOL:
        return struct.pack("<B", 1 if v else 0)
    if t == T_STR:
        return s_str(v)
    raise ValueError(t)


def write_gguf(path, kvs, tensors):
    """kvs: list of (key, type, value) or (key, T_ARR, (elem_type, list));
    tensors: list of (name, ne list, ggml type, raw bytes)."""
    out = bytearray()
    out += b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs))
    for k, t, v in kvs:
        out += s_str(k) + struct.pack("<I", t)
        if t == T_ARR:
            et, items = v
            out += struct.pack("<IQ", et, len(items))
            for it in items:
                out += s_val(et, it)
        else:
            out += s_val(t, v)
    off = 0
    offs = []
    for name, ne, typ, raw in tensors:
        offs.append(off)
        out += s_str(name) + struct.pack("<I", len(ne))
        for d in ne:
            out += struct.pack("<Q", d)
        out += struct.pack("<IQ", typ, off)
        off += (len(raw) + ALIGN - 1) // ALIGN * ALIGN
    out += b"\0" * ((-len(out)) % ALIGN)
    for (name, ne, typ, raw), o in zip(tensors, offs):
        out += raw + b"\0" * ((-len(raw)) % ALIGN)
    with open(path, "wb") as f:
        f.write(out)


def q4_0(rng, rows, cols, scale):
    """Random Q4_0 matrix: random nibbles and f16 scales (exact values)."""
    assert cols % 32 == 0
    nb = cols // 32
    d = (rng.uniform(0.5, 1.5, size=(rows, nb)) * scale).astype(np.float16)
    q = rng.integers(0, 16, size=(rows, nb, 32), dtype=np.uint8)
    blk = np.zeros((rows, nb, 18), dtype=np.uint8)
    blk[:, :, 0:2] = d.view(np.uint8).reshape(rows, nb, 2)
    blk[:, :, 2:] = q[:, :, :16] | (q[:, :, 16:] << 4)
    return blk.tobytes()


def q8_0(rng, rows, cols, scale):
    nb = cols // 32
    d = (rng.uniform(0.5, 1.5, size=(rows, nb)) * scale / 16).astype(np.float16)
    q = rng.integers(-127, 128, size=(rows, nb, 32)).astype(np.int8)
    blk = np.zeros((rows, nb, 34), dtype=np.uint8)
    blk[:, :, 0:2] = d.view(np.uint8).reshape(rows, nb, 2)
    blk[:, :, 2:] = q.view(np.uint8)
    return blk.tobytes()


def f32(a):
    return np.asarray(a, dtype=np.float32).tobytes()


def gpt2_vocab(n):
    """GPT2 byte-level token strings: 256 single bytes, then some merges."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    k = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + k)
            k += 1
    byte2u = {b: chr(c) for b, c in zip(bs, cs)}
    toks = [byte2u[b] for b in range(256)]
    words = [" the", " and", "ing", " of", "er", " is", " 你好", " rose", "\n\n", " AMD"]
    i = 0
    while len(toks) < n:
        w = words[i % len(words)] + ("" if i < len(words) else str(i))
        toks.append("".join(byte2u[b] for b in w.encode("utf-8")))
        i += 1
    return toks


def spm_vocab(n):
    toks = ["<unk>", "<s>", "</s>"] + ["<0x%02X>" % b for b in range(256)]
    words = ["▁the", "▁and", "ing", "▁of", "er", "▁is", "▁rose", "▁AMD"]
    i = 0
    while len(toks) < n:
        toks.append(words[i % len(words)] + ("" if i < len(words) else str(i)))
        i += 1
    return toks


def make(path, arch, seed, n_layer, dim, ffn, n_head, n_kv, vocab, ctx, biases, tied, embd_type, q8_layer_v):
    rng = np.random.default_rng(seed)
    hd = dim // n_head
    kvd = n_kv * hd
    kvs = [
        ("general.architecture", T_STR, arch),
        ("general.name", T_STR, "tiny-" + arch),
        ("general.alignment", T_U32, ALIGN),
        (arch + ".block_count", T_U32, n_layer),
        (arch + ".context_length", T_U32, ctx),
        (arch + ".embedding_length", T_U32, dim),
        (arch + ".feed_forward_length", T_U32, ffn),
        (arch + ".attention.head_count", T_U32, n_head),
        (arch + ".attention.head_count_kv", T_U32, n_kv),
        (arch + ".attention.layer_norm_rms_epsilon", T_F32, 1e-6 if arch == "qwen2" else 1e-5),
        (arch + ".rope.freq_base", T_F32, 1000000.0 if arch == "qwen2" else 10000.0),
        ("general.some_f64", T_F64, 0.25),
        ("general.some_bools", T_ARR, (T_BOOL, [True, False, True])),
    ]
    if arch == "qwen2":
        toks = gpt2_vocab(vocab)
        kvs += [("tokenizer.ggml.model", T_STR, "gpt2"), ("tokenizer.ggml.tokens", T_ARR, (T_STR, toks)),
                ("tokenizer.ggml.token_type", T_ARR, (T_I32, [1] * vocab)),
                ("tokenizer.ggml.merges", T_ARR, (T_STR, ["Ġ t", "h e"]))]
    else:
        toks = spm_vocab(vocab)
        kvs += [("tokenizer.ggml.model", T_STR, "llama"), ("tokenizer.ggml.tokens", T_ARR, (T_STR, toks)),
                ("tokenizer.ggml.scores", T_ARR, (T_F32, [0.0] * vocab)),
                ("tokenizer.ggml.bos_token_id", T_U32, 1)]

    T = []
    if embd_type == GGML_Q4_0:
        T.append(("token_embd.weight", [dim, vocab], GGML_Q4_0, q4_0(rng, vocab, dim, 0.15)))
    else:
        T.append(("token_embd.weight", [dim, vocab], GGML_F16,
                  (rng.standard_normal((vocab, dim)) * 1.0).astype(np.float16).tobytes()))
    for l in range(n_layer):
        p = "blk.%d." % l
        sc = lambda k: 0.2 / np.sqrt(k)
        T.append((p + "attn_norm.weight", [dim], GGML_F32, f32(1 + 0.1 * rng.standard_normal(dim))))
        T.append((p + "attn_q.weight", [dim, dim], GGML_Q4_0, q4_0(rng, dim, dim, sc(dim))))
        T.append((p + "attn_k.weight", [dim, kvd], GGML_Q4_0, q4_0(rng, kvd, dim, sc(dim))))
        if l == q8_layer_v:
            T.append((p + "attn_v.weight", [dim, kvd], GGML_Q8_0, q8_0(rng, kvd, dim, sc(dim))))
        else:
            T.append((p + "attn_v.weight", [dim, kvd], GGML_Q4_0, q4_0(rng, kvd, dim, sc(dim))))
        if biases:
            T.append((p + "attn_q.bias", [dim], GGML_F32, f32(0.3 * rng.standard_normal(dim))))
            T.append((p + "attn_k.bias", [kvd], GGML_F32, f32(0.3 * rng.standard_normal(kvd))))
            T.append((p + "attn_v.bias", [kvd], GGML_F32, f32(0.3 * rng.standard_normal(kvd))))
        T.append((p + "attn_output.weight", [dim, dim], GGML_Q4_0, q4_0(rng, dim, dim, sc(dim))))
        T.append((p + "ffn_norm.weight", [dim], GGML_F32, f32(1 + 0.1 * rng.standard_normal(dim))))
        T.append((p + "ffn_gate.weight", [dim, ffn], GGML_Q4_0, q4_0(rng, ffn, dim, sc(dim))))
        T.append((p + "ffn_up.weight", [dim, ffn], GGML_Q4_0, q4_0(rng, ffn, dim, sc(dim))))
        T.append((p + "ffn_down.weight", [ffn, dim], GGML_Q4_0, q4_0(rng, dim, ffn, sc(ffn))))
    T.append(("output_norm.weight", [dim], GGML_F32, f32(1 + 0.1 * rng.standard_normal(dim))))
    if not tied:
        T.append(("output.weight", [dim, vocab], GGML_Q4_0, q4_0(rng, vocab, dim, 4 * sc(dim))))
    write_gguf(path, kvs, T)
    print("wrote", path)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "build"
    make(out + "/tiny-qwen2.gguf", "qwen2", 1, n_layer=2, dim=96, ffn=352, n_head=6, n_kv=2, vocab=300, ctx=256,
         biases=True, tied=True, embd_type=GGML_Q4_0, q8_layer_v=-1)
    make(out + "/tiny-llama.gguf", "llama", 2, n_layer=2, dim=1152, ffn=1376, n_head=8, n_kv=4, vocab=301, ctx=256,
         biases=False, tied=False, embd_type=GGML_F16, q8_layer_v=1)


if __name__ == "__main__":
    main()
