#!/usr/bin/env python3
"""Independent numpy float64 reference forward pass (Llama / Qwen2 GGUF).

It reads the GGUF with the `gguf` pip package (not with the engine's
parser), dequantizes the weights in the GGUF layout (not the repacked one),
and runs a plain float64 forward pass.

Modes:
  float : float activations everywhere.
  q8    : before each Q4_0 or Q8_0 matrix (not the embedding), the input is
          quantized per block of 32 as the engine does (d = amax/127,
          q = round(x/d)); the product is sum_b d_w * d_a * sum_j w_j * q_a
          in float64, with w_j = q_w - 8 (Q4_0) or the int8 weight (Q8_0).

usage: ref_forward.py MODEL.gguf DUMP --mode q8|float [--tol REL]
Relative error = max |engine - ref| / max |ref| per position.

In q8 mode the engine (float32) and the reference (float64) agree to about
1e-7, except where an activation lands within float32 rounding of a .5
boundary of x/d: then one int8 differs by 1 and the logits move by ~1e-3.
So the q8 test checks the median per-position error tightly and the max
loosely.

DUMP is the --dump-logits file of ie-run: "pos token l0 l1 ...". The
reference follows the same tokens, compares the logits of every position,
and checks that its greedy token is the token ie-run chose next.
"""
import argparse
import sys

import numpy as np
from gguf import GGUFReader

Q4_0, Q8_0, F16, F32 = 2, 8, 1, 0


def field(r, key, default=None):
    f = r.get_field(key)
    if f is None:
        return default
    v = f.parts[f.data[0]]
    if f.types[0] == 8:  # string
        return bytes(v).decode("utf-8")
    return v[0].item()


def dequant(t):
    """Tensor -> float64 array of shape (rows, cols) (GGUF ne = [cols, rows])."""
    ne = [int(x) for x in t.shape]
    cols = ne[0]
    rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
    raw = np.asarray(t.data).view(np.uint8).reshape(-1)
    tt = int(t.tensor_type)
    if tt == F32:
        a = raw.view(np.float32).astype(np.float64)
    elif tt == F16:
        a = raw.view(np.float16).astype(np.float64)
    elif tt == Q8_0:
        b = raw.reshape(-1, 34)
        d = b[:, :2].copy().view(np.float16).astype(np.float64)
        a = (d * b[:, 2:].view(np.int8).astype(np.float64)).reshape(-1)
    elif tt == Q4_0:
        b = raw.reshape(-1, 18)
        d = b[:, :2].copy().view(np.float16).astype(np.float64)
        q = np.concatenate([b[:, 2:] & 15, b[:, 2:] >> 4], axis=1).astype(np.float64) - 8.0
        a = (d * q).reshape(-1)
    else:
        raise SystemExit("unsupported type %d in %s" % (tt, t.name))
    return a.reshape(rows, cols)


class Mat:
    def __init__(self, t):
        self.w = dequant(t)
        tt = int(t.tensor_type)
        # keep the integer form for the q8 mode (the engine's integer paths)
        self.qint = tt in (Q4_0, Q8_0)
        rows, cols = self.w.shape
        nb = cols // 32
        if tt == Q4_0:
            raw = np.asarray(t.data).view(np.uint8).reshape(-1, 18)
            self.d = raw[:, :2].copy().view(np.float16).astype(np.float64).reshape(rows, nb)
            q = np.concatenate([raw[:, 2:] & 15, raw[:, 2:] >> 4], axis=1).astype(np.int64) - 8
            self.q = q.reshape(rows, nb, 32)
        elif tt == Q8_0:
            raw = np.asarray(t.data).view(np.uint8).reshape(-1, 34)
            self.d = raw[:, :2].copy().view(np.float16).astype(np.float64).reshape(rows, nb)
            self.q = raw[:, 2:].copy().view(np.int8).astype(np.int64).reshape(rows, nb, 32)


def quant_q8(x):
    """Per block of 32: d = amax/127 (float32, as the engine), q = round half away from zero."""
    xb = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(xb).max(axis=1)
    d = (amax / np.float32(127.0)).astype(np.float32)
    safe = np.where(d == 0, np.float32(1), d)
    r = (xb / safe[:, None]).astype(np.float32)
    q = (np.sign(r) * np.floor(np.abs(r) + 0.5)).astype(np.int64)
    q[d == 0] = 0
    return d.astype(np.float64), q


def matvec(m, x, mode):
    if mode == "q8" and getattr(m, "qint", False):
        d, q = quant_q8(x)
        dots = np.einsum("rbj,bj->rb", m.q, q).astype(np.float64)
        return (m.d * d[None, :] * dots).sum(axis=1)
    return m.w @ x


def rmsnorm(x, w, eps):
    # the engine computes the norm of the f32 values; float64 here
    return x / np.sqrt(np.mean(x * x) + eps) * w


def rope(x, nh, hd, pos, base, neox):
    x = x.reshape(nh, hd).copy()
    h2 = hd // 2
    i = np.arange(h2)
    th = pos * base ** (-2.0 * i / hd)
    c, s = np.cos(th), np.sin(th)
    if neox:
        a, b = x[:, :h2].copy(), x[:, h2:].copy()
        x[:, :h2], x[:, h2:] = a * c - b * s, a * s + b * c
    else:
        a, b = x[:, 0::2].copy(), x[:, 1::2].copy()
        x[:, 0::2], x[:, 1::2] = a * c - b * s, a * s + b * c
    return x.reshape(-1)


class Model:
    def __init__(self, path):
        r = GGUFReader(path)
        self.arch = field(r, "general.architecture")
        a = self.arch
        self.n_layer = field(r, a + ".block_count")
        self.dim = field(r, a + ".embedding_length")
        self.n_head = field(r, a + ".attention.head_count")
        self.n_kv = field(r, a + ".attention.head_count_kv")
        self.eps = field(r, a + ".attention.layer_norm_rms_epsilon", 1e-5)
        self.base = field(r, a + ".rope.freq_base", 10000.0)
        self.hd = self.dim // self.n_head
        self.neox = a == "qwen2"
        T = {t.name: t for t in r.tensors}
        self.tok = dequant(T["token_embd.weight"])
        self.out = Mat(T["output.weight"] if "output.weight" in T else T["token_embd.weight"])
        self.out_norm = dequant(T["output_norm.weight"]).reshape(-1)
        self.L = []
        for l in range(self.n_layer):
            p = "blk.%d." % l
            g = lambda n: T.get(p + n)
            L = {k: Mat(g(k + ".weight")) for k in
                 ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]}
            L["attn_norm"] = dequant(g("attn_norm.weight")).reshape(-1)
            L["ffn_norm"] = dequant(g("ffn_norm.weight")).reshape(-1)
            for b in ["attn_q", "attn_k", "attn_v"]:
                t = g(b + ".bias")
                L[b + ".bias"] = dequant(t).reshape(-1) if t is not None else 0.0
            self.L.append(L)
        self.kc = [[] for _ in range(self.n_layer)]
        self.vc = [[] for _ in range(self.n_layer)]

    def step(self, tok, pos, mode):
        hd, nh, nkv = self.hd, self.n_head, self.n_kv
        x = self.tok[tok].copy()
        for l, L in enumerate(self.L):
            h = rmsnorm(x, L["attn_norm"], self.eps)
            q = matvec(L["attn_q"], h, mode) + L["attn_q.bias"]
            k = matvec(L["attn_k"], h, mode) + L["attn_k.bias"]
            v = matvec(L["attn_v"], h, mode) + L["attn_v.bias"]
            q = rope(q, nh, hd, pos, self.base, self.neox)
            k = rope(k, nkv, hd, pos, self.base, self.neox)
            self.kc[l].append(k.reshape(nkv, hd))
            self.vc[l].append(v.reshape(nkv, hd))
            K = np.stack(self.kc[l])  # (P, nkv, hd)
            V = np.stack(self.vc[l])
            qh = q.reshape(nh, hd)
            grp = nh // nkv
            att = np.zeros((nh, hd))
            for hh in range(nh):
                s = K[:, hh // grp, :] @ qh[hh] / np.sqrt(hd)
                s = np.exp(s - s.max())
                s /= s.sum()
                att[hh] = s @ V[:, hh // grp, :]
            x = x + matvec(L["attn_output"], att.reshape(-1), mode)
            h = rmsnorm(x, L["ffn_norm"], self.eps)
            g = matvec(L["ffn_gate"], h, mode)
            u = matvec(L["ffn_up"], h, mode)
            m = g / (1 + np.exp(-g)) * u
            x = x + matvec(L["ffn_down"], m, mode)
        h = rmsnorm(x, self.out_norm, self.eps)
        return matvec(self.out, h, mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dump")
    ap.add_argument("--mode", choices=["q8", "float"], default="q8")
    ap.add_argument("--tol", type=float, default=None, help="max relative error allowed (all positions)")
    ap.add_argument("--tol-median", type=float, default=None, help="median of the per-position errors allowed")
    a = ap.parse_args()
    rows = [l.split() for l in open(a.dump) if l.strip()]
    m = Model(a.model)
    worst_abs = worst_rel = 0.0
    rels = []
    tok_ok = True
    for i, row in enumerate(rows):
        pos, tok = int(row[0]), int(row[1])
        c = np.array([float(v) for v in row[2:]])
        ref = m.step(tok, pos, a.mode)
        err = np.abs(c - ref).max()
        rel = err / np.abs(ref).max()
        worst_abs, worst_rel = max(worst_abs, err), max(worst_rel, rel)
        rels.append(rel)
        # the token ie-run feeds next: from the dump (prompt or its argmax)
        if i + 1 < len(rows):
            nxt = int(rows[i + 1][1])
            is_gen = nxt == int(np.argmax(c))
            if is_gen and int(np.argmax(ref)) != nxt:
                srt = np.sort(ref)
                print("  pos %d: greedy token differs (ref %d, engine %d; ref top-2 gap %.3g)"
                      % (pos, int(np.argmax(ref)), nxt, srt[-1] - srt[-2]))
                tok_ok = False
    med = float(np.median(rels))
    ok = tok_ok and (a.tol is None or worst_rel <= a.tol) and (a.tol_median is None or med <= a.tol_median)
    print("%s mode=%s: %d positions, max abs err %.3e, rel err max %.3e (tol %s) median %.3e (tol %s), "
          "greedy tokens %s -> %s" % (a.model.split("/")[-1], a.mode, len(rows), worst_abs, worst_rel, a.tol, med,
                                     a.tol_median, "identical" if tok_ok else "DIFFER", "PASS" if ok else "FAIL"))
    print("  per position rel err: " + " ".join("%.1e" % r for r in rels))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
