#!/usr/bin/env python3
"""Kernel timeline of the engine on the GPU: an interactive HTML page (and a
Perfetto trace) from a launch log and a rocprofv3 kernel trace.

usage: ie_viz.py TRACE.csv KERNEL_TRACE.csv [-o out.html] [--perfetto trace.json]
                 [--bw 640] [--title TEXT]

TRACE.csv is the launch log that ie-run writes with IE_TRACE=TRACE.csv: one
row per kernel launch, in launch order, with the run (call, phase, first
position, tokens), the op, its layer, the grid and the weight bytes of the op.
KERNEL_TRACE.csv is `rocprofv3 --kernel-trace --output-format csv` of the same
run. All launches go to one stream, so the n-th ie_ dispatch (by start time)
of the kernel trace is launch n of the log; the script checks the names.

tools/viz.sh runs both and calls this script.

The page shows, for one run (a decode step, a prompt chunk, ...):
  - the kernels on a time axis, in lanes by kernel class, by layer, or in
    one lane, with the idle gaps between kernels;
  - per kernel: duration, gap before it, weight bytes, GB/s and the share of
    the nominal bandwidth (--bw, GB/s);
  - per run: wall time, kernel time, gap time, launches, and the time to read
    the weights at the nominal bandwidth (the floor);
  - tables by kernel class and by layer (mean over the runs of the phase).
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def load_log(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        sys.exit(f"{path}: no launches")
    need = {"call", "pos", "T_run", "launch", "phase", "op", "layer", "kernel", "groups", "T", "weight_bytes"}
    if not need <= set(rows[0]):
        sys.exit(f"{path}: not an IE_TRACE file (columns {sorted(rows[0])})")
    return rows


def load_rocprof(path):
    out = []
    for r in csv.DictReader(open(path)):
        name = r["Kernel_Name"].strip().strip('"')
        if name.endswith(".kd"):
            name = name[:-3]
        if not name.startswith("ie_"):
            continue  # runtime copies and fills
        out.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), name))
    out.sort()
    return out


def join(log, disp):
    if len(log) != len(disp):
        sys.exit(
            f"{len(log)} launches in the log, {len(disp)} ie_ dispatches in the kernel trace.\n"
            "Run ie-run once, under rocprofv3, with IE_TRACE set (tools/viz.sh does this)."
        )
    for i, (row, d) in enumerate(zip(log, disp)):
        if row["kernel"] != d[2]:
            sys.exit(f"launch {i}: the log has {row['kernel']}, the kernel trace has {d[2]}")
    kernels, kidx = [], {}
    runs, cur = [], None
    for row, (t0, t1, name) in zip(log, disp):
        call = int(row["call"])
        if cur is None or cur["call"] != call:
            cur = {"call": call, "phase": row["phase"], "pos": int(row["pos"]), "T": int(row["T_run"]), "t0": t0, "l": []}
            runs.append(cur)
        if name not in kidx:
            kidx[name] = len(kernels)
            kernels.append(name)
        cur["l"].append([
            t0 - cur["t0"], t1 - t0, kidx[name], int(row["op"]), int(row["layer"]),
            int(row["groups"]), int(row["T"]), float(row["weight_bytes"]),
        ])
    t_first = runs[0]["t0"]
    for r in runs:
        r["start"] = r["t0"] - t_first
        del r["t0"]
    return kernels, runs


def perfetto(kernels, runs, cls, path):
    ev = []
    lanes = {}
    for r in runs:
        base = r["start"]
        ev.append({"name": f"{r['phase']} pos={r['pos']} T={r['T']}", "ph": "X", "pid": 1, "tid": 0,
                   "ts": base / 1e3, "dur": (r["l"][-1][0] + r["l"][-1][1]) / 1e3, "args": {"call": r["call"]}})
        for t, d, k, op, layer, groups, T, wb in r["l"]:
            c = cls(kernels[k])
            tid = lanes.setdefault(c, len(lanes) + 1)
            args = {"op": op, "layer": layer, "groups": groups, "T": T}
            if wb > 0:
                args["MB"] = round(wb / 1e6, 3)
                args["GB/s"] = round(wb / d, 1) if d else 0
            ev.append({"name": kernels[k], "cat": c, "ph": "X", "pid": 1, "tid": tid,
                       "ts": (base + t) / 1e3, "dur": d / 1e3, "args": args})
    meta = [{"name": "thread_name", "ph": "M", "pid": 1, "tid": 0, "args": {"name": "run"}}]
    meta += [{"name": "thread_name", "ph": "M", "pid": 1, "tid": v, "args": {"name": k}} for k, v in lanes.items()]
    meta.append({"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "GPU stream"}})
    json.dump({"traceEvents": meta + ev, "displayTimeUnit": "ns"}, open(path, "w"))


def kclass(name):
    n = name[3:]
    if n.startswith("gemv"):
        return "gemv"
    if n.startswith("gemm"):
        return "gemm"
    if n.startswith("attn"):
        return "attn"
    if n.startswith(("gdn", "gnorm", "ring_store")):
        return "gdn"
    if n.startswith(("rmsnorm", "quant", "requant")):
        return "norm"
    if n.startswith(("argmax", "topk")):
        return "sample"
    return "elem"


def summary(kernels, runs, bw):
    by = {}
    for r in runs:
        by.setdefault(r["phase"], []).append(r)
    for ph, rs in by.items():
        use = rs[1:] if len(rs) > 1 else rs  # the first run of a phase warms up
        wall = sum(r["l"][-1][0] + r["l"][-1][1] for r in use) / len(use) / 1e3
        busy = sum(sum(x[1] for x in r["l"]) for r in use) / len(use) / 1e3
        wb = sum(sum(x[7] for x in r["l"]) for r in use) / len(use)
        n = sum(len(r["l"]) for r in use) / len(use)
        print(f"{ph}: {len(rs)} runs; mean of {len(use)}: {wall:.1f} us wall, {busy:.1f} us in kernels, "
              f"{wall - busy:.1f} us gaps, {n:.0f} launches, {wb / 1e6:.1f} MB weights "
              f"(floor {wb / bw / 1e3:.1f} us at {bw:g} GB/s)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="IE_TRACE launch log (CSV)")
    ap.add_argument("kernel_trace", help="rocprofv3 kernel trace (CSV)")
    ap.add_argument("-o", "--output", default="ie_timeline.html")
    ap.add_argument("--perfetto", help="also write a Chrome/Perfetto trace (JSON)")
    ap.add_argument("--bw", type=float, default=640.0, help="nominal memory bandwidth, GB/s (R9700: 640)")
    ap.add_argument("--title", default="", help="text under the page title (model, flags)")
    a = ap.parse_args()

    kernels, runs = join(load_log(a.trace), load_rocprof(a.kernel_trace))
    summary(kernels, runs, a.bw)
    data = {"kernels": kernels, "classes": [kclass(k) for k in kernels], "runs": runs, "bw": a.bw, "title": a.title}
    page = (Path(__file__).with_name("ie_viz.html")).read_text()
    page = page.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    Path(a.output).write_text(page)
    print(f"wrote {a.output} ({len(runs)} runs, {sum(len(r['l']) for r in runs)} launches)", file=sys.stderr)
    if a.perfetto:
        perfetto(kernels, runs, kclass, a.perfetto)
        print(f"wrote {a.perfetto} (open it in https://ui.perfetto.dev)", file=sys.stderr)


if __name__ == "__main__":
    main()
