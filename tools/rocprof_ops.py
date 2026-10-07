#!/usr/bin/env python3
"""Map a rocprofv3 kernel trace (and optional counter collection) to the ops
of the engine graph, and summarise per op, per kernel and per layer.

usage: rocprof_ops.py OPS.csv TRACE_kernel_trace.csv [COUNTERS_counter_collection.csv] > out.csv

OPS.csv is the per-op file that ie-run writes with IE_PROFILE=1
IE_PROFILE_CSV=OPS.csv (op index, layer, kernel, rows, cols, weight bytes).
Each decode step dispatches the same ops in the same order, so the n-th
engine kernel of a step is op n. Kernels whose names do not start with
"ie_" (runtime copies) are ignored. The first step (warm up) is skipped.

Output: one CSV row per op with the mean hardware duration (ns), the mean
gap before it (ns, idle time between the previous kernel's end and its
start), and the mean of each counter per dispatch.
Summary lines go to stderr.
"""
import csv
import sys
from collections import defaultdict


def load_ops(path):
    ops = list(csv.DictReader(open(path)))
    for o in ops:
        o["op"] = int(o["op"])
        o["layer"] = int(o["layer"])
        o["weight_bytes"] = float(o["weight_bytes"])
    return ops


def steps_of(rows, n_ops, key):
    """Split the ie_ dispatches (sorted by key) into steps of n_ops."""
    rows = sorted((r for r in rows if r["Kernel_Name"].strip('"').startswith("ie_")), key=key)
    if len(rows) % n_ops:
        sys.exit(f"{len(rows)} engine dispatches is not a multiple of {n_ops} ops")
    return [rows[i : i + n_ops] for i in range(0, len(rows), n_ops)]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ops = load_ops(sys.argv[1])
    n = len(ops)
    trace = list(csv.DictReader(open(sys.argv[2])))
    steps = steps_of(trace, n, key=lambda r: int(r["Start_Timestamp"]))
    for s in steps:
        for i, r in enumerate(s):
            if r["Kernel_Name"] != ops[i]["kernel"]:
                sys.exit(f"op {i}: trace has {r['Kernel_Name']}, graph has {ops[i]['kernel']}")
    use = steps[1:] if len(steps) > 1 else steps
    dur = [0.0] * n
    gap = [0.0] * n
    step_wall = 0.0
    for s in use:
        for i, r in enumerate(s):
            dur[i] += int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
            if i:
                gap[i] += int(r["Start_Timestamp"]) - int(s[i - 1]["End_Timestamp"])
        step_wall += int(s[-1]["End_Timestamp"]) - int(s[0]["Start_Timestamp"])
    k = len(use)
    dur = [d / k for d in dur]
    gap = [g / k for g in gap]

    # Counters: one row per (dispatch, counter); dispatches in Dispatch_Id order.
    cnames, cval = [], [defaultdict(float) for _ in range(n)]
    if len(sys.argv) > 3:
        rows = list(csv.DictReader(open(sys.argv[3])))
        by_disp = defaultdict(dict)
        names = {}
        for r in rows:
            d = int(r["Dispatch_Id"])
            names[d] = r["Kernel_Name"]
            by_disp[d][r["Counter_Name"]] = by_disp[d].get(r["Counter_Name"], 0.0) + float(r["Counter_Value"])
        cnames = sorted({c for v in by_disp.values() for c in v})
        disp = [{"Kernel_Name": names[d], "id": d, **by_disp[d]} for d in by_disp]
        csteps = steps_of(disp, n, key=lambda r: r["id"])
        cuse = csteps[1:] if len(csteps) > 1 else csteps
        for s in cuse:
            for i, r in enumerate(s):
                for c in cnames:
                    cval[i][c] += r.get(c, 0.0) / len(cuse)

    w = csv.writer(sys.stdout)
    w.writerow(["op", "layer", "kernel", "rows", "cols", "weight_bytes", "hw_ns", "gap_ns"] + cnames)
    for i, o in enumerate(ops):
        w.writerow([i, o["layer"], o["kernel"], o["rows"], o["cols"], int(o["weight_bytes"]), round(dur[i], 1),
                    round(gap[i], 1)] + [round(cval[i][c], 1) for c in cnames])

    busy, idle = sum(dur), sum(gap)
    print(f"steps used: {k}; per token: kernels {busy / 1e6:.3f} ms, gaps {idle / 1e6:.3f} ms, "
          f"first-to-last {step_wall / k / 1e6:.3f} ms", file=sys.stderr)
    per = defaultdict(lambda: [0.0, 0.0, 0, 0.0])
    for i, o in enumerate(ops):
        key = (o["kernel"], f"{o['rows']}x{o['cols']}" if o["weight_bytes"] else "-")
        per[key][0] += dur[i]
        per[key][1] += gap[i]
        per[key][2] += 1
        per[key][3] += o["weight_bytes"]
    print(f"{'kernel':14} {'shape':>12} {'ops':>4} {'hw ms':>8} {'share':>6} {'us/op':>7} {'GB/s':>7}", file=sys.stderr)
    for (kn, shape), (d, g, c, b) in sorted(per.items(), key=lambda kv: -kv[1][0]):
        bw = f"{b / d:7.1f}" if b else "      -"
        print(f"{kn:14} {shape:>12} {c:4d} {d / 1e6:8.4f} {100 * d / busy:5.1f}% {d / c / 1e3:7.2f} {bw}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
