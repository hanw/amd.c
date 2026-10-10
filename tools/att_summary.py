#!/usr/bin/env python3
"""Where the cycles of one kernel dispatch go, from a rocprofv3 thread trace.

  rocprofv3 --att --kernel-include-regex '^ie_gemm_q4kr$' --kernel-iteration-range '[3-3]' \
      --output-format csv -d OUT -o run -- ./build/ie-run MODEL --backend gpu --tokens-file P.txt --n 1 --chunk 512
  att_summary.py OUT/stats_ui_output_agent_*_dispatch_*.csv [N]

The stats file has per instruction: hit count, latency (cycles from issue to
the next instruction of the wave), stall and idle cycles. This prints the
latency and stall by instruction class (WMMA, VALU, waits for memory and
LDS, barriers, ...) and the N instructions with the most latency."""
import csv, sys, collections
rows = [r for r in csv.DictReader(open(sys.argv[1])) if int(r["Hitcount"] or 0) > 0]
tot = sum(int(r["Latency"]) for r in rows); st = sum(int(r["Stall"]) for r in rows); idle = sum(int(r["Idle"]) for r in rows)
print(f"instructions hit: {len(rows)}, latency cycles {tot}, stall {st} ({100*st/tot:.1f}%), idle {idle}")
def cls(ins):
    m = ins.split()[0] if ins.split() else ins
    if m.startswith("v_wmma"): return "wmma"
    if m.startswith("s_wait_dscnt"): return "s_wait_dscnt (LDS wait)"
    if m.startswith("s_wait_loadcnt"): return "s_wait_loadcnt (memory wait)"
    if m.startswith("s_wait"): return "s_wait other"
    if m.startswith("s_barrier"): return "barrier"
    if m.startswith("ds_load"): return "ds_load"
    if m.startswith("ds_store"): return "ds_store"
    if m.startswith(("global_load", "buffer_load")): return "global_load"
    if m.startswith(("global_store")): return "global_store"
    if m.startswith("v_dual"): return "valu dual"
    if m.startswith("v_"): return "valu"
    if m.startswith("s_"): return "salu/other"
    return m
c = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    k = cls(r["Instruction"]); c[k][0] += int(r["Latency"]); c[k][1] += int(r["Stall"]); c[k][2] += int(r["Hitcount"])
print(f"{'class':32s} {'latency%':>8s} {'stall%':>7s} {'issued':>9s}")
for k, (l, s, h) in sorted(c.items(), key=lambda x: -x[1][0]):
    print(f"{k:32s} {100*l/tot:8.1f} {100*s/tot:7.1f} {h:9d}")
print("top instructions by latency:")
for r in sorted(rows, key=lambda r: -int(r["Latency"]))[:int(sys.argv[2]) if len(sys.argv) > 2 else 15]:
    print(f"  {int(r['Latency']):9d} lat {int(r['Stall']):9d} stall {int(r['Hitcount']):7d} hit  {r['Vaddr']:>7s}  {r['Instruction'][:70]}")
