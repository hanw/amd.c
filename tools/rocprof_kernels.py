"""Time per kernel of one phase (default: prefill) from an IE_TRACE log and a
rocprofv3 kernel trace of the same run (as tools/ie_viz.py).

usage: rocprof_kernels.py TRACE.csv KERNEL_TRACE.csv [phase]

Prints, as the mean over the runs of the phase (the first run left out when
there are more), the time of each kernel, its share, its launches per run
and the time per launch."""
import sys, collections
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ie_viz
log = ie_viz.load_log(sys.argv[1]); disp = ie_viz.load_rocprof(sys.argv[2])
kernels, runs = ie_viz.join(log, disp)
phase = sys.argv[3] if len(sys.argv) > 3 else "prefill"
sel = [r for r in runs if r["phase"] == phase][1:] or [r for r in runs if r["phase"] == phase]
n = len(sel)
by = collections.defaultdict(lambda: [0.0, 0])
tot = 0.0
for r in sel:
    for (t0, dur, k, op, layer, groups, T, wb) in r["l"]:
        by[kernels[k]][0] += dur / 1e6; by[kernels[k]][1] += 1; tot += dur / 1e6
print(f"{phase}: {n} runs (pos {sel[0]['pos']}..{sel[-1]['pos']}), {tot/n:.1f} ms per run in kernels")
for k, (ms, c) in sorted(by.items(), key=lambda x: -x[1][0])[:16]:
    print(f"  {k:24s} {ms/n:9.2f} ms/run {100*ms/tot:5.1f}%  {c//n:4d} launches/run  {ms/c*1e3:8.1f} us each")
