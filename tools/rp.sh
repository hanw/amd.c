#!/bin/sh
# rp.sh MODEL [N]: hardware kernel trace (rocprofv3) of N decode steps,
# summarised per kernel by tools/rocprof_ops.py. ROCPROF: the rocprofv3 path.
M=$1; N=${2:-16}
R=${ROCPROF:-rocprofv3}
D=$(mktemp -d)
IE_PROFILE=1 IE_PROFILE_CSV=$D/ops.csv ./build/ie-run "$M" --backend gpu --tokens 760 --n "$N" >/dev/null 2>&1
$R --kernel-trace --output-format csv -d $D/rp -o run -- ./build/ie-run "$M" --backend gpu --tokens 760 --n "$N" >/dev/null 2>&1
python3 tools/rocprof_ops.py $D/ops.csv $D/rp/run_kernel_trace.csv 2>&1 >/dev/null
rm -rf $D
