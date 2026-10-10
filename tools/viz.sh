#!/bin/sh
# viz.sh MODEL [N] [OUT.html]: kernel timeline of N decode steps.
# Runs ie-run once under rocprofv3 with IE_TRACE set, then tools/ie_viz.py.
# Writes OUT.html (default ie_timeline.html) and OUT.json (Perfetto).
# ROCPROF: the rocprofv3 path. VIZ_ARGS: more ie-run options (for example
# "--draft 7" for MTP steps, or "--tokens-file p.txt" for a prompt).
set -e
M=$1; N=${2:-8}; O=${3:-ie_timeline.html}
[ -n "$M" ] || { echo "usage: tools/viz.sh MODEL.gguf [N] [OUT.html]" >&2; exit 2; }
R=${ROCPROF:-rocprofv3}
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT
IE_TRACE=$D/launch.csv $R --kernel-trace --output-format csv -d "$D/rp" -o run -- \
  ./build/ie-run "$M" --backend gpu --tokens 760 --n "$N" $VIZ_ARGS >/dev/null 2>"$D/err" || { cat "$D/err" >&2; exit 1; }
K=$(find "$D/rp" -name 'run_kernel_trace.csv' | head -1)
[ -n "$K" ] || { echo "viz.sh: no kernel trace in $D/rp" >&2; ls -R "$D/rp" >&2; exit 1; }
python3 tools/ie_viz.py "$D/launch.csv" "$K" -o "$O" --perfetto "${O%.html}.json" \
  --title "$(basename "$M") · ie-run --n $N $VIZ_ARGS · $(git rev-parse --short HEAD 2>/dev/null)"
