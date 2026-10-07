#!/bin/sh
# Translate core/ie_core.h + laws/ie_laws.cpp to Lean (cpp-lean, EDG
# cpfe-lean) and check every proof. Needs: CPP_LEAN (the cpp-lean checkout),
# EDG_DIR (edg-cpp with build/gcc-release/bin/cpfe-lean), lean 4.34.0 on PATH.
set -e
cd "$(dirname "$0")"
CPP_LEAN=${CPP_LEAN:-$HOME/cpp-lean}
"$CPP_LEAN/src/tools/cpp2lean" ../laws/ie_laws.cpp IeCore.lean IeLaws.lean
"$CPP_LEAN/src/tools/leancheck" IeCore IeLaws IeLemmas IeProof
