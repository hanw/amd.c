#!/bin/sh
# Check the integer GEMV kernels of the hsaco: each must use the dot4
# instruction (v_dot4_i32_iu8, 8 per block), 128-bit global loads
# (global_load_b128), and no scratch (private) memory. The Q8_0 kernel must
# use signed A operands (neg_lo:[1,1,0]); the Q4 kernel unsigned A
# (neg_lo:[0,1,0]).
set -e
H=${1:-build/ie_kernels.hsaco}
OBJDUMP=${OBJDUMP:-llvm-objdump}
ok=1
check() { # kernel, min b128 loads, the neg_lo form
  D=$($OBJDUMP -d --disassemble-symbols="$1" "$H")
  dot=$(printf '%s\n' "$D" | grep -c 'v_dot4_i32_iu8' || true)
  sgn=$(printf '%s\n' "$D" | grep 'v_dot4_i32_iu8' | grep -c "$3" || true)
  b128=$(printf '%s\n' "$D" | grep -c 'global_load_b128' || true)
  scratch=$(printf '%s\n' "$D" | grep -c 'scratch_' || true)
  echo "$1: v_dot4_i32_iu8 x$dot ($3 x$sgn), global_load_b128 x$b128, scratch x$scratch"
  printf '%s\n' "$D" | grep -E 'v_dot4_i32_iu8|global_load' | sed 's|//.*||; s/^[[:space:]]*/    /'
  [ "$dot" -ge 8 ] && [ "$sgn" -eq "$dot" ] && [ "$b128" -ge "$2" ] && [ "$scratch" -eq 0 ] || ok=0
}
check ie_gemv_q4q8 1 'neg_lo:\[0,1,0\]'
check ie_gemv_q8q8 2 'neg_lo:\[1,1,0\]'
echo "kernels in $H:"
${READELF:-llvm-readelf} --notes "$H" | grep -E '^ +\.name: +ie_' | sed 's/^ *\.name: */    /'
[ "$ok" -eq 1 ] && echo "objdump check: ok"
