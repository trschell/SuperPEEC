#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Verification baseline: the full validator suite + the 3 byte anchors.
# Usage:   bash validation/run_baseline.sh [python]
#
# A SKIPPED VALIDATOR IS NOT A PASSING ONE. Several validators print
# 'SKIP: ...' and exit 0 when their inputs are absent -- five of them
# when the VoxHenry corpus is missing, which this tree does not ship,
# and one when there is no usable GPU. This runner used to score exit 0
# as PASS, so it reported 38/38 while actually testing 33. That is how a
# plain IndexError in conduction_weights (equiterminal.py, k <= 3) lived
# in a tree with a fully green suite: the only validator that reaches it
# is one of the five that silently skipped. Skips are now counted and
# named, and the script exits non-zero if anything FAILS.
#
# TO ENABLE THE SKIPPED FIVE you need a VoxHenry checkout in TWO places
# and one scratch directory:
#     ln -s /path/to/VoxHenry validation/VoxHenry   # the skip guard
#     ln -s /path/to/VoxHenry VoxHenry              # the data paths
#     mkdir -p validation/studies                   # scratch .vhr output
# The two links are needed because the guard tests an absolute path
# (relative to __file__) while the data paths are resolved from the
# REPO ROOT -- an inconsistency inside the validators, not here.
set -u
cd "$(dirname "$0")/.."
PY="${1:-python3}"
echo "python: $($PY -V 2>&1)   repo: $(pwd)"
export PYTHONPATH=src:validation
n_pass=0; n_skip=0; n_fail=0; skipped=""; failed=""
for v in validation/validate_*.py; do
  name="$(basename ${v%.py})"
  out=$("$PY" "$v" 2>&1); rc=$?
  reason=$(echo "$out" | grep -m1 '^SKIP' | cut -c7-72)
  if [ $rc -ne 0 ]; then
    n_fail=$((n_fail+1)); failed="$failed $name"
    printf '%-36s FAIL(rc=%d)  %s\n' "$name" $rc \
       "$(echo "$out" | grep -E '^FAIL|Error' | head -1)"
  elif [ -n "$reason" ]; then
    n_skip=$((n_skip+1)); skipped="$skipped $name"
    printf '%-36s SKIP  %s\n' "$name" "$reason"
  else
    n_pass=$((n_pass+1)); printf '%-36s PASS\n' "$name"
  fi
done
echo "--- $n_pass passed, $n_skip SKIPPED (not tested), $n_fail failed ---"
if [ $n_skip -gt 0 ]; then
  echo "NOT TESTED:$skipped"
  echo "  ^ these exited 0 WITHOUT RUNNING. See the header for how to"
  echo "    enable them; until then this suite is not a full gate."
fi
[ $n_fail -gt 0 ] && echo "FAILED:$failed"
echo "=== ANCHORS ==="
# main.py picks the setup via a hard-coded `sourcetype`; patch a temp copy.
# SPPEEC_GPU=0 ON PURPOSE: anchors pin the deterministic CPU path (GPU
# scatter_add atomics reorder at rounding level run to run), per the
# 2026-08-12 anchor policy. The values below are MACHINE-LOCAL residual
# norms of a converged solve (~1e-30) on the machine they were recorded; they detect
# regressions on ONE machine, nothing more, and shift with
# OPENBLAS_NUM_THREADS.
for s in 1 2 3; do
  sed "s/^sourcetype = 3$/sourcetype = $s/" validation/main.py > _anchor_tmp.py
  printf 'setup%s: ' $s
  SPPEEC_GPU=0 "$PY" _anchor_tmp.py 2>&1 | grep '^norm: ' | tail -1
done
rm -f _anchor_tmp.py
# TIMING (2026-08-07, idle box): setup1 ~1-2 min, setup2 ~15 min (the
# long one; high CPU from threaded BLAS is HEALTHY), setup3 a few min.
# Anchor lineage 2026-08-14: the edge-scheme REMOVAL left these
# bit-identical (proof the removed code was inert). The dead-code
# PRUNE then REBUILT mp_fortran, and a rebuild re-rolls the ~1e-30
# rounding dust these converged residual norms are made of.
# Anchor lineage (thread port): adding sppeec_threads re-based setup1
# only -- 1.0201733741400102e-30 -> 1.0141635224321326e-30 -- while
# setup2 and setup3 came back BIT-IDENTICAL. That is the shift this
# header already predicted ("shift with OPENBLAS_NUM_THREADS"): pinning
# OpenBLAS to one thread changes the reduction ORDER of the dense
# kernels, so the ~1e-30 rounding dust these norms are made of re-rolls.
# setup1 moved by 0.6% OF THE DUST, not of any answer, and the physics
# gate for the change is the validator suite: 38/38 PASS, verdicts
# identical to the pre-port run.
# Lineage 2026-08-15: rebuilt again to ADD the MID_M2L_C64
# subroutine (fp32 phase 3c); anchors re-rolled as expected -- the
# anchor setups are lv2 (no MidLevel), so the changed code paths
# are not even exercised there. Physics gate for the change: 34/34
# validators incl. oracle tolerances to 1e-13 (baseline_phase3c).
echo "expected (this machine, stock env, binaries as built fresh in"
echo "the SuperPEEC tree 2026-08-16; a rebuild or another machine"
echo "re-bases these -- record your own values on first green run):"
echo "          setup1 1.0141635224321326e-30"
echo "          setup2 3.6833436970020444e-30"
echo "          setup3 2.7418785553237524e-31"
echo ALL DONE
# Non-zero exit on any FAIL so CI and callers can gate on this script.
# Skips deliberately do NOT fail the run -- they are legitimate when the
# corpus or a GPU is genuinely unavailable -- but they are counted and
# named above so they can never again be mistaken for coverage.
exit $(( n_fail > 0 ))
