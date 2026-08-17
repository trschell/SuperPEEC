#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Verification baseline: the full validator suite + the 3 byte anchors.
# Usage:   bash studies/run_baseline.sh [python]
set -u
cd "$(dirname "$0")/.."
PY="${1:-python3}"
echo "python: $($PY -V 2>&1)   repo: $(pwd)"
export PYTHONPATH=src:validation
for v in validation/validate_*.py; do
  out=$("$PY" "$v" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then printf '%-36s PASS\n' "$(basename ${v%.py})"
  else printf '%-36s FAIL(rc=%d)  %s\n' "$(basename ${v%.py})" $rc \
       "$(echo "$out" | grep -E '^FAIL|Error' | head -1)"; fi
done
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
# Lineage 2026-08-15: rebuilt again to ADD the MID_M2L_C64
# subroutine (fp32 phase 3c); anchors re-rolled as expected -- the
# anchor setups are lv2 (no MidLevel), so the changed code paths
# are not even exercised there. Physics gate for the change: 34/34
# validators incl. oracle tolerances to 1e-13 (baseline_phase3c).
echo "expected (this machine, stock env, binaries as built fresh in"
echo "the SuperPEEC tree 2026-08-16; a rebuild or another machine"
echo "re-bases these -- record your own values on first green run):"
echo "          setup1 1.0201733741400102e-30"
echo "          setup2 3.6833436970020444e-30"
echo "          setup3 2.7418785553237524e-31"
echo ALL DONE
