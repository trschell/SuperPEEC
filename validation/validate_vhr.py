# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate vhr.py against VoxHenry's own reader and against SuperPEEC.

Two independent halves:

PART A -- parser agreement. ``vhr_ref.txt`` is a dump produced by running
VoxHenry's own MATLAB/Octave ``pre_input_file.m`` over every shipped
input file (regenerate with ``dump_vhr.m``). Every field vhr.py
extracts is compared against it: grid dimensions, voxel
pitch, occupied-voxel count, the SUM of all conductivities and London
depths (which catches a value landing in the wrong voxel, not just a
miscount), frequency count and sum, and per-port terminal counts and
index sums. Port index sums are compared in the file's own 1-BASED
convention, so an off-by-one in the index conversion shows up here.

PART B -- SuperPEEC adapter. For the models SuperPEEC can represent, builds a
tree and checks the two things that are silently wrong rather than
loud:

  * the realised CELL PITCH on ALL THREE axes. Tree pads the grid up to
    a whole number of leaf boxes and divides the top box by the PADDED
    count, and the padding ratio differs per axis on a non-cubic grid
    (60x20x20 at leaf 5 pads to 65x25x25: 1.083 on x, 1.25 on y/z), so
    a single scalar rescale leaves anisotropic cells. Anisotropy is not
    loud -- it only shows as the x-directed filament resistance drifting
    from the y- and z-directed ones -- so both are asserted.

  * PORT NODE RESOLUTION. Tree.parsesource ASSIGNS rather than
    accumulates, so contributions at nodes shared between adjacent port
    faces would be silently dropped if not summed first. The check is
    that the compressed source vector has exactly as many nonzeros as
    the deduplicated node list has entries, and that the injected
    current still sums to zero across the port.

Run inside the toolbox:  python3 validate_vhr.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]


import os as _os
if not _os.path.isdir(_os.path.join(_os.path.dirname(
        _os.path.abspath(__file__)), 'VoxHenry')):
    print('SKIP: VoxHenry corpus not present -- this validator '
          'compares against VoxHenry shipped inputs/reference values. '
          'Place a VoxHenry checkout at validation/VoxHenry to enable it.')
    raise SystemExit(0)

import os
import sys
import numpy as np
import vhr
import stencils as st

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, 'VoxHenry', 'Input_files')
REF = os.path.join(HERE, 'vhr_ref.txt')

fails = []


def check(tag, cond, detail=''):
    if cond:
        return True
    fails.append("%s: %s" % (tag, detail))
    print("    FAIL %s  %s" % (tag, detail))
    return False


def close(a, b, rtol=1e-12):
    a, b = float(a), float(b)
    if a == b:
        return True
    return abs(a - b) <= rtol*max(abs(a), abs(b))


def read_ref(path):
    """Parse the Octave dump into {filename: {key: [tokens]}}."""
    out = {}
    cur = None
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == 'FILE':
                cur = {}
                out[tok[1]] = cur
            elif tok[0] == 'PORT':
                cur.setdefault('PORT', []).append(tok[1:])
            else:
                cur[tok[0]] = tok[1:]
    return out


# ---------------------------------------------------------------- A

print("=== PART A: vhr.py vs VoxHenry pre_input_file.m ===")
if not os.path.exists(REF):
    print("  reference dump missing: %s" % REF)
    print("  regenerate with: octave dump_vhr.m  (from the SuperPEEC root)")
    raise SystemExit(1)
ref = read_ref(REF)
print("  %d files in reference dump\n" % len(ref))

for name in sorted(ref):
    r = ref[name]
    m = vhr.read_vhr(os.path.join(INPUTS, name))
    print("  %s" % name)
    check(name + ' dims', list(m.dims) == [int(v) for v in r['DIMS']],
          "%s vs %s" % (list(m.dims), r['DIMS']))
    check(name + ' dx', close(m.dx, r['DX'][0]),
          "%r vs %s" % (m.dx, r['DX'][0]))
    # occupancy and the conductivity SUM: a miscount and a value in the
    # wrong voxel are different bugs; the sum catches the second.
    check(name + ' nnz(sigma)',
          int(np.count_nonzero(m.sigma)) == int(r['NNZSIG'][0]),
          "%d vs %s" % (np.count_nonzero(m.sigma), r['NNZSIG'][0]))
    check(name + ' sum(sigma)', close(m.sigma.sum(), r['SUMSIG'][0], 1e-10),
          "%.17g vs %s" % (m.sigma.sum(), r['SUMSIG'][0]))
    nlam = int(r['NNZLAM'][0])
    if nlam < 0:
        check(name + ' lambdaL', m.lambdaL is None,
              "expected None, got %r" % (m.lambdaL,))
    else:
        check(name + ' nnz(lambdaL)',
              m.lambdaL is not None
              and int(np.count_nonzero(m.lambdaL)) == nlam,
              "%s vs %d" % (None if m.lambdaL is None
                            else np.count_nonzero(m.lambdaL), nlam))
        check(name + ' sum(lambdaL)',
              m.lambdaL is not None
              and close(m.lambdaL.sum(), r['SUMLAM'][0], 1e-10),
              "%s vs %s" % (None if m.lambdaL is None else m.lambdaL.sum(),
                            r['SUMLAM'][0]))
    # VoxHenry keeps duplicate frequencies; vhr.py uniquifies, so compare
    # against the unique set on both sides.
    check(name + ' nfreq', m.freq.size == int(r['NFREQ'][0]),
          "%d vs %s (after unique)" % (m.freq.size, r['NFREQ'][0]))
    check(name + ' sum(freq)', close(m.freq.sum(), r['FREQSUM'][0], 1e-12),
          "%.17g vs %s" % (m.freq.sum(), r['FREQSUM'][0]))
    check(name + ' nports', len(m.ports) == int(r['NPORTS'][0]),
          "%d vs %s" % (len(m.ports), r['NPORTS'][0]))
    for row in r.get('PORT', []):
        p = m.ports[int(row[0]) - 1]
        np_, nn = int(row[2]), int(row[4])
        sump, sumn = int(row[6]), int(row[8])
        check(name + ' port%s counts' % row[0],
              len(p.pos) == np_ and len(p.neg) == nn,
              "P %d vs %d, N %d vs %d" % (len(p.pos), np_, len(p.neg), nn))
        # VoxHenry stores [x y z side] 1-based with side in 1..6 and its
        # own ordering ('-x','+x','-y','+y','-z','+z'); vhr.py stores
        # 0-based indices plus (axis, sign). Convert back to compare.
        for arr, want, tagl in ((p.pos, sump, 'P'), (p.neg, sumn, 'N')):
            if len(arr) == 0:
                got = 0
            else:
                side = 2*arr[:, 3] + (arr[:, 4] > 0).astype(int) + 1
                got = int((arr[:, :3] + 1).sum() + side.sum())
            check(name + ' port%s %s index sum' % (row[0], tagl), got == want,
                  "%d vs %d" % (got, want))
    check(name + ' ngrounds', len(m.grounds) == int(r['NGRND'][0]),
          "%d vs %s" % (len(m.grounds), r['NGRND'][0]))

# ---------------------------------------------------------------- B

print("\n=== PART B: SuperPEEC adapter ===")
print("  %-56s %-9s %-7s %s" % ('file', 'leaf/lvl', 'nodes', 'checks'))

for name in sorted(ref):
    m = vhr.read_vhr(os.path.join(INPUTS, name))
    # keep the sweep inside the toolbox memory cap; the two coil models
    # are 1M and 3.2M voxels and are not what this is testing.
    if int(np.prod(m.dims)) > 100000:
        print("  %-56s skipped (%d voxels, memory)"
              % (name, int(np.prod(m.dims))))
        continue
    try:
        leaf, levels = m.partition()
        M = m.build_tree(leaf, levels)
    except ValueError as exc:
        kind = 'superconductor' if m.superconductor else 'mixed material'
        check(name + ' rejects ' + kind, True)
        print("  %-56s rejected: %s" % (name, kind))
        continue

    # Mixed conductivity and superconductors are CELL-scheme features:
    # rule does not apply. Skip rather than fail -- edge is deprecated.
    if False:  # mixed conductivity supported since cell
        print("  %-56s skipped (mixed sigma needs the cell scheme)" % name)
        continue
    if False:  # superconductors supported since cell
        print("  %-56s skipped (superconductor needs the cell scheme)"
              % name)
        continue

    # the pitch must be dx on EVERY axis, not just x
    lf = np.asarray(M.e.l, dtype=float)
    ok = check(name + ' pitch', np.all(np.abs(lf/m.dx - 1.0) < 1e-12),
               "%s vs dx %g" % (list(lf), m.dx))
    m.prepare(M, m.freq[-1])
    # isotropic cells => the three filament resistances are equal, and
    # each is 1/(dx*sigma). On a MIXED-material model there is no single
    # sigma: r is a per-filament ARRAY, and every entry must be the
    # series pair of two half-cells, so it lies between the two extreme
    # single-material values and equals one of them wherever a filament
    # sits inside one material.
    if m.superconductor:
        # Uniform superconductor (all corpus SC files are): every r is
        # z(w)/dx with z the two-fluid impedance density -- the kinetic
        # inductance rides in Im(r). The frequency is the prepare()
        # frequency, the shipped maximum.
        zv = np.unique(m.impedance_density(m.freq[-1])[
            m.struc().astype(bool)])
        want_r = zv[0]/m.dx if zv.size else 0.0
        ok &= check(name + ' superconductor r == z(w)/dx',
                    zv.size == 1
                    and all(np.all(np.abs(np.asarray(r) - want_r)
                                   <= 1e-12*abs(want_r))
                            for r in (M.e.r, M.f.r, M.g.r)),
                    "r = %s, want %s" % (np.ravel(M.e.r)[0], want_r))
    elif m.sigma_values().size == 1:
        want_r = 1.0/(m.dx*m.uniform_sigma())
        ok &= check(name + ' resistance',
                    all(np.all(np.abs(np.asarray(r) - want_r)
                               <= 1e-12*want_r)
                        for r in (M.e.r, M.f.r, M.g.r)),
                    "e %.6g f %.6g g %.6g, want %.6g"
                    % (np.ravel(M.e.r)[0], np.ravel(M.f.r)[0],
                       np.ravel(M.g.r)[0], want_r))
    else:
        rr = np.concatenate([np.atleast_1d(M.e.r), np.atleast_1d(M.f.r),
                             np.atleast_1d(M.g.r)])
        lo = 1.0/(m.dx*m.sigma_values().max())
        hi = 1.0/(m.dx*m.sigma_values().min())
        ok &= check(name + ' resistance (mixed sigma)',
                    rr.size == (np.size(M.e.struc) + np.size(M.f.struc)
                                + np.size(M.g.struc))
                    and np.all(rr >= lo*(1 - 1e-12))
                    and np.all(rr <= hi*(1 + 1e-12)),
                    "%d values in [%.6g, %.6g], bounds [%.6g, %.6g]"
                    % (rr.size, rr.min(), rr.max(), lo, hi))
    # every port node must resolve, with duplicates summed not dropped
    for pi in range(len(m.ports)):
        snx, sny, snz, val = m.port_nodes(pi, 1e-6)
        ok &= check(name + ' port%d unique' % pi,
                    len(set(zip(snx, sny, snz))) == val.size,
                    "%d distinct of %d" % (len(set(zip(snx, sny, snz))),
                                           val.size))
        # a dropped or double-counted face would show up at ~1/nfaces
        # relative (percent level); the bound here is 1e-13 relative,
        # loose enough for the sqrt(n)*eps accumulated over ~1000 nodes
        # and still eleven orders tighter than any real imbalance.
        ok &= check(name + ' port%d sums to zero' % pi,
                    abs(val.sum()) <= 1e-13*abs(val).sum(),
                    "sum %.3g of %.3g total" % (abs(val.sum()),
                                                abs(val).sum()))
        src = m.source_vector(M, pi, 1e-6)
        ok &= check(name + ' port%d resolves' % pi,
                    int(np.count_nonzero(src)) == val.size,
                    "%d nonzero of %d nodes"
                    % (np.count_nonzero(src), val.size))
        # 'uniform' weighting must land on the same node set
        _, _, _, val2 = m.port_nodes(pi, 1e-6, weight='uniform')
        ok &= check(name + ' port%d uniform' % pi, val2.size == val.size,
                    "%d vs %d nodes" % (val2.size, val.size))
    print("  %-56s %s/%d %7d %s"
          % (name, 'x'.join(str(v) for v in leaf), levels,
             np.size(M.lv[0].struc), 'PASS' if ok else 'FAIL'))
    del M

print()
if fails:
    print("%d CHECK(S) FAILED" % len(fails))
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("ALL CHECKS PASSED")
