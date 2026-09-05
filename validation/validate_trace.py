# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""DIAGONAL TRACES: the ``[[trace]]`` primitive and the section cut
(docs/trace_plan.md). The gate of that program, in five parts.

A  an axis-aligned trace is BIT-IDENTICAL to the equivalent block:
   the fill record, the conductivity grid, R and L at one frequency.
B  the sampled fill of a rotated bar sums to its true area w*L to
   1e-3 at 45 and 30 degrees (the staircase's -11% at 4 cells across
   is what the cut exists to remove).
C  the 45-degree ladder: one copper bar (100 x 50 um x 1.6 mm) at
   4/8/16 cells across the width, axis-aligned (the rotation-invariant
   reference; DC closed form to 1e-5) against the same bar at 45
   degrees, driven at the staircased end cut on the prescribed-current
   path. The diagonal/aligned ratios are gated per phase (the plan's
   section 7): the baseline before any of this was built is

       cells across           4        8       16
       DC   R diag/aligned  1.420    1.132    1.029
       DC   L diag/aligned  1.061    1.021    1.005
       1e9  R (deep skin)   1.690    1.385    1.204
       1e9  L               1.060    1.017    1.003

D  the dogleg (x pad, 45-degree run, x pad) with ports on the pads,
   both port paths, R within the DC bound of the aligned equivalent.
E  the union rule: a trace ending inside a pad leaves every pad cell
   whole (fill == 1, no cut record on it).

Until phase 1 lands the primitive this validator SKIPS: a skip is
counted by the runner as NOT TESTED, which is the truth.
"""
import os as _op
import sys as _sp
import time

import numpy as np

_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sppeec_input                       # noqa: E402
import voxmodel                           # noqa: E402
import port_impedance as pz               # noqa: E402

FAIL = []


def check(name, ok, note=''):
    print('    %s %s%s' % ('ok  ' if ok else 'FAIL', name,
                           ('  ' + note) if note else ''), flush=True)
    if not ok:
        FAIL.append(name)


SIGMA = 5.8e7
W, T, LEN = 100e-6, 50e-6, 1.6e-3      # T = 2 cells at the coarsest pitch
MU0 = 4e-7*np.pi

# Gate thresholds on the diagonal/aligned ratio by program phase
# (docs/trace_plan.md section 7). PHASE is bumped as each phase closes;
# a threshold of None is not yet gated.
PHASE = 1
GATE = {                       # (nw, metric): {phase: max ratio}
    (8, 'R_dc'): {1: 1.01}, (16, 'R_dc'): {1: 1.005},
    (8, 'L_dc'): {2: 1.005}, (16, 'L_dc'): {2: 1.002},
    (8, 'R_1e9'): {3: 1.10}, (16, 'R_1e9'): {3: 1.05},
    (8, 'L_1e9'): {3: 1.005}, (16, 'L_1e9'): {3: 1.002},
}


def gate(nw, metric):
    g = GATE.get((nw, metric), {})
    lim = [v for ph, v in g.items() if ph <= PHASE]
    return min(lim) if lim else None


# --- geometry through the TOML path ----------------------------------

def aligned_doc(nw, freq, pad=0.0):
    """The bar along x, as a [[block]] with face ports on its ends."""
    h = W/nw
    nt = int(round(T/h))
    nl = int(round((LEN + 2*pad)/h))
    pf = ', '.join('[0, %d, %d, "-x"]' % (j, k)
                   for j in range(nw) for k in range(nt))
    nf = ', '.join('[%d, %d, %d, "+x"]' % (nl - 1, j, k)
                   for j in range(nw) for k in range(nt))
    return '\n'.join([
        '[grid]', 'dims = [%d, %d, %d]' % (nl, nw, nt), 'pitch = %r' % h,
        '[[block]]', 'from_m = [0.0, 0.0, 0.0]',
        'to_m = [%r, %r, %r]' % (nl*h, W, T), 'sigma = %r' % SIGMA,
        '[port]', 'p_faces = [%s]' % pf, 'n_faces = [%s]' % nf,
        '[solve]', 'freq = [%r]' % freq, 'enrich = "off"'])


def rotated_frame(nw, angle_deg):
    """Grid side, pitch, and the bar's start point for a bar of length
    LEN rotated by ``angle_deg`` about z, with a 2-cell margin."""
    h = W/nw
    a = np.deg2rad(angle_deg)
    c, s = float(np.cos(a)), float(np.sin(a))
    margin = 2*h
    ext = LEN*c + W*s + 2*margin
    exty = LEN*s + W*c + 2*margin
    n = int(np.ceil(max(ext, exty)/h))
    x0 = margin + W*s/2
    y0 = margin + W*c/2 if s >= 0 else margin
    return h, n, (x0, y0), (c, s)


def rotated_doc(nw, freq, angle_deg, ports=''):
    """The same bar as a [[trace]] at ``angle_deg``; ports optional
    (part C attaches the staircased end cut programmatically)."""
    h, n, (x0, y0), (c, s) = rotated_frame(nw, angle_deg)
    nt = int(round(T/h))
    x1, y1 = x0 + LEN*c, y0 + LEN*s
    return '\n'.join([
        '[grid]', 'dims = [%d, %d, %d]' % (n, n, nt), 'pitch = %r' % h,
        '[[trace]]', 'path_m = [[%r, %r], [%r, %r]]' % (x0, y0, x1, y1),
        'width_m = %r' % W, 'z_m = [0.0, %r]' % T, 'sigma = %r' % SIGMA,
        ports,
        '[solve]', 'freq = [%r]' % freq, 'enrich = "off"'])


def end_cut_port(m, nw, angle_deg):
    """Attach the staircased end-cut port: every boundary face of the
    metal whose outside neighbour lies beyond an end plane (u < 0 at P,
    u > LEN at N), x- and y-faces alike -- the mixed-orientation
    prescribed-current port."""
    h, n, (x0, y0), (c, s) = rotated_frame(nw, angle_deg)
    # the port's cells are the staircase's own (fill >= 1/2): the
    # prescribed-current port hands every face an EQUAL share, and a
    # sliver face at 1/fill terminal resistance would dominate the
    # answer (measured: R ratio 1.03 -> 1.09 at DC, 1.20 -> 2.13 deep
    # skin). Slivers beyond the port faces hang as dead-end stubs.
    fill = np.asarray(m.struc()) > 0 if m.fill is None else np.asarray(m.fill)
    inside = fill[:, :, 0] >= 0.5
    xc = (np.arange(n) + 0.5)*h
    nt = int(m.dims[2])
    p = voxmodel.Port('p1')
    for i, j in zip(*np.nonzero(inside)):
        for ax, dlt in ((0, -1), (0, +1), (1, -1), (1, +1)):
            ni, nj = (i + dlt, j) if ax == 0 else (i, j + dlt)
            if 0 <= ni < n and 0 <= nj < n and inside[ni, nj]:
                continue
            xn = xc[i] + (dlt*h if ax == 0 else 0.0)
            yn = xc[j] + (dlt*h if ax == 1 else 0.0)
            un = (xn - x0)*c + (yn - y0)*s
            for k in range(nt):
                if un < 0:
                    p._add('P', (int(i), int(j), k, ax, dlt))
                elif un > LEN:
                    p._add('N', (int(i), int(j), k, ax, dlt))
    p._freeze()
    m.ports = [p]


def solve_lpr(pr, m, freq):
    M = pr.tree(m)
    m.prepare(M, freq)
    sol = pz.LpRSolver(M)
    Z, infos = pz.impedance_matrix(m, M, sol, freq)
    z = complex(Z[0, 0])
    return z.real, z.imag/(2*np.pi*freq), infos[0]


# --- parts -------------------------------------------------------------

def part_a():
    print("A: axis-aligned trace == block, bit for bit")
    nw, freq = 4, 1e8
    h = W/nw
    nt, nl = int(round(T/h)), int(round(LEN/h))
    pb = sppeec_input.loads(aligned_doc(nw, freq))
    mb = pb.model()
    doc_t = aligned_doc(nw, freq).replace(
        '[[block]]\nfrom_m = [0.0, 0.0, 0.0]\nto_m = [%r, %r, %r]'
        % (nl*h, W, T),
        '[[trace]]\npath_m = [[0.0, %r], [%r, %r]]\nwidth_m = %r\n'
        'z_m = [0.0, %r]' % (W/2, nl*h, W/2, W, T))
    pt = sppeec_input.loads(doc_t)
    mt = pt.model()
    check('sigma grid identical', np.array_equal(mb.sigma, mt.sigma))
    check('no cut record on an aligned trace',
          mt.cut is None or not mt.cut.get('cells'),
          'cut=%r' % (None if mt.cut is None else mt.cut['kind']))
    check('fill whole everywhere',
          mt.fill is None or bool(np.all(mt.fill[mt.sigma != 0] == 1.0)))
    rb, lb, _ = solve_lpr(pb, mb, freq)
    rt, lt, _ = solve_lpr(pt, mt, freq)
    check('R bit-identical', rb == rt, '%.6e vs %.6e' % (rb, rt))
    check('L bit-identical', lb == lt, '%.6e vs %.6e' % (lb, lt))


def part_b():
    print("B: sampled fill sums to the true area")
    for ang in (45.0, 30.0):
        for nw in (4, 8):
            pr = sppeec_input.loads(rotated_doc(nw, 1e6, ang))
            m = pr.model()
            h = W/nw
            area = float(np.asarray(m.fill)[:, :, 0].sum())*h*h
            stair = float((np.asarray(m.struc()) > 0)[:, :, 0].sum())*h*h
            check('%g deg, %d across: fill area' % (ang, nw),
                  abs(area/(W*LEN) - 1) < 1e-3,
                  'fill %.4f staircase %.4f of w*L'
                  % (area/(W*LEN), stair/(W*LEN)))


def part_c():
    print("C: the 45-degree ladder (diagonal/aligned ratios)")
    print("    %4s %8s | %8s %8s | %s"
          % ('nw', 'f', 'R_d/R_a', 'L_d/L_a', 'gate'))
    r_dc = LEN/(SIGMA*W*T)
    for nw in (4, 8, 16):
        for f, tag in ((1e3, 'dc'), (1e9, '1e9')):
            pa = sppeec_input.loads(aligned_doc(nw, f))
            ma = pa.model()
            ra, la, _ = solve_lpr(pa, ma, f)
            pd = sppeec_input.loads(rotated_doc(nw, f, 45.0))
            md = pd.model()
            end_cut_port(md, nw, 45.0)
            t0 = time.perf_counter()
            rd, ld, info = solve_lpr(pd, md, f)
            dt = time.perf_counter() - t0
            if tag == 'dc':
                check('nw=%d aligned DC R exact' % nw,
                      abs(ra/r_dc - 1) < 1e-5, '%.6f' % (ra/r_dc))
            for metric, ratio in (('R_' + tag, rd/ra), ('L_' + tag, ld/la)):
                lim = gate(nw, metric)
                note = ('<= %.3f' % lim) if lim is not None else 'ungated'
                print("    %4d %8.0e | %8.4f %-8s | %s %s  (%d mv, %.0f s)"
                      % (nw, f, ratio, metric, note,
                         '' if lim is None else ('ok' if ratio <= lim
                                                 else 'FAIL'),
                         info['matvecs'], dt), flush=True)
                if lim is not None and ratio > lim:
                    FAIL.append('ladder nw=%d %s %.4f > %.3f'
                                % (nw, metric, ratio, lim))


def dogleg_doc(nw, freq, equipotential, pad_cells=6):
    """x pad, 45-degree run, x pad; ports on the pads' outer faces."""
    h = W/nw
    nt = int(round(T/h))
    # in-plane projection of the run, snapped to whole cells so the
    # far pad stays on-grid (the segment is exactly 45 degrees)
    run = round(LEN/float(np.sqrt(2.0))/h)*h
    pad = pad_cells*h
    margin = 2*h
    x0, y0 = margin, margin + W/2
    x1 = x0 + pad
    x2, y2 = x1 + run, y0 + run
    x3 = x2 + pad
    n1 = int(np.ceil((x3 + margin)/h))
    n2 = int(np.ceil((y2 + W/2 + margin)/h))
    j0, j1 = int(round((y0 - W/2)/h)), int(round((y0 + W/2)/h))
    j2, j3 = int(round((y2 - W/2)/h)), int(round((y2 + W/2)/h))
    i0, i3 = int(round(x0/h)), int(round(x3/h)) - 1
    pf = ', '.join('[%d, %d, %d, "-x"]' % (i0, j, k)
                   for j in range(j0, j1) for k in range(nt))
    nf = ', '.join('[%d, %d, %d, "+x"]' % (i3, j, k)
                   for j in range(j2, j3) for k in range(nt))
    return '\n'.join([
        '[grid]', 'dims = [%d, %d, %d]' % (n1, n2, nt), 'pitch = %r' % h,
        '[[block]]', 'from_m = [%r, %r, 0.0]' % (x0, y0 - W/2),
        'to_m = [%r, %r, %r]' % (x1, y0 + W/2, T), 'sigma = %r' % SIGMA,
        '[[trace]]', 'path_m = [[%r, %r], [%r, %r]]' % (x1, y0, x2, y2),
        'width_m = %r' % W, 'z_m = [0.0, %r]' % T, 'sigma = %r' % SIGMA,
        '[[block]]', 'from_m = [%r, %r, 0.0]' % (x2, y2 - W/2),
        'to_m = [%r, %r, %r]' % (x3, y2 + W/2, T), 'sigma = %r' % SIGMA,
        '[port]', 'p_faces = [%s]' % pf, 'n_faces = [%s]' % nf,
        'equipotential = %s' % ('true' if equipotential else 'false'),
        '[solve]', 'freq = [%r]' % freq, 'enrich = "off"'])


def part_d():
    print("D: the dogleg with pads, both port paths")
    nw, f = 8, 1e3
    h = W/nw
    run = round(LEN/float(np.sqrt(2.0))/h)*h*float(np.sqrt(2.0))
    r_ref = (run + 2*6*h)/(SIGMA*W*T)        # pads + run, aligned bound
    for eq in (False, True):
        pr = sppeec_input.loads(dogleg_doc(nw, f, eq))
        m = pr.model()
        M = pr.tree(m)
        sw = pr.sweeper(m, M)
        Z = sw.solve(f)
        r = float(np.real(np.atleast_2d(Z)[0, 0]))
        check('%s path: DC R within 6%% of the aligned bound'
              % ('equipotential' if eq else 'prescribed'),
              abs(r/r_ref - 1) < 0.06, 'R/R_ref = %.4f' % (r/r_ref))


def part_e():
    print("E: a trace ending inside a pad leaves the pad whole")
    nw = 8
    pr = sppeec_input.loads(dogleg_doc(nw, 1e6, False))
    m = pr.model()
    h = W/nw
    margin = 2*h
    i0, i1 = int(round(margin/h)), int(round((margin + 6*h)/h))
    j0 = int(round(margin/h))
    j1 = int(round((margin + W)/h))
    pad = np.asarray(m.fill)[i0:i1, j0:j1, :]
    check('pad cells all whole', bool(np.all(pad == 1.0)),
          'min fill %.3f' % float(pad.min()))
    cut_cells = set(m.cut['cells']) if m.cut else set()
    inpad = [c for c in cut_cells if i0 <= c[0] < i1 and j0 <= c[1] < j1]
    check('no cut record inside the pad', not inpad, '%d' % len(inpad))


def main():
    try:
        sppeec_input.loads(rotated_doc(4, 1e6, 45.0))
    except Exception as e:                      # the primitive is not there
        msg = str(e).splitlines()[0][:60]
        print("SKIP: [[trace]] not parsed yet (%s) -- docs/trace_plan.md "
              "phase 1" % msg)
        return 0
    for part in (part_a, part_b, part_c, part_d, part_e):
        part()
    print("%d checks failed" % len(FAIL))
    for f in FAIL:
        print("  FAIL", f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    _sp.exit(main())
