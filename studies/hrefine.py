# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""h-REFINEMENT: does the DISCRETE answer converge to the physical one?

THE GAP THIS FILLS. Every other check in the suite holds the mesh
FIXED and tests the solver against it:
  * the dense oracle (squid_washer.oracle_z) agrees to 11 digits
    precisely BECAUSE it assembles the same discretisation;
  * Foster's reactance theorem is satisfied by ANY lossless LC network,
    including a badly discretised one;
  * losslessness and the frequency-independence of L are structural
    invariants of the same discrete system.
None of them can see discretisation error. Only refinement can, and
without it every number is "the answer at the resolution I happened to
run" rather than a physical prediction with an error bar.

Precedent: the dielectric accuracy law was established exactly this way
(4-point BEM Richardson at fixed W/d), and it turned an apparent "fill
overshoot bug" into a documented +-5% finite-h law that is NON-MONOTONE
and sign-configuration dependent. Expect the same character here.

WHAT IT MEASURES
  * L(h) on a fixed PHYSICAL washer as the cell size halves, the
    observed order p from successive differences, and the Richardson
    extrapolate L(h->0) -- the actual physical prediction.
  * THE PORT PREDICTION. The terminal correction is a HALF-CELL
    filament, so it must vanish as h -> 0, and with it the LpR/LpPR
    port gap. If those do NOT shrink, the port model is wrong in a way
    no amount of solver validation could reveal.

A SUPERCONDUCTOR is the right vehicle: a normal metal at 10 GHz has a
skin depth (0.66 um in Cu) comparable to the coarse cells, so refining
would improve the skin resolution and confound geometric convergence
with a physical one. The two-fluid law is per-cell exact and carries no
such scale.

ROUTE NOTE -- and it is a trap this study fell into. Small rungs get a
single-level tree (exact W), large ones the lean multilevel tree (band
W), so an unforced sequence can switch FORMULATION mid-study. MEASURED
at fixed h: multilevel/band-W gives -0.051 pH at dxy 1 um and -0.045 pH
at dxy 0.5 um, i.e. a consistent -0.5% -- comparable to the refinement
steps themselves. A sequence that switches route is measuring two
things at once, and the code now WARNS when it happens. Set ROUTE=
single|multi for a clean sequence.

Usage:
    PYTHONPATH=.:studies python3 studies/hrefine.py
Env: OUTER/HOLE/SLIT (um, default 20/6/1), LAM (nm, 90), FREQ (1e10),
     RUNGS (default 3), LPR=1 also run the LpR equipotential path.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import stencils as st
from port_impedance import LpPRSolver, terminal_impedance
from squid_washer import build_washer


def tree_for(m, force=None):
    """(tree, route). Single-level exact-W while it is affordable, the
    lean multilevel band-W tree once it is not."""
    leaf, levels = m.partition()
    if force == 'multi' and levels >= 2:
        return m.build_tree(leaf, levels, capacitive=True, fftnear=True,
                            keep_n2n=False), 'lean-multi/band-W'
    if force == 'single' or (force is None and np.prod(m.dims) <= 8000):
        return m.build_tree(st.single_level_nleaf(m.dims), 1,
                            capacitive=True), 'single/exact-W'
    if levels < 2:
        return m.build_tree(st.single_level_nleaf(m.dims), 1,
                            capacitive=True), 'single/exact-W'
    return m.build_tree(leaf, levels, capacitive=True, fftnear=True,
                        keep_n2n=False), 'lean-multi/band-W'


def run_rung(dxy_um, dz_nm, film_cells, geo, freq, force=None,
             want_lpr=False):
    m = build_washer(hole_um=geo['hole'], outer_um=geo['outer'],
                     slit_um=geo['slit'], dxy_um=dxy_um, dz_nm=dz_nm,
                     film_cells=film_cells, ground=False,
                     lam_nm=geo['lam'], eps_bg=geo['eps_bg'],
                     port_um=geo.get('port'))
    t0 = time.perf_counter()
    M, route = tree_for(m, force)
    m.prepare(M, freq)
    # ccap='band' on the multilevel route. The DEFAULT 'diag' needs a
    # dense P_ext^-1 eye probe and dies by SIGKILL, with no traceback,
    # once the external-node count gets large (measured: fine at 51k
    # cells, killed at 410k). Unlike the W route this is only the
    # PRECONDITIONER, so the converged answer is unchanged and mixing
    # it across rungs does not contaminate the convergence study.
    S = LpPRSolver(m, M, **({'ccap': 'band'}
                            if route.startswith('lean') else {}))
    z, _, info = S.solve(freq, tol=1e-11, restrt=100, maxiter=3)
    zt, _, _ = S.solve(freq, tol=1e-11, restrt=100,
                       maxiter=3, terminals=True)
    w = 2*np.pi*freq
    out = dict(dxy=dxy_um, dims=tuple(int(v) for v in m.dims), route=route,
               L=1e12*z.imag/w, L_term=1e12*zt.imag/w,
               Zterm=1e12*np.imag(terminal_impedance(m, M, 0, freq))/w,
               mv=info['matvecs'], sec=time.perf_counter() - t0,
               nport=len(m.port(0).pos))
    if want_lpr:
        import equiterminal as eq
        m2 = build_washer(hole_um=geo['hole'], outer_um=geo['outer'],
                          slit_um=geo['slit'], dxy_um=dxy_um, dz_nm=dz_nm,
                          film_cells=film_cells, ground=False,
                          lam_nm=geo['lam'], eps_bg=geo['eps_bg'],
                          port_um=geo.get('port'))
        lf, lv = m2.partition()
        M2 = m2.build_tree(lf, lv)
        m2.prepare(M2, freq)
        z1, _, _ = eq.EquiTerminalSolver(m2, M2, 0).solve(freq)
        out['L_lpr'] = 1e12*z1.imag/w
    return out


def richardson(vals):
    """(order, extrapolate) from three successive halvings.

    REFUSES to extrapolate unless the successive differences actually
    DECAY. Two ways it can fail, both hit in the first run of this
    study:
      * d1/d2 < 0 -- the sequence is non-monotone, so there is no
        single power law to extrapolate along;
      * p <= 0 -- the differences GROW, i.e. the sequence is not (yet)
        converging. My first version guarded only the first case and
        cheerfully printed 'Richardson h->0 = 10.12090 pH' off an
        observed order of -0.707, which is a fabricated number: a
        growing increment extrapolates to nothing. Being pre-asymptotic
        is a legitimate outcome of a refinement study and must be
        REPORTED, not smoothed over.
    """
    a, b, c = vals[-3], vals[-2], vals[-1]
    d1, d2 = a - b, b - c
    if d2 == 0 or d1/d2 <= 0:
        return float('nan'), float('nan')
    p = float(np.log2(d1/d2))
    if p <= 0:
        return p, float('nan')
    return p, float(c + d2/(2**p - 1))


def main():
    geo = dict(outer=float(os.environ.get('OUTER', '20')),
               hole=float(os.environ.get('HOLE', '6')),
               slit=float(os.environ.get('SLIT', '1')),
               lam=float(os.environ.get('LAM', '90')),
               eps_bg=3.9,
               port=(float(os.environ['PORT'])
                     if os.environ.get('PORT') else None))
    freq = float(os.environ.get('FREQ', '1e10'))
    nr = int(os.environ.get('RUNGS', '3'))
    want_lpr = os.environ.get('LPR', '1') == '1'
    print("h-refinement: FIXED washer outer %g / hole %g / slit %g um, "
          "Nb lambda %g nm, %.3g Hz"
          % (geo['outer'], geo['hole'], geo['slit'], geo['lam'], freq),
          flush=True)
    print("%-7s %-14s %-18s %9s %9s %9s %9s %6s %6s"
          % ("dxy um", "dims", "route", "L(LpPR)", "L(+term)", "L(LpR)",
             "Z_term", "mv", "s"), flush=True)
    rows = []
    for k in range(nr):
        dxy = 1.0/2**k
        dz = 100.0/2**k
        fc = 2*2**k
        try:
            r = run_rung(dxy, dz, fc, geo, freq,
                         force=os.environ.get('ROUTE'),
                         want_lpr=want_lpr)
        except Exception as e:
            print("  rung dxy=%g FAILED: %s: %s"
                  % (dxy, type(e).__name__, str(e)[:120]), flush=True)
            continue
        rows.append(r)
        print("%-7.4g %-14s %-18s %9.5f %9.5f %9s %9.5f %6d %6.0f"
              % (r['dxy'], str(r['dims']), r['route'], r['L'],
                 r['L_term'],
                 ("%.5f" % r['L_lpr']) if 'L_lpr' in r else "-",
                 r['Zterm'], r['mv'], r['sec']), flush=True)
    if len(rows) >= 3:
        routes = {r['route'] for r in rows}
        if len(routes) > 1:
            # MEASURED: single/exact-W vs lean-multi/band-W is worth
            # -0.5% of L at fixed h (-0.051 pH at dxy 1, -0.045 at 0.5,
            # consistently negative). That is comparable to the
            # refinement steps themselves, so a sequence that switches
            # route mid-study is measuring two things at once. Force
            # one route with ROUTE= to get a clean sequence.
            print("  WARNING routes differ across rungs (%s): the "
                  "single->multi switch is worth ~0.5%% of L at fixed "
                  "h, comparable to the refinement itself. Re-run with "
                  "ROUTE=single or ROUTE=multi for a clean sequence."
                  % ", ".join(sorted(routes)), flush=True)
        keys = [('L', 'L(LpPR prescribed)'), ('L_term', 'L(LpPR + term)')]
        if 'L_lpr' in rows[0]:
            keys.append(('L_lpr', 'L(LpR equipotential)'))
        for key, label in keys:
            p, ext = richardson([r[key] for r in rows])
            if not np.isfinite(ext):
                print("  %-22s NO VALID EXTRAPOLATION (observed order "
                      "p = %s): the differences do not decay, so this "
                      "sequence is PRE-ASYMPTOTIC at these resolutions "
                      "-- refine further before quoting an h->0 value."
                      % (label, ("%.3f" % p) if np.isfinite(p)
                         else "non-monotone"), flush=True)
            else:
                print("  %-22s observed order p = %.3f, Richardson "
                      "h->0 = %.5f pH  (finest rung is %+.2f%% off it)"
                      % (label, p, ext, 100*(rows[-1][key]/ext - 1)),
                      flush=True)
    if rows and 'L_lpr' in rows[0]:
        print("  PORT PREDICTION (these must shrink as h -> 0):",
              flush=True)
        for r in rows:
            print("    dxy %-6.4g gap(LpR-LpPR) %8.5f pH   Z_term "
                  "%8.5f pH   port faces/side %d"
                  % (r['dxy'], r['L_lpr'] - r['L'], r['Zterm'],
                     r['nport']), flush=True)
    # Jaycox-Ketchen, with the discretisation error now separable
    jk = 1.25*4e-7*np.pi*geo['hole']*1e-6
    if len(rows) >= 3:
        best = 'L_lpr' if 'L_lpr' in rows[0] else 'L'
        p, ext = richardson([r[best] for r in rows])
        print("  Jaycox-Ketchen 1.25 mu0 d = %.5f pH: coarsest rung "
              "ratio %.3f" % (1e12*jk, rows[0][best]/(1e12*jk)),
              flush=True)
        if np.isfinite(ext):
            print("    extrapolated (%s) ratio %.3f -- the difference "
                  "between those two IS the discretisation error the "
                  "raw comparison hides, and note it can move the "
                  "ratio AWAY from 1: agreement at a coarse mesh may "
                  "be two errors cancelling."
                  % (best, ext/(1e12*jk)), flush=True)


if __name__ == '__main__':
    main()
