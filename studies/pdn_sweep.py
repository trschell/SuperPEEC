# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Frequency-sweep driver: Z11(f) of a FILLED pdn_planes board.

The dielectric program's phase C. One board, one tree, one solver
setup; the frequency loop re-uses everything that does not depend on w
and pays only the per-point cost:

  amortized once   board + capacitive tree, W rescale (band; pure
                   electrostatics), diagschur assembly (A, diag(Lp),
                   C_cap band -- geometry only)
  per frequency    filament impedances z(w) (dielectric branches are
                   frequency dependent), S_d = -jw C_cap - A^T D^-1 A
                   and its sparse LU, the fgmres solve

so a sweep costs roughly (setup once) + N x (refactor + solve) rather
than N cold runs -- measured on the 160^2 board: 75 s/point against a
229 s cold end-to-end run.

WARM STARTS (WARM=1, default) carry the previous point's solution
forward with the node block scaled by w_prev/w -- see the long note in
LpPRSolver.solve. The naive version of this lever is a MEASURED NULL
(a globally-rescaled warm start costs +1 matvec per point and saves
none, at every frequency of the 1e5-1e9 sweep); the block-scaled form
works because the two blocks obey different frequency laws. It pays
only on FINE sweeps: initial residual 0.005 at a 2% step, 0.11 at
1.5x, 0.79 at 10x, so a 3-points-per-decade sweep gains little and a
resonance-resolving sweep gains a lot. fgmres's stopping test is
||r|| < tol*||b||, relative to the RHS rather than to the initial
residual, so a warm start can never buy a looser answer.

WHAT THE CURVE SHOWS. Below the plane-pair series resonance the port
is a capacitor: |Z| falls at -20 dB/decade with C the static value
(84.07 pF on the 320^2 board), Re(Z) is the plane/via spreading
resistance plus dielectric loss if a loss tangent is set. Approaching
resonance Im(Z) turns inductive; the crossing 1/(2 pi sqrt(L C)) is
the number a PDN designer reads off first. CAVEAT at the top of the
band: these boards use 50 um cells, so above ~1 GHz the cell scheme's
uniform-current assumption breaks (copper skin depth 2 um at 1 GHz)
and R is underestimated -- the sweep still converges, but treat the
high decade as qualitative. The diagschur preconditioner also loses
its regime above the wL/R crossover, visible as growing matvec counts.

Usage (the three rungs):
    NPLANE=48  ... python3 studies/pdn_sweep.py      # smoke, minutes
    NPLANE=160 ... python3 studies/pdn_sweep.py      # the curve
    NPLANE=320 ... python3 studies/pdn_sweep.py      # flagship
with PYTHONPATH=src in front.

Env:
  NPLANE   cells per plane side (default 160)
  FMIN/FMAX/NPTS   sweep band (default 1e5 .. 1e9 Hz, 13 points)
  FREQS    explicit comma-separated list, overrides FMIN/FMAX/NPTS
  EPS      dielectric constant (default 4.2 = FR4)
  DF       loss tangent (default 0; 0.02 is FR4's -- eps_r becomes
           complex and rides the SAME Ruehli material law, so the
           dielectric's conduction loss enters Re(Z11) for free)
  WARM     1 (default) warm-start each point from the previous
  CCAP     'band' (default) or 'diag' C_cap block of the Schur
  RESTRT/MAXITER   fgmres restart length / cycles (default 100/3)
  TOL      fgmres tolerance (default 1e-10)
  TAG      output basename (default pdn_sweep_<nplane>)

Writes results/<TAG>.csv (flushed per point, so a killed run keeps
its points) and results/<TAG>.png (|Z|, phase, matvec count).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from port_impedance import LpPRSolver
from pdn_planes import build_pdn, stats


def _freqs():
    explicit = os.environ.get('FREQS', '')
    if explicit:
        return np.array([float(v) for v in explicit.split(',')])
    fmin = float(os.environ.get('FMIN', '1e5'))
    fmax = float(os.environ.get('FMAX', '1e9'))
    npts = int(os.environ.get('NPTS', '13'))
    return np.logspace(np.log10(fmin), np.log10(fmax), npts)


def build(nplane, eps_r, df):
    """Board + capacitive tree, lean where the partition allows it."""
    eps = eps_r*(1.0 - 1j*df) if df else eps_r
    m = build_pdn(nplane=nplane, eps_r=eps)
    nc, nd = stats(m)
    print("pdn_sweep: %s  %d copper + %d FR4 = %d cells  (eps_r %s)"
          % (m.dims, nc, nd, nc + nd, eps), flush=True)
    leaf, levels = m.partition()
    t0 = time.perf_counter()
    if levels > 1:
        # LEAN capacitive tree: no stored near-field n2n (fftnear
        # supplies the near field, kernel-direct blocks supply the
        # band W). 320^2: 33.1 GB -> 0.2 GB.
        M = m.build_tree(leaf, levels, capacitive=True, fftnear=True,
                         keep_n2n=False)
    else:
        M = m.build_tree(leaf, levels, capacitive=True)
    print("  tree: nleaf %s, %d level(s), %.1f s%s"
          % (list(int(v) for v in leaf), levels,
             time.perf_counter() - t0,
             ", lean" if levels > 1 else ""), flush=True)
    return m, M


def replot(tag):
    """Redraw (and refit) from an existing results/<tag>.csv."""
    path = os.path.join('results', tag + '.csv')
    with open(path) as fh:
        head = fh.readline()
    d = np.genfromtxt(path, delimiter=',', names=True, skip_header=1)
    meta = dict(kv.split('=') for kv in head.split()[1:] if '=' in kv)
    rows = [(float(f), complex(re, im), 0.0, 0.0, dict(matvecs=int(mv)),
             0.0)
            for f, re, im, mv in zip(d['freq_Hz'], d['re_Z'], d['im_Z'],
                                     d['matvecs'])]
    plot(rows, tag, int(meta.get('nplane', 0)),
         float(meta.get('eps_r', 1.0)), float(meta.get('df', 0.0)))


def main():
    if os.environ.get('REPLOT'):
        replot(os.environ['REPLOT'])
        return
    nplane = int(os.environ.get('NPLANE', '160'))
    eps_r = float(os.environ.get('EPS', '4.2'))
    df = float(os.environ.get('DF', '0'))
    warm = os.environ.get('WARM', '1') == '1'
    ccap = os.environ.get('CCAP', 'band')
    restrt = int(os.environ.get('RESTRT', '100'))
    maxiter = int(os.environ.get('MAXITER', '3'))
    tol = float(os.environ.get('TOL', '1e-10'))
    tag = os.environ.get('TAG', 'pdn_sweep_%d' % nplane)
    freqs = _freqs()

    t_all = time.perf_counter()
    m, M = build(nplane, eps_r, df)
    m.prepare(M, float(freqs[0]))
    t0 = time.perf_counter()
    S = LpPRSolver(m, M, ccap=ccap)
    print("  solver setup %.1f s (wsolve %s, ccap %s)"
          % (time.perf_counter() - t0, S.wsolve, ccap), flush=True)

    os.makedirs('results', exist_ok=True)
    csv = os.path.join('results', tag + '.csv')
    fh = open(csv, 'w')
    fh.write("# pdn_sweep nplane=%d eps_r=%g df=%g warm=%d ccap=%s\n"
             % (nplane, eps_r, df, int(warm), ccap))
    fh.write("freq_Hz,re_Z,im_Z,abs_Z,deg_Z,C_eff_F,L_eff_H,matvecs,"
             "resid,true_resid,seconds\n")
    fh.flush()

    rows = []
    x_prev = f_prev = None
    print("      f [Hz]        Re Z         Im Z      |Z|      C_eff"
          "     matvecs   true   sec", flush=True)
    for f in freqs:
        t0 = time.perf_counter()
        z, x, info = S.solve(float(f), tol=tol, restrt=restrt,
                             maxiter=maxiter,
                             x0=x_prev if warm else None,
                             x0_freq=f_prev)
        dt = time.perf_counter() - t0
        if warm:
            x_prev, f_prev = x, float(f)
        w = 2*np.pi*float(f)
        # C_eff is meaningful while the port is capacitive (Im Z < 0),
        # L_eff once it turns inductive; report both, let the reader
        # pick the branch (the sign flip IS the series resonance).
        ceff = float(np.imag(1.0/z)/w)
        leff = float(np.imag(z)/w)
        rows.append((float(f), z, ceff, leff, info, dt))
        fh.write("%.6g,%.8g,%.8g,%.8g,%.4f,%.8g,%.8g,%d,%.3e,%.3e,"
                 "%.1f\n"
                 % (f, z.real, z.imag, abs(z),
                    np.degrees(np.angle(z)), ceff, leff,
                    info['matvecs'], info['residual'],
                    info['true_residual'], dt))
        fh.flush()
        w0 = info.get('warm_residual')
        print("  %11.4g  %11.4g  %11.4g  %8.4g  %9.4g  %6d  %7.1e"
              "  %5.0f%s"
              % (f, z.real, z.imag, abs(z), ceff, info['matvecs'],
                 info['true_residual'], dt,
                 '' if w0 is None else '   warm r0 %.3f' % w0),
              flush=True)
    fh.close()
    print("  wrote %s   (total %.0f s)"
          % (csv, time.perf_counter() - t_all), flush=True)
    plot(rows, tag, nplane, eps_r, df)


def fit_rlc(f, z):
    """Least-squares (L, C) of the lumped port model Im Z = wL - 1/(wC).

    TWO equilibrations are mandatory and the fit is garbage without
    either (both measured on the 160^2 sweep: raw lstsq returns a
    NEGATIVE inductance):
      * ROWS -- Im Z spans 7.5e4 ohm at 100 kHz down to 7 ohm at 1 GHz,
        so an unweighted fit is 100% low-frequency and the wL term is
        invisible. Weight by 1/|Im Z| (fit RELATIVE error).
      * COLUMNS -- the two columns are w ~ 1e9 and 1/w ~ 1e-9, 19
        orders apart: cond(A) 4e15, and the L column falls under
        lstsq's rank cutoff. Normalise both columns (cond -> 1.5).
    The same scaling trap as the part-D MNA oracle and the BEM system.
    """
    w = 2*np.pi*np.asarray(f, dtype=float)
    b = np.asarray(z).imag
    A = np.column_stack([w, -1.0/w])
    rw = 1.0/np.abs(b)
    Aw, bw = A*rw[:, None], b*rw
    cs = np.linalg.norm(Aw, axis=0)
    sol, *_ = np.linalg.lstsq(Aw/cs, bw, rcond=None)
    L, invc = sol/cs
    C = 1.0/invc
    dev = float(np.max(np.abs((A@(sol/cs) - b)/b)))
    fres = 1.0/(2*np.pi*np.sqrt(L*C)) if L > 0 and C > 0 else float('nan')
    return L, C, fres, dev


def plot(rows, tag, nplane, eps_r, df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    f = np.array([r[0] for r in rows])
    z = np.array([r[1] for r in rows])
    mv = np.array([r[4]['matvecs'] for r in rows])
    # low-frequency capacitance: the lowest point, where the port is
    # purely capacitive
    c0 = float(np.imag(1.0/z[0])/(2*np.pi*f[0]))
    L, C, fres, dev = fit_rlc(f, z)
    print("  lumped port model: C %.5g pF, ESL %.4g pH, series "
          "resonance %.4g GHz (max rel dev %.1e over the band)"
          % (1e12*C, 1e12*L, 1e-9*fres, dev), flush=True)

    # ONE MEASURE PER PANEL, each on its own scale: |Z| spans 5 decades
    # while Re Z sits flat at ~5e-4 ohm, so sharing an axis buries the
    # loss curve in 8 decades of empty space (and a twin y-axis is
    # never the answer). Colours are the validated categorical slots
    # 1/2/3/7 in fixed order; text stays in ink, not series colour.
    C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
    INK, MUTED = '#0b0b0b', '#52514e'
    mk = dict(marker='o', ms=5.5, lw=1.8)
    fig, ax = plt.subplots(4, 1, figsize=(7.0, 9.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 2, 2, 1.6]))
    ax[0].loglog(f, 1.0/(2*np.pi*f*c0), ls=':', lw=1.4, color=MUTED,
                 label='ideal 1/($\\omega$C), C = %.3g pF' % (1e12*c0))
    ax[0].loglog(f, np.abs(z), color=C1, label='|Z11|', **mk)
    ax[0].set_ylabel('|Z11|   [$\\Omega$]', color=MUTED)
    ax[0].set_title('pdn_planes %d$^2$, $\\epsilon_r$ %.4g%s '
                    '— port impedance Z11(f)\n'
                    'lumped fit:  C %.4g pF,  ESL %.3g pH,  '
                    '$f_{series}$ %.3g GHz'
                    % (nplane, eps_r,
                       ', tan$\\delta$ %.3g' % df if df else '',
                       1e12*C, 1e12*L, 1e-9*fres), color=INK)
    ax[0].legend(fontsize=8, frameon=False, labelcolor=MUTED)
    ax[1].loglog(f, z.real, color=C2, **mk)
    ax[1].set_ylabel('Re Z11  (ESR)   [$\\Omega$]', color=MUTED)
    # LOSS ANGLE: a flat -90 deg phase line carries no information, so
    # plot its departure instead. Note what this panel does and does
    # NOT show -- series L changes |Z| (panel 1 bending below the ideal
    # dotted line), not the phase; the departure from -90 deg is
    # Re Z/|Im Z| alone, i.e. the total loss angle. With tan d = 0 it
    # is pure copper loss (~0.005 deg here); an FR4 tan d of 0.02 would
    # put it near 1.1 deg.
    ax[2].semilogx(f, np.degrees(np.angle(z)) + 90.0, color=C3, **mk)
    ax[2].axhline(0.0, color=MUTED, lw=0.8, ls=':')
    ax[2].set_ylabel('loss angle\nphase(Z11) + 90$^\\circ$   [deg]',
                     color=MUTED)
    ax[3].semilogx(f, mv, color=C4, **mk)
    ax[3].set_ylabel('matvecs', color=MUTED)
    ax[3].set_xlabel('frequency   [Hz]', color=MUTED)
    for a in ax:
        a.grid(True, which='major', alpha=0.25, lw=0.6)
        a.grid(True, which='minor', alpha=0.12, lw=0.5)
        a.tick_params(colors=MUTED, labelsize=9)
        for s in a.spines.values():
            s.set_color('#d5d4cf')
    fig.tight_layout()
    png = os.path.join('results', tag + '.png')
    fig.savefig(png, dpi=130)
    print("  wrote %s" % png, flush=True)


if __name__ == '__main__':
    main()
