# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""dc-SQUID washer in Nb, solved through the LpPR (capacitive) path.

WHY LpPR AND NOT LpR. SQUID and RSFQ extraction is conventionally an
INDUCTANCE problem -- InductEx, 3D-MLSI, VoxHenry all return L and stop
there, and SuperPEEC's own LpR/equiterminal path does the same. But the
quantity that limits a real washer is not L alone: a washer over a
groundplane is an LC resonator, and its self-resonance (Enpuku's washer
resonance) sits on top of the SQUID's transfer function. The capacitive
path returns L AND C from one solve, so the resonance falls out of a
frequency sweep -- something the inductive formulation structurally
cannot show. Same for RSFQ passive transmission lines, where the design
quantity is Z0 = sqrt(L'/C'), not L' alone.

WHAT MAKES THIS RUN (the composition matters, 2026-08-08):
  * Superconductor + PER-CELL dielectric is explicitly refused by
    ``impedance_density``. It is not needed: Nb wiring is embedded in a
    UNIFORM SiO2 stack, so ``eps_bg`` -- a background permittivity that
    divides the coefficients of potential at tree build -- is the
    physically correct model and composes freely with the two-fluid
    superconductor law.
  * Every metal layer is >= 2 cells thick. A 1-cell film has ZERO
    filaments in one orientation and the fortran filament2node/connect
    kernels reject that (the same trap the pdn capstone hit with 1-cell
    plates).
  * ANISOTROPIC cells are what make a realistic stack affordable: a
    150 um washer over a 300 nm dielectric gap is a ~10:1 cell aspect.
    Verified to compose with superconductors + groundplane + LpPR.

VERIFIED BEHAVIOUR
  * Lossless by construction: sigma = 0 with lambdaL > 0 gives
    Re Z11 ~ 1e-17 ohm, i.e. numerically exact zero -- the two-fluid
    London channel is purely reactive.
  * L is frequency-independent across 1e9-1e11 Hz to all printed
    digits, kinetic contribution included (it tracks lambdaL: 0.61 pH
    at lambda -> 0, 0.99 pH at 90 nm, 6.68 pH at 400 nm on the same
    20-cell washer).
  * A groundplane REDUCES L (1.92 -> 1.57 pH on the test washer):
    image-current screening, as it must.
  * LpR vs LpPR differ by a FIXED PORT-LOCAL term, not a formulation
    error: measured 0.0609 / 0.0651 / 0.0689 pH as the washer grows and
    L itself goes 0.99 -> 1.51 -> 2.30 pH, so the RELATIVE gap falls
    6.1% -> 4.3% -> 3.0% and is ~0.06% on a realistic 100 pH washer.
    The two paths define the port differently (equipotential terminal
    vs nodal injection); neither is wrong, and the difference is
    port-local. Do NOT chase it -- size the loop so it does not matter.

Usage:
    PYTHONPATH=src python3 studies/squid_washer.py
Env:
  HOLE/OUTER/SLIT   washer hole, outer side, slit width, in um
                    (default 20 / 60 / 2)
  DXY/DZ            cell pitch in um / nm (default 1 um, 200 nm)
  GROUND            1 (default) put a Nb groundplane under the washer
  GAP               washer-to-groundplane dielectric, nm (default 400)
  LAM               Nb London depth, nm (default 90)
  EPSBG             background permittivity (default 3.9 = SiO2)
  FREQ              extraction frequency (default 1e10)
  SWEEP=1           frequency sweep -> lumped L, C and the washer
                    self-resonance
  VALIDATE=1        bare washer (no groundplane) vs the Jaycox-Ketchen
                    square-washer formula L = 1.25 mu0 d
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import stencils as st
import voxmodel
from port_impedance import LpPRSolver

MU0 = voxmodel.MU0


def build_washer(hole_um=20.0, outer_um=60.0, slit_um=2.0, dxy_um=1.0,
                 dz_nm=200.0, film_cells=2, ground=True, gap_nm=400.0,
                 lam_nm=90.0, eps_bg=3.9, port_um=None):
    """Square Nb washer with a slit, optionally over a groundplane.

    The washer occupies the TOP ``film_cells`` layers; below it (when
    ``ground``) sit the dielectric gap and a solid groundplane of the
    same thickness. The port straddles the slit at the hole edge --
    where the junctions go on a real dc SQUID.
    """
    dxy = dxy_um*1e-6
    dz = dz_nm*1e-9
    n = int(round(outer_um/dxy_um))
    nh = int(round(hole_um/dxy_um))
    ns = max(int(round(slit_um/dxy_um)), 1)
    ngap = max(int(round(gap_nm/dz_nm)), 1) if ground else 0
    nz = film_cells + (ngap + film_cells if ground else 0)

    occ = np.ones((n, n), dtype=bool)
    c0 = (n - nh)//2
    occ[c0:c0+nh, c0:c0+nh] = False                  # the hole
    sy = (n - ns)//2                                 # slit: hole -> edge
    occ[sy:sy+ns, :c0] = False

    lam = lam_nm*1e-9
    lamg = np.zeros((n, n, nz))
    for z in range(nz - film_cells, nz):             # the washer film
        lamg[:, :, z] = np.where(occ, lam, 0.0)
    if ground:
        for z in range(film_cells):                  # solid groundplane
            lamg[:, :, z] = lam

    m = voxmodel.VoxelModel('squid_washer_%d' % n)
    m.dims = (n, n, nz)
    m.d = np.array([dxy, dxy, dz])
    m.sigma = np.zeros((n, n, nz))       # lossless: sigma 0, lambdaL > 0
    m.lambdaL = lamg
    m.superconductor = True
    m.eps_bg = eps_bg
    m.freq = np.array([1e10])

    # Port across the slit at the hole edge, on the washer film only --
    # where a dc SQUID's junctions sit. The slit runs in -y with its
    # WIDTH in x, so its two lips are at x = sy-1 and x = sy+ns and the
    # drive is x-directed (axis index 0).
    #
    # ``port_um`` is the port's IN-PLANE length along the lip, in
    # microns, and it matters enormously in a REFINEMENT study. The
    # default (None = one cell) makes the contact SHRINK as the mesh is
    # refined, and a contact of vanishing width has a logarithmically
    # divergent spreading impedance -- so L never converges. Measured
    # on the h-refinement sequence (studies/hrefine.py, dxy 1 -> 0.125):
    # increments 0.054 / 0.124 / 0.134 pH, i.e. tending to a CONSTANT
    # per halving, which is exactly that logarithm. Hold the port at a
    # fixed PHYSICAL size (port_um) whenever you refine; a port pinned
    # to voxel indices is not the same port at two resolutions.
    p = voxmodel.Port('slit')
    npw = max(int(round(port_um/dxy_um)), 1) if port_um else 1
    for z in range(nz - film_cells, nz):
        for k in range(npw):
            y = c0 - 1 - k
            if y < 0:
                break
            p._add('P', (sy - 1, y, z, 0, 1))
            p._add('N', (sy + ns, y, z, 0, -1))
    p._freeze()
    m.ports = [p]
    return m


def foster_check(f, z, verbose=True):
    """FOSTER'S REACTANCE THEOREM -- the rigorous, geometry-free test.

    A LOSSLESS one-port (which this is by construction: sigma = 0 with
    lambdaL > 0 makes every branch purely reactive, so the discretised
    model is an exact LC network) must have dX/dw > 0 at every
    frequency, with poles and zeros strictly ALTERNATING. X therefore
    climbs to +inf at a parallel resonance, jumps to -inf, climbs back
    through zero at a series resonance, and repeats.

    So MULTIPLE sign changes are mandatory, not suspicious -- a
    reactance that crossed zero only once would be the unphysical
    result. What a sign change costs you is the validity of a one-pole
    lumped fit above it, nothing more.

    Returns (violations, poles, zeros). Any violation is a real defect:
    an unconverged solve, a sign error, or a preconditioner returning
    garbage will break monotonicity long before it breaks anything a
    single-frequency sanity check would notice.
    """
    f = np.asarray(f, dtype=float)
    x = np.asarray(z).imag
    w = 2*np.pi*f
    viol, poles, zeros = [], [], []
    for i in range(1, len(x)):
        pole = x[i-1] > 0 and x[i] < 0
        zero = x[i-1] < 0 and x[i] > 0
        if pole:
            poles.append((f[i-1], f[i]))
        elif zero:
            zeros.append((f[i-1], f[i]))
        elif (x[i] - x[i-1])/(w[i] - w[i-1]) <= 0:
            viol.append((f[i-1], f[i]))
    if verbose:
        print("  Foster check (lossless one-port: dX/dw > 0, poles and "
              "zeros alternate): %d violation(s), %d pole(s), %d zero(s)"
              % (len(viol), len(poles), len(zeros)), flush=True)
        for a, b in poles:
            print("    parallel resonance bracketed in %.4g - %.4g Hz"
                  % (a, b), flush=True)
        for a, b in zeros:
            print("    series resonance bracketed in %.4g - %.4g Hz"
                  % (a, b), flush=True)
        for a, b in viol:
            print("    VIOLATION between %.4g and %.4g Hz -- the "
                  "reactance is not monotone; suspect convergence"
                  % (a, b), flush=True)
    return viol, poles, zeros


def oracle_z(m, freq):
    """Z11 from a DENSE, independently assembled system -- the check
    that validates the ANSWER, not just self-consistency.

    Same MNA shape the production path solves, but every piece built
    from ground truth: exact box-integral partial inductances, the
    tree's dense n2n for P, the model's own complex branch impedances,
    and a dense equilibrated least-squares solve instead of
    W-rescaling + preconditioned fgmres. Nothing is shared with
    SystemMat/LpPRSolver except the geometry, so agreement exercises
    the whole production stack.

    Adapted from validate_dielectric._capacitance, whose drive is
    hard-wired to a plate pair; this one drives the model's PORT.
    O(n^3) -- small washers only.
    """
    import terminal as tm
    import equiterminal as eq
    M = m.build_tree(st.single_level_nleaf(m.dims), 1, capacitive=True)
    m.prepare(M, freq)
    nfe = int(np.size(M.e.struc))
    nff = int(np.size(M.f.struc))
    nfg = int(np.size(M.g.struc))
    B, _ = eq.sparse_incidence(M, M._vhr_whole, nfe + nff + nfg,
                               int(np.size(M.lv[0].struc)))
    fa, fc = eq.filament_cells(M)
    nfil, nn = B.shape
    l_t = m.d
    lo, hi = fc*l_t[None, :], (fc + 1)*l_t[None, :]
    Lp = np.zeros((nfil, nfil))
    for ax in range(3):
        s = np.flatnonzero(fa == ax)
        if s.size:
            alo, ahi = lo[s].copy(), hi[s].copy()
            alo[:, ax] = (fc[s, ax] + 0.5)*l_t[ax]
            ahi[:, ax] = (fc[s, ax] + 1.5)*l_t[ax]
            Lp[np.ix_(s, s)] = tm.box_mutual_matrix(alo, ahi, ax)
    r = np.concatenate([
        np.broadcast_to(np.atleast_1d(M.e.r), (nfe,)),
        np.broadcast_to(np.atleast_1d(M.f.r), (nff,)),
        np.broadcast_to(np.atleast_1d(M.g.r), (nfg,))])
    w = 2*np.pi*freq
    Z = np.diag(r.astype(np.complex128)) + 1j*w*Lp
    P = M.n2n.toarray() if hasattr(M.n2n, 'toarray') else np.asarray(M.n2n)
    Cnode = np.zeros((nn, nn), dtype=np.complex128)
    ext = M.external
    Cnode[np.ix_(ext, ext)] = np.linalg.inv(P[np.ix_(ext, ext)])
    A = B.toarray().astype(np.complex128)
    K = np.block([[Z, -A], [A.T, 1j*w*Cnode]])
    src = m.source_vector(M, 0, 1.0)
    rhs = np.zeros(nfil + nn, dtype=np.complex128)
    rhs[nfil:] = src
    # row-equilibrate: branch and node rows span many orders and
    # lstsq's rank cutoff would discard the small-scale PHYSICS
    # (the trap that cost the dielectric program a week)
    rs = 1.0/np.maximum(np.abs(K).max(axis=1), 1e-300)
    x = np.linalg.lstsq(K*rs[:, None], rhs*rs, rcond=None)[0]
    phi = x[nfil:]
    return complex(np.dot(phi, src))


def solve_at(m, freq, verbose=False):
    M = m.build_tree(st.single_level_nleaf(m.dims), 1, capacitive=True)
    m.prepare(M, freq)
    S = LpPRSolver(m, M)
    z, x, info = S.solve(freq, verbose=verbose)
    return z, info, S, M


def stats(m):
    occ = np.asarray(m.lambdaL) != 0.0
    return int(occ.sum())


def main():
    hole = float(os.environ.get('HOLE', '20'))
    outer = float(os.environ.get('OUTER', '60'))
    slit = float(os.environ.get('SLIT', '2'))
    dxy = float(os.environ.get('DXY', '1'))
    dz = float(os.environ.get('DZ', '200'))
    ground = os.environ.get('GROUND', '1') == '1'
    gap = float(os.environ.get('GAP', '400'))
    lam = float(os.environ.get('LAM', '90'))
    epsbg = float(os.environ.get('EPSBG', '3.9'))
    freq = float(os.environ.get('FREQ', '1e10'))

    if os.environ.get('VALIDATE', '0') == '1':
        validate(hole, outer, slit, dxy, dz, lam, epsbg)
        return
    if os.environ.get('ORACLE', '0') == '1':
        validate_oracle(dxy, dz, epsbg)
        return

    kw = dict(hole_um=hole, outer_um=outer, slit_um=slit, dxy_um=dxy,
              dz_nm=dz, ground=ground, gap_nm=gap, lam_nm=lam,
              eps_bg=epsbg)
    m = build_washer(**kw)
    print("squid_washer: %s cells, %d Nb, d=[%.3g,%.3g,%.3g] m, "
          "aspect %.0f:1, eps_bg %.3g, lambda %.0f nm, ground %s"
          % (m.dims, stats(m), m.d[0], m.d[1], m.d[2], m.d[0]/m.d[2],
             epsbg, lam, ground), flush=True)

    if os.environ.get('SWEEP', '0') == '1':
        sweep(m, kw)
        return
    t0 = time.perf_counter()
    z, info, _, _ = solve_at(m, freq, verbose=True)
    w = 2*np.pi*freq
    print("  Z11(%.3g Hz) = %.6g%+.6gj ohm   L %.5g pH   "
          "(%d matvecs, true %.1e, %.1f s)"
          % (freq, z.real, z.imag, 1e12*z.imag/w, info['matvecs'],
             info['true_residual'], time.perf_counter() - t0), flush=True)
    jk = 1.25*MU0*hole*1e-6
    print("  Jaycox-Ketchen square-washer L = 1.25 mu0 d = %.5g pH "
          "(bare-washer formula; a groundplane screens it DOWN)"
          % (1e12*jk), flush=True)


def fit_parallel_lc(f, z):
    """Lumped (L, C) of a PARALLEL resonator: Im Z = wL/(1 - w^2 LC).

    NOT the series form pdn_sweep.fit_rlc uses. A PDN port sees plate
    capacitance in SERIES with the via/plane inductance; a washer port
    across the slit sees the loop inductance SHUNTED by the slit
    capacitance, so Im Z rises with w, diverges at the self-resonance
    and comes back NEGATIVE above it -- exactly what the sweep shows
    (+17.3 ohm at 316 GHz, -11.1 ohm at 562 GHz). Fitting the series
    model to that data returns nonsense.

    The parallel form linearises exactly:
        1/Im Z = (1/L)(1/w) - C w
    which is linear least squares in (1/L, C) on the basis (1/w, -w).
    Column-equilibrated for the same reason fit_rlc is: 1/w and w are
    ~24 orders apart at these frequencies.

    BAND SENSITIVITY, measured on the 60 um demo washer -- report L
    confidently, the resonance as a RANGE. L is robust (5.394 / 5.386 /
    5.403 pH from points below the pole / through it / including the
    second mode, against a direct low-frequency read of 5.397 pH). C is
    NOT: 17.1 fF -> f_res 524 GHz from below the pole, 33.1 fF ->
    377 GHz once the bracketing point is included, and a meaningless
    3.9 fF if the second-mode point is left in. The honest statement is
    that the sign change brackets the resonance in 316-562 GHz and the
    fits place it at 380-520 GHz; tightening it needs points sampled
    inside that interval, not a better fit.
    """
    w = 2*np.pi*np.asarray(f, dtype=float)
    b = 1.0/np.asarray(z).imag
    A = np.column_stack([1.0/w, -w])
    cs = np.linalg.norm(A, axis=0)
    sol, *_ = np.linalg.lstsq(A/cs, b, rcond=None)
    invL, C = sol/cs
    L = 1.0/invL
    # NORM-relative, not max-per-point: 1/Im Z passes through zero at
    # the pole, so a per-point relative deviation is meaningless there
    # (it reads ~0.5 on a fit whose L matches the direct low-frequency
    # read to 0.2%).
    dev = float(np.linalg.norm(A@(sol/cs) - b)/np.linalg.norm(b))
    fres = 1.0/(2*np.pi*np.sqrt(L*C)) if L > 0 and C > 0 else float('nan')
    return L, C, fres, dev


def sweep(m, kw):
    """Frequency sweep -> lumped L, C and the washer self-resonance."""
    fmin = float(os.environ.get('FMIN', '1e9'))
    fmax = float(os.environ.get('FMAX', '1e12'))
    npts = int(os.environ.get('NPTS', '10'))
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), npts)
    M = m.build_tree(st.single_level_nleaf(m.dims), 1, capacitive=True)
    m.prepare(M, float(freqs[0]))
    S = LpPRSolver(m, M)
    zs = []
    x_prev = f_prev = None
    print("      f [Hz]         Re Z         Im Z       L_eff [pH]"
          "   matvecs   true", flush=True)
    for f in freqs:
        z, x, info = S.solve(float(f), x0=x_prev, x0_freq=f_prev)
        x_prev, f_prev = x, float(f)
        zs.append(z)
        print("  %11.4g  %11.4g  %11.4g   %10.5g   %6d  %7.1e"
              % (f, z.real, z.imag, 1e12*z.imag/(2*np.pi*f),
                 info['matvecs'], info['true_residual']), flush=True)
    zs = np.array(zs)
    L, C, fres, dev = fit_parallel_lc(freqs, zs)
    print("  lumped washer (PARALLEL L||C): L %.5g pH, C %.5g fF, "
          "self-resonance %.5g GHz (fit dev %.1e)"
          % (1e12*L, 1e15*C, 1e-9*fres, dev), flush=True)
    viol, poles, zeros = foster_check(freqs, zs)
    if not poles:
        print("  NOTE no parallel resonance in band: the fitted "
              "resonance is an EXTRAPOLATION.", flush=True)
    if len(poles) + len(zeros) > 1:
        # Poles and zeros ALTERNATE (Foster), so a second crossing is
        # required physics, not a defect -- it only ends the validity
        # of the ONE-POLE fit. Measured on the 60 um demo washer:
        # +17.3 ohm at 316 GHz, -11.1 at 562 GHz (pole), +11.3 at
        # 1 THz (zero), the last of which also sits where the SiO2
        # wavelength (152 um) is only 2.5x the washer and the
        # quasi-static approximation is marginal anyway.
        print("  NOTE %d resonances in band: the one-pole fit above "
              "is valid only up to the FIRST one; C and f_res from a "
              "wider band are meaningless (measured: 33 fF vs 3.9 fF)."
              % (len(poles) + len(zeros)), flush=True)


def validate_oracle(dxy, dz, epsbg, tol=1e-8):
    """Production LpPRSolver vs the dense independently-assembled
    system (:func:`oracle_z`) -- the strongest available check.

    MEASURED 2026-08-08: agreement to 1.5e-12 - 2.5e-11 RELATIVE on
    two washers x two frequencies x weak/strong kinetic regimes, i.e.
    ~11 significant digits. Nothing but the geometry is shared, so
    this exercises W-rescaling, the band preconditioner, the FMM
    traversals and the Krylov solve all at once.
    """
    print("validate: production LpPRSolver vs dense oracle", flush=True)
    worst = 0.0
    for outer, hole, lam in ((12.0, 4.0, 90.0), (16.0, 6.0, 360.0)):
        for freq in (1e10, 1e11):
            kw = dict(hole_um=hole, outer_um=outer, slit_um=1.0,
                      dxy_um=dxy, dz_nm=dz, ground=False, lam_nm=lam,
                      eps_bg=epsbg)
            zp, _, _, _ = solve_at(build_washer(**kw), freq)
            zo = oracle_z(build_washer(**kw), freq)
            rel = abs(zp - zo)/abs(zo)
            worst = max(worst, rel)
            print("  outer %4.1f hole %3.1f lambda %5.0f nm  f %.0e:  "
                  "L prod %.6g / oracle %.6g pH   rel %.2e"
                  % (outer, hole, lam, freq, 1e12*zp.imag/(2*np.pi*freq),
                     1e12*zo.imag/(2*np.pi*freq), rel), flush=True)
    print("  worst relative deviation %.2e (tol %.0e): %s"
          % (worst, tol, "PASS" if worst < tol else "FAIL"), flush=True)
    return worst < tol


def validate(hole, outer, slit, dxy, dz, lam, epsbg):
    """Bare washer vs Jaycox-Ketchen, and the kinetic-inductance trend.

    JK assumes a NARROW slit and outer >> hole; SuperPEEC models the real
    (finite) slit and finite outer size, so agreement is expected at
    the tens-of-percent level, tightening as outer/hole grows.
    """
    print("validate: bare Nb washer vs Jaycox-Ketchen L = 1.25 mu0 d",
          flush=True)
    freq = 1e10
    w = 2*np.pi*freq
    for h in (hole, 1.5*hole):
        m = build_washer(hole_um=h, outer_um=outer, slit_um=slit,
                         dxy_um=dxy, dz_nm=dz, ground=False,
                         lam_nm=lam, eps_bg=epsbg)
        z, info, _, _ = solve_at(m, freq)
        jk = 1.25*MU0*h*1e-6
        print("  hole %5.1f um (outer/hole %.2f): L %8.5g pH   "
              "JK %8.5g pH   ratio %.3f   (%d mv)"
              % (h, outer/h, 1e12*z.imag/w, 1e12*jk,
                 (z.imag/w)/jk, info['matvecs']), flush=True)
    # kinetic trend: L must RISE with lambda, and Re Z must stay zero
    for lm in (1e-3, lam, 4*lam):
        m = build_washer(hole_um=hole, outer_um=outer, slit_um=slit,
                         dxy_um=dxy, dz_nm=dz, ground=False, lam_nm=lm,
                         eps_bg=epsbg)
        z, info, _, _ = solve_at(m, freq)
        print("  lambda %7.4g nm: L %8.5g pH   Re Z %.2e ohm "
              "(lossless check)" % (lm, 1e12*z.imag/w, z.real),
              flush=True)


if __name__ == '__main__':
    main()
