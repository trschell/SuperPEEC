# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for the LpPR/dielectric TOML input extension (2026-08-15).

Covers the schema contract (formulation auto-resolution and every
rejection the doctrine promises is loud) and the physics of the
plate_pair example end to end: solved C above the parallel-plate
formula by a plausible fringing margin, frequency-flat below
resonance, warm-started, true-residual-clean; a loss tangent shows
up in Re(Z) as dielectric loss.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys

import numpy as np

sys.path.insert(0, 'src')

FAIL = []
EPS0 = 8.8541878128e-12


def check(name, ok, note=''):
    print('    %s %s%s' % ('ok  ' if ok else 'FAIL', name,
                           ('  ' + note) if note else ''), flush=True)
    if not ok:
        FAIL.append(name)


PLATES = '\n'.join([
    '[grid]', 'dims = [12, 12, 6]', 'pitch = 100e-6',
    '[[block]]', 'from = [0, 0, 0]', 'to = [12, 12, 2]',
    'sigma = 5.8e7',
    '[[block]]', 'from = [0, 0, 2]', 'to = [12, 12, 4]',
    'epsilon = 4.2', '{DF}',
    '[[block]]', 'from = [0, 0, 4]', 'to = [12, 12, 6]',
    'sigma = 5.8e7',
    '[port]',
    'p_faces = [[5, 5, 5, "+z"], [5, 6, 5, "+z"],'
    ' [6, 5, 5, "+z"], [6, 6, 5, "+z"]]',
    'n_faces = [[5, 5, 0, "-z"], [5, 6, 0, "-z"],'
    ' [6, 5, 0, "-z"], [6, 6, 0, "-z"]]',
    '[solve]', 'freq = [1e7, 1e8]', '{EXTRA}'])


def doc(df='', extra=''):
    return PLATES.replace('{DF}', df).replace('{EXTRA}', extra)


def expect_error(name, text, needle):
    """Parse AND build -- doctrine errors fire at either stage."""
    import sppeec_input
    try:
        sppeec_input.loads(text).model()
        check(name, False, 'no exception')
    except ValueError as exc:
        check(name, needle in str(exc), str(exc)[:70])


def main():
    import sppeec_input

    # -- schema contract ---------------------------------------------
    prob = sppeec_input.loads(doc())
    check("epsilon block resolves formulation auto -> LpPR",
          prob.formulation == 'LpPR' and prob.rtol == 1e-10,
          '%s rtol %g' % (prob.formulation, prob.rtol))
    p3 = sppeec_input.load('examples/module3wire.toml')
    check("wire input still resolves auto -> LpR",
          p3.formulation == 'LpR' and p3.rtol == 1e-4)
    expect_error("explicit LpR + epsilon rejected",
                 doc(extra="formulation = 'LpR'"), 'need formulation')
    expect_error("LpPR + [[wire]] rejected",
                 doc() + '\n[[wire]]\npoints = [[0,0,0],[1e-4,0,0]]\n'
                 'radius = 1e-5\nsigma = 3e7', 'not supported')
    expect_error("bad face string rejected",
                 doc().replace('"+z"', '"+q"'), 'face one of')
    expect_error("mixed port styles rejected",
                 doc().replace('[solve]',
                               'p_cells = [[0, 0, 0]]\n[solve]'),
                 'not both')
    expect_error("loss_tangent without epsilon rejected",
                 doc().replace('sigma = 5.8e7',
                               'sigma = 5.8e7\nloss_tangent = 0.02',
                               1), 'needs epsilon')
    expect_error("empty block rejected",
                 doc().replace('epsilon = 4.2', ''), 'sigma (conductor)')
    # face on a non-conductor is caught at model(); trigger it
    try:
        sppeec_input.loads(doc().replace('[5, 5, 5, "+z"]',
                                         '[5, 5, 2, "+z"]')).model()
        check("face-on-dielectric caught at model()", False)
    except ValueError as exc:
        check("face-on-dielectric caught at model()",
              'not a conductor' in str(exc))

    # -- physics: the plate pair end to end --------------------------
    prob = sppeec_input.loads(doc())
    m = prob.model()
    check("epsilon array built (real, 4.2 in the gap)",
          m.epsilon is not None
          and m.epsilon.dtype == np.float64
          and float(np.real(m.epsilon[5, 5, 3])) == 4.2
          and float(np.real(m.epsilon[5, 5, 0])) == 1.0)
    M = prob.tree(m)
    sw = prob.sweeper(m, M)
    z1, i1 = sw.solve(1e7)
    z2, i2 = sw.solve(1e8)
    c1 = float(np.imag(1.0/z1)/(2*np.pi*1e7))
    c2 = float(np.imag(1.0/z2)/(2*np.pi*1e8))
    c_formula = EPS0*4.2*(12*100e-6)**2/(2*100e-6)
    check("port is capacitive at both points",
          z1.imag < 0 and z2.imag < 0)
    check("C above the plate formula by fringing (0..50%)",
          c_formula < c1 < 1.5*c_formula,
          'C %.4g vs formula %.4g F' % (c1, c_formula))
    check("C frequency-flat below resonance (1e-3)",
          abs(c1 - c2)/c1 < 1e-3, 'dC/C %.1e' % (abs(c1 - c2)/c1))
    check("true residual clean at both points",
          i1['true_residual'] < 1e-8 and i2['true_residual'] < 1e-8,
          '%.1e / %.1e' % (i1['true_residual'], i2['true_residual']))
    check("i_f exposed for field export",
          'i_f' in i2 and np.asarray(i2['i_f']).size > 0)

    # -- multi-port: the 2x2 Z matrix on coupled_plates ---------------
    bar = '\n'.join(['[grid]', 'dims = [8, 8, 4]', 'pitch = 1e-4',
                     '[[block]]', 'from = [0, 0, 0]',
                     'to = [8, 8, 4]', 'sigma = 5.8e7'])
    expect_error("two cell-style ports rejected",
                 bar + '\n[[port]]\np_cells = [[0, 0, 0]]\n'
                 'n_cells = [[7, 7, 0]]\n[[port]]\n'
                 'p_cells = [[0, 1, 0]]\nn_cells = [[7, 6, 0]]',
                 'two-terminal')
    expect_error("mixed styles across ports rejected",
                 bar + '\n[[port]]\np_faces = [[0, 0, 3, "+z"]]\n'
                 'n_faces = [[7, 7, 3, "+z"]]\n[[port]]\n'
                 'p_cells = [[0, 1, 0]]\nn_cells = [[7, 6, 0]]',
                 'share one style')
    pm = sppeec_input.load('examples/coupled_plates.toml')
    check("[[port]] parses, names kept",
          pm.formulation == 'LpPR'
          and [n for n, _, _ in pm.ports_faces] == ['left', 'right'])
    mm = pm.model()
    check("model carries 2 ports",
          len(mm.ports) == 2 and mm.ports[0].name == 'left')
    Mm = pm.tree(mm)
    Zm, im = pm.sweeper(mm, Mm).solve(1e8)
    check("Z is a 2x2 matrix", np.shape(Zm) == (2, 2))
    recip = abs(Zm[0, 1] - Zm[1, 0])/abs(Zm[0, 1])
    check("reciprocity Z12 == Z21 (solved, not imposed)",
          recip < 1e-4, 'rel %.1e' % recip)
    mirror = abs(Zm[0, 0] - Zm[1, 1])/abs(Zm[0, 0])
    check("mirror symmetry Z11 == Z22", mirror < 1e-6,
          'rel %.1e' % mirror)
    check("coupling below self-impedance",
          abs(Zm[0, 1]) < 0.5*abs(Zm[0, 0]),
          '|Z12|/|Z11| %.3f' % (abs(Zm[0, 1])/abs(Zm[0, 0])))
    check("both ports capacitive",
          Zm[0, 0].imag < 0 and Zm[1, 1].imag < 0)
    check("multi-port info aggregated + i_f kept",
          im['matvecs'] > 0 and im['true_residual'] < 1e-8
          and 'i_f' in im)

    # -- loss tangent enters Re(Z) -----------------------------------
    probl = sppeec_input.loads(doc(df='loss_tangent = 0.02'))
    ml = probl.model()
    check("lossy epsilon is complex",
          ml.epsilon.dtype == np.complex128
          and abs(ml.epsilon[5, 5, 3].imag + 4.2*0.02) < 1e-12)
    Ml = probl.tree(ml)
    zl, _ = probl.sweeper(ml, Ml).solve(1e8)
    # dielectric conduction loss: Re(Z) ~ df*|Im Z| >> the lossless
    # plate resistance (~6e-5 ohm measured)
    check("loss tangent dominates Re(Z)",
          zl.real > 100*z2.real and abs(zl.imag - z2.imag)
          / abs(z2.imag) < 0.05,
          'Re %.4g (lossless %.4g), Im drift %.1e'
          % (zl.real, z2.real,
             abs(zl.imag - z2.imag)/abs(z2.imag)))

    # -- superconductor blocks (lambda_l) ------------------------------
    def bar_doc(material):
        return '\n'.join([
            '[grid]', 'dims = [12, 2, 2]', 'pitch = 200e-9',
            '[[block]]', 'from = [0, 0, 0]', 'to = [12, 2, 2]',
            material,
            '[port]',
            'p_faces = [[0, 0, 0, "-x"], [0, 0, 1, "-x"],'
            ' [0, 1, 0, "-x"], [0, 1, 1, "-x"]]',
            'n_faces = [[11, 0, 0, "+x"], [11, 0, 1, "+x"],'
            ' [11, 1, 0, "+x"], [11, 1, 1, "+x"]]',
            '[solve]', 'freq = [1e9]'])

    def bar_z(material):
        p = sppeec_input.loads(bar_doc(material))
        mb = p.model()
        return p.sweeper(mb, p.tree(mb)).solve(1e9)[0], p

    expect_error("epsilon + lambda_l rejected",
                 bar_doc('lambda_l = 90e-9\nepsilon = 4.2'),
                 'cannot combine')
    expect_error("lambda_l <= 0 rejected", bar_doc('lambda_l = 0.0'),
                 'must be > 0')
    expect_error("explicit LpR + lambda_l rejected",
                 bar_doc('lambda_l = 90e-9')
                 + "\nformulation = 'LpR'", 'ohmic pads')
    zsc, psc = bar_z('lambda_l = 90e-9')
    check("lambda_l resolves auto -> LpPR, model flagged",
          psc.formulation == 'LpPR')
    w = 2*np.pi*1e9
    check("pure London bar is a lossless inductor",
          zsc.imag > 0 and abs(zsc.real) < 1e-9*zsc.imag,
          'Z %.3g%+.3gj' % (zsc.real, zsc.imag))
    # the exact kinetic-inductance identity (validate_superconductor
    # part A, through the TOML/LpPR path): on a symmetry-pinned 2x2
    # cross-section the current split cannot depend on lambda, so
    # L(lam2) - L(lam1) == MU0*(lam2^2 - lam1^2)*l_eff/A exactly,
    # with l_eff the centre-to-centre span (LpPR's convention)
    z1, _ = bar_z('lambda_l = 50e-9')
    z2, _ = bar_z('lambda_l = 100e-9')
    dx = 200e-9
    dl = (z2.imag - z1.imag)/w
    dl_exact = (4e-7*np.pi)*(100e-9**2 - 50e-9**2)*(11*dx)/(4*dx*dx)
    check("kinetic-inductance identity dL == MU0*dlam2*l/A",
          abs(dl - dl_exact)/dl_exact < 1e-3,
          'dL %.6g vs %.6g H (rel %.1e)'
          % (dl, dl_exact, abs(dl - dl_exact)/dl_exact))
    # normal-metal limit: lambda = 1 m two-fluid == plain copper
    zn, _ = bar_z('sigma = 5.8e7\nlambda_l = 1.0')
    zc, _ = bar_z('sigma = 5.8e7')
    check("lambda -> inf two-fluid matches plain copper",
          abs(zn - zc)/abs(zc) < 1e-8,
          'rel %.1e' % (abs(zn - zc)/abs(zc)))

    # -- sweep expressions + equipotential ports (shipped together,
    # -- tested together on examples/equibar.toml) ---------------------
    pe = sppeec_input.load('examples/equibar.toml')
    check("sweep table expands to exact logspace",
          len(pe.freqs) == 5 and np.allclose(
              pe.freqs, np.logspace(5, 9, 5), rtol=1e-13))
    plin = sppeec_input.loads(
        open('examples/equibar.toml').read().replace(
            'freq = { from = 1e5, to = 1e9, points = 5 }',
            'freq = { from = 1e5, to = 5e5, points = 3, '
            'spacing = "lin" }'))
    check("lin spacing expands to exact linspace",
          np.allclose(plin.freqs, [1e5, 3e5, 5e5], rtol=1e-13))
    eqtxt = open('examples/equibar.toml').read()
    expect_error("freq table missing points rejected",
                 eqtxt.replace(', points = 5', ''), "needs 'points'")
    expect_error("bad spacing rejected",
                 eqtxt.replace('points = 5',
                               'points = 5, spacing = "cubic"'),
                 "'log' or 'lin'")
    expect_error("from >= to rejected",
                 eqtxt.replace('to = 1e9', 'to = 1e4'),
                 '0 < from < to')
    expect_error("equipotential + dielectric rejected",
                 eqtxt.replace('sigma = 5.8e7', 'epsilon = 4.2'),
                 'prescribed injection')
    expect_error("equipotential + wires rejected",
                 eqtxt + '\n[[wire]]\n'
                 'points = [[0, 0, 0], [1e-5, 0, 0]]\n'
                 'radius = 1e-6\nsigma = 3e7', 'do not combine')
    expect_error("equipotential + explicit LpPR rejected",
                 eqtxt + '\nformulation = "LpPR"',
                 'no equipotential terminal')
    expect_error("equipotential on cell-style port rejected",
                 '\n'.join(['[grid]', 'dims = [8, 2, 2]',
                            'pitch = 1e-5', '[[block]]',
                            'from = [0, 0, 0]', 'to = [8, 2, 2]',
                            'sigma = 5.8e7', '[port]',
                            'equipotential = true',
                            'p_cells = [[0, 0, 0]]',
                            'n_cells = [[7, 1, 1]]']),
                 'face-style port')

    me = pe.model()
    Me = pe.tree(me)
    swe = pe.sweeper(me, Me)
    zlo, ie = swe.solve(1e5)
    # the razor gate: equiterminal's terminal treatment recovers the
    # FULL physical length, so DC-regime R is l/(sigma*A) exactly
    r_dc = (24*10e-6)/(5.8e7*(4*10e-6)**2)
    check("equipotential R(1e5) == l/(sigma*A) analytic",
          abs(zlo.real - r_dc)/r_dc < 1e-3,
          '%.6g vs %.6g (rel %.1e)'
          % (zlo.real, r_dc, abs(zlo.real - r_dc)/r_dc))
    check("i_f exposed on the equipotential path",
          'i_f' in ie and np.asarray(ie['i_f']).size > 0)
    zhi, _ = swe.solve(1e9)
    check("skin effect: R rises, L falls across the sweep",
          zhi.real > 1.5*zlo.real
          and zhi.imag/(2*np.pi*1e9) < zlo.imag/(2*np.pi*1e5),
          'R %.3g->%.3g, L %.4g->%.4g'
          % (zlo.real, zhi.real, zlo.imag/(2*np.pi*1e5),
             zhi.imag/(2*np.pi*1e9)))
    # equivalence: the TOML route is EXACTLY the direct solver
    from equiterminal import EquiTerminalSolver
    m2 = pe.model()
    M2 = pe.tree(m2)
    m2.prepare(M2, 1e7)
    zd, _, _ = EquiTerminalSolver(m2, M2, 0).solve(
        1e7, rtol=pe.rtol, method=pe.method)
    zt, _ = swe.solve(1e7)
    check("TOML equipotential == direct EquiTerminalSolver",
          abs(zt - zd)/abs(zd) < 1e-9,
          'rel %.1e' % (abs(zt - zd)/abs(zd)))
    # lambda_l + equipotential is the VALIDATED superconductor combo
    psc2 = sppeec_input.loads(
        bar_doc('lambda_l = 90e-9').replace(
            '[port]', '[port]\nequipotential = true'))
    check("lambda_l + equipotential resolves to LpR",
          psc2.formulation == 'LpR' and psc2.equipotential)
    msc = psc2.model()
    zsc2, _ = psc2.sweeper(msc, psc2.tree(msc)).solve(1e9)
    check("equipotential London bar: lossless inductor",
          zsc2.imag > 0 and abs(zsc2.real) < 1e-6*zsc2.imag,
          'Z %.3g%+.3gj' % (zsc2.real, zsc2.imag))

    # -- dispersive dielectrics (Djordjevic-Sarkar) --------------------
    import voxmodel
    dstxt = open('examples/plate_pair_ds.toml').read()
    expect_error("unknown dispersion model rejected",
                 dstxt.replace('"djordjevic"', '"debye9000"'),
                 "must be 'djordjevic'")
    import re
    expect_error("dispersion without f_ref rejected",
                 re.sub(r'f_ref.*\n', '', dstxt), 'needs f_ref')
    expect_error("dispersion without loss rejected",
                 dstxt.replace('loss_tangent = 0.02',
                               'loss_tangent = 0.0'),
                 'loss_tangent > 0')
    expect_error("f_ref outside (f1, f2) rejected",
                 dstxt.replace('f_ref = 1e8', 'f_ref = 1e8\nf1 = 1e9'),
                 '0 < f1 < f_ref < f2')
    expect_error("f1 without dispersion rejected",
                 doc(df='loss_tangent = 0.02\nf1 = 1e3'),
                 'only means something under dispersion')
    expect_error("too-lossy fit (eps_inf < 1) rejected",
                 dstxt.replace('loss_tangent = 0.02',
                               'loss_tangent = 0.5')
                 .replace('f_ref = 1e8',
                          'f_ref = 1e8\nf1 = 1e7\nf2 = 1e9'),
                 'eps_inf')

    pds = sppeec_input.loads(dstxt)
    mds = pds.model()
    check("dispersion registered on the model",
          len(mds.epsilon_dispersion) == 1)
    lo, hi, einf, deps, f1, f2 = mds.epsilon_dispersion[0]
    e_ref = voxmodel.VoxelModel.ds_epsilon(1e8, einf, deps, f1, f2)
    check("fit reproduces epsilon*(1 - j*df) at f_ref exactly",
          abs(e_ref - 4.2*(1 - 0.02j)) < 1e-12,
          '%s' % e_ref)
    e_lo = voxmodel.VoxelModel.ds_epsilon(1e6, einf, deps, f1, f2)
    e_hi = voxmodel.VoxelModel.ds_epsilon(1e10, einf, deps, f1, f2)
    check("causal signature: eps' falls with frequency",
          e_lo.real > e_ref.real > e_hi.real,
          "eps' %.4f > %.4f > %.4f"
          % (e_lo.real, e_ref.real, e_hi.real))
    tans = [-voxmodel.VoxelModel.ds_epsilon(f, einf, deps, f1,
                                            f2).imag
            / voxmodel.VoxelModel.ds_epsilon(f, einf, deps, f1,
                                             f2).real
            for f in (1e6, 1e8, 1e10)]
    check("tan-delta near-flat inside the band (< 30% spread)",
          max(tans) < 1.3*min(tans),
          'tan %.4f..%.4f' % (min(tans), max(tans)))

    # the razor: at f_ref the dispersive and constant-df models are
    # the SAME material, so the solves must agree
    Mds = pds.tree(mds)
    swds = pds.sweeper(mds, Mds)
    zds, _ = swds.solve(1e8)
    rel = abs(zds - zl)/abs(zl)      # zl: constant-df plate pair @1e8
    check("dispersive solve at f_ref == constant-df solve",
          rel < 1e-6, 'rel %.1e' % rel)
    # and away from f_ref the dispersion shows: C falls, loss stays
    zd1, _ = swds.solve(1e6)
    zd2, _ = swds.solve(1e9)
    c_1 = np.imag(1.0/zd1)/(2*np.pi*1e6)
    c_2 = np.imag(1.0/zd2)/(2*np.pi*1e9)
    check("C declines across the band (eps' dispersion)",
          c_1 > c_2, 'C %.4g -> %.4g F' % (c_1, c_2))
    df1 = zd1.real/abs(zd1.imag)
    df2 = zd2.real/abs(zd2.imag)
    check("effective loss tangent near-flat in the solve",
          0.7 < df1/df2 < 1.3, 'df_eff %.4f vs %.4f' % (df1, df2))

    print('\n%d checks failed' % len(FAIL))
    raise SystemExit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
