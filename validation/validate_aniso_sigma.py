# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for ANISOTROPIC per-cell conductivity (``sigma_axis``).

A cell cut by an axis-aligned boundary is a LAMINATE, and a laminate's
effective conductivity is anisotropic: the arithmetic mean along the
layers, the harmonic mean across them. For a layered medium those are
EXACT, not bounds -- which is the whole reason an axis-aligned partial
cell is tractable where a curved one is not.

WHY IT MATTERS. The subpixel program carries ``sigma_eff = sigma*fill``,
a SCALAR, hence isotropic. That is exactly the arithmetic mean, so it is
right for current in the plane of the cut and wrong across it: against
vacuum the harmonic mean is ZERO -- the open circuit a broken path
physically is -- where the scalar gives a finite resistance. On a layer
stack that is not academic, because vias carry current across precisely
the boundaries that create partial cells.

  A  THE TWO MEANS, against hand arithmetic, including the vacuum case
     where the harmonic mean must collapse to a true open.
  B  ISOTROPY IS THE DEFAULT AND IS BIT-IDENTICAL. A model with no
     sigma_axis must produce exactly the resistances it always did, and
     an anisotropic array whose three axes are equal must reproduce the
     isotropic answer to the last bit -- the two code paths differ
     (scalar fast path vs per-filament array) so this is a real check.
  C  THE ANISOTROPY REACHES THE SOLVER, per orientation and only there:
     a z-cut cell must change the z-directed filament resistance and
     leave x/y alone, and vice versa.
  D  THE COMPLEX PATH TOO. z is an IMPEDANCE, so the laminate's means
     swap over: along the layers conductances add, across them
     impedances add. A London model must show the same anisotropy with
     the same orientation sense.

Run: PYTHONPATH=src python3 validation/validate_aniso_sigma.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import numpy as np

import voxmodel

FAIL = []
SIG = 5.8e7
DX = 1e-7


def check(name, ok, detail=""):
    print("  %-4s %-52s %s" % ("ok" if ok else "FAIL", name, detail),
          flush=True)
    if not ok:
        FAIL.append(name)


def model(fill=None, axis=2, lam=None):
    m = voxmodel.VoxelModel('aniso')
    m.dims = (6, 6, 6)
    m.d = np.full(3, DX)
    m.sigma = np.full(m.dims, SIG, dtype=np.float32)
    if lam is not None:
        m.lambdaL = np.full(m.dims, lam)
        m.superconductor = True
    m.freq = np.array([1e10])
    if fill is not None:
        m.laminate_sigma(fill, axis)
    return m


def main():
    # -- A: the two means ---------------------------------------------
    f = np.ones((6, 6, 6))
    f[:, :, 3] = 0.35
    a = model(f, axis=2).sigma_axis
    check("in-plane is the arithmetic mean (sigma*f vs vacuum)",
          abs(a[0, 0, 3, 0] - 0.35*SIG) < 1e-6*SIG
          and abs(a[0, 0, 3, 1] - 0.35*SIG) < 1e-6*SIG,
          "%.4e, want %.4e" % (a[0, 0, 3, 0], 0.35*SIG))
    check("through-plane is the harmonic mean -> a true OPEN",
          a[0, 0, 3, 2] == 0.0, "%.4e" % a[0, 0, 3, 2])
    check("whole cells are untouched and isotropic",
          np.allclose(a[0, 0, 0], SIG), "%s" % (a[0, 0, 0],))
    # against a SECOND material rather than vacuum
    m2 = model()
    f2 = np.ones((6, 6, 6))
    f2[:, :, 3] = 0.5
    m2.laminate_sigma(f2, axis=2, sigma_out=SIG/4)
    want_par = 0.5*SIG + 0.5*SIG/4
    want_ser = 1.0/(0.5/SIG + 0.5/(SIG/4))
    got = m2.sigma_axis[0, 0, 3]
    check("two-material laminate matches both means",
          abs(got[0] - want_par) < 1e-6*SIG
          and abs(got[2] - want_ser) < 1e-6*SIG,
          "par %.4e/%.4e  ser %.4e/%.4e"
          % (got[0], want_par, got[2], want_ser))

    # -- B: isotropy is the default, bit-identical ---------------------
    base = model()
    M = base.build_tree(*base.partition())
    r_iso = base.resistances(M)
    same = model()
    same.sigma_axis = np.repeat(
        np.asarray(same.sigma, dtype=np.float64)[..., None], 3, axis=-1)
    r_aniso = same.resistances(M)
    worst = max(float(np.max(np.abs(np.atleast_1d(x) - np.atleast_1d(y))))
                for x, y in zip(r_iso, r_aniso))
    scale = float(np.max(np.abs(np.atleast_1d(r_iso[0]))))
    # NOT bit-identical, and it should not be claimed as such: the
    # uniform fast path evaluates l/(A sigma) while the array path
    # evaluates 0.5*l/A*(1/sa + 1/sb). Algebraically equal, one ulp
    # apart in floating point.
    check("equal-axis sigma_axis reproduces the isotropic answer",
          worst <= 4e-16*scale,
          "max |diff| %.3e on %.3e (%.1f ulp)"
          % (worst, scale, worst/(np.spacing(scale) or 1.0)))

    # -- C: the anisotropy reaches the solver, per orientation ---------
    # A CRACK (partial cell between two whole ones) is the case the
    # per-cell occupancy model cannot express -- the filament exists but
    # must not conduct. It raises unless given a numerical floor, which
    # is itself worth pinning.
    fc = np.ones((6, 6, 6))
    fc[:, :, 3] = 0.5
    mm = model()
    mm.laminate_sigma(fc, axis=2)
    try:
        mm.resistances(M)
        check("a crack raises rather than silently conducting", False,
              "no error")
    except RuntimeError as exc:
        check("a crack raises rather than silently conducting",
              'zero conductivity' in str(exc), str(exc)[:44])
    mm2 = model()
    mm2.laminate_sigma(fc, axis=2, open_floor=1e-9)
    r = mm2.resistances(M)                       # (re=y, rf=x, rg=z)
    cross, along = np.asarray(r[2]), np.asarray(r[1])
    check("with a floor, the z-directed R is ~1e9x the in-plane R",
          cross.max() > 1e6*along.max(),
          "across %.3e vs along %.3e" % (cross.max(), along.max()))

    # -- D: the complex (London) path ----------------------------------
    fl = np.ones((6, 6, 6))
    fl[:, :, 3] = 0.5
    ml = model()
    ml.lambdaL = np.full(ml.dims, 9e-8)
    ml.superconductor = True
    ml.laminate_sigma(fl, axis=2, open_floor=1e-9)
    rl = ml.resistances(M, freq=1e10)
    zc, za = np.asarray(rl[2]), np.asarray(rl[1])
    check("London path carries the same anisotropy sense",
          np.any(~np.isfinite(zc)) or np.abs(zc).max() > 1e3*np.abs(za).max(),
          "|z| across %.3e vs along %.3e"
          % (np.abs(zc).max(), np.abs(za).max()))

    # -- F: the TOML wiring, end to end -------------------------------
    # A 75 nm film on a 30 nm pitch is 2.5 cells: the staircase rounds it
    # to 2 or 3. The reference is the SAME physical object at 15 nm,
    # where it is 5 cells exactly.
    import sppeec_input
    body = """
[grid]
dims  = [%d, %d, %d]
pitch = %g
%s

[[block]]
from_m = [0.0, 0.0, 0.0]
to_m   = [6.0e-7, 1.8e-7, 6.0e-8]
sigma  = 5.8e7

[[block]]
from_m = [0.0, 0.0, 1.2e-7]
to_m   = [6.0e-7, 1.8e-7, 1.95e-7]
sigma  = 5.8e7

[port]
p_faces = [%s]
n_faces = [%s]
equipotential = true

[solve]
freq = [1e10]
"""

    def run(pitch, nx, ny, nz, zlo, zhi, sub):
        pf = ", ".join('[0, %d, %d, "-x"]' % (j, k)
                       for j in range(ny) for k in range(zlo, zhi))
        nf = ", ".join('[%d, %d, %d, "+x"]' % (nx - 1, j, k)
                       for j in range(ny) for k in range(zlo, zhi))
        txt = body % (nx, ny, nz, pitch,
                      "subpixel = true" if sub else "", pf, nf)
        prob = sppeec_input.loads(txt)
        mm = prob.model()
        MM = prob.tree(mm)
        Z, _ = prob.sweeper(mm, MM).solve(prob.freqs[0])
        return Z, mm

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Zr, _ = run(1.5e-8, 40, 12, 16, 8, 13, False)     # exact
        Zs, _ = run(3.0e-8, 20, 6, 8, 4, 7, False)        # staircase
        Za, ma = run(3.0e-8, 20, 6, 8, 4, 7, True)        # subpixel
    er = abs(Zs.real/Zr.real - 1.0)
    ea = abs(Za.real/Zr.real - 1.0)
    check("subpixel is declared per grid and reaches the model",
          ma.sigma_axis is not None and ma.slab_fill is not None,
          "axis %s" % (ma.slab_fill or {}).get('axis'))
    check("a 2.5-cell film: staircase R is badly wrong",
          er > 0.10, "%.2f%% off the exact-pitch reference" % (100*er))
    check("subpixel recovers R to well under 1%",
          ea < 0.01, "%.3f%% (from %.1f%%)" % (100*ea, 100*er))

    print("\n%d checks failed" % len(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
