# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for the c64 LEAF GATHER BUFFERS (default; SPPEEC_LEAF_FP64=1 off).

``LeafLevel.p2m``/``l2p`` contract per-filament copies of the P2M/L2P
operators. Those gathers are the largest single store in a large run
-- 10.09 GB at R4, more than the whole documented build peak -- and
they are built LAZILY on the first matvec, which is why
docs/memory_census_r4.md never saw them. Storing them complex64 with
the magnitude factored into an fp64 scalar halves them.

WHY A PLAIN .astype(complex64) WOULD BE WRONG, and what this gates:

  A  DYNAMIC RANGE. The tables carry r**n in SI metres and ynmr also
     carries m0 = mu0/(4 pi) l**2, so the smallest non-zero entry runs
     7.6 decades above float32's smallest normal at R4, 4.6 at R6, and
     UNDERFLOWS at a 1 um pitch. Underflow would silently zero the
     high-order harmonics -- a far field that just gets quietly worse,
     which no residual check catches. The normalised table must stay
     inside float32 at pitches where the raw one does not.
  B  THE SCALE IS EXACT. buffer*scale reproduces the fp64 gather to
     fp32 storage rounding, for both gather axes.
  C  THE CONTRACTION IS UNCHANGED. p2m and l2p against the fp64 path
     agree to fp32 rounding on real leaf geometry -- this is the
     operator, not just the table.
  D  MEMORY IS ACTUALLY SAVED, and built chunkwise: materialising the
     fp64 gather first would spend 3x the final buffer in transients
     and hand most of the saving back.
  E  THE ESCAPE HATCH WORKS: SPPEEC_LEAF_FP64=1 really restores fp64
     with scale 1.0, so an A/B is always available.

Run: PYTHONPATH=src python3 validation/validate_leaf_fp32.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import subprocess
import sys

import numpy as np

import levels

FAIL = []
MU0 = 4e-7*np.pi


def check(name, ok, detail=""):
    print("  %-4s %-54s %s" % ("ok" if ok else "FAIL", name, detail),
          flush=True)
    if not ok:
        FAIL.append(name)


def synth(pitch, nmax=4, ncell=324, nfil=20000, seed=0):
    """A table with the real r**n * m0 magnitude structure at `pitch`."""
    rng = np.random.default_rng(seed)
    nn = (nmax + 1)**2
    r = np.abs(rng.standard_normal(ncell))*pitch*6.0
    m0 = MU0/(4*np.pi)*pitch**2
    tab = np.empty((nn, ncell), dtype=np.complex128)
    for n in range(nmax + 1):
        for m in range(-n, n + 1):
            k = n*n + n + m
            tab[k, :] = m0*(r**n)*(rng.standard_normal(ncell)
                                   + 1j*rng.standard_normal(ncell))
    return tab, rng.integers(0, ncell, nfil)


def main():
    tiny = np.finfo(np.float32).tiny

    # -- A: dynamic range, at pitches where the RAW table underflows --
    rows = []
    for pitch in (0.5e-3, 0.125e-3, 0.0625e-3, 0.02e-3, 1e-6):
        tab, idx = synth(pitch)
        raw = np.abs(tab[tab != 0]).min()
        buf, sc = levels._gather_single(tab, idx, 1)
        got = np.abs(buf[buf != 0])
        rows.append((pitch, raw, float(got.min()) if got.size else 0.0))
    raw_bad = [p for p, r, _ in rows if r < tiny]
    norm_bad = [p for p, _, g in rows if g < tiny or g == 0.0]
    check("raw tables really do underflow at fine pitch "
          "(the trap is real)", bool(raw_bad),
          "raw min %.2e at %g m vs f32 tiny %.2e"
          % (rows[-1][1], rows[-1][0], tiny))
    check("normalised c64 table stays inside float32 at every pitch",
          not norm_bad,
          "worst normalised min %.2e" % min(g for _, _, g in rows))

    # -- B: the scale reproduces the fp64 gather, both axes ----------
    tab, idx = synth(0.0625e-3)
    worst = 0.0
    for axis in (0, 1):
        src = tab.T.copy() if axis == 0 else tab
        ref = src[idx, :] if axis == 0 else src[:, idx]
        buf, sc = levels._gather_single(src, idx, axis)
        err = (np.abs(buf.astype(np.complex128)*sc - ref).max()
               / np.abs(ref).max())
        worst = max(worst, err)
        check("axis %d gather is c64 with an fp64 scale" % axis,
              buf.dtype == np.complex64 and sc > 0.0,
              "dtype %s scale %.4g" % (buf.dtype, sc))
    check("buffer*scale == fp64 gather to fp32 rounding",
          worst < 1e-6, "max rel err %.2e" % worst)

    # -- C: the CONTRACTION, not just the table ----------------------
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((idx.size,))
         + 1j*rng.standard_normal((idx.size,)))
    ref64 = tab[:, idx] @ x
    buf, sc = levels._gather_single(tab, idx, 1)
    got = sc*(buf @ x)
    err = np.abs(got - ref64).max()/np.abs(ref64).max()
    check("p2m-style contraction matches fp64", err < 1e-6,
          "rel %.2e" % err)

    src = tab.T.copy()
    y = (rng.standard_normal(tab.shape[0])
         + 1j*rng.standard_normal(tab.shape[0]))
    ref64 = src[idx, :] @ y
    buf, sc = levels._gather_single(src, idx, 0)
    err = np.abs(sc*(buf @ y) - ref64).max()/np.abs(ref64).max()
    check("l2p-style contraction matches fp64", err < 1e-6,
          "rel %.2e" % err)

    # -- D: memory halves, and the BUILD does not spend it back ------
    buf, _ = levels._gather_single(tab, idx, 1)
    ref = np.ascontiguousarray(tab[:, idx])
    check("buffer is half the fp64 gather",
          buf.nbytes*2 == ref.nbytes,
          "%d vs %d bytes" % (buf.nbytes, ref.nbytes))
    big = np.random.default_rng(2).integers(0, 324, 3_000_000)
    import tracemalloc
    tracemalloc.start()
    b2, _ = levels._gather_single(tab, big, 1)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    check("gather built chunkwise (transient < 1.4x the buffer)",
          peak < 1.4*b2.nbytes,
          "peak %.0f MB for a %.0f MB buffer"
          % (peak/1e6, b2.nbytes/1e6))
    del b2

    # -- E: the escape hatch -----------------------------------------
    r = subprocess.run(
        [sys.executable, '-c',
         "import sys; sys.path.insert(0,'src'); import levels, numpy as np;"
         "t=np.ones((4,8),dtype=np.complex128);"
         "b,s=levels._gather_single(t,np.arange(8),1);"
         "print(b.dtype, s)"],
        capture_output=True, text=True,
        cwd=_op.path.join(_op.path.dirname(_op.path.abspath(__file__)),
                          '..'),
        env=dict(_op.environ, SPPEEC_LEAF_FP64='1'))
    check("SPPEEC_LEAF_FP64=1 restores fp64 with scale 1.0",
          'complex128' in r.stdout and '1.0' in r.stdout,
          r.stdout.strip() or r.stderr.strip()[:50])

    print("\n%d checks failed" % len(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
