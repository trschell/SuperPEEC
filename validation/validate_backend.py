# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Acceptance for the device backend seam and the OpenCL operator.

Three things are checked.

*Selection.* ``SPPEEC_BACKEND`` and the ``SPPEEC_GPU`` master gate must
resolve the way the documented table says, and the default must keep
choosing CUDA-or-host so that no existing install changes behaviour by
installing PyOpenCL. Each case runs in its own process because the
resolution is cached.

*Agreement.* The OpenCL near field and top-level M2L must reproduce the
host Fortran kernels. They are not bitwise equal and cannot be: the
transforms are a different library and the contraction a different
order, so the bar is fp64 rounding.

*Reproducibility.* The OpenCL kernels must return exactly the same bits
on a repeated call with the same input. This is not a nicety. A
preconditioner whose map drifts between applies breaks the Arnoldi
relation of a long GMRES cycle, which is how the streamed basis stalled
on R4; the CUDA near field cannot make this promise because it reduces
with atomics, and the OpenCL kernels are written output-centric so that
they can.

Skips cleanly (exit 0) where the backend under test is absent.

Run inside the toolbox:  python3 validate_backend.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import os
import subprocess
import sys

import numpy as np

import multipole as mp

FAIL = []
CELL = 1e-5


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name
          + ("  " + detail if detail else ""), flush=True)
    if not ok:
        FAIL.append(name)


def resolved(env):
    """backend.name() in a fresh process with ``env`` applied."""
    e = dict(os.environ)
    e.pop('SPPEEC_BACKEND', None)
    e.pop('SPPEEC_GPU', None)
    e.update(env)
    e['PYTHONPATH'] = (_op.path.join(_op.path.dirname(
        _op.path.abspath(__file__)), '../src') + os.pathsep
        + e.get('PYTHONPATH', ''))
    out = subprocess.run(
        [sys.executable, '-c',
         'import warnings; warnings.simplefilter("ignore");'
         ' import backend; print(backend.name())'],
        capture_output=True, text=True, env=e)
    return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else \
        ('ERROR: ' + out.stderr.strip()[-200:])


def selection_cases():
    check("SPPEEC_GPU=0 disables every backend",
          resolved({'SPPEEC_GPU': '0'}) == 'none')
    check("SPPEEC_GPU=0 wins over an explicit backend",
          resolved({'SPPEEC_GPU': '0', 'SPPEEC_BACKEND': 'opencl'}) == 'none')
    check("SPPEEC_BACKEND=none disables every backend",
          resolved({'SPPEEC_BACKEND': 'none'}) == 'none')
    got = resolved({})
    check("the default never selects OpenCL implicitly", got != 'opencl',
          "default resolved to %r" % got)
    check("the default is cuda or none", got in ('cuda', 'none'),
          "default resolved to %r" % got)
    got = resolved({'SPPEEC_BACKEND': 'nonsense'})
    check("an unknown backend name falls back to auto",
          got in ('cuda', 'none'), "resolved to %r" % got)


def small_tree():
    """A small inductive tree with a usable top level."""
    f = np.ones((8, 8, 8), np.int8)
    f[7, 7, 7] = 0
    return mp.Tree(f, np.array([4, 4, 4]), np.array([8, 8, 8])*CELL,
                   2, 1e0, 2)


def opencl_present():
    """True when the box has an OpenCL GPU, whatever the default is."""
    try:
        import pyopencl as cl
        for plat in cl.get_platforms():
            if plat.get_devices(cl.device_type.GPU):
                return True
    except Exception:
        pass
    return False


def operator_cases_in_child():
    """Run the operator checks in a process that selects OpenCL.

    The suite runs every validator at the default backend, which is
    CUDA here, and scores any validator that prints a SKIP line as
    untested. So rather than skipping whenever OpenCL is not the
    default, this re-runs itself with the backend selected, and only
    reports a genuine skip when the box has no OpenCL device at all.
    """
    if not opencl_present():
        print('SKIP: no OpenCL GPU device -- the operator agreement and '
              'reproducibility checks need one')
        return
    e = dict(os.environ)
    e['SPPEEC_BACKEND'] = 'opencl'
    e.pop('SPPEEC_GPU', None)
    e['SPPEEC_BACKEND_CHILD'] = '1'
    out = subprocess.run([sys.executable, _op.path.abspath(__file__)],
                         capture_output=True, text=True, env=e)
    for line in out.stdout.splitlines():
        if line.startswith(('PASS  ', 'FAIL  ')):
            print(line, flush=True)
            if line.startswith('FAIL  '):
                FAIL.append(line[6:].split('  ')[0])
    if out.returncode != 0 and not FAIL:
        FAIL.append('opencl operator child (rc=%d): %s'
                    % (out.returncode, out.stderr.strip()[-300:]))


def mode_cases():
    """The mode-block apply, against the host path on a small model.

    Needs a model that actually builds mode spectra, so it solves the
    enriched bar example at its lowest frequency with the device apply
    switched off, keeps the enrichment, and then applies both ways.
    The bar is tolerant: the spectra are stored complex64, so the two
    paths agree to about 1e-8, not to fp64 rounding.
    """
    import numpy as _np
    model = _op.path.join(_op.path.dirname(_op.path.abspath(__file__)),
                          '..', 'examples', 'equibar.toml')
    if not _op.path.exists(model):
        return
    keep = os.environ.get('SPPEEC_MODE_APPLY_GPU')
    os.environ['SPPEEC_MODE_APPLY_GPU'] = '0'       # host reference path
    found = []
    try:
        import sppeec_input
        import enrich
        orig = enrich.Enrichment.apply_fft

        def spy(self, u, i_f):
            if not found and getattr(self, 'Fu', None) is not None:
                found.append(self)
            return orig(self, u, i_f)
        enrich.Enrichment.apply_fft = spy
        try:
            pr = sppeec_input.load(model)
            m = pr.model()
            M = pr.tree(m)
            pr.sweeper(m, M).solve(float(pr.freqs[0]))
        finally:
            enrich.Enrichment.apply_fft = orig
    except Exception as exc:
        check("mode apply: enriched model builds", False,
              "%s: %s" % (type(exc).__name__, exc))
        return
    finally:
        if keep is None:
            os.environ.pop('SPPEEC_MODE_APPLY_GPU', None)
        else:
            os.environ['SPPEEC_MODE_APPLY_GPU'] = keep
    if not found:
        check("mode apply: enriched model builds spectra", False,
              "no Enrichment with spectra was applied")
        return
    enr = found[0]
    ncell = int(enr._g3[0].size)
    rng = _np.random.default_rng(11)
    u = (rng.standard_normal(enr.nmode)
         + 1j*rng.standard_normal(enr.nmode)).astype(_np.complex128)
    i_f = (rng.standard_normal(ncell)
           + 1j*rng.standard_normal(ncell)).astype(_np.complex128)
    ru, rf = enr.apply_fft(u, i_f)
    import ocl_modes
    op = ocl_modes.ModeApply(enr)
    gu, gf = op.apply(u, i_f)
    eu = float(_np.abs(gu - ru).max())/max(1e-300, float(_np.abs(ru).max()))
    ef = float(_np.abs(gf - rf).max())/max(1e-300, float(_np.abs(rf).max()))
    check("mode apply: OpenCL agrees with the host path", eu < 1e-6
          and ef < 1e-6, "rel err modes=%.3e filaments=%.3e" % (eu, ef))
    check("mode apply: repeated call is bit-identical",
          all(_np.array_equal(a, b)
              for a, b in zip((gu, gf), op.apply(u, i_f))))


def wire_cases():
    """The wire-bond particular current, device against host.

    A synthetic wire graph: a spanning tree plus chords, which is the
    shape the bond-wire assembly produces. Both solves run the same
    preconditioned conjugate gradient on the same graph Laplacian, so
    they agree to solver tolerance, and the device one must also report
    a clean KCL residual.
    """
    import numpy as _np
    import scipy.sparse as _sp
    import ocl_wire
    from wireassembly import _laplacian_current_cpu
    rng = _np.random.default_rng(41)
    nn = 4000
    parent = _np.full(nn, -1, _np.int64)
    parent[1:] = rng.integers(0, _np.arange(1, nn))      # a random tree
    edges = [(int(parent[i]), i) for i in range(1, nn)]
    for _ in range(600):                                  # chords
        a, b = int(rng.integers(0, nn)), int(rng.integers(0, nn))
        if a != b:
            edges.append((a, b))
    ne = len(edges)
    rows = _np.repeat(_np.arange(ne), 2)
    cols = _np.array([c for e in edges for c in e], _np.int64)
    vals = _np.tile(_np.array([-1.0, 1.0]), ne)
    B = _sp.csr_matrix((vals, (rows, cols)), shape=(ne, nn))
    rhs = rng.standard_normal(nn)
    rhs[parent < 0] = 0.0
    rhs -= rhs.mean()

    got, resid = ocl_wire.laplacian_current(B, parent, rhs)
    ref, resid_h = _laplacian_current_cpu(B, parent, rhs)
    check("wire Laplacian: device residual is clean", resid < 1e-9,
          "max |B^T ihat - rhs| = %.2e (host %.2e)" % (resid, resid_h))
    scale = max(1e-30, float(_np.abs(ref).max()))
    rel = float(_np.abs(got - ref).max())/scale
    check("wire Laplacian: device agrees with the host solve",
          rel < 1e-6, "rel diff=%.3e" % rel)
    again, _ = ocl_wire.laplacian_current(B, parent, rhs)
    check("wire Laplacian: repeated solve is bit-identical",
          _np.array_equal(got, again))


def sparse_product_cases():
    """The device sparse-sparse product and transpose, against scipy.

    Both are build-phase products: the Gram when no certified stencil
    is available, and a Galerkin coarse operator. They must agree with
    scipy exactly, structure included, so a hierarchy does not depend
    on which backend assembled it.
    """
    import numpy as _np
    import scipy.sparse as _sp
    import ocl_sparse
    rng = _np.random.default_rng(31)
    A = _sp.random(4000, 3000, 0.002, format='csr', random_state=1)
    B = _sp.random(3000, 2500, 0.002, format='csr', random_state=2)
    try:
        C = ocl_sparse.spgemm(A, B, _np.float32, maxc=128)
    except ocl_sparse.SpGEMMOverflow as exc:
        check("sparse product: random case", False, str(exc))
        return
    ref = (A.astype(_np.float32) @ B.astype(_np.float32)).tocsr()
    ref.sort_indices()
    C.sort_indices()
    check("sparse product: structure and values match scipy",
          C.shape == ref.shape and C.nnz == ref.nnz
          and _np.array_equal(C.indices, ref.indices)
          and _np.array_equal(C.indptr, ref.indptr)
          and float(abs(C - ref).max()) == 0.0,
          "nnz %d vs %d" % (C.nnz, ref.nnz))

    # a plaquette-like basis: four entries of plus or minus one per row
    nr, nc = 20000, 9000
    rows = _np.repeat(_np.arange(nr), 4)
    cols = rng.integers(0, nc, nr*4)
    vals = rng.choice([-1.0, 1.0], nr*4)
    Y = _sp.csr_matrix((vals, (rows, cols)), shape=(nr, nc))
    Y.sum_duplicates()
    YT = ocl_sparse.transpose(Y, _np.float32)
    refT = _sp.csr_matrix(Y.T.astype(_np.float32))
    refT.sort_indices()
    YT.sort_indices()
    check("sparse transpose: matches scipy exactly",
          YT.shape == refT.shape and YT.nnz == refT.nnz
          and _np.array_equal(YT.indices, refT.indices)
          and float(abs(YT - refT).max()) == 0.0)
    try:
        G = ocl_sparse.spgemm(Y, YT, _np.float32, maxc=128)
    except ocl_sparse.SpGEMMOverflow as exc:
        check("sparse product: Gram case", False, str(exc))
        return
    refG = (Y.astype(_np.float32) @ refT).tocsr()
    refG.sort_indices()
    G.sort_indices()
    check("sparse product: a Gram matches scipy exactly",
          G.nnz == refG.nnz and _np.array_equal(G.indices, refG.indices)
          and float(abs(G - refG).max()) == 0.0,
          "nnz %d vs %d" % (G.nnz, refG.nnz))
    G2 = ocl_sparse.spgemm(Y, YT, _np.float32, maxc=128)
    check("sparse product: repeated build is bit-identical",
          _np.array_equal(G.data, G2.data)
          and _np.array_equal(G.indices, G2.indices))


def precond_cases():
    """The multigrid preconditioner apply, against the host apply.

    Both are the same V-cycle on the same hierarchy in float32, so they
    agree to rounding; the device one must also repeat exactly, which
    is the property a long GMRES cycle depends on and the reason the
    products are hand-written.

    ``SPPEEC_KEEP_HOST_COPIES`` keeps the host stencil path alive next
    to the device one so both can be applied, the same arrangement the
    CUDA GeoMG validator uses.
    """
    import numpy as _np
    model = _op.path.join(_op.path.dirname(_op.path.abspath(__file__)),
                          '..', 'examples', 'dbc_halfbridge.toml')
    if not _op.path.exists(model):
        return
    keep = os.environ.get('SPPEEC_KEEP_HOST_COPIES')
    os.environ['SPPEEC_KEEP_HOST_COPIES'] = '1'
    seen = []

    class _Stop(Exception):
        pass
    try:
        import sppeec_input
        import port_impedance as pi
        orig = pi._GeoMGFactor.__init__

        def spy(self, *a, **k):
            orig(self, *a, **k)
            seen.append(self)
            raise _Stop()                 # the factor is all we need
        pi._GeoMGFactor.__init__ = spy
        try:
            pr = sppeec_input.load(model)
            m = pr.model()
            M = pr.tree(m)
            pr.sweeper(m, M).solve(float(pr.freqs[0]))
        except _Stop:
            pass
        finally:
            pi._GeoMGFactor.__init__ = orig
    except Exception as exc:
        check("preconditioner: factor builds", False,
              "%s: %s" % (type(exc).__name__, exc))
        return
    finally:
        if keep is None:
            os.environ.pop('SPPEEC_KEEP_HOST_COPIES', None)
        else:
            os.environ['SPPEEC_KEEP_HOST_COPIES'] = keep
    if not seen:
        return
    f = seen[0]
    blk = f._gpu
    check("preconditioner: the OpenCL block is the one built",
          blk is not None and type(blk).__module__ == 'ocl_geomg',
          "gpu_state=%r" % getattr(f, 'gpu_state', None))
    if blk is None:
        return
    # this model takes the stencil path, so level 0 must be applied as
    # a stencil and not fall back to forming the Gram on the host
    check("preconditioner: level 0 is applied as a stencil",
          getattr(blk.core, 'level0', None) == 'stencil',
          "level0=%r" % getattr(blk.core, 'level0', None))
    sten = getattr(f.mg, '_sten0', None)
    if sten is not None:
        import ocl_core as _oc
        import ocl_stencil as _os
        op = _os.Stencil0(sten, f.mg._wdi0_t, blk.core.dtype)
        rng0 = _np.random.default_rng(29)
        xv = rng0.standard_normal(sten.n).astype(sten.dtype)
        ref = sten.matvec(xv)
        xd = _oc.to_device(xv)
        yd = _oc.zeros((sten.n,), sten.dtype)
        got = op.spmv(xd, yd).get()
        check("level-0 stencil: OpenCL matches the host kernel exactly",
              _np.array_equal(got, ref),
              "max diff=%.3e" % float(_np.abs(got - ref).max()))
    rng = _np.random.default_rng(17)
    b = rng.standard_normal(f.n).astype(_np.float32)
    f._gpu = None                          # the host apply, for reference
    ref = f(b)
    f._gpu = blk
    got = blk(b)
    rel = float(_np.abs(got - ref).max())/max(1e-30,
                                              float(_np.abs(ref).max()))
    check("preconditioner: OpenCL apply agrees with the host apply",
          rel < 1e-5, "rel diff=%.3e" % rel)
    check("preconditioner: repeated apply is bit-identical",
          _np.array_equal(got, blk(b)))


def operator_cases():
    import backend
    if backend.name() != 'opencl':
        FAIL.append('operator child did not select OpenCL (got %r)'
                    % backend.name())
        return
    import ocl_m2l
    import ocl_p2p
    M = small_tree()
    rng = np.random.default_rng(7)

    for nm in ('e', 'f', 'g'):
        lf = getattr(M, nm, None)
        if lf is None or getattr(lf, 'p2p_transfer', None) is None:
            continue
        # the bare tree builds the near-field tables but leaves the
        # filament buffer to the model's prepare; one per index entry
        nfil = int(np.size(lf.idx))
        data = (rng.standard_normal(nfil)
                + 1j*rng.standard_normal(nfil)).astype(np.complex128)
        lf.data = data.copy()
        lf.p2pcpu()
        ref = lf.data.copy()
        op = ocl_p2p.NearField(lf)
        got = op.apply(data)
        scale = max(1e-300, float(np.abs(ref).max()))
        err = float(np.abs(got - ref).max())/scale
        check("near field %s: OpenCL agrees with the host kernel" % nm,
              err < 1e-12, "rel err=%.3e" % err)
        again = op.apply(data)
        check("near field %s: repeated call is bit-identical" % nm,
              np.array_equal(got, again))

    mode_cases()
    wire_cases()
    sparse_product_cases()
    precond_cases()

    top = M.lv[int(M.numlevels) - 1]
    data = (rng.standard_normal(top.data.shape)
            + 1j*rng.standard_normal(top.data.shape)).astype(np.complex128)
    top.data = data.copy()
    top.m2lfortran()
    ref = top.data.copy()
    op = ocl_m2l.TopM2L(top)
    got = op.apply(data)
    scale = max(1e-300, float(np.abs(ref).max()))
    err = float(np.abs(got - ref).max())/scale
    check("top-level M2L: OpenCL agrees with the host kernel", err < 1e-12,
          "rel err=%.3e" % err)
    check("top-level M2L: repeated call is bit-identical",
          np.array_equal(got, op.apply(data)))


if os.environ.get('SPPEEC_BACKEND_CHILD') == '1':
    operator_cases()                     # the OpenCL half, in its own process
else:
    selection_cases()
    operator_cases_in_child()
    print("\n" + ("ALL PASS" if not FAIL else "FAILURES: " + ", ".join(FAIL)))
sys.exit(1 if FAIL else 0)
