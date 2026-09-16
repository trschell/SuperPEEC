# -*- coding: utf-8 -*-
"""Restarted GMRES with the Arnoldi basis streamed to disk.

The Krylov basis is the largest solve-time residency once the apply
side has been slimmed (memory survey 2026-09-15: 2.5 GiB on the DBC
R4, 3.4 GiB on the RSFQ XNOR, growing 0.18-0.26 GiB per lgmres
iteration). This solver keeps that basis in a file and holds only a
fixed handful of vectors in memory, so the solve-phase footprint no
longer grows with the iteration count at all:

    in memory   the current Arnoldi vector, its image under A P^-1,
                one read buffer, and the solution update (~4 vectors)
    on disk     v_0 .. v_k, one contiguous record per vector, read
                back one vector at a time for the orthogonalisation
                and for the end-of-cycle combination

Left preconditioning (Arnoldi on P^-1 A), like lgmres, and NOT the
textbook right-preconditioned form: measured on the DBC R3 at rtol
1e-4 (2026-09-15), a right-preconditioned solve stopping at the same
TRUE residual norm as lgmres left 40x more error in R (0.125% vs
0.003% against a 1e-7 reference) -- GMRES minimises the norm it
iterates in, and the preconditioned residual |P^-1 r| is close to
the error itself, so minimising it shapes the error away from the
slow global directions the impedance functional reads, whereas
minimising |r| leaves them. Termination stays on the true residual
(the user's rtol semantics): one extra matvec, and a pass over the
basis to form the iterate, whenever the running ratio of the two
norms predicts convergence. One basis is stored either way.

The default restart length is the whole matvec budget, i.e. FULL
GMRES within budget: with the basis on disk there is no memory reason
to restart, and restarting only costs matvecs. The reads grow as
k^2/2 vectors over a cycle; ``SPPEEC_STREAM_RESTART`` bounds that for
experiments.

Where the file goes: ``SPPEEC_STREAM_DIR``, else ``~/.cache/sppeec/
krylov``. NOT the system temp directory: on this box /tmp is a
tmpfs, and a basis on a RAM-backed file system is still RAM (a
warning is issued if the chosen directory is on tmpfs/ramfs). The
file is unlinked right after creation, so it lives exactly as long
as the solve and a crash leaves nothing behind. Reads and writes are
positional (pread/pwrite) into ONE reusable buffer rather than a
memory map, because pages touched through a map are charged to the
process RSS as file-backed memory -- the very figure this solver is
meant to keep flat. The kernel's page cache still serves repeated
reads from RAM while it has room, and evicts under pressure.
"""
import os
import tempfile
import time
import warnings

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import blas, lapack, solve_triangular

try:                                   # OpenMP projections (Makefile_multipole)
    import krylov_kernels as _kk
except ImportError:                    # numpy fallback, ~5x slower per vector
    _kk = None

_IO_CHUNK = 64 << 20
# the basis is orthogonalised against in BLOCKS read from the file: at
# most BLOCK_BYTES of complex64 vectors (R4: 5 x 96 MB) on READ_THREADS
# threads (page-cache copies are single-core bound at ~4 GB/s), then one
# kernel pass for the projections and one for the update
BLOCK_BYTES = int(float(os.environ.get('SPPEEC_STREAM_BLOCK_MB', '512'))*2**20)
BLOCK_MAX = 16
READ_THREADS = int(os.environ.get('SPPEEC_STREAM_READ_THREADS', '4'))
CHECK_EVERY = int(os.environ.get('SPPEEC_STREAM_CHECK_EVERY', '10'))
# the ratio drifts down as the solve closes in (R3: 0.154 -> 0.109 over
# the last 20 steps), so a prediction taken at the last check is a
# little optimistic; without a margin the R3 solve paid four
# near-miss checks (true residual 1.10, 1.03, 1.008 x tol) in a row
CHECK_MARGIN = float(os.environ.get('SPPEEC_STREAM_CHECK_MARGIN', '0.8'))
_VERBOSE = os.environ.get('SPPEEC_STREAM_VERBOSE') == '1'


def stream_dir():
    d = os.environ.get('SPPEEC_STREAM_DIR')
    if not d:
        d = os.path.join(os.path.expanduser('~'), '.cache', 'sppeec',
                         'krylov')
    os.makedirs(d, exist_ok=True)
    return d


def _on_ram_fs(path):
    """True when ``path`` lives on tmpfs/ramfs (Linux /proc/mounts)."""
    path = os.path.realpath(path)
    best, ram = -1, False
    try:
        with open('/proc/mounts') as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mp, fstype = parts[1], parts[2]
                if path == mp or path.startswith(mp.rstrip('/') + '/'):
                    if len(mp) > best:
                        best, ram = len(mp), fstype in ('tmpfs', 'ramfs')
    except OSError:
        pass
    return ram


def _pwrite_all(fd, arr, offset):
    mv = memoryview(np.ascontiguousarray(arr)).cast('B')
    pos = 0
    while pos < len(mv):
        n = os.pwrite(fd, mv[pos:pos + _IO_CHUNK], offset + pos)
        if n <= 0:
            raise OSError("short write on the streamed Krylov basis")
        pos += n


def _pread_all(fd, arr, offset):
    mv = memoryview(arr).cast('B')
    pos = 0
    while pos < len(mv):
        n = os.preadv(fd, [mv[pos:pos + _IO_CHUNK]], offset + pos)
        if n <= 0:
            raise OSError("short read on the streamed Krylov basis")
        pos += n


class BasisFile(object):
    """``v_0 .. v_{count-1}`` of one dtype in one unlinked file."""

    def __init__(self, n, dtype, workdir=None):
        self.n = int(n)
        self.dtype = np.dtype(dtype)
        self.nbytes = self.n*self.dtype.itemsize
        workdir = workdir or stream_dir()
        if _on_ram_fs(workdir):
            warnings.warn("SPPEEC_STREAM_DIR=%s is on a RAM-backed file "
                          "system: the streamed Krylov basis still "
                          "occupies memory there" % (workdir,))
        fd, path = tempfile.mkstemp(prefix='sppeec_krylov_', dir=workdir)
        os.unlink(path)
        self.fd = fd
        self.count = 0
        self.buf = np.empty(self.n, self.dtype)
        self._pool = None
        self.t_read = 0.0          # seconds inside reads (summed over threads)
        self.t_block = 0.0         # wall seconds inside read_block

    def append(self, v):
        """Write ``v`` as the next vector; returns the stored copy."""
        v = np.ascontiguousarray(v, self.dtype)
        _pwrite_all(self.fd, v, self.count*self.nbytes)
        self.count += 1
        return v

    def read(self, j, out=None):
        out = self.buf if out is None else out
        t = time.time()
        _pread_all(self.fd, out, j*self.nbytes)
        self.t_read += time.time() - t
        return out

    def read_block(self, j0, m, blk):
        """Vectors ``j0 .. j0+m-1`` into the rows of ``blk``."""
        t = time.time()
        if m <= 1 or READ_THREADS <= 1:
            for i in range(m):
                self.read(j0 + i, out=blk[i])
        else:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=READ_THREADS)
            list(self._pool.map(lambda i: self.read(j0 + i, out=blk[i]),
                                range(m)))
        self.t_block += time.time() - t

    def reset(self):
        self.count = 0

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            self.buf = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


_T_KERNEL = [0.0]                  # seconds inside the projections/updates


def _block_dots(blk, w):
    """h[i] = <blk[i], w> (conjugate-linear in blk), complex128."""
    t = time.time()
    if _kk is not None and blk.dtype == np.complex64:
        h = _kk.block_dots(blk.T, w)
    else:
        h = np.array([np.vdot(blk[i], w) for i in range(blk.shape[0])],
                     np.complex128)
    _T_KERNEL[0] += time.time() - t
    return h


def _block_update(blk, h, w):
    """w -= blk^T h, in place."""
    t = time.time()
    if _kk is not None and blk.dtype == np.complex64:
        _kk.block_update(blk.T, np.ascontiguousarray(h, np.complex128), w)
    else:
        for i in range(blk.shape[0]):
            w -= h[i]*blk[i]
    _T_KERNEL[0] += time.time() - t


def _orthogonalise(V, k, vk, w, blk, H_col):
    """MGS of ``w`` against v_0..v_{k-1} (file, in blocks) and ``vk``
    (the newest, in memory); the coefficients go to ``H_col[:k+1]``."""
    nb = blk.shape[0]
    for j0 in range(0, k, nb):
        m = min(nb, k - j0)
        V.read_block(j0, m, blk)
        hb = _block_dots(blk[:m], w)
        _block_update(blk[:m], hb, w)
        H_col[j0:j0 + m] = hb
    hk = _block_dots(vk[None, :], w)
    _block_update(vk[None, :], hk, w)
    H_col[k] = hk[0]


def _combine(V, y, n, blk, dtype=np.complex128):
    """u = sum_j y[j] v_j over the file, in blocks."""
    u = np.zeros(n, dtype)
    nb = blk.shape[0]
    for j0 in range(0, len(y), nb):
        m = min(nb, len(y) - j0)
        V.read_block(j0, m, blk)
        _block_update(blk[:m], -y[j0:j0 + m], u)
    return u


def gmres_stream(A, b, M, rtol=1e-4, budget=300, restart=None, x0=None,
                 callback=None, on_residual=None, workdir=None,
                 basis_dtype=np.complex64):
    """Left-preconditioned restarted GMRES, basis on disk.

    ``A``/``M`` expose ``matvec`` (in complex128); ``budget`` is the
    total number of operator applies allowed (the same quantity as
    lgmres's maxiter*inner_m); ``restart`` is the cycle length
    (default: the budget, i.e. full GMRES). ``callback(x)`` is called
    with each cycle's iterate, ``on_residual(relres)`` after every
    Arnoldi step with the predicted true relative residual. Returns
    ``(x, flag, nmv)`` with flag 0 = converged, 1 = budget exhausted,
    like scipy.

    PRECISION: the stored basis is ``basis_dtype`` (complex64 by
    default: each vector is rounded once, when written), but every
    piece of Arnoldi arithmetic -- inner products, norms, the
    Gram-Schmidt updates, the Hessenberg entries and the iterate --
    runs in complex128. Measured on the DBC R3 (2026-09-15): a full
    cycle in complex64 arithmetic stalled at |M r|/|b| ~ 1e-4 for
    hundreds of steps (the Arnoldi relation only held to single
    precision over a long cycle; lgmres survives in single because it
    restarts every ten steps). The in-memory working set is ~5
    complex128 vectors plus one basis_dtype read buffer, flat in the
    iteration count.

    Termination is on the TRUE residual, ``|b - A x| <= rtol |b|``,
    the same quantity lgmres tests. The Arnoldi recurrence only knows
    the PRECONDITIONED residual ``|M(b - A x)|``, so the true one is
    measured (one matvec, plus a pass over the basis to form x) when
    the running ratio of the two norms predicts convergence; the
    ratio is refreshed at every measurement. On a well-preconditioned
    system that costs one to three extra matvecs per solve.
    """
    dt = np.complex128
    b = np.asarray(b, dt)
    n = b.shape[0]
    bnorm = float(np.linalg.norm(b))
    if bnorm == 0.0:
        return np.zeros(n, dt), 0, 0
    tol = rtol*bnorm
    restart = int(budget if restart is None else min(restart, budget))
    restart = max(1, restart)
    nmv = 0
    if x0 is not None:
        x = np.array(x0, dt, copy=True)
        r = b - np.asarray(A.matvec(x), dt)
        nmv += 1
    else:
        x = np.zeros(n, dt)
        r = b.copy()
    flag = 1
    H = np.zeros((restart + 1, restart), np.complex128)
    cs = np.zeros(restart, np.float64)
    sn = np.zeros(restart, np.complex128)
    g = np.zeros(restart + 1, np.complex128)
    t_start = time.time()
    t_ops = 0.0                    # seconds inside A and M applies
    nchecks = 0
    _T_KERNEL[0] = 0.0
    with BasisFile(n, basis_dtype, workdir) as V:
        nb = max(1, min(BLOCK_MAX, BLOCK_BYTES//(n*V.dtype.itemsize)))
        blk = np.empty((nb, n), V.dtype)
        while True:
            beta = float(np.linalg.norm(r))
            if beta <= tol:
                flag = 0
                break
            if nmv >= budget:
                break
            V.reset()
            vk = np.asarray(M.matvec(r), dt)
            del r
            betap = float(np.linalg.norm(vk))
            ratio = betap/beta            # |M r| / |r|, refreshed below
            vk /= betap
            # vk is kept in the STORED precision from here on, so the
            # Arnoldi relation is exact for what the file holds
            vk = V.append(vk)
            H[:] = 0.0
            g[:] = 0.0
            g[0] = betap
            k = 0
            k_checked = 0
            done = False
            while True:
                t = time.time()
                w = np.asarray(M.matvec(np.asarray(A.matvec(
                    np.asarray(vk, dt)), dt)), dt)
                t_ops += time.time() - t
                nmv += 1
                # modified Gram-Schmidt against the streamed basis in
                # blocks (classical within a block of orthonormal
                # vectors, modified across blocks); the newest vector
                # is still in memory
                _orthogonalise(V, k, vk, w, blk, H[:, k])
                hk = float(np.linalg.norm(w))
                H[k + 1, k] = hk
                for j in range(k):
                    t = cs[j]*H[j, k] + sn[j]*H[j + 1, k]
                    H[j + 1, k] = -np.conj(sn[j])*H[j, k] + cs[j]*H[j + 1, k]
                    H[j, k] = t
                c, s, rr = lapack.zlartg(H[k, k], H[k + 1, k])
                cs[k], sn[k] = c, s
                H[k, k] = rr
                H[k + 1, k] = 0.0
                g[k + 1] = -np.conj(s)*g[k]
                g[k] = c*g[k]
                est = abs(g[k + 1])         # |M (b - A x_k)|
                if on_residual is not None:
                    on_residual(est/ratio/bnorm)
                k += 1
                breakdown = hk <= 1e-14*betap
                if not breakdown:
                    vk = V.append(w/hk)
                cycle_end = (k >= restart or nmv >= budget or breakdown)
                # measure the true residual when the prediction says
                # converged, and in any case every CHECK_EVERY steps:
                # the ratio drifts as the residual moves into the
                # smooth directions the preconditioner amplifies most,
                # and a prediction taken from the rough initial
                # residual can be pessimistic by a large factor (an
                # early R3 run marched blind to the budget)
                if not (cycle_end or est <= tol*ratio*CHECK_MARGIN
                        or k - k_checked >= CHECK_EVERY):
                    continue
                k_checked = k
                # form the iterate and measure the true residual
                y = solve_triangular(H[:k, :k], g[:k], lower=False,
                                     check_finite=False)
                u = _combine(V, y, n, blk)
                u += x
                t = time.time()
                r = b - np.asarray(A.matvec(u), dt)
                t_ops += time.time() - t
                nmv += 1
                nchecks += 1
                true = float(np.linalg.norm(r))
                if _VERBOSE:
                    print("gmres_stream: k %d nmv %d |Mr|/|b| %.3e "
                          "predicted %.3e true %.3e ratio %.3g"
                          % (k, nmv, est/bnorm, est/ratio/bnorm,
                             true/bnorm, ratio), flush=True)
                if true <= tol:
                    x = u
                    done = True
                    break
                ratio = est/true if true > 0 else ratio
                if cycle_end or nmv >= budget:
                    x = u
                    break
                del u, r
            del vk, w
            if callback is not None:
                callback(x)
            if done:
                flag = 0
                break
        if _VERBOSE:
            print("gmres_stream: done flag %d, %d matvecs (%d residual "
                  "checks), %.0f s: operators %.0f, basis reads %.0f "
                  "(%.0f thread-s), projections %.0f, block %d x %.0f MB"
                  % (flag, nmv, nchecks, time.time() - t_start, t_ops,
                     V.t_block, V.t_read, _T_KERNEL[0], nb,
                     V.nbytes/2**20), flush=True)
    return x, flag, nmv
