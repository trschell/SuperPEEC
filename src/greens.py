# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""PEEC Green's-function / partial-element kernels and octree node sizing.

Closed-form partial-element expressions for the PEEC (Partial Element
Equivalent Circuit) discretization on a uniform rectangular grid:

* :func:`genL3D`   -- partial *inductance* between current filaments (uses
  the magnetic constant mu0); this feeds the inductive ``Lp`` path.
* :func:`p_self`, :func:`p_par`, :func:`gen_p_par`, :func:`gen_p_parz`,
  :func:`gen2_p_parz`, :func:`gen_p_per` -- coefficients of *potential*
  between charge panels (the capacitive ``P`` path; each uses the electric
  constant eps0). "par"/"per" denote panels that are parallel or
  perpendicular to one another.

Every kernel is translation-invariant on the grid, so it returns the value
as a function of the integer separation between two elements -- i.e. the
generating vector/array of a (block-)Toeplitz partial-element matrix, which
the FMM/FFT machinery then applies without ever forming the dense matrix.

sizing the tree. Split out of multipole.py; depends only on numpy.
"""
# Thread defaults -- see sppeec_threads.py. The library defaults cost
# 6.3x on the FMM path (OpenBLAS spawning a dozen threads for each of
# many small gemv calls) and 1.8x even on the dense LpPR path, and the
# penalty is LARGEST on the coarse meshes the skin/corner studies use.
# The runtime call works whatever the import order; the environment
# block inside the module covers OMP and FFTW for anyone importing
# early.
try:
    import sppeec_threads as _spthreads
    _spthreads.enforce_blas()
except Exception:                    # tuning must never break a solve
    _spthreads = None

import numpy as np


def genL3D(dw, dl, dt, NW, NL, NT):
    """Generate the partial mutual-inductance Toeplitz vector for filaments.

    Builds the array ``M[i, j, k]`` of partial mutual inductances between two
    identical rectangular current filaments (cross section ``dw x dt``, length
    ``dl``) separated by ``i``, ``j``, ``k`` filament steps along width,
    length and thickness. Because the geometry is translation-invariant this
    single array is the generating block of the (three-level) Toeplitz partial
    inductance matrix.

    The routine first evaluates the exact self-inductance ``L`` for every
    filament size from ``1 x 1 x 1`` up to ``NW x NL x NT`` (closed form), then
    forms the mutual inductances by the finite-difference / superposition
    identity of Zhong et al., "Exact Closed Form Formula for Partial Mutual
    Inductances of On-Chip Interconnects".

    Parameters
    ----------
    dw, dl, dt : float
        Filament dimensions: width, length, thickness (metres).
    NW, NL, NT : int
        Number of filament steps of separation to tabulate along width,
        length and thickness.

    Returns
    -------
    ndarray, shape (NW, NL, NT)
        Partial mutual inductance as a function of integer separation
        (henries), normalized by ``dw**2 * dt**2``.
    """
    # NW = 11
    # NL = 11
    # NT = 5
    # dw = 50e-6/NW
    # dl = 50e-6/NL
    # dt = 10e-6/NT

    mu0 = 4*np.pi*1e-7

    # This first code block generates a 2-D array of self-inductances whose
    # filaments range in size from the smallest, a regular filament, to the
    # largest, a filament that is as large as the plate being evaluated
    # (NW*dw by NL*dl).
    [W, l, T] = np.meshgrid(np.r_[1:NW+1], np.r_[1:NL+1], np.r_[1:NT+1],
                            indexing='ij')
    W = dw*W.astype(float)
    l = dl*l.astype(float)
    T = dt*T.astype(float)
    L = box_selfind(W, l, T)
    Ltemp = np.zeros((NW+1, NL+1, NT+1))
    Ltemp[1:, 1:, 1:] = L
    L = Ltemp
    return _mutual_stencil(L, NW, NL, NT)/(dw**2*dt**2)


def box_selfind(W, l, T):
    """Self partial inductance of a rectangular bar, times ``W**2 T**2``.

    The exact closed form (Zhong et al.) for a bar of cross-section
    ``W x T`` and length ``l``, pre-multiplied by ``W**2 T**2`` -- the
    scaling in which the mutual inductances are tensor second differences
    of it, which is how :func:`genL3D` uses it.

    Factored out of ``genL3D`` (byte-identical: same expressions, same
    order) so it can be evaluated at lengths that are NOT multiples of
    the lattice pitch. That is what an arbitrary-length port terminal
    needs: the mutual between bars of UNEQUAL axial extent is the same
    stencil with the axial second difference replaced by the mixed
    difference ``1/2 [Psi(s+q) - Psi(s) - Psi(s+q-p) + Psi(s-p)]``, whose
    four arguments are generally off-lattice. See
    ``terminal.unequal_kernel``.

    Any zero dimension gives zero (a degenerate bar), matching the
    ``Ltemp[1:, 1:, 1:]`` padding genL3D relies on.

    Parameters
    ----------
    W, l, T : ndarray or float
        Width, length and thickness (metres), broadcastable.

    Returns
    -------
    ndarray
    """
    mu0 = 4*np.pi*1e-7
    W, l, T = np.broadcast_arrays(np.asarray(W, dtype=float),
                                  np.asarray(l, dtype=float),
                                  np.asarray(T, dtype=float))
    good = (W > 0) & (l > 0) & (T > 0)
    if not good.all():
        out = np.zeros(np.shape(good))
        if good.any():
            out[good] = box_selfind(W[good], l[good], T[good])
        return out
    w = W/l
    t = T/l

    r = np.sqrt(w**2 + t**2)
    aw = np.sqrt(w**2 + 1)
    at = np.sqrt(t**2 + 1)
    ar = np.sqrt(w**2 + t**2 + 1)

    L = 1/4 * (1/w*np.arcsinh(w/at) + 1/t*np.arcsinh(t/aw) + np.arcsinh(1/r))
    L = L + 1/24*(t**2/w*np.arcsinh(w/(t*at*(r + ar))) +
                  w**2/t*np.arcsinh(t/(w*aw*(r + ar))))
    L = L + 1/24*(t**2/(w**2)*np.arcsinh(w**2/(t*r*(at + ar))) +
                  w**2/(t**2)*np.arcsinh(t**2/(w*r*(aw + ar))))
    L = L + 1/24*(1/(w*t**2)*np.arcsinh(w*t**2/(at*(aw + ar))) +
                  1/(t*w**2)*np.arcsinh(t*w**2/(aw*(at + ar))))
    L = L - 1/6*(1/(w*t)*np.arctan(w*t/ar) + t/w*np.arctan(w/(t*ar)) +
                 w/t*np.arctan(t/(w*ar)))
    L = L - 1/60*((ar + r + t + at)*t**2/((ar + r)*(r + t)*(t + at)*(at + ar)))
    L = L - 1/60*((ar + r + w + aw)*w**2/((ar + r)*(r + w)*(w + aw)*(aw + ar)))
    L = L - 1/60*((ar + aw + 1 + at)/((ar + aw)*(aw + 1)*(1 + at)*(at + ar)))
    L = L - 1/20*(1/(r + ar) + 1/(aw + ar) + 1/(at + ar))
    L = 2*mu0/np.pi*L*l
    return W**2*T**2*L


def box_pair_stencil(lo, hi, chunk=256):
    """Mixed second difference of :func:`box_selfind` over box pairs.

    The double integral ``I = Int_a Int_b dV dV'/|r-r'|`` between two
    axis-aligned boxes is a mixed second difference, in EACH axis, of the
    box function. For an axis where box A spans ``[0,p]`` and B spans
    ``[s,s+q]`` the four terms are ``|s+q|, |s|, |s+q-p|, |s-p|`` with
    coefficients ``+1/2,-1/2,-1/2,+1/2``; the tensor product over three
    axes gives ``4**3 = 64`` terms. With equal boxes on a lattice each
    axis collapses to genL3D's symmetric ``1/2 [1,-2,1]``.

    Returns ``S = (mu0/4pi) * I``, the quantity both physical constants
    are reached from:

        partial mutual inductance   Mp = S/(A_a A_b)
        coefficient of potential    P  = S*c**2/(V_a V_b)

    Parameters
    ----------
    lo, hi : ndarray, shape (n, 3)
        Lower and upper corners of each box (metres).
    chunk : int, optional
        Row block size, to bound peak memory.

    Returns
    -------
    ndarray, shape (n, n)
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    n = lo.shape[0]
    ext = hi - lo
    coef = np.array([0.5, -0.5, -0.5, 0.5])
    S = np.zeros((n, n))
    for a0 in range(0, n, chunk):
        a1 = min(a0 + chunk, n)
        lens = []
        for d in range(3):
            p = ext[a0:a1, d][:, None]
            q = ext[:, d][None, :]
            s = lo[None, :, d] - lo[a0:a1, None, d]
            lens.append(np.stack([np.abs(s + q), np.abs(s),
                                  np.abs(s + q - p), np.abs(s - p)],
                                 axis=-1))
        blk = np.zeros((a1 - a0, n))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    blk += (coef[i]*coef[j]*coef[k])*box_selfind(
                        lens[0][:, :, i], lens[1][:, :, j],
                        lens[2][:, :, k])
        S[a0:a1] = blk
    return S


def box_pair_stencil_pairs(loa, hia, lob, hib):
    """:func:`box_pair_stencil` for an explicit LIST of box pairs.

    ``loa/hia`` and ``lob/hib`` are aligned ``(n, 3)`` arrays; entry ``i``
    is the pair ``(A_i, B_i)``. This is the primitive for assembling
    TRUNCATED (sparse) blocks: the dense form costs ``O(n**2)`` boxes
    even when only near pairs are wanted.

    Returns ``S`` as a ``(n,)`` array; see :func:`box_pair_stencil` for
    the constants that turn it into inductance or potential.
    """
    loa = np.asarray(loa, dtype=float)
    hia = np.asarray(hia, dtype=float)
    lob = np.asarray(lob, dtype=float)
    hib = np.asarray(hib, dtype=float)
    lens = []
    for d in range(3):
        p = hia[:, d] - loa[:, d]
        q = hib[:, d] - lob[:, d]
        s = lob[:, d] - loa[:, d]
        lens.append(np.stack([np.abs(s + q), np.abs(s),
                              np.abs(s + q - p), np.abs(s - p)], axis=-1))
    coef = np.array([0.5, -0.5, -0.5, 0.5])
    S = np.zeros(loa.shape[0])
    for i in range(4):
        for j in range(4):
            for k in range(4):
                S += (coef[i]*coef[j]*coef[k])*box_selfind(
                    lens[0][:, i], lens[1][:, j], lens[2][:, k])
    return S


def _mutual_stencil(L, NW, NL, NT):
    """Tensor second difference of the self-inductance table.

    The mutual inductance between identical parallel bars at integer
    separations is the product of three ``1/2 [1, -2, 1]`` operators
    applied to :func:`box_selfind`, one per axis -- 27 terms with
    coefficients -1 / +0.5 (6) / -0.25 (12) / +0.125 (8).

    ``L`` is the ZERO-PADDED table, shape ``(NW+1, NL+1, NT+1)``, index 0
    holding the degenerate (zero-extent) bar.
    """
    # This block of code adds and subtracts various self inductances to compute
    # the mutual inductance according to the paper "Exact Closed Form Formula
    # for Partial Mutual Inductances of On-Chip Interconnects" by Guoan Zhong
    M = np.zeros((NW, NL, NT))
    M[0, 0, 0] = L[1, 1, 1]
    M[0, 0, 1:NT] = -L[1, 1, 1:NT] + 0.5*(L[1, 1, :NT-1] + L[1, 1, 2:NT+1])
    M[0, 1:NL, 0] = -L[1, 1:NL, 1] + 0.5*(L[1, :NL-1, 1] + L[1, 2:NL+1, 1])
    M[1:NW, 0, 0] = -L[1:NW, 1, 1] + 0.5*(L[:NW-1, 1, 1] + L[2:NW+1, 1, 1])
    M[0, 1:NL, 1:NT] = L[1, 1:NL, 1:NT] - \
        0.5*(L[1, 1:NL, :NT-1] + L[1, 1:NL, 2:NT+1] +
             L[1, :NL-1, 1:NT] + L[1, 2:NL+1, 1:NT]) + \
        0.25*(L[1, :NL-1, :NT-1] + L[1, 2:NL+1, 2:NT+1] +
              L[1, :NL-1, 2:NT+1] + L[1, 2:NL+1, :NT-1])
    M[1:NW, 0, 1:NT] = L[1:NW, 1, 1:NT] - \
        0.5*(L[1:NW, 1, :NT-1] + L[1:NW, 1, 2:NT+1] +
             L[:NW-1, 1, 1:NT] + L[2:NW+1, 1, 1:NT]) + \
        0.25*(L[:NW-1, 1, :NT-1] + L[2:NW+1, 1, 2:NT+1] +
              L[:NW-1, 1, 2:NT+1] + L[2:NW+1, 1, :NT-1])
    M[1:NW, 1:NL, 0] = L[1:NW, 1:NL, 1] - \
        0.5*(L[1:NW, :NL-1, 1] + L[1:NW, 2:NL+1, 1] +
             L[:NW-1, 1:NL, 1] + L[2:NW+1, 1:NL, 1]) + \
        0.25*(L[:NW-1, :NL-1, 1] + L[2:NW+1, 2:NL+1, 1] +
              L[:NW-1, 2:NL+1, 1] + L[2:NW+1, :NL-1, 1])
    M[1:NW, 1:NL, 1:NT] = -L[1:NW, 1:NL, 1:NT] + \
        0.5*(L[1:NW, 1:NL, :NT-1] + L[1:NW, :NL-1, 1:NT] +
             L[:NW-1, 1:NL, 1:NT] + L[1:NW, 1:NL, 2:NT+1] +
             L[1:NW, 2:NL+1, 1:NT] + L[2:NW+1, 1:NL, 1:NT]) - \
        0.25*(L[:NW-1, :NL-1, 1:NT] + L[:NW-1, 1:NL, :NT-1] +
              L[1:NW, :NL-1, :NT-1] + L[2:NW+1, 2:NL+1, 1:NT] +
              L[2:NW+1, 1:NL, 2:NT+1] + L[1:NW, 2:NL+1, 2:NT+1] +
              L[:NW-1, 1:NL, 2:NT+1] + L[:NW-1, 2:NL+1, 1:NT] +
              L[1:NW, :NL-1, 2:NT+1] + L[2:NW+1, :NL-1, 1:NT] +
              L[1:NW, 2:NL+1, :NT-1] + L[2:NW+1, 1:NL, :NT-1]) + \
        0.125*(L[:NW-1, :NL-1, :NT-1] + L[:NW-1, :NL-1, 2:NT+1] +
               L[:NW-1, 2:NL+1, :NT-1] + L[:NW-1, 2:NL+1, 2:NT+1] +
               L[2:NW+1, :NL-1, :NT-1] + L[2:NW+1, :NL-1, 2:NT+1] +
               L[2:NW+1, 2:NL+1, :NT-1] + L[2:NW+1, 2:NL+1, 2:NT+1])

    return M


def p_self(w, l):
    """Self coefficient of potential of a single square-section panel.

    Closed-form self term for a panel of width `w` and length `l` (an
    approximation valid for the ``dw x dl`` charge panels of the PEEC mesh).

    Parameters
    ----------
    w : float
        Panel width (metres).
    l : float
        Panel length (metres).

    Returns
    -------
    float
        The self coefficient of potential ``P_ii`` (units 1/F), the diagonal
        entry relating the panel's potential to its own charge.
    """
    u = l/w
    pii = 3*np.log(u + np.sqrt(u**2 + 1)) + u**2 + 1/u + \
        3*u*np.log(1/u + np.sqrt(1/u**2 + 1)) - (u**(4/3) + u**(-2/3))**(3/2)
    c = 299792458
    mu0 = 4*np.pi*1e-7
    eps0 = 1/(mu0 * c**2)
    return l/w**2*(4*np.pi*eps0) * 2/3 * pii


def gen_p_par(ax, ay, bx, by, dx, dy):
    """Mutual coefficient of potential between two coplanar parallel panels.

    Evaluates ``P_ij`` for two rectangular charge panels lying in the same
    plane (zero out-of-plane separation): a source panel of size ``ax x ay``
    and a target panel of size ``bx x by`` whose centres are offset by
    ``(dx, dy)`` in the plane. Computed by the standard four-corner
    alternating-sum evaluation of the integrated 1/r kernel.

    Parameters
    ----------
    ax, ay : float
        Source-panel dimensions (metres).
    bx, by : float
        Target-panel dimensions (metres).
    dx, dy : float
        Centre-to-centre offset in the shared plane (metres).

    Returns
    -------
    float
        Mutual coefficient of potential ``P_ij`` (units 1/F).
    """
    x = np.zeros((2,))
    x[1] = ax
    y = np.zeros((2,))
    y[1] = ay
    u = np.zeros((2,))
    u[:] = [dx + ax/2 - bx/2, dx + ax/2 + bx/2]
    v = np.zeros((2,))
    v[:] = [dy + ay/2 - by/2, dy + ay/2 + by/2]
    S = np.arcsinh
    pij = 0
    for k in range(2):
        for l in range(2):
            w = np.abs(y[k] - v[l])
            if w > 0:
                for m in range(2):
                    for n in range(2):
                        ell = np.abs(x[m] - u[n])
                        if ell > 0:
                            r = np.sqrt(ell**2 + w**2)
                            I = -2/3*ell**2*w**2*(1/(ell + r) + 1/(w + r)) + \
                                2*ell**2*w*S(w/ell) + 2*w**2*ell*S(ell/w)
                            pij += (-1)**(k + l + m + n) * I
    c = 299792458
    mu0 = 4*np.pi*1e-7
    eps0 = 1/(mu0 * c**2)
    return pij/(16*np.pi*eps0*x[1]*y[1]*(u[1]-u[0])*(v[1]-v[0]))


def gen_p_parz(ax, ay, bx, by, dx, dy, z):
    """Mutual coefficient of potential between two offset parallel panels.

    Like :func:`gen_p_par` but for panels lying in two parallel planes
    separated by an out-of-plane distance `z` (``z > 0``), with in-plane
    centre offset ``(dx, dy)``. Uses the four-corner alternating sum of the
    z-separated closed form.

    Parameters
    ----------
    ax, ay : float
        Source-panel dimensions (metres).
    bx, by : float
        Target-panel dimensions (metres).
    dx, dy : float
        In-plane centre-to-centre offset (metres).
    z : float
        Out-of-plane separation between the two planes (metres, nonzero).

    Returns
    -------
    float
        Mutual coefficient of potential ``P_ij`` (units 1/F).
    """
    x = np.zeros((2,))
    x[1] = ax
    y = np.zeros((2,))
    y[1] = ay
    u = np.zeros((2,))
    u[:] = [ax/2 + dx - bx/2, ax/2 + dx + bx/2]
    v = np.zeros((2,))
    v[:] = [ay/2 + dy - by/2, ay/2 + dy + by/2]
    S = np.arcsinh
    T = np.arctan
    pij = 0
    for k in range(2):
        for l in range(2):
            q = np.abs(y[k] - v[l]) / z
            for m in range(2):
                for n in range(2):
                    p = np.abs(x[m] - u[n]) / z
                    I = 6*(p**2 - 1)*q*S(q/np.sqrt(p**2 + 1)) + \
                        6*(q**2 - 1)*p*S(p/np.sqrt(q**2 + 1)) + \
                        6*p*S(p) + 6*q*S(q) - \
                        12*p*q*T(p*q/np.sqrt(p**2 + q**2 + 1)) + \
                        4*p**2*(1/(np.sqrt(p**2 + q**2 + 1) +
                                   np.sqrt(q**2 + 1)) -
                                1/(1 + np.sqrt(1 + p**2))) + \
                        2*q**2*np.sqrt(q**2 + 1) + \
                        2*p**2*np.sqrt(p**2 + 1) - \
                        2*(p**2 + q**2)*np.sqrt(p**2 + q**2 + 1)
                    I *= z**3/3
                    pij += (-1)**(k + l + m + n) * I
    c = 299792458
    mu0 = 4*np.pi*1e-7
    eps0 = 1/(mu0 * c**2)
    return pij/(16*np.pi*eps0*ax*ay*bx*by)


def gen2_p_parz(ax, ay, az, nx, ny, nz):
    """Vectorized parallel-panel potential Toeplitz array over a grid.

    Batched counterpart of :func:`gen_p_par`/:func:`gen_p_parz`: builds the
    whole ``nx x ny x nz`` array of mutual coefficients of potential between
    identical parallel panels of size ``ax x ay`` as a function of integer
    separation ``(i, j, k)`` in units of ``(ax, ay, az)``. The ``k = 0`` slab
    (coplanar case, ``z = 0``) is handled by a separate closed form. This is
    the generating array of the capacitive Toeplitz block.

    Parameters
    ----------
    ax, ay, az : float
        Panel dimensions / grid strides along x, y, z (metres).
    nx, ny, nz : int
        Number of separation steps to tabulate along x, y, z.

    Returns
    -------
    ndarray, shape (nx, ny, nz)
        Mutual coefficients of potential (units 1/F) by integer separation.
    """
    ix = ax*np.r_[:nx+1]
    iy = ay*np.r_[:ny+1]
    iz = az*np.r_[1:nz]
    mx, my, mz = np.meshgrid(ix, iy, iz, indexing='ij')
    S = np.arcsinh
    T = np.arctan
    pij = np.zeros((nx, ny, nz), dtype=float)
    p = mx / mz
    q = my / mz
    I = mz**3/3 * (6*(p**2 - 1)*q*S(q/np.sqrt(p**2 + 1)) + \
                   6*(q**2 - 1)*p*S(p/np.sqrt(q**2 + 1)) + \
                   6*p*S(p) + 6*q*S(q) - \
                   12*p*q*T(p*q/np.sqrt(p**2 + q**2 + 1)) + \
                   4*p**2*(1/(np.sqrt(p**2 + q**2 + 1) + np.sqrt(q**2 + 1)) -
                           1/(1 + np.sqrt(1 + p**2))) + \
                   2*q**2*np.sqrt(q**2 + 1) + 2*p**2*np.sqrt(p**2 + 1) - \
                   2*(p**2 + q**2)*np.sqrt(p**2 + q**2 + 1))
    pij[1:, 1:, 1:] = 4*I[1:-1, 1:-1, :] - 2*I[1:-1, 2:, :] - \
                      2*I[2:, 1:-1, :] + I[2:, 2:, :] - \
                      2*I[:-2, 1:-1, :] - 2*I[1:-1, :-2, :] + \
                      I[:-2, 2:, :] + I[2:, :-2, :] + I[:-2, :-2, :]
    pij[0, 1:, 1:] = 4*I[1, 1:-1, :] - 2*I[1, 2:, :] - 2*I[1, :-2, :]
    pij[1:, 0, 1:] = 4*I[1:-1, 1, :] - 2*I[2:, 1, :] - 2*I[:-2, 1, :]
    pij[0, 0, 1:] = 4*I[1, 1, :]
    # special case: same plane
    mx = mx[1:, 1:, 0]
    my = my[1:, 1:, 0]
    r = np.sqrt(mx**2 + my**2)
    I = np.zeros((nx+1, ny+1), dtype=float)
    I[1:, 1:] = -2/3*mx**2*my**2*(1/(mx + r) + 1/(my + r)) + \
                2*mx**2*my*S(my/mx) + 2*my**2*mx*S(mx/my)
    pij[1:, 1:, 0] = 4*I[1:-1, 1:-1] - 2*I[1:-1, 2:] - \
                      2*I[2:, 1:-1] + I[2:, 2:] - \
                      2*I[:-2, 1:-1] - 2*I[1:-1, :-2] + \
                      I[:-2, 2:] + I[2:, :-2] + I[:-2, :-2]
    # print(pij[7, 15, 0])
    pij[0, 1:, 0] = 2*I[1, 2:] + 2*I[1, :-2] - 4*I[1, 1:-1]
    pij[1:, 0, 0] = 2*I[2:, 1] + 2*I[:-2, 1] - 4*I[1:-1, 1]
    pij[0, 0, 0] = 4*I[1, 1]
    c = 299792458
    mu0 = 4*np.pi*1e-7
    eps0 = 1/(mu0 * c**2)
    return np.abs(pij/(16*np.pi*eps0*ax**2*ay**2))

def gen_p_per(ax, ay, az, nx, ny, nz):
    """Mutual coefficients of potential between perpendicular panels.

    Gives the ``nx x ny x nz`` array of mutual coefficients of potential
    between a single z-oriented panel and a grid of y-oriented panels (i.e.
    panels whose normals are perpendicular), as a function of integer
    separation. Each panel is ``ax x ay x az`` in size and the grid strides
    follow the same dimensions. This is the "perpendicular" capacitive
    Toeplitz block, complementing the parallel case in :func:`gen2_p_parz`.

    Uses the exact, numerically stable closed form of Jain, Koh and
    Balakrishnan, "Exact and numerically stable closed-form expressions for
    potential coefficients of rectangular conductors" (IEEE TCAS-II Express
    Briefs, 2006; correction 2007).

    Parameters
    ----------
    ax, ay, az : float
        Panel dimensions / grid strides along x, y, z (metres).
    nx, ny, nz : int
        Number of separation steps to tabulate along x, y, z.

    Returns
    -------
    ndarray, shape (nx, ny, nz)
        Mutual coefficients of potential (units 1/F) by integer separation.
    """
    ix = ax*np.r_[1:nx+1]
    iy = ay*np.r_[1:ny+1]
    iz = az*np.r_[1:nz+1]
    x, b, c = np.meshgrid(ix, iy, iz, indexing='ij')
    S = np.arcsinh
    T = np.arctan
    rho = np.sqrt(x**2 + b**2 + c**2)
    tau = np.sqrt(b**2 + c**2)
    J = np.zeros((nx+1, ny+1, nz+1), dtype=float)
    I = np.zeros((nx+1, ny, nz), dtype=float)
    J[1:, 1:, 1:] = x**2*c*np.log(b + rho) - \
                    c**3/3*np.log((b + rho)/(b + tau)) + \
                    x**2*b*np.log(c + rho) - \
                    b**3/3*np.log((c + rho)/(c + tau)) + \
                    2*x*b*c * S(x/tau) - 2*b*c/3*(rho - tau) - \
                    x**3/3*T(b*c/(x*rho)) - b**2*x*T(x*c/(b*rho)) - \
                    c**2*x*T(x*b/(c*rho))
    rho = np.sqrt(x**2 + c**2)
    J[1:, 0, 1:] = x[:, 0, :]**2*c[:, 0, :]*np.log(rho[:, 0, :]) - \
                   c[:, 0, :]**3/3*np.log(rho[:, 0, :] / c[:, 0, :])
    rho = np.sqrt(x**2 + b**2)
    J[1:, 1:, 0] = x[:, :, 0]**2*b[:, :, 0]*np.log(rho[:, :, 0]) - \
                   b[:, :, 0]**3/3*np.log(rho[:, :, 0] / b[:, :, 0])
    I = J[:, :-1, :-1] - J[:, :-1, 1:] - J[:, 1:, :-1] + J[:, 1:, 1:]
    pij = np.zeros((nx, ny, nz), dtype=float)
    pij[1:, :, :] = I[2:, :, :] - 2*I[1:-1, :, :] + I[:-2, :, :]
    pij[0, :, :] = 2*I[1, :, :]
    c = 299792458
    mu0 = 4*np.pi*1e-7
    eps0 = 1/(mu0 * c**2)
    return pij/(8*np.pi*eps0*ax**2*ay*az)

def p_par(ax, ay, bx, by, dx, dy):
    """Mutual coefficient of potential between two coplanar parallel panels.

    Alternative closed form for the same coplanar configuration as
    :func:`gen_p_par` (source ``ax x ay``, target ``bx x by``, in-plane centre
    offset ``(dx, dy)``), evaluated as a 16-term alternating sum over the four
    x-edges and four y-edges of the two panels, with explicit guards for the
    ``a + rho <= 0`` / ``b + rho <= 0`` cases where the log term is dropped for
    numerical stability.

    Parameters
    ----------
    ax, ay : float
        Source-panel dimensions (metres).
    bx, by : float
        Target-panel dimensions (metres).
    dx, dy : float
        Centre-to-centre offset in the shared plane (metres).

    Returns
    -------
    float
        Mutual coefficient of potential ``P_ij`` (units 1/F).
    """
    a = np.ndarray((4,))
    b = np.ndarray((4,))
    a[0] = dx - ax/2 - bx/2
    a[1] = dx + ax/2 - bx/2
    a[2] = dx + ax/2 + bx/2
    a[3] = dx - ax/2 + bx/2
    b[0] = dy - ay/2 - by/2
    b[1] = dy + ay/2 - by/2
    b[2] = dy + ay/2 + by/2
    b[3] = dy - ay/2 + by/2
    pij = 0
    for k in range(4):
        for m in range(4):
            rho = np.sqrt(a[k]**2 + b[m]**2)
            if a[k]+rho <= 0 and b[m]+rho <= 0:
                pij -= (-1)**(m+k)*rho/6*(b[m]**2 + a[k]**2)
            elif a[k]+rho <= 0:
                pij += (-1)**(m+k)*(a[k]**2*b[m]/2*np.log(b[m] + rho) -
                                    rho/6*(b[m]**2 + a[k]**2))
            elif b[m]+rho <= 0:
                pij += (-1)**(m+k)*(b[m]**2*a[k]/2*np.log(a[k] + rho) -
                                    rho/6*(b[m]**2 + a[k]**2))
            else:
                pij += (-1)**(m+k)*(b[m]**2*a[k]/2*np.log(a[k] + rho) +
                                    a[k]**2*b[m]/2*np.log(b[m] + rho) -
                                    rho/6*(b[m]**2 + a[k]**2))
    c = 299792458
    mu0 = 4*np.pi*1e-7
    eps0 = 1/(mu0 * c**2)
    return pij/(4*np.pi*eps0*ax*ay*bx*by)




def _J_per(x, b, c):
    """Primitive behind the perpendicular-panel coefficient of potential.

    The same closed form :func:`gen_p_per` uses (Jain, Koh and
    Balakrishnan 2006), evaluated for ``x, b, c >= 0`` with its
    degenerate limits. It is EVEN in ``x`` but neither even nor odd in
    ``b`` or ``c`` -- it is a primitive with a branch choice valid only
    on the positive side -- which is why :func:`p_per_rect` never
    evaluates it at a negative argument.

    ``x`` is folded through ``abs`` (legitimate, by that evenness), but
    a negative ``b`` or ``c`` is an ERROR and is raised rather than
    returned. Without the check the ``np.where(b > 0, ...)`` below takes
    its false branch and hands back the ``b = 0`` limit -- a plausible
    number, indistinguishable from a correct one, with nothing to
    indicate anything went wrong. The whole correctness argument for
    :func:`p_per_rect` rests on this never happening, so it is enforced
    rather than documented.

    Raises
    ------
    ValueError
        If any ``b`` or ``c`` is negative.
    """
    x = np.abs(np.asarray(x, dtype=float))
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    if np.any(b < 0) or np.any(c < 0):
        bad = min(float(np.min(b)), float(np.min(c)))
        raise ValueError(
            "_J_per requires b >= 0 and c >= 0 (most negative argument "
            "%.6g). The primitive's branch is only valid on the positive "
            "side; callers must split the integration range at its sign "
            "change first -- see _split()." % bad)
    S, T = np.arcsinh, np.arctan
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = np.sqrt(x*x + b*b + c*c)
        tau = np.sqrt(b*b + c*c)
        full = (x*x*c*np.log(b + rho) - c**3/3*np.log((b + rho)/(b + tau))
                + x*x*b*np.log(c + rho) - b**3/3*np.log((c + rho)/(c + tau))
                + 2*x*b*c*S(x/tau) - 2*b*c/3*(rho - tau)
                - x**3/3*T(b*c/(x*rho)) - b*b*x*T(x*c/(b*rho))
                - c*c*x*T(x*b/(c*rho)))
        r_xc = np.sqrt(x*x + c*c)
        r_xb = np.sqrt(x*x + b*b)
        b0 = x*x*c*np.log(r_xc) - c**3/3*np.log(r_xc/np.where(c > 0, c, 1))
        c0 = x*x*b*np.log(r_xb) - b**3/3*np.log(r_xb/np.where(b > 0, b, 1))
    out = np.where(b > 0, np.where(c > 0, full, c0), np.where(c > 0, b0, 0.0))
    return np.where(x > 0, out, 0.0)


def _Jn_per(x, b, c):
    """:func:`_J_per` normalised to vanish whenever ``b`` or ``c`` is 0.

    Needed so the two single-integral directions can be split
    independently at their sign changes and still compose.
    """
    z = np.zeros_like(np.asarray(b, dtype=float))
    return (_J_per(x, b, c) - _J_per(x, z, c)
            - _J_per(x, b, z) + _J_per(x, z, z))


def _split(t_lo, t_hi):
    """Signed magnitudes for ``Int_{t_lo}^{t_hi} h dt``, ``h`` even.

    Always exactly two terms. The integrand is even, so an interval that
    straddles zero splits into two one-sided pieces rather than needing
    the primitive at a negative argument; collecting the three cases
    (wholly positive, wholly negative, straddling) gives the branchless
    signs ``(-sgn(t_lo), +sgn(t_hi))`` on ``(|t_lo|, |t_hi|)``.
    """
    t_lo = np.asarray(t_lo, dtype=float)
    t_hi = np.asarray(t_hi, dtype=float)
    return ((-np.sign(t_lo), np.abs(t_lo)), (np.sign(t_hi), np.abs(t_hi)))


def p_per_rect(a, b):
    """Coefficient of potential between two PERPENDICULAR rectangles.

    Independent panel sizes at an arbitrary offset --- the general form
    that :func:`gen_p_per` provides only for equal panels on a lattice
    whose stride equals the panel size. The cell-centred discretisation
    needs whole ``dx`` faces on lattices staggered by ``dx/2``, where
    size and stride differ, so that specialisation does not apply.

    The parallel counterpart is already general: see
    :func:`gen_p_parz`.

    Parameters
    ----------
    a, b : tuple
        ``(axis, (u0, u1), (v0, v1), plane)`` as built by
        ``panel_quad.rect``: ``axis`` is the normal, the two intervals
        are the extents on the other two axes in ASCENDING AXIS ORDER,
        and ``plane`` is the coordinate along the normal. The two
        normals must differ.

    Returns
    -------
    float
        ``P_ij`` in 1/F.

    Notes
    -----
    Structure: the two panels share the axis that is neither normal, so
    that direction carries a genuine double integral and contributes a
    four-term corner difference; each of the other two directions is an
    extent of one panel against the other's plane and contributes two
    terms. Sixteen evaluations of :func:`_Jn_per` in all. The symmetric
    second difference in :func:`gen_p_per` is what this collapses to
    when both shared extents are equal and offsets are integer
    multiples.

    Validated in ``validate_panel_kernel.py`` against direct quadrature
    (``panel_quad.py``, no shared code), by reciprocity, and by the
    subdivision identity -- which holds for TOUCHING panels, where
    quadrature cannot follow.
    """
    na, nb = int(a[0]), int(b[0])
    if na == nb:
        raise ValueError("p_per_rect needs perpendicular panels; use "
                         "gen_p_parz for parallel ones")
    shared = [i for i in range(3) if i != na and i != nb][0]

    def ext(rc, axis):
        other = [i for i in range(3) if i != rc[0]]
        return rc[1] if other[0] == axis else rc[2]

    axl, axh = ext(a, shared)
    bxl, bxh = ext(b, shared)
    bal, bah = ext(b, na)      # B's extent along A's normal
    abl, abh = ext(a, nb)      # A's extent along B's normal
    pa, pb = float(a[3]), float(b[3])
    xa, xb = (axl, axh), (bxl, bxh)
    total = 0.0
    for i in (0, 1):
        for j in (0, 1):
            sx = -(-1.0)**(i + j)
            X = abs(xa[i] - xb[j])
            for sb, tb in _split(pa - bah, pa - bal):
                for sc, tc in _split(abl - pb, abh - pb):
                    total += sx*sb*sc*_Jn_per(X, tb, tc)
    mu0 = 4*np.pi*1e-7
    eps0 = 1/(mu0 * 299792458.0**2)
    aa = (a[1][1] - a[1][0])*(a[2][1] - a[2][0])
    ab = (b[1][1] - b[1][0])*(b[2][1] - b[2][0])
    return float(total/(8*np.pi*eps0*aa*ab))


def p_per_table(l, so, to, off):
    """Perpendicular whole-face coupling table for the cell scheme.

    Vectorised :func:`p_per_rect` over a lattice of integer cell
    separations, for the cell-centred discretisation where a panel is a
    whole cell face. The half-cell staggering between differently
    oriented panel lattices is NOT encoded here -- it falls out of
    placing each rectangle at its actual cell indices, which is why this
    needs no offset bookkeeping of its own.

    Parameters
    ----------
    l : sequence of 3 float
        Cell dimensions.
    so, to : int
        Source and target panel normals (0/1/2); must differ.
    off : sequence of 3 int
        Table half-width per axis; separations ``-off[c] .. +off[c]``
        are tabulated, so index ``sep + off[c]``. SIGNED, unlike the
        parallel table: a perpendicular coupling is not symmetric under
        reflection of a single axis.

    Returns
    -------
    ndarray, shape ``2*off + 1``
        Coefficients of potential in 1/F.
    """
    l = np.asarray(l, dtype=float)
    off = np.asarray(off, dtype=int)
    if so == to:
        raise ValueError("p_per_table is for perpendicular panels")
    shared = [c for c in range(3) if c != so and c != to][0]
    grids = np.meshgrid(*[np.arange(-off[c], off[c] + 1) for c in range(3)],
                        indexing='ij')
    # Source: normal so, plane at 0, occupying cell 0 on the other axes.
    # Target: normal to, plane at sep[to]*l[to], occupying cell sep[c].
    axl, axh = 0.0, l[shared]
    bxl = grids[shared]*l[shared]
    bxh = bxl + l[shared]
    bal = grids[so]*l[so]                  # B's extent along A's normal
    bah = bal + l[so]
    abl, abh = 0.0, l[to]                  # A's extent along B's normal
    pa = 0.0
    pb = grids[to]*l[to]
    total = np.zeros(grids[0].shape, dtype=float)
    for i, xa in enumerate((axl, axh)):
        for j, xb in enumerate((bxl, bxh)):
            sx = -(-1.0)**(i + j)
            X = np.abs(xa - xb)
            for sb, tb in _split(pa - bah, pa - bal):
                for sc, tc in _split(abl - pb, abh - pb):
                    total += sx*sb*sc*_Jn_per(X, tb, tc)
    mu0 = 4*np.pi*1e-7
    eps0 = 1/(mu0 * 299792458.0**2)
    aa = l[[c for c in range(3) if c != so]].prod()
    ab = l[[c for c in range(3) if c != to]].prod()
    return total/(8*np.pi*eps0*aa*ab)


def panel_tables(l, n_src, n_tgt, orientation):
    """The three panel kernel tables for one source orientation.

    THE single construction of the capacitive near-field tables, shared
    by ``leaf_poten.LeafPoten.p2pinit3`` (the dense/sparse assembly) and
    ``circulant_poten`` (the FFT one). Those two used to carry
    independent copies -- circulant_poten's docstring said "lifted
    verbatim" -- which was already a duplication and would have become a
    three-way one once the cell scheme needed different kernels.

    Parameters
    ----------
    l : sequence of 3 float
        Panel dimensions / lattice pitch of the SOURCE leaf.
    n_src : sequence of 3 int
        Source panel lattice size.
    n_tgt : sequence of 3 sequences of 3 int
        Target lattice sizes, for the x-, y- and z-oriented targets.
    orientation : {'x', 'y', 'z'}
        Source panel normal. Panels are WHOLE FACES: parallel blocks
        stay in registry with size equal to stride, but cross blocks
        are whole faces on lattices staggered by half a cell, which
        ``gen_p_per`` cannot express -- they come from
        :func:`p_per_table` and are indexed by SIGNED separation
        shifted by ``offsets``.

    Returns
    -------
    so : int
        Source normal axis.
    tables : tuple of 3 ndarray
        Kernel tables to the x-, y- and z-oriented targets, in the
        leaf's own (x, y, z) axis order.
    offsets : list of 3 ndarray
        The per-target index shift for the SIGNED cross-block lookup
        (the parallel block still uses ``abs``).
    """
    l = np.asarray(l, dtype=np.float64)
    n_src = np.asarray(n_src, dtype=int)
    n2 = (2*n_src).astype(int)
    m2 = (2*n_src).astype(int)
    l2 = l/2
    so = {'x': 0, 'y': 1, 'z': 2}[orientation]
    if orientation == 'x':
        tab_x = np.transpose(
            gen2_p_parz(l[1], l[2], l[0], n2[1], n2[2], n2[0]), (2, 0, 1))
        tab_y = np.transpose(
            gen_p_per(l[2], l2[0], l[1], m2[2], 2*m2[0], m2[1]), (1, 2, 0))
        tab_z = np.transpose(
            gen_p_per(l[1], l2[0], l[2], m2[1], 2*m2[0], m2[2]), (1, 0, 2))
    elif orientation == 'y':
        tab_x = np.transpose(
            gen_p_per(l[2], l2[1], l[0], m2[2], 2*m2[1], m2[0]), (2, 1, 0))
        tab_y = np.transpose(
            gen2_p_parz(l[2], l[0], l[1], n2[2], n2[0], n2[1]), (1, 2, 0))
        tab_z = gen_p_per(l[0], l2[1], l[2], m2[0], 2*m2[1], m2[2])
    elif orientation == 'z':
        tab_x = np.transpose(
            gen_p_per(l[1], l[0], l2[2], m2[1], m2[0], 2*m2[2]), (1, 0, 2))
        tab_y = gen_p_per(l[0], l[1], l2[2], m2[0], m2[1], 2*m2[2])
        tab_z = gen2_p_parz(l[0], l[1], l[2], n2[0], n2[1], n2[2])
    else:
        raise ValueError("invalid orientation")
    tables = [tab_x, tab_y, tab_z]
    offsets = []
    for tt in range(3):
        nt = np.asarray(n_tgt[tt], dtype=int)
        off = (n_src + nt + 2).astype(int)
        offsets.append(off)
        if tt != so:
            tables[tt] = p_per_table(l, so, tt, off)
    return so, tuple(tables), offsets
