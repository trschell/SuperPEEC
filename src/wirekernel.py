# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Partial inductance kernels for ARBITRARILY ORIENTED prismatic
filaments -- the piece SuperPEEC needs to model bond wires properly.

WHY THIS EXISTS. Voxels staircase a curved, skewed bond wire, and the
staircase is the dominant error on wire geometry (the skin-engine study
measured a direction-dependent +7.4%/-3.6% swing that turned out to be
staircase-vs-circle changing sign). FastHenry and InductEx handle these
problems well because their basis is a thin filament of arbitrary
orientation. This module supplies the same kind of basis so a wire can
be a chain of straight segments coupled into the voxel lattice, rather
than a staircase carved out of it.

CROSS-SECTION: TRUE ANNULAR SECTORS. A round wire is represented by a
circular core plus a ring of curved sectors (12 by default). Curved
inner/outer edges cost nothing over straight-edged trapezoids -- both
are tensor products in (r, theta) with the r Jacobian -- and they give
the exact area AND the exact perimeter. A straight-edged 12-gon can
match only one: sized to equal area it carries 1.17% too much
perimeter, which lands as ~1.2% low in the deep-skin R_ac. There is no
reason to accept that once the quadrature is polar anyway.

THE AXIAL INTEGRALS ARE ANALYTIC; ONLY THE CROSS-SECTIONS ARE
QUADRATURED. Two regimes, and the split is what keeps this robust:
  * PARALLEL filaments -- exact closed form, the second antiderivative
    of 1/sqrt(u^2+d^2). Covers same-wire segment pairs and any parallel
    pair, at machine precision with no quadrature error at all.
  * SKEWED filaments -- the inner integral (a POINT to a straight
    segment) is analytic; only the outer one is quadratured. This
    deliberately avoids Grover's general skew formula, whose closed
    form carries a minefield of degenerate branches (parallel,
    collinear, intersecting) -- exactly what FastHenry needs a separate
    deg_mutual.c to handle. Trading one 1-D quadrature for that whole
    class of branch bugs is a good deal, and this project has already
    lost time to a principal-branch error in the BEM arctan.

SELF TERMS ride the same parallel closed form with the transverse
distance replaced by the cross-section's GEOMETRIC MEAN DISTANCE.
:func:`gmd` computes it by quadrature and is checked against the disc's
closed form a*exp(-1/4).
"""
import numpy as np

MU0 = 4e-7*np.pi


# ---------------------------------------------------------------- shapes

def sector_points(r0, r1, th0, th1, nr=4, nth=4):
    """Quadrature over an annular sector: (offsets in the (v,w) plane,
    areas). Tensor Gauss-Legendre in (r, theta) with the r Jacobian, so
    the curved edges are exact -- there is no polygonal approximation
    anywhere in this routine."""
    xr, wr = np.polynomial.legendre.leggauss(nr)
    xt, wt = np.polynomial.legendre.leggauss(nth)
    r = 0.5*(r1 - r0)*xr + 0.5*(r1 + r0)
    jr = 0.5*(r1 - r0)*wr
    t = 0.5*(th1 - th0)*xt + 0.5*(th1 + th0)
    jt = 0.5*(th1 - th0)*wt
    R, T = np.meshgrid(r, t, indexing='ij')
    WR, WT = np.meshgrid(jr, jt, indexing='ij')
    area = (WR*R*WT).ravel()          # r dr dtheta
    return np.column_stack([(R*np.cos(T)).ravel(),
                            (R*np.sin(T)).ravel()]), area


def rect_points(w, h, nw=4, nh=4):
    """Quadrature over a centred rectangle -- only needed so the
    degenerate axis-aligned case can be checked against the existing
    exact bar formula (terminal.box_mutual_matrix)."""
    xw, ww = np.polynomial.legendre.leggauss(nw)
    xh, wh = np.polynomial.legendre.leggauss(nh)
    V, W = np.meshgrid(0.5*w*xw, 0.5*h*xh, indexing='ij')
    A, B = np.meshgrid(0.5*w*ww, 0.5*h*wh, indexing='ij')
    return np.column_stack([V.ravel(), W.ravel()]), (A*B).ravel()


class Filament:
    """A prismatic filament: cross-section swept along a straight axis.

    ``p0`` is the start point, ``u`` the (unit) axis, ``length`` the
    span, and (``off``, ``area``) the cross-section quadrature in the
    local transverse frame (v, w).
    """

    def __init__(self, p0, u, length, off, area, v=None, shape=None):
        self.p0 = np.asarray(p0, dtype=float)
        u = np.asarray(u, dtype=float)
        self.u = u/np.linalg.norm(u)
        self.length = float(length)
        # transverse frame: v chosen stably, w = u x v
        if v is None:
            a = np.array([1.0, 0.0, 0.0])
            if abs(self.u @ a) > 0.9:
                a = np.array([0.0, 1.0, 0.0])
            v = np.cross(self.u, a)
        v = np.asarray(v, dtype=float)
        self.v = v/np.linalg.norm(v)
        self.w = np.cross(self.u, self.v)
        self.off = np.asarray(off, dtype=float)
        self.area = np.asarray(area, dtype=float)
        # SHAPE FUNCTION phi over the cross-section. The element carries
        # J = I phi / int(phi dA) instead of uniform current, so ONE
        # element can hold a skin profile that would otherwise need
        # several radial rings. shape=None is phi=1 and everything below
        # reduces to the uniform-current case bit-for-bit.
        self.shape = (np.ones_like(self.area) if shape is None
                      else np.asarray(shape))
        self.complex_shape = bool(np.iscomplexobj(self.shape))
        # NB: self.w is the transverse frame vector u x v; the
        # quadrature weights are self.wt
        self.wt = self.area*self.shape
        self.A = float(self.wt.sum())
        self.geom_area = float(self.area.sum())
        # resistance factor: R = rho l int(phi^2)/int(phi)^2, from
        # P = int rho |J|^2 dA l = I^2 R
        # NON-CONJUGATED pairing throughout. A PEEC system is complex
        # SYMMETRIC, not Hermitian: the Galerkin/reaction integral is
        # <b_k, L b_j> with no conjugate, so a complex shape makes the
        # matrix entries themselves complex rather than producing an
        # |phi|^2 energy form. The dissipated power still comes out
        # correctly as Re(Z)|I|^2 once the system is solved.
        num = (self.area*self.shape*self.shape).sum()
        self.rfac = num/(self.A*self.A)

    def set_shape(self, shape):
        """Retune the cross-section shape IN PLACE (frequency retune:
        the skin profile follows delta(f)). Geometry (off/area/frame)
        is untouched, so any cached point-resolved kernels stay valid
        -- this is what makes a frequency sweep's coupling rebuild a
        REWEIGHTING instead of a requadrature."""
        self.shape = (np.ones_like(self.area) if shape is None
                      else np.asarray(shape))
        self.complex_shape = bool(np.iscomplexobj(self.shape))
        self.wt = self.area*self.shape
        self.A = float(self.wt.sum().real) if not self.complex_shape \
            else complex(self.wt.sum())
        num = (self.area*self.shape*self.shape).sum()
        self.rfac = num/(self.A*self.A)

    def resistance(self, rho, length=None):
        """Series resistance of this element under its own shape.
        Complex when the shape is complex -- see the note on the
        non-conjugated pairing above."""
        r = rho*(self.length if length is None else length)*self.rfac
        return complex(r) if self.complex_shape else float(r.real)

    def points(self):
        """Global 3-D positions of the cross-section quadrature nodes,
        on the START face."""
        return (self.p0[None, :] + self.off[:, 0:1]*self.v[None, :]
                + self.off[:, 1:2]*self.w[None, :])


# ------------------------------------------------------- line kernels

def _G(u, d):
    """Second antiderivative of 1/sqrt(u^2+d^2)."""
    return u*np.arcsinh(u/d) - np.sqrt(u*u + d*d)


def mutual_parallel(s1a, s1b, s2a, s2b, d):
    """EXACT double axial integral for two parallel line filaments,
    axial spans [s1a,s1b] and [s2a,s2b] on a common axis, transverse
    distance d. No quadrature error."""
    return (_G(s1b - s2a, d) - _G(s1b - s2b, d)
            - _G(s1a - s2a, d) + _G(s1a - s2b, d))


def _segment_potential(p, a, u2, l2, eps):
    """Analytic inner integral: int_0^l2 ds / |p - (a + s u2)|."""
    wv = p - a
    t = wv @ u2
    h2 = (wv*wv).sum(-1) - t*t
    h = np.sqrt(np.maximum(h2, eps*eps))
    return np.arcsinh((l2 - t)/h) - np.arcsinh(-t/h)


def mutual_skew_lines(a1, u1, l1, a2, u2, l2, ng=16, eps=1e-12):
    """Two arbitrarily oriented line filaments. Inner integral analytic
    (point-to-segment), outer by Gauss -- see the module note on why the
    general closed form is avoided.

    THE SCHEME IS ASYMMETRIC BY CONSTRUCTION (segment 2 analytic,
    segment 1 quadratured), so M(1,2) and M(2,1) differ by exactly the
    quadrature error -- which makes their disagreement a FREE error
    estimator needing no reference solution. Measured on a skewed pair:

        ng        4       8       16      32
        rel asym  2.9e-4  2.8e-7  3.2e-12 3e-16

    ng=16 is the default because it is where the value itself stops
    moving in every printed digit and reciprocity reaches 1e-12; ng=8
    is already good to 3e-7 if cost ever matters. Explicit
    symmetrisation is deliberately NOT applied -- it would hide the
    estimator without improving the answer."""
    x, w = np.polynomial.legendre.leggauss(ng)
    s = 0.5*l1*(x + 1.0)
    jac = 0.5*l1*w
    pts = a1[None, :] + s[:, None]*u1[None, :]
    inner = _segment_potential(pts, a2, u2, l2, eps)
    return float((inner*jac).sum())


# --------------------------------------------------- filament mutuals

def mutual(f1, f2, ng=16, parallel_tol=1e-9, self_gmd=None):
    """Partial mutual inductance (H) between two prismatic filaments.

    Uniform current density over each cross-section; the axial
    integrals are analytic and the cross-sections are quadratured.
    ``self_gmd`` overrides the transverse distance for a self term.
    """
    dot = float(f1.u @ f2.u)
    p1, p2 = f1.points(), f2.points()
    a1, a2 = f1.wt, f2.wt
    if abs(abs(dot) - 1.0) < parallel_tol:
        sgn = 1.0 if dot > 0 else -1.0
        u = f1.u
        # axial coordinates along u
        s1a = p1 @ u
        s1b = s1a + f1.length
        s2a = p2 @ u
        s2b = s2a + sgn*f2.length
        lo2 = np.minimum(s2a, s2b)
        hi2 = np.maximum(s2a, s2b)
        dp = p1[:, None, :] - p2[None, :, :]
        dax = dp @ u
        dperp = dp - dax[..., None]*u[None, None, :]
        d = np.linalg.norm(dperp, axis=-1)
        if self_gmd is not None:
            d = np.full_like(d, float(self_gmd))
        d = np.maximum(d, 1e-30)
        M = mutual_parallel(s1a[:, None], s1b[:, None],
                            lo2[None, :], hi2[None, :], d)
        val = (a1[:, None]*a2[None, :]*M).sum()
        out = sgn*MU0/(4*np.pi)*val/(f1.A*f2.A)
        return (complex(out) if (f1.complex_shape or f2.complex_shape)
                else float(np.real(out)))
    # SKEW, vectorised over both cross-sections at once. The scalar
    # loop this replaces was O(n1*n2) Python calls, which is fine for a
    # unit test and hopeless for the near field of a real wire: a
    # module has ~1e5-1e6 filament/voxel pairs inside the 27-box
    # neighbourhood, and they all take this branch.
    x, w = np.polynomial.legendre.leggauss(ng)
    sq = 0.5*f1.length*(x + 1.0)
    jac = 0.5*f1.length*w
    P = p1[:, None, :] + sq[None, :, None]*f1.u[None, None, :]
    W = P[:, None, :, :] - p2[None, :, None, :]
    t = W @ f2.u
    h2 = (W*W).sum(-1) - t*t
    h = np.sqrt(np.maximum(h2, 1e-24))
    inner = np.arcsinh((f2.length - t)/h) - np.arcsinh(-t/h)
    tot = np.einsum('ijk,k,i,j->', inner, jac, a1, a2)
    out = dot*MU0/(4*np.pi)*tot/(f1.A*f2.A)
    return (complex(out) if (f1.complex_shape or f2.complex_shape)
            else float(np.real(out)))


def gmd(off, area, ng_pair=None):
    """Geometric mean distance of a cross-section from itself.

    ``exp( (1/A^2) int int ln|r1-r2| )``. The log singularity is
    integrable, so plain quadrature converges -- but slowly, which is
    why this is meant to be computed ONCE per wire gauge and cached.
    Validated against the disc's closed form a*exp(-1/4).
    """
    d = np.linalg.norm(off[:, None, :] - off[None, :, :], axis=-1)
    np.fill_diagonal(d, np.nan)
    wgt = area[:, None]*area[None, :]
    m = np.isfinite(d)
    A2 = float((wgt[m]).sum())
    return float(np.exp((wgt[m]*np.log(d[m])).sum()/A2))


def self_inductance(f, gmd_value):
    """Self partial inductance via the parallel kernel with d = GMD."""
    return mutual(f, f, self_gmd=gmd_value)


def round_wire(p0, u, length, radius, nsect=12, core_frac=0.5,
               nr=3, nth=3, v=None, nring=1, grade=0.5, delta=None,
               complex_shape=False):
    """A round wire cross-section: circular CORE + ``nring`` rings of
    ``nsect`` curved annular sectors each.

    ``delta`` (skin depth) switches on the per-element SKIN SHAPE
    ``phi(r) = exp(-(a - r)/delta)`` -- the physical profile, restricted
    to whichever element a quadrature point falls in, with the element
    amplitudes free to redistribute on top of it. As delta -> inf it
    tends to 1 and the uniform-current case is recovered exactly, so the
    low-frequency limit is safe by construction.

    NOTE the shape is deliberately the ASYMPTOTIC exponential, not the
    exact Bessel profile: using the exact solution as the basis would
    reproduce the isolated-wire answer by construction and prove
    nothing, and would not generalise to proximity, where the profile is
    not the isolated-wire one.

    MEASURED (studies/wire_bessel.py) with UNIFORM elements: one ring
    tracks R_ac/R_dc to ~1% only while a/delta < 1, is 12% low at
    a/delta = 2.4 and SATURATES near 2 beyond a/delta ~ 5 while the
    truth climbs past 12; 2-3 graded rings (25-37 elements) hold 3-8%
    over the 1-10 MHz power-module band.
    """
    out = []
    rc = core_frac*radius
    # nsect may be a PER-RING sequence. Nothing in the connectivity
    # forbids it: every element of a segment is a parallel branch
    # between the segment's two end nodes (FastHenry does the same --
    # each filament spans node0+offset to node1+offset), so there is no
    # element-to-element mesh to keep consistent. That allows the
    # angular resolution to be spent only where proximity crowding
    # lives, in the outer rings, and keeps element aspect ratios near 1
    # instead of making slivers near the centre.
    ns = ([int(nsect)]*nring if np.isscalar(nsect)
          else [int(v) for v in nsect])
    if len(ns) != nring:
        raise ValueError("nsect sequence must have one entry per ring "
                         "(got %d for %d rings)" % (len(ns), nring))

    def _mk(r0, r1, t0, t1, nrq, ntq):
        off, ar = sector_points(r0, r1, t0, t1, nrq, ntq)
        sh = None
        if delta is not None and delta > 0:
            rr = np.linalg.norm(off, axis=1)
            # complex_shape=True is a MEASURED DEAD END, kept only so
            # it is not retried. The exact profile does decay AND rotate
            # with the same length scale, J ~ exp(-(1+j)(a-r)/d), so a
            # complex shape looks like the obvious refinement. It is
            # not: Galerkin PEEC pairs basis functions WITHOUT
            # conjugation (that is what makes the system complex
            # SYMMETRIC), so R = rho l int(phi^2)/int(phi)^2 suffers
            # cancellation once phi rotates phase inside its own
            # element, and Re(R) GOES NEGATIVE -- measured -0.78 ohm on
            # the core at 1 MHz and -54 ohm on the inner ring at
            # 100 MHz, with the damage scaling as the rotation angle
            # across the element (0.59, 1.86, 5.88 rad). Errors vs the
            # exact Bessel solution ran to hundreds of percent.
            # Repairing it needs a conjugated pairing for the resistive
            # term, which breaks the complex-symmetric structure the
            # rest of the solver assumes. The REAL magnitude-decay
            # shape already reaches -0.7% at a/delta 23.5 with 2 rings:
            # the (complex) element amplitudes carry the phase BETWEEN
            # elements, and intra-element rotation turns out not to
            # matter at this element count.
            kk = (1.0 + 1j) if complex_shape else 1.0
            sh = np.exp(-kk*(radius - rr)/delta)
        return Filament(p0, u, length, off, ar, v=v, shape=sh)

    out.append(_mk(0.0, rc, 0.0, 2*np.pi, nr, max(nth, 8)))
    t = np.array([grade**k for k in range(nring)], dtype=float)
    t = t/t.sum()
    edges = rc + (radius - rc)*np.concatenate([[0.0], np.cumsum(t)])
    for m in range(nring):
        for k in range(ns[m]):
            out.append(_mk(edges[m], edges[m + 1], 2*np.pi*k/ns[m],
                           2*np.pi*(k + 1)/ns[m], nr, nth))
    return out


# ------------------------------------------------- batched near field
#
# The scalar :func:`mutual` costs ~750 us per filament-voxel pair, all
# of it per-call setup and small-array einsum -- hopeless for a real
# near field (~1e5-1e6 pairs inside the 27-box neighbourhood of a
# module's wires). These evaluate one whole SEGMENT (every cross-section
# element at once, sharing p0/u/length) against many voxel bars or
# another whole segment, with the same two-regime split as `mutual`:
# parallel pairs exact, skewed pairs analytic-inner + Gauss-outer.


def _concat_seg(fs):
    """Stack a segment's cross-section quadratures into flat arrays.

    Every element of a segment shares ``p0``, ``u`` and ``length`` by
    construction (:func:`round_wire` builds them that way -- elements
    are parallel branches between the segment's end nodes, differing
    only in transverse offsets). Asserted rather than assumed: a mixed
    list would silently produce garbage geometry.
    """
    f0 = fs[0]
    for f in fs[1:]:
        if (np.linalg.norm(f.p0 - f0.p0) > 1e-15
                or abs(f.u @ f0.u - 1.0) > 1e-12
                or abs(f.length - f0.length) > 1e-15):
            raise ValueError("segment elements disagree on p0/u/length")
    pts = np.concatenate([f.points() for f in fs])
    wt = np.concatenate([f.wt for f in fs])
    eid = np.concatenate([np.full(f.wt.size, k, dtype=np.int64)
                          for k, f in enumerate(fs)])
    A = np.array([f.A for f in fs])
    cplx = any(f.complex_shape for f in fs)
    return f0, pts, wt, eid, A, cplx


def _block_reduce(vals, wt1, eid1, n1, wt2, eid2, n2):
    """Contract pointwise pair potentials into per-element blocks:
    ``B[e1, e2] = sum_{i in e1, j in e2} wt1_i wt2_j vals_ij``."""
    S1 = np.zeros((n1, wt1.size), dtype=wt1.dtype)
    S1[eid1, np.arange(wt1.size)] = wt1
    S2 = np.zeros((n2, wt2.size), dtype=wt2.dtype)
    S2[eid2, np.arange(wt2.size)] = wt2
    return S1 @ vals @ S2.T


def voxel_xsec(d, axis, nq=4):
    """The voxel-bar cross-section quadrature in GLOBAL 3-D offsets.

    Mirrors :func:`voxel_bar`'s frame exactly (v = first other axis,
    w = u x v) so the batched path is bit-consistent with the scalar
    one. Returns (offsets (nq*nq, 3), areas, total area).
    """
    d = np.asarray(d, dtype=float)
    if d.ndim == 0:
        d = np.full(3, float(d))
    others = [k for k in range(3) if k != axis]
    off2, ar = rect_points(d[others[0]], d[others[1]], nq, nq)
    u = np.zeros(3)
    u[axis] = 1.0
    v = np.zeros(3)
    v[others[0]] = 1.0
    w = np.cross(u, v)
    off3 = off2[:, 0:1]*v[None, :] + off2[:, 1:2]*w[None, :]
    return off3, ar, float(ar.sum())


def mutual_voxels(fs, cells, d, axis, nq=4, ng=16, parallel_tol=1e-9,
                  chunk=256, return_kernel=False):
    """Partial mutuals (H) between one wire segment and many voxel bars.

    ``fs`` is the segment's element list (shared p0/u/length), ``cells``
    the (m, 3) LOW-cell lattice coordinates of the bars (SuperPEEC's
    convention: the bar at cell c runs from the centre of c to the
    centre of c+1 along ``axis``, i.e. ``p0 = (c + 0.5)*d`` -- the same
    frame :func:`voxel_bar` uses, with the world origin at the corner
    of voxel (0,0,0)). Returns a (len(fs), m) array.

    Vectorised over every element and every bar at once; the skew path
    is chunked over cells to bound the (n1, ng, mc*n2) intermediate.
    Agreement with the scalar :func:`mutual` is gated in
    validate_wirecoupler.py part A.
    """
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
    m = cells.shape[0]
    f0, p1, wt1, eid1, A1, cplx = _concat_seg(fs)
    E, n1 = len(fs), p1.shape[0]
    dv = np.asarray(d, dtype=float)
    if dv.ndim == 0:
        dv = np.full(3, float(dv))
    off3, ar2, A2 = voxel_xsec(dv, axis, nq)
    n2 = ar2.size
    l2 = dv[axis]
    base = (cells + 0.5)*dv[None, :]          # bar start-face centres
    out = np.zeros((E, m), dtype=(np.complex128 if cplx else np.float64))
    if m == 0:
        return out
    dot = float(f0.u[axis])
    if abs(dot) < parallel_tol:
        # perpendicular straight filaments have zero mutual -- the fact
        # the whole e/f/g orientation split (and the three-point-source
        # far field) rests on. Skip the arithmetic, not just the result.
        if return_kernel:
            return np.zeros((p1.shape[0], m)), eid1
        return out
    if abs(abs(dot) - 1.0) < parallel_tol:
        # exact parallel path, all pairs at once
        sgn = 1.0 if dot > 0 else -1.0
        u = f0.u
        s1a = p1 @ u                          # (n1,)
        s1b = s1a + f0.length
        p2 = base[:, None, :] + off3[None, :, :]      # (m, n2, 3)
        s2a = p2 @ u
        s2b = s2a + sgn*l2
        lo2 = np.minimum(s2a, s2b)
        hi2 = np.maximum(s2a, s2b)
        dp = p1[:, None, None, :] - p2[None, :, :, :]
        dax = dp @ u
        dperp = dp - dax[..., None]*u[None, None, None, :]
        dist = np.maximum(np.linalg.norm(dperp, axis=-1), 1e-30)
        M = mutual_parallel(s1a[:, None, None], s1b[:, None, None],
                            lo2[None, :, :], hi2[None, :, :], dist)
        vals = np.einsum('imj,j->im', M, ar2)         # (n1, m)
        K = sgn*MU0/(4*np.pi)/A2*vals                 # pointwise rows
        if return_kernel:
            return K, eid1
        S1 = np.zeros((E, n1), dtype=wt1.dtype)
        S1[eid1, np.arange(n1)] = wt1
        out = (S1 @ K)/A1[:, None]
        return out if cplx else np.real(out)
    # skew path: outer Gauss along the wire, analytic point-to-segment
    # inner along the bar. u2 is an axis unit vector, so the axial/
    # transverse split needs no dot products -- t is the axis component.
    others = [k for k in range(3) if k != axis]
    x, w = np.polynomial.legendre.leggauss(ng)
    sq = 0.5*f0.length*(x + 1.0)
    jac = 0.5*f0.length*w
    P = p1[:, None, :] + sq[None, :, None]*f0.u[None, None, :]  # (n1,ng,3)
    Kbuf = (np.zeros((n1, m)) if return_kernel else None)
    S1 = np.zeros((E, n1), dtype=wt1.dtype)
    S1[eid1, np.arange(n1)] = wt1
    for c0 in range(0, m, chunk):
        sl = np.s_[c0:c0 + chunk]
        p2 = (base[sl][:, None, :] + off3[None, :, :])  # (mc, n2, 3)
        mc = p2.shape[0]
        t = P[:, :, None, None, axis] - p2[None, None, :, :, axis]
        h2 = np.zeros(t.shape)
        for k in others:
            dk = P[:, :, None, None, k] - p2[None, None, :, :, k]
            h2 += dk*dk
        h = np.sqrt(np.maximum(h2, 1e-24))
        inner = np.arcsinh((l2 - t)/h) - np.arcsinh(-t/h)
        vals = np.einsum('iqmj,q,j->im', inner, jac, ar2)   # (n1, mc)
        if return_kernel:
            Kbuf[:, sl] = vals
        else:
            out[:, sl] = S1 @ vals
    if return_kernel:
        return dot*MU0/(4*np.pi)/A2*Kbuf, eid1
    out *= dot*MU0/(4*np.pi)/(A1[:, None]*A2)
    return out if cplx else np.real(out)


def mutual_block(fs1, fs2, ng=16, parallel_tol=1e-9,
                 return_kernel=False):
    """Per-element block of partial mutuals between two DISTINCT wire
    segments: returns (len(fs1), len(fs2)).

    Same-segment blocks need the GMD self terms -- use
    :func:`segment_self_block` for those. Parallel segment pairs take
    the exact axial closed form; skewed pairs the analytic-inner +
    Gauss-outer scheme, whose M12/M21 asymmetry remains the free
    quadrature-error estimator (see :func:`mutual_skew_lines`).
    """
    f1, p1, wt1, eid1, A1, cplx1 = _concat_seg(fs1)
    f2, p2, wt2, eid2, A2, cplx2 = _concat_seg(fs2)
    cplx = cplx1 or cplx2
    dot = float(f1.u @ f2.u)
    if abs(dot) < parallel_tol:      # perpendicular: exactly zero
        if return_kernel:
            return (np.zeros((p1.shape[0], p2.shape[0])), eid1, eid2)
        return np.zeros((len(fs1), len(fs2)),
                        dtype=(np.complex128 if cplx else np.float64))
    if abs(abs(dot) - 1.0) < parallel_tol:
        sgn = 1.0 if dot > 0 else -1.0
        u = f1.u
        s1a = p1 @ u
        s1b = s1a + f1.length
        s2a = p2 @ u
        s2b = s2a + sgn*f2.length
        lo2 = np.minimum(s2a, s2b)
        hi2 = np.maximum(s2a, s2b)
        dp = p1[:, None, :] - p2[None, :, :]
        dax = dp @ u
        dperp = dp - dax[..., None]*u[None, None, :]
        dist = np.maximum(np.linalg.norm(dperp, axis=-1), 1e-30)
        M = mutual_parallel(s1a[:, None], s1b[:, None],
                            lo2[None, :], hi2[None, :], dist)
        K = sgn*MU0/(4*np.pi)*M
        if return_kernel:
            return K, eid1, eid2
        B = _block_reduce(K, wt1, eid1, len(fs1), wt2, eid2, len(fs2))
        B = B/(A1[:, None]*A2[None, :])
        return B if cplx else np.real(B)
    x, w = np.polynomial.legendre.leggauss(ng)
    sq = 0.5*f1.length*(x + 1.0)
    jac = 0.5*f1.length*w
    P = p1[:, None, :] + sq[None, :, None]*f1.u[None, None, :]
    W = P[:, :, None, :] - p2[None, None, :, :]
    t = W @ f2.u
    h2 = (W*W).sum(-1) - t*t
    h = np.sqrt(np.maximum(h2, 1e-24))
    inner = np.arcsinh((f2.length - t)/h) - np.arcsinh(-t/h)
    vals = np.einsum('iqj,q->ij', inner, jac)
    K = dot*MU0/(4*np.pi)*vals
    if return_kernel:
        return K, eid1, eid2
    B = _block_reduce(K, wt1, eid1, len(fs1), wt2, eid2, len(fs2))
    B = B/(A1[:, None]*A2[None, :])
    return B if cplx else np.real(B)


def segment_self_block(fs, gmds=None):
    """The dense (E, E) self block of one segment.

    Off-diagonal pairs are exact parallel mutuals between distinct
    cross-section regions (their quadrature points never coincide, and
    the analytic axial integral keeps touching elements benign -- the
    touching-pair study measured 0.008% at n=3 already). The diagonal
    uses the parallel kernel with d -> each element's cross-section GMD,
    computed here (and cheap) unless supplied.
    """
    E = len(fs)
    if E > 1:
        B = mutual_block(fs, fs)
    else:
        B = np.zeros((1, 1), dtype=(np.complex128 if fs[0].complex_shape
                                    else np.float64))
    for k, f in enumerate(fs):
        g = gmd(f.off, f.area) if gmds is None else gmds[k]
        B[k, k] = mutual(f, f, self_gmd=g)
    return B




def segment_self_kernel(fs, gmds=None):
    """Pointwise (N1, N1) kernel of a segment's self block: parallel
    closed form at actual transverse distances, with WITHIN-ELEMENT
    point pairs evaluated at the element's cross-section GMD (the same
    convention segment_self_block realises after weighting). Geometry
    only -- reweight with the current shapes via reweight_block."""
    f0, p1, wt1, eid1, A1, cplx = _concat_seg(fs)
    u = f0.u
    s1a = p1 @ u
    s1b = s1a + f0.length
    dp = p1[:, None, :] - p1[None, :, :]
    dax = dp @ u
    dperp = dp - dax[..., None]*u[None, None, :]
    dist = np.maximum(np.linalg.norm(dperp, axis=-1), 1e-30)
    for k, f in enumerate(fs):
        g = gmd(f.off, f.area) if gmds is None else gmds[k]
        sel = (eid1 == k)
        dist[np.ix_(sel, sel)] = float(g)
    M = mutual_parallel(s1a[:, None], s1b[:, None],
                        s1a[None, :], s1b[None, :], dist)
    return MU0/(4*np.pi)*M, eid1


def reweight_block(K, eid1, eid2, fs1, fs2):
    """Contract a cached pointwise kernel with the CURRENT shapes:
    B[e1, e2] = sum_{i in e1, j in e2} wt1_i K_ij wt2_j / (A1 A2)."""
    wt1 = np.concatenate([f.wt for f in fs1])
    wt2 = np.concatenate([f.wt for f in fs2])
    A1 = np.array([f.A for f in fs1])
    A2 = np.array([f.A for f in fs2])
    B = _block_reduce(K, wt1, eid1, len(fs1), wt2, eid2, len(fs2))
    B = B/(A1[:, None]*A2[None, :])
    cplx = any(f.complex_shape for f in list(fs1) + list(fs2))
    return B if cplx else np.real(B)


def reweight_rows(K, eid1, fs1):
    """Contract cached wire->voxel kernel rows with the current
    shapes: C[e, c] = sum_{i in e} wt_i K_ic / A_e."""
    wt1 = np.concatenate([f.wt for f in fs1])
    A1 = np.array([f.A for f in fs1])
    S1 = np.zeros((len(fs1), wt1.size), dtype=wt1.dtype)
    S1[eid1, np.arange(wt1.size)] = wt1
    C = (S1 @ K)/A1[:, None]
    cplx = any(f.complex_shape for f in fs1)
    return C if cplx else np.real(C)


def voxel_bar(cell, d, axis, nq=4):
    """A lattice voxel filament as a :class:`Filament`.

    SuperPEEC's convention: a filament runs from one cell centre to the
    next along ``axis``, with a ``d x d`` cross-section. This is the
    other half of the hybrid -- everything a wire couples to in the near
    field is one of these.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim == 0:
        d = np.full(3, float(d))
    c = (np.asarray(cell, dtype=float) + 0.5)*d
    u = np.zeros(3)
    u[axis] = 1.0
    others = [k for k in range(3) if k != axis]
    off, ar = rect_points(d[others[0]], d[others[1]], nq, nq)
    v = np.zeros(3)
    v[others[0]] = 1.0
    return Filament(c, u, d[axis], off, ar, v=v)


def foot_constriction(r0, rho, equipotential=True):
    """Analytic constriction (spreading) resistance of a bond foot.

    WHY ANALYTIC HERE AND MESHED IN FastHenry. FastHenry resolves the
    constriction by locally refining its ground plane -- `contact point`
    cuts cells down to the contact width, `decay_rect` grades cells away
    from it, `equiv_rect` shorts a patch into an equipotential pad. That
    works because its planes are QUADTREE meshed. SuperPEEC's lattice is
    UNIFORM by construction: the FFT/Toeplitz acceleration needs
    translation invariance (the same reason layered Green's functions
    were rejected outright), so local refinement is unavailable and the
    singularity has to be handled in closed form.

    MEASURED JUSTIFICATION (studies/bond_foot.py): a single-node
    attachment has NO converged value -- R ~ 1/dx over four
    refinements, 8.4x too resistive at dx = 6.25 um and worsening. Even
    a correctly sized footprint is ~20% low at r0/dx ~ 3 and converges
    slowly, because current density at a contact edge carries an
    inverse-square-root singularity.

    ``equipotential`` selects the boundary condition. A metal bond is an
    equipotential disc: rho/(4 r0). Uniform current density over the
    same disc gives 8 rho/(3 pi^2 r0), 8.1% higher. Measured ratio
    0.9365 against the exact 0.9255 at r0/dx = 3.2, so the machinery
    reproduces the distinction.

    HOW TO COMBINE THIS WITH A MESH -- NOT YET SETTLED, DO NOT GUESS.
    A first attempt subtracted a radial shell, rho/(2 pi b), for the
    part the mesh supposedly carries beyond radius b. That is WRONG
    whenever b < r0: inside the contact there is no spreading to
    subtract, and the expression went negative for every practical cell
    size. The mesh does not "take over at a radius" -- it resolves the
    whole patch and misplaces only the edge singularity, so the
    correction is MULTIPLICATIVE on the meshed result and must be
    calibrated against r0/dx. That calibration needs the finite-size
    bias in studies/bond_foot.py separated from the discretisation error
    first (currently ~-21% at L/r0 = 30, still moving). Until then use
    this as the physical REFERENCE VALUE, not as an additive series
    term.
    """
    return (rho/(4.0*r0) if equipotential
            else 8.0*rho/(3.0*np.pi**2*r0))
