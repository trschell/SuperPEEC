# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Reference coefficients of potential between rectangular panels.

Direct numerical quadrature of the defining integral

    P_ij = 1/(4 pi eps0 A_i A_j) Int_i Int_j dS dS' / |r - r'|

for two rectangles, parallel or perpendicular, of INDEPENDENT sizes at an
ARBITRARY offset. Slow, and singular when the panels touch, but it shares
no code with the closed forms in :mod:`greens` -- which is the point. It
is the oracle against which a closed-form kernel can be checked, and it
is what makes it safe to write one.

Validated against ``greens.gen_p_per`` to 2e-15 in the case that routine
covers (equal panels of side ``a`` on an integer lattice, with the
z-normal panel at ``x,y in [0,a], z = 0`` and the y-normal panel at
``x in [p a, (p+1) a], y = q a, z in [r a, (r+1) a]``) -- which also
pins down that routine's otherwise undocumented geometric convention.

WHY THIS EXISTS
---------------
``greens.gen_p_per(ax, ay, az, nx, ny, nz)`` uses its arguments as BOTH
the panel dimensions AND the grid strides, because it evaluates a
primitive on a lattice whose points sit at multiples of the panel size
and then takes fixed differences. That is fine when size and stride
coincide -- quarter panels of ``dx/2`` on a ``dx/2`` lattice -- but the
cell-centred discretisation needs whole ``dx`` faces on lattices
staggered by ``dx/2``, where they do not. The parallel case already has
a general form (``greens.gen_p_parz`` takes independent sizes and an
arbitrary offset); the perpendicular case does not, and this module is
the reference for writing one.

Accuracy: Gauss-Legendre converges fast while the panels are separated,
and degrades as they approach. :func:`converged` reports the difference
between two rules so a caller can see the error rather than assume it.
"""

import numpy as np

_C0 = 299792458.0
_MU0 = 4*np.pi*1e-7
_EPS0 = 1/(_MU0 * _C0**2)


def _rule(lo, hi, npt):
    u, w = np.polynomial.legendre.leggauss(npt)
    return 0.5*(hi - lo)*u + 0.5*(lo + hi), 0.5*(hi - lo)*w


def _panel_points(rect, npt):
    """Quadrature points and weights on a rectangle.

    ``rect`` is ``(axis, (u0, u1), (v0, v1), w)``: ``axis`` is the
    normal (0/1/2), the two intervals are the extents on the other two
    axes IN ASCENDING AXIS ORDER, and ``w`` is the plane coordinate.
    """
    axis, iu, iv, plane = rect
    other = [i for i in range(3) if i != axis]
    pu, wu = _rule(iu[0], iu[1], npt)
    pv, wv = _rule(iv[0], iv[1], npt)
    gu, gv = np.meshgrid(pu, pv, indexing='ij')
    pts = np.zeros((gu.size, 3))
    pts[:, other[0]] = gu.ravel()
    pts[:, other[1]] = gv.ravel()
    pts[:, axis] = plane
    return pts, np.outer(wu, wv).ravel()


def area(rect):
    """Area of a rectangle in :func:`_panel_points` form."""
    _, iu, iv, _ = rect
    return (iu[1] - iu[0])*(iv[1] - iv[0])


def p_panels(a, b, npt=12):
    """Coefficient of potential between two rectangles, in 1/F.

    Parameters
    ----------
    a, b : tuple
        ``(axis, (u0, u1), (v0, v1), plane)`` as in
        :func:`_panel_points`. Any two orientations; parallel and
        perpendicular are both handled, since nothing here depends on
        the relative orientation.
    npt : int, optional
        Gauss-Legendre points per dimension (4-D total cost ``npt**4``).

    Returns
    -------
    float
    """
    pa, wa = _panel_points(a, npt)
    pb, wb = _panel_points(b, npt)
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    if not np.all(d > 0):
        raise ValueError("panels touch or intersect: the integrand is "
                         "singular and quadrature will not converge")
    integral = np.einsum('i,j,ij->', wa, wb, 1.0/d)
    return integral/(4*np.pi*_EPS0*area(a)*area(b))


def converged(a, b, npt=12, coarse=None):
    """``(value, relative difference against a coarser rule)``.

    The second number is an honest error estimate: use it rather than
    assuming the quadrature is converged, because it is not when the
    panels are close.
    """
    coarse = coarse if coarse is not None else max(4, npt - 4)
    fine = p_panels(a, b, npt)
    crude = p_panels(a, b, coarse)
    return fine, abs(fine - crude)/abs(fine)


def rect(axis, u, v, plane):
    """Convenience constructor mirroring :func:`_panel_points`'s form."""
    return (int(axis), (float(u[0]), float(u[1])),
            (float(v[0]), float(v[1])), float(plane))


def face(cell, axis, sign, dx):
    """The outward face of a cubic cell, as a rectangle.

    ``cell`` is the integer cell index triple, ``axis`` the face normal
    and ``sign`` -1 or +1. Cell ``(i, j, k)`` occupies
    ``[i dx, (i+1) dx]`` on each axis, so this is the whole-face panel
    of the cell-centred discretisation.
    """
    lo = [c*dx for c in cell]
    other = [i for i in range(3) if i != axis]
    plane = lo[axis] + (dx if sign > 0 else 0.0)
    return rect(axis,
                (lo[other[0]], lo[other[0]] + dx),
                (lo[other[1]], lo[other[1]] + dx), plane)
