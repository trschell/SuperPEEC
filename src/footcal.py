# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""The bond-foot DISCRETISATION CALIBRATION -- exact, automatic.

THE PROBLEM. A wire foot injects its chain current into the voxel
lattice. The lattice then develops its own discrete spreading
resistance around the injection patch -- which never converges to the
physical constriction (the contact-edge current density has an
inverse-square-root singularity no uniform lattice resolves; measured
in studies/bond_foot.py: a single-node foot DIVERGES as 1/dx, a
resolved footprint is still 23-52%% low at r0/dx = 1.6-6.4). Adding
the analytic constriction on top double-counts whatever the lattice
does carry, by a mesh-dependent amount.

THE FIX. Give the model term exactly the DEFICIT:

    R_foot(model) = R_disc(r0, rho) * h(r0/dx),   h = 1 - g

where R_disc = 8 rho/(3 pi^2 r0) is the uniform-flux disc constriction
(the boundary condition a PRESCRIBED-split patch actually imposes --
matched BCs, no mixing with the equipotential value) and

    g(r0/dx) = R_lattice_patch / R_disc

is the fraction the lattice itself carries. Then for the half-space
ideal, lattice + model == R_disc IDENTICALLY at every resolution: the
calibration cancels the lattice's patch self-resistance and replaces
it with the continuum value. What remains approximate is only that a
real pad's near-foot field is not a half-space (finite thickness and
extent); the refinement gate in validate_footcal.py measures that.

THE MEASUREMENT IS EXACT, NOT EXTRAPOLATED. R_lattice_patch is
computed from the INFINITE simple-cubic lattice Green's function --
the classic Bessel-integral representation

    G(r) = 1/2 * int_0^inf  ive(rx,u) * ive(ry,u) * ive(rz,u) du

(ive = exponentially scaled modified Bessel I) with a two-term
asymptotic tail, and the insulating pad surface enters by a single
image (reflect across the plane between the surface layer and the
outside: G_half(dr) = G(dx,dy,0) + G(dx,dy,1) for cells in one
layer). No finite block, no boundary bias, no extrapolation -- which
is what dissolved the "separate the finite-size bias first" blocker
recorded in the bond-foot study. Gated in validate_footcal.py against
the exact adjacent-node resistance of the infinite 1-ohm cubic
network (1/3 ohm) and an independent finite-block solve.

WEIGHTS MUST MATCH THE SOLVER. The patch is the set of surface cells
the physical contact disc covers, with DISC-COVERAGE area weights --
:func:`patch_offsets` is used both here and by WireBondSolver's foot
construction, so the calibrated g and the injected split are the same
object by construction.

CUBIC CELLS ONLY (the lattice Green's function above is the isotropic
one); anisotropic pitches raise, matching the skin-engine precedent.
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import ive

_GCACHE = {}


def green(rx, ry, rz, T=200.0):
    """Infinite simple-cubic lattice Green's function G(r).

    Unit-conductance edges; the potential at r for unit current
    injected at 0 (extracted at infinity) is G(r), and the two-node
    resistance is 2*(G(0) - G(r)). Exact up to quadrature: finite
    integral on [0, T] plus the two-term asymptotic tail
    prod(ive) ~ (2 pi u)^{-3/2} (1 - (sum(4 r_i^2 - 1))/(8u)).
    """
    key = tuple(sorted((abs(int(rx)), abs(int(ry)), abs(int(rz)))))
    if key in _GCACHE:
        return _GCACHE[key]
    a, b, c = key

    def f(u):
        return ive(a, u)*ive(b, u)*ive(c, u)

    val, _ = quad(f, 0.0, T, limit=400)
    # tail: (2 pi u)^{-3/2} (1 - s/(8u)), s = sum(4 n^2 - 1)
    s = 4.0*(a*a + b*b + c*c) - 3.0
    t1 = (2.0*np.pi)**-1.5 * 2.0*T**-0.5
    t2 = (2.0*np.pi)**-1.5 * (s/8.0) * (2.0/3.0)*T**-1.5
    out = 0.5*(val + t1 - t2)
    _GCACHE[key] = out
    return out


def patch_offsets(r0_over_dx, nsub=32):
    """Surface-cell offsets and DISC-COVERAGE weights for a contact of
    radius ``r0`` (in cells) centred at the origin of a cell-centre
    lattice offset by (0.5, 0.5).

    Returns (offsets (m, 2) int, weights (m,) summing to 1). The
    contact centre sits at a CELL CENTRE (the doctrine's tie rule);
    coverage is the sampled fraction of the disc over each cell.
    """
    x = float(r0_over_dx)
    n = int(np.ceil(x)) + 1
    off, wt = [], []
    ss = (np.arange(nsub) + 0.5)/nsub
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            # cell (i, j) spans [i-0.5, i+0.5) x [j-0.5, j+0.5) around
            # the contact centre at (0, 0)
            px = i - 0.5 + ss[:, None]
            py = j - 0.5 + ss[None, :]
            cov = float((px*px + py*py <= x*x).mean())
            if cov > 0.0:
                off.append((i, j))
                wt.append(cov)
    off = np.array(off, dtype=np.int64)
    wt = np.array(wt)
    return off, wt/wt.sum()


def lattice_patch_R(r0_over_dx):
    """Dimensionless lattice patch spreading resistance R * sigma * dx
    (energy definition, prescribed coverage-weight split) of the
    half-space with an insulating surface."""
    off, w = patch_offsets(r0_over_dx)
    m = off.shape[0]
    R = 0.0
    for j in range(m):
        for k in range(m):
            d = off[j] - off[k]
            gh = (green(d[0], d[1], 0) + green(d[0], d[1], 1))
            R += w[j]*w[k]*gh
    return R


def gh_curve(r0_over_dx):
    """(g, h) at one ratio: g = lattice-carried fraction of the
    uniform-flux disc constriction, h = 1 - g = the model's share."""
    x = float(r0_over_dx)
    Rlat = lattice_patch_R(x)                  # R * sigma * dx
    Rdisc = 8.0/(3.0*np.pi**2*x)               # (8 rho/(3 pi^2 r0)) * sigma * dx
    g = Rlat/Rdisc
    return g, 1.0 - g


def foot_resistance(r0, dx, rho):
    """The CALIBRATED foot model resistance for one contact (ohms):
    R_disc(r0, rho) * h(r0/dx). Composed with the lattice's own patch
    spreading this reproduces the uniform-flux disc constriction
    exactly on the half-space ideal, at every resolution."""
    g, h = gh_curve(r0/dx)
    if h < 0.0:
        # cannot happen for the half-space ideal (the lattice converges
        # to the continuum FROM BELOW -- the edge singularity it cannot
        # carry); guard it anyway so a future change fails loudly
        # rather than injecting negative resistance
        raise RuntimeError("calibrated foot deficit went negative "
                           "(h = %g at r0/dx = %g)" % (h, r0/dx))
    return (8.0*rho/(3.0*np.pi**2*r0))*h
