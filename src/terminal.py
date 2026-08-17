# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Terminal (half-length) port filaments and their mutual inductance.

With nodes at cell centres, a conductor run of ``L`` cells carries ``L``
nodes but only ``L-1`` filaments between them, so the chain spans
``(L-1)*dx`` while the conductor is ``L*dx`` long. A terminal filament of
length ``dx/2`` at each port face closes the gap; see
``docs/unitcell_whitepaper.tex`` Section IV.

THE COUPLING PROBLEM, AND WHY IT NEEDS NO NEW FORMULA
-----------------------------------------------------
A terminal filament is half as long as an ordinary one, so the coupling
between them is a mutual partial inductance between bars of UNEQUAL
axial extent. The superposition identity behind ``greens.genL3D``
assumes IDENTICAL bars on a lattice whose pitch equals the bar size, so
it cannot produce that coupling directly, and the obvious route is the
general unequal-bar closed form (Hoer--Love).

That turns out to be unnecessary. Halve the lattice along the filament
axis only. Every bar in the problem is then a contiguous run of
half-slots of length ``dx/2``:

    terminal filament  = 1 half-slot
    ordinary filament  = 2 half-slots

and all the half-slots are IDENTICAL bars on a uniform pitch-``dx/2``
axial lattice -- exactly what ``genL3D`` already handles. Because the
partial inductance is

    Mp(A,B) = (mu0/4pi) / (Aa*Ab) * Int_A Int_B 1/|r-r'|

and every bar here has the SAME cross-section, splitting a bar
axially splits the integral with no change of normalisation:

    Mp(A,B) = sum_i sum_j Mp(a_i, b_j)                            (*)

over the half-slots of A and B. Series segments carry the same current,
so this is the ordinary series-partial-inductance sum, self terms
included. ``validate_terminal.py`` checks (*) against the full-pitch
kernel to machine precision, and checks the result against an
independent 6-D quadrature.

The cost is confined: the axial lattice doubles only for the couplings
that involve a terminal, and terminals number with the PORT AREA, not
the conductor volume. The interior--interior block stays identical-bar
and keeps its Toeplitz/FFT treatment untouched.

HALF-SLOT LAYOUT AT A PORT
--------------------------
Measuring from the port face in units of ``dx/2``, a run of ``L`` cells
driven face to face lays out as::

    slot:   0 | 1   2 | 3   4 | ...        | 2L-2 | 2L-1
            T |  fil 0  |  fil 1 |          | fil L-2 |  T

one terminal, ``L-1`` ordinary filaments, one terminal: ``1 + 2(L-1) + 1
= 2L`` half-slots, i.e. exactly ``L*dx``. See :func:`port_run_slots`.
"""

import numpy as np
from greens import genL3D, box_selfind
from stencils import FILAMENT_AXIS

# Tensor second-difference weights. Transversely both bars have the same
# extent, so each axis keeps genL3D's symmetric 1/2 [1, -2, 1] at sizes
# (|m-1|, m, m+1). Axially the two bars differ, and the operator becomes
# the MIXED difference
#     1/2 [ Psi(|s+q|) - Psi(|s|) - Psi(|s+q-p|) + Psi(|s-p|) ]
# for bars spanning [0, p] and [s, s+q]: the double integral of a kernel
# of the coordinate difference over two intervals is a second difference
# in their endpoints. At p == q it collapses to 1/2 [1, -2, 1] (the two
# middle arguments coincide), which is genL3D's own axial stencil -- so
# this is a strict generalisation, not a parallel implementation.
_TRANSVERSE_C = np.array([0.5, -1.0, 0.5])
_AXIAL_C = np.array([0.5, -0.5, -0.5, 0.5])

# A terminal filament is half as long as an ordinary one.
TERMINAL_SLOTS = 1
INTERIOR_SLOTS = 2


def axial_halfstep_kernel(l, orientation, n):
    """Mutual partial inductance on an axially halved lattice.

    Returns the Toeplitz generating block for bars of cross-section
    ``l`` transversely and length ``l[d]/2`` along their own axis ``d``,
    tabulated at integer separations: full cells transversely, HALF
    cells axially.

    Parameters
    ----------
    l : sequence of 3 float
        Cell dimensions (metres).
    orientation : {'e', 'f', 'g'}
    n : sequence of 3 int
        Separations to tabulate, in cells. The axial entry is doubled
        internally (plus one) so half-slot separations up to ``2*n[d]``
        are available.

    Returns
    -------
    ndarray
        Indexed ``[sx, sy, sz]`` in the tree's own axis order, with the
        axial index counted in HALF cells.

    Notes
    -----
    The argument shuffle mirrors ``LeafInduct.p2pinit``: ``genL3D``
    takes ``(dw, dl, dt)`` with ``dl`` the axial direction, so the
    element's own axis goes in the middle slot and the result is
    permuted back to ``(x, y, z)``.
    """
    d = FILAMENT_AXIS[orientation]
    others = [i for i in range(3) if i != d]
    dims = (l[others[0]], l[d]/2.0, l[others[1]])
    counts = (int(n[others[0]]), 2*int(n[d]) + 1, int(n[others[1]]))
    K = genL3D(dims[0], dims[1], dims[2], counts[0], counts[1], counts[2])
    perm = np.argsort([others[0], d, others[1]])
    return np.transpose(K, perm)


def unequal_kernel(l, orientation, n, p, q, s0):
    """Mutual partial inductance between parallel bars of UNEQUAL length.

    Both bars have the cell cross-section; bar A has axial extent ``p``
    and bar B extent ``q``, and their axial START positions differ by
    ``s0 + k*l[d]`` for integer ``k``. This is the arbitrary-length
    generalisation of :func:`axial_halfstep_kernel`, which only reaches
    lengths commensurate with a half cell.

    Unlike the half-step kernel this is NOT symmetric in the axial
    index -- the two bars sit at different sub-cell offsets -- so ``k``
    is SIGNED and the table is indexed ``k + n[d]``.

    Parameters
    ----------
    l : sequence of 3 float
        Cell dimensions (metres).
    orientation : {'e', 'f', 'g'}
    n : sequence of 3 int
        Separations to tabulate, in cells. The axial entry gives
        ``k = -n[d] .. +n[d]``.
    p, q : float
        Axial extents of the two bars (metres).
    s0 : float
        Axial offset of bar B's start from bar A's, at ``k = 0``.

    Returns
    -------
    ndarray
        Indexed ``[sx, sy, sz]`` in the tree's own axis order; the axial
        index is ``k + n[d]`` and the transverse ones are absolute cell
        separations.
    """
    d = FILAMENT_AXIS[orientation]
    others = [i for i in range(3) if i != d]
    n = [int(v) for v in n]
    ks = np.arange(-n[d], n[d] + 1)
    s = s0 + ks*l[d]
    ax_len = np.stack([np.abs(s + q), np.abs(s),
                       np.abs(s + q - p), np.abs(s - p)], axis=-1)
    m0 = np.arange(n[others[0]])
    m1 = np.arange(n[others[1]])
    w_len = np.stack([np.abs(m0 - 1), m0, m0 + 1], axis=-1)*l[others[0]]
    t_len = np.stack([np.abs(m1 - 1), m1, m1 + 1], axis=-1)*l[others[1]]
    out = np.zeros((m0.size, ks.size, m1.size))
    for a, ca in enumerate(_TRANSVERSE_C):
        for b, cb in enumerate(_AXIAL_C):
            for c, cc in enumerate(_TRANSVERSE_C):
                out += ca*cb*cc*box_selfind(
                    w_len[:, a][:, None, None],
                    ax_len[:, b][None, :, None],
                    t_len[:, c][None, None, :])
    out /= l[others[0]]**2 * l[others[1]]**2
    perm = np.argsort([others[0], d, others[1]])
    return np.transpose(out, perm)


def mutual_offset(kern, orientation, k, n_axial, transverse=(0, 0)):
    """Look up :func:`unequal_kernel` at signed axial separation ``k``."""
    d = FILAMENT_AXIS[orientation]
    others = [i for i in range(3) if i != d]
    idx = [0, 0, 0]
    idx[d] = int(k) + int(n_axial)
    idx[others[0]] = abs(int(transverse[0]))
    idx[others[1]] = abs(int(transverse[1]))
    return float(kern[idx[0], idx[1], idx[2]])


def mutual_segments(kern, orientation, span_a, span_b, transverse=(0, 0)):
    """Mutual partial inductance between two parallel bars.

    Each bar is given as a contiguous run of half-slots on the shared
    axial lattice, so this covers terminal-to-terminal,
    terminal-to-ordinary and ordinary-to-ordinary uniformly. Evaluates
    the sum (*) in the module docstring.

    Parameters
    ----------
    kern : ndarray
        From :func:`axial_halfstep_kernel`.
    orientation : {'e', 'f', 'g'}
    span_a, span_b : (int, int)
        ``(offset, length)`` in half-slots.
    transverse : (int, int)
        Separation in CELLS along the two non-axial directions, in
        ascending axis order.

    Returns
    -------
    float
        Partial inductance in henries. Passing a bar as both arguments
        gives its self partial inductance.
    """
    d = FILAMENT_AXIS[orientation]
    others = [i for i in range(3) if i != d]
    idx = [0, 0, 0]
    idx[others[0]] = abs(int(transverse[0]))
    idx[others[1]] = abs(int(transverse[1]))
    oa, la = span_a
    ob, lb = span_b
    total = 0.0
    for i in range(la):
        for j in range(lb):
            idx[d] = abs((ob + j) - (oa + i))
            total += kern[idx[0], idx[1], idx[2]]
    return float(total)


def port_run_slots(ncells, terminals=(True, True)):
    """Half-slot spans for a conductor run driven face to face.

    Parameters
    ----------
    ncells : int
        Cells along the run (``>= 1``).
    terminals : (bool, bool)
        Whether to close the low and high ends with a terminal
        filament. Both True is the face-to-face port; both False leaves
        the model half a cell short at each end (self-consistent, but a
        different geometry -- see the whitepaper).

    Returns
    -------
    list of (offset, length, kind)
        ``kind`` is ``'terminal'`` or ``'interior'``. Offsets are in
        half-slots measured from the low port face.
    """
    if ncells < 1:
        raise ValueError("ncells must be >= 1")
    lo, hi = terminals
    spans = []
    pos = 0 if lo else 1
    if lo:
        spans.append((pos, TERMINAL_SLOTS, 'terminal'))
        pos += TERMINAL_SLOTS
    for _ in range(ncells - 1):
        spans.append((pos, INTERIOR_SLOTS, 'interior'))
        pos += INTERIOR_SLOTS
    if hi:
        spans.append((pos, TERMINAL_SLOTS, 'terminal'))
        pos += TERMINAL_SLOTS
    return spans


def run_length_slots(ncells, terminals=(True, True)):
    """Total half-slots spanned by :func:`port_run_slots`.

    ``2*ncells`` with both terminals, i.e. exactly ``ncells*dx``.
    """
    return sum(s[1] for s in port_run_slots(ncells, terminals))


def box_mutual_matrix(lo, hi, axis):
    """Partial mutual inductance between PARALLEL axis-aligned bars.

    Bars of arbitrary size and position, all carrying current along
    ``axis`` (perpendicular bars have zero mutual partial inductance).
    ``Mp = S/(A_a A_b)`` with ``S`` the box-pair stencil and ``A`` each
    bar's cross-section perpendicular to ``axis`` -- the same identity
    ``genL3D`` uses, generalised to unequal boxes.

    This is what lets a full-cell filament and a half-width sub-filament
    couple: they are simply two boxes.

    Parameters
    ----------
    lo, hi : ndarray, shape (n, 3)
    axis : {0, 1, 2}

    Returns
    -------
    ndarray, shape (n, n)
        Partial mutual inductances (henries).
    """
    from greens import box_pair_stencil
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    ext = hi - lo
    others = [c for c in range(3) if c != int(axis)]
    area = ext[:, others[0]]*ext[:, others[1]]
    S = box_pair_stencil(lo, hi)
    return S/(area[:, None]*area[None, :])


def terminal_resistance(l, orientation, sigma, t_l=None):
    """Resistance of one terminal filament of axial length ``t_l``.

    Full cross-section, so simply ``t_l/(A sigma)``. ``t_l=None`` gives
    the half-cell default, i.e. half an ordinary filament's resistance.
    """
    d = FILAMENT_AXIS[orientation]
    others = [i for i in range(3) if i != d]
    if t_l is None:
        t_l = 0.5*l[d]
    return t_l/(l[others[0]]*l[others[1]]*sigma)
