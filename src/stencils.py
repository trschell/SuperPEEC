# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""The single definition of which discrete elements exist.

Seven element sets are derived from the voxel occupancy grid: three
filament orientations (``e`` y-directed, ``f`` x-directed, ``g``
z-directed -- matching main.py's resistance formulas, where ``e.r`` uses
``l[1]``, ``f.r`` uses ``l[0]`` and ``g.r`` uses ``l[2]``), the circuit
nodes, and three surface-panel orientations.

Those seven rules were written out three times -- once in tree.py's
single-level branch, once in its multilevel depth-first scan, and once
more in the Fortran ``get_node_size`` (which counts them to size the
arrays). They are collected here so there is ONE place to read them and
ONE place to change them.

THE SCHEME: CELL-CENTRED
------------------------
Nodes sit at cell CENTRES; a filament runs between two adjacent
centres and exists where BOTH cells conduct; a panel is a whole cell
face and exists where occupancy changes across it. Cross-sections tile
the conductor exactly, so ``R_DC = l/(sigma A)`` with no boundary
correction, and filaments share their lattice with same-normal panels.
This is the scheme of docs/unitcell_whitepaper.tex.

(A corner-node/edge-filament scheme predated this one; it over-counted
the conductor cross-section by ``(N+1)^2/N^2`` and was removed
2026-08-14 after a year deprecated. Its history lives in
docs/cell_centred_scope.md.)

THE PADDED-BLOCK CONVENTION
---------------------------
Every routine here takes ``pad``, a cell-occupancy block padded by one
layer, and ``shape``, the element lattice size to produce. Element
``(i, j, k)`` reads cells ``pad[i+a, j+b, k+c]`` for ``a, b, c`` in
``{0, 1}``, so ``a = 0`` selects the cell BELOW the element on that axis
and ``a = 1`` the cell above. Out-of-range cells must already be zero in
``pad``.

Both callers already have such a block: the single-level branch pads the
whole grid (``np.pad(fullstruc, 1)``), and the multilevel scan builds a
per-box ``singlegroup`` of shape ``nleaf + 1`` with the box's cells
placed inside it.
"""

import os as _os
import numpy as np

# The edge scheme was REMOVED 2026-08-14. The env knob is refused
# loudly rather than silently ignored: a pinned SPPEEC_SCHEME=edge in
# an old script would otherwise run cell and report different numbers
# with no hint why.
if _os.environ.get('SPPEEC_SCHEME', 'cell') != 'cell':
    raise RuntimeError(
        "SPPEEC_SCHEME=%r: the edge scheme was removed 2026-08-14 "
        "(deprecated since 2026-08-05; see docs/cell_centred_scope.md). "
        "Unset SPPEEC_SCHEME -- cell is the only scheme."
        % _os.environ['SPPEEC_SCHEME'])

# The axis each filament orientation runs along.
FILAMENT_AXIS = {'e': 1, 'f': 0, 'g': 2}

# The axis each panel orientation is normal to.
PANEL_AXIS = {'x': 0, 'y': 1, 'z': 2}

# Every element's own axis, and which orientations are panels. Used both
# here and by levels.leafinit, which needs the same notion of "own axis"
# to place elements for the FMM -- see its docstring for the stagger.
AXIS_OF = dict(FILAMENT_AXIS, **PANEL_AXIS)
PANEL_ORIENTATIONS = frozenset(PANEL_AXIS)


def element_shapes(nt):
    """Lattice dimensions of the seven element sets.

    The one place lattice sizes are defined, in the leaf order
    ``e, f, g, node, px, py, pz``.

    Parameters
    ----------
    nt : sequence of 3 int
        Cell-grid dimensions.

    Returns
    -------
    list of 7 tuples
    """
    x, y, z = (int(v) for v in nt)
    # one node per cell; a filament per adjacent pair, so one fewer
    # along its own axis; a panel per face, so one more
    return [(x, y-1, z), (x-1, y, z), (x, y, z-1), (x, y, z),
            (x+1, y, z), (x, y+1, z), (x, y, z+1)]


def single_level_nleaf(nt):
    """Leaf-box size for a single-level tree: one box holding everything.

    ``Tree`` sizes its near-field workspaces from ``nleaf``, so a
    single-level tree must be told a box big enough for the whole
    element lattice -- the cell grid itself; getting it wrong
    undersizes the P2P buffers, which corrupts the heap rather than
    raising.

    Parameters
    ----------
    nt : sequence of 3 int
        Cell-grid dimensions.

    Returns
    -------
    ndarray of int, shape (3,)
    """
    return np.asarray(nt, dtype=int).copy()


def port_node(cell, axis, sign):
    """The node a port face drives, as a single node coordinate.

    A port is naturally described where the geometry is -- a voxel FACE,
    ``(cell, axis, sign)`` with ``sign`` -1 for the low face and +1 for
    the high one. Nodes are cell centres, so the face drives ITS OWN
    CELL regardless of sign. (An off-grid node coordinate would not
    raise -- it aliases onto a different node or surfaces as
    ``parsesource``'s "Location is not within structure!" far from the
    cause -- which is why this stays a function rather than inline
    arithmetic at every port site.)

    Parameters
    ----------
    cell : sequence of 3 int
    axis : {0, 1, 2}
    sign : {-1, +1}

    Returns
    -------
    tuple of 3 int
    """
    return tuple(int(v) for v in cell)


def _axis_slice(a, axis, lo, hi):
    """``a`` sliced ``[lo:hi]`` along ``axis``, other axes untouched."""
    sl = [slice(None)]*3
    sl[axis] = slice(lo, hi)
    return a[tuple(sl)]


def cell_node_struc(fullstruc):
    """Cell scheme: one node per conductor cell."""
    return np.asarray(fullstruc).copy()


def cell_filament_struc(fullstruc, orientation):
    """Cell scheme: a filament where BOTH cells sharing a face conduct.

    The minimum rather than a boolean AND so a material tag carried in
    ``fullstruc`` survives as the weaker of the two cells.
    """
    d = FILAMENT_AXIS[orientation]
    a = np.asarray(fullstruc)
    return np.minimum(_axis_slice(a, d, 0, -1), _axis_slice(a, d, 1, None))


def cell_panel_struc(fullstruc, orientation):
    """Cell scheme: a panel on every face where occupancy changes.

    One value per cell face, so ``nt[d] + 1`` along the normal. No
    transverse refinement: a whole face belongs to the single cell
    behind it, which is what makes panels congruent with cell-centred
    filaments.
    """
    d = PANEL_AXIS[orientation]
    a = np.asarray(fullstruc)
    pw = [(0, 0)]*3
    pw[d] = (1, 1)
    p = np.pad(a, pw)
    return np.abs(_axis_slice(p, d, 1, None).astype(np.int16)
                  - _axis_slice(p, d, 0, -1).astype(np.int16)).astype(a.dtype)


def struc_from_cells(fullstruc):
    """The seven element occupancies for a whole cell block.

    Shapes are :func:`element_shapes` of the block's own dimensions.

    Parameters
    ----------
    fullstruc : ndarray
        Cell occupancy, used directly (no padding needed).

    Returns
    -------
    list of 7 ndarrays
        In leaf order ``e, f, g, node, px, py, pz``.
    """
    a = np.asarray(fullstruc)
    # NORMALISE to {0, 1} exactly as struc_from_block does: struc
    # VALUES become node2panel projection weights and beta scalings
    # downstream, and a material-ID grid (dielectrics carry id 2)
    # otherwise leaks the id into them. Measured on the half-filled
    # plate model (SINGLE-LEVEL trees take this path -- every
    # phase-2 validator did): n2n blocks gained exactly 2x per
    # dielectric index (4x diel-diel), which crushed the free-
    # surface bound-charge dipole to ~5% of its true effect
    # (2-terminal C ratio 1.06 vs the BEM reference 1.59).
    one = np.asarray(1, dtype=a.dtype)
    return [np.minimum(x, one)
            for x in (cell_filament_struc(a, 'e'),
                      cell_filament_struc(a, 'f'),
                      cell_filament_struc(a, 'g'),
                      cell_node_struc(a),
                      cell_panel_struc(a, 'x'),
                      cell_panel_struc(a, 'y'),
                      cell_panel_struc(a, 'z'))]


def _blk(b, off, shape):
    """``b`` offset by ``off`` and cropped to ``shape``."""
    return b[off[0]:off[0]+shape[0],
             off[1]:off[1]+shape[1],
             off[2]:off[2]+shape[2]]


def struc_from_block(block, shape):
    """The seven element occupancies for one box of the multilevel scan.

    Multilevel stores a flat ``nleaf**3`` block of EVERY element type per
    box, with each element's overhang living in the neighbouring box. So
    ``block`` must carry the box's cells plus ONE OVERHANG LAYER ON EACH
    SIDE: ``block[i+1, j+1, k+1]`` is box-local cell ``(i, j, k)``.

    Both sides are needed because the two element families reach
    opposite ways. A cell-centred filament at local ``i`` joins cells
    ``i`` and ``i+1``, so it reaches HIGH; a panel at local ``i`` is the
    face between cells ``i-1`` and ``i``, so it reaches LOW. One layer
    each side covers both, and costs nothing -- ``block`` is a scratch
    buffer.

    Parameters
    ----------
    block : ndarray
        Cell occupancy, shape ``shape + 2`` on every axis.
    shape : sequence of 3 int
        The box's element block size (``nleaf``).

    Returns
    -------
    list of 7 ndarrays
        In leaf order ``e, f, g, node, px, py, pz``, all of the given
        shape.
    """
    b = np.asarray(block)
    shape = tuple(int(v) for v in shape)
    core = _blk(b, (1, 1, 1), shape)
    out = []
    # MATERIAL-ID grids (dielectric support phase 2, 2026-08-07): the
    # occupancy may carry ids (0 empty, 1 conductor, 2 dielectric)
    # rather than booleans. The rules below then give the right
    # EXISTENCE sets -- in particular |a - b| fires panels at BURIED
    # conductor-dielectric interfaces, which is exactly where bound
    # charge must live -- but their VALUES leak into projection
    # weights downstream, so every output is normalised to {0, 1}.
    # For binary inputs the normalisation is a bit-level no-op.
    for o in ('e', 'f', 'g'):
        hi = [1, 1, 1]
        hi[FILAMENT_AXIS[o]] = 2
        out.append(np.minimum(np.minimum(core, _blk(b, hi, shape)),
                              np.asarray(1, dtype=b.dtype)))
    out.append(np.minimum(core, np.asarray(1, dtype=b.dtype)))
    for o in ('x', 'y', 'z'):
        lo = [1, 1, 1]
        lo[PANEL_AXIS[o]] = 0
        out.append(np.minimum(np.abs(core.astype(np.int16)
                                     - _blk(b, lo, shape).astype(np.int16)
                                     ).astype(b.dtype),
                              np.asarray(1, dtype=b.dtype)))
    return out


def count_elements(fullstruc, nleaf, ngroups):
    """Element counts used to size the multilevel arrays.

    Drop-in replacement for ``mp_fortran.get_node_size``, computed from
    the SAME rules the depth-first scan then fills the arrays with. That
    matters: these counts allocate the arrays, so a counter that
    disagrees with the filler truncates silently. Deriving both from
    this module removes that failure mode by construction rather than
    by testing for it.

    Parameters
    ----------
    fullstruc : ndarray
        Voxel occupancy, in the tree's own ``(x, y, z)`` order (NOT
        transposed as the Fortran routine wanted it).
    nleaf : sequence of 3 int
        Cells per leaf box.
    ngroups : sequence of 3 int
        Leaf boxes per axis, so the padded lattice is
        ``ngroups * nleaf``.

    Returns
    -------
    ndarray, shape (7, 2), dtype int
        Column 0 is the total count of each element set in the leaf
        order ``e, f, g, node, px, py, pz``; panels are UNREFINED, so
        the caller applies the factor of 4 for the 2x2 face split, as
        the Fortran routine also required. Column 1 is the number of
        non-empty boxes, of which only the node row is consumed.
    """
    nleaf = np.asarray(nleaf, dtype=int)
    ngroups = np.asarray(ngroups, dtype=int)
    full = np.asarray(ngroups*nleaf, dtype=int)
    nt = np.asarray(fullstruc.shape, dtype=int)
    # The boxes tile the padded lattice, so counting per box and summing
    # is the same as applying the rules once over the whole lattice --
    # done here through struc_from_block, the very routine the scan uses,
    # with the same one-layer-each-side overhang convention.
    block = np.zeros(tuple(full + 2), dtype=fullstruc.dtype)
    hi = np.minimum(nt, full)
    block[1:1+hi[0], 1:1+hi[1], 1:1+hi[2]] = \
        fullstruc[:hi[0], :hi[1], :hi[2]]
    shape = tuple(int(v) for v in full)
    strucs = struc_from_block(block, shape)
    out = np.zeros((7, 2), dtype=int)
    occupied = None
    for i, s in enumerate(strucs):
        out[i, 0] = int(np.count_nonzero(s))
        blocks = s.reshape(ngroups[0], nleaf[0], ngroups[1], nleaf[1],
                           ngroups[2], nleaf[2]).any(axis=(1, 3, 5))
        out[i, 1] = int(np.count_nonzero(blocks))
        occupied = blocks if occupied is None else (occupied | blocks)
    # tree.py stores a box holding ANY element, so the group count --
    # which sizes idx0 -- must be counted the same way. A box can carry
    # surface panels without carrying nodes; see the comment at the
    # storage gate in Tree.__init__.
    out[3, 1] = int(np.count_nonzero(occupied))
    return out


