# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Export SuperPEEC results to Blender (glTF 2.0 binary, ``.glb``).

ParaView gets the volume field (:mod:`vtkout`); Blender gets the
SURFACE. The two are different questions. ParaView answers "what is
the field doing inside the metal"; Blender answers "what does this
module look like, and where is the current crowding on the parts you
can see" -- lit, shaded, camera-able, and renderable to a figure or a
turntable animation without anyone owning a VTK licence or knowing
what a threshold filter is.

WHY glTF AND NOT A .blend. ``.glb`` is a published, self-contained
binary format that Blender imports natively (File > Import > glTF
2.0) with no addon, no Blender on the solving machine, and no version
coupling -- the same file also opens in Windows 3D Viewer, macOS
Quick Look, Godot, three.js and every DCC tool that matters. A
``.blend`` would need Blender in the solve environment and would rot
against Blender releases. The companion ``studies/blender_view.py``
turns the ``.glb`` into a shaded, camera'd scene when Blender IS
available.

WHAT IS EXPORTED

* The CONDUCTOR SURFACE as axis-aligned quads: exactly those voxel
  faces where metal meets air (or the domain boundary). Interior
  faces are skipped, so a 10-million-cell module becomes ~1e6 quads,
  not 1e7 boxes. Each quad carries the TANGENTIAL current density of
  the cell behind it -- the component parallel to the surface, which
  is the physical surface current; the normal component through an
  exposed face is zero by construction (current cannot leave the
  metal) and including it would only add discretisation noise.

  Dropping the normal component also sidesteps a real artefact.
  ``vtkout.current_density`` averages the two filaments on opposite
  faces of a cell and counts an absent neighbour as zero, so at a
  surface the component ALONG the surface normal comes out halved --
  correct for the volume field (the current terminates there), and
  wrong by a factor of two if read as a surface current. The
  tangential components have live filaments on both sides and are
  unaffected, so |J_tan| at the skin is the honest number where
  |J| would not be.

* The BOND WIRES as real swept tubes at their physical radius, one
  colour band per solved segment, with sphere joints at the polyline
  corners so the elbows close. These are the same chain currents
  ``vtkout.export_wires`` writes as ``.vtp`` lines -- here they have
  thickness, so they read as wire in a rendered image.

* The FEET: each contact flares from the wire radius out to the
  model's contact radius ``foot_r0`` (default twice the wire radius)
  and seats on a filled disc lying on the pad. This is not dressing.
  The foot constriction the solver adds is computed FOR that disc --
  ``R_disc(r0, rho) * h(r0/dx)`` from :mod:`footcal` -- so a wire
  drawn as a bare cylinder ending in mid-air omits a term that is in
  the answer. The disc seats on the outward face of the solver's own
  anchor cell, so it also shows WHERE each wire was landed.

COLOUR IS BAKED, AND ALSO NOT. Every vertex carries both:

* ``COLOR_0`` -- an RGBA the viewer shows immediately, from a
  logarithmic map over four decades below the maximum (|J| on a
  conductor spans orders of magnitude; a linear ramp shows one hot
  pad and a black board). Blender wires this into Base Color on
  import, so the file is correct the moment it opens.
* ``_JMAG`` -- the RAW scalar in A/m^2 (A for wires), as a glTF
  custom attribute. Blender 3.4+ imports it as a mesh attribute, so
  the ramp can be rebuilt in shader nodes against the true numbers
  without re-exporting. The colour is a convenience; this is the
  data.

UNITS AND ORIENTATION. Geometry is metres, and a 40 mm module at 1
Blender unit = 1 m sits inside the default 0.1 m near clip -- it is
invisible until you fight the viewport. So positions are scaled by
``scale`` (default 1000: one Blender unit = one millimetre) and the
factor is recorded in the asset ``extras``. Coordinates are written
Y-up per the glTF convention, which Blender's importer converts back
to Z-up with its default settings -- so the model arrives in
Blender's axes matching the solver's.

Usage as a library::

    import blendout
    blendout.export_scene(
        m, M, info['i_f'], 'module.glb',
        parts=prob.block_cells(m),          # one object per [[block]]
        wires=sw.sol.wires, i_w=info['i_w'],
        seg0=sw.sol.wc.seg0, wire_of_seg=sw.sol.wire_of_seg,
        foot_cell=sw.sol.foot_cell, foot_r0=sw.sol.foot_r0,
        freq=freq)

Everything after ``path`` is optional: without ``parts`` the skin is a
single object, and without the wire arguments there are no wires.

From the CLI, ``sppeec_cli.py --export-glb`` writes one per solved
frequency alongside the ``.vti``/``.vtp`` exports; the rendering half
is ``studies/blender_view.py``. Both are documented in the README.
"""
import json
import os
import struct

import numpy as np

import vtkout

RESULTS_DIR = 'results'

# Decades of dynamic range shown below the maximum. Four matches the
# LogNorm floor the README figure uses (studies/dbc_r4_plot.py); below
# that the board is black and above it the crowding washes out.
DECADES = 4.0

# 17-point control tables, linearly interpolated at shading time --
# visually indistinguishable from the 256-entry originals and keeps
# src/ free of a matplotlib dependency (vtkout has none either).
_CMAPS = {
    'inferno': (
        (0.0015, 0.0005, 0.0139), (0.0423, 0.0281, 0.1411),
        (0.1293, 0.0473, 0.2908), (0.2383, 0.0366, 0.3964),
        (0.3415, 0.0623, 0.4294), (0.4412, 0.0993, 0.4316),
        (0.5409, 0.1347, 0.4151), (0.6401, 0.1714, 0.3811),
        (0.7357, 0.2159, 0.3302), (0.8224, 0.2752, 0.2661),
        (0.8943, 0.3534, 0.1936), (0.9470, 0.4492, 0.1153),
        (0.9784, 0.5579, 0.0349), (0.9879, 0.6753, 0.0653),
        (0.9746, 0.7977, 0.2063), (0.9476, 0.9174, 0.4107),
        (0.9884, 0.9984, 0.6449)),
    'viridis': (
        (0.2670, 0.0049, 0.3294), (0.2823, 0.0950, 0.4173),
        (0.2788, 0.1755, 0.4834), (0.2590, 0.2515, 0.5247),
        (0.2297, 0.3224, 0.5457), (0.1994, 0.3876, 0.5546),
        (0.1727, 0.4488, 0.5579), (0.1490, 0.5081, 0.5573),
        (0.1276, 0.5669, 0.5506), (0.1206, 0.6258, 0.5335),
        (0.1579, 0.6838, 0.5017), (0.2461, 0.7389, 0.4520),
        (0.3692, 0.7889, 0.3829), (0.5160, 0.8312, 0.2943),
        (0.6785, 0.8637, 0.1895), (0.8456, 0.8873, 0.0997),
        (0.9932, 0.9062, 0.1439)),
    'turbo': (
        (0.1900, 0.0718, 0.2322), (0.2511, 0.2524, 0.6337),
        (0.2763, 0.4212, 0.8912), (0.2586, 0.5796, 0.9988),
        (0.1584, 0.7355, 0.9231), (0.0927, 0.8655, 0.7623),
        (0.1966, 0.9490, 0.5947), (0.4278, 0.9942, 0.3857),
        (0.6436, 0.9900, 0.2336), (0.8047, 0.9245, 0.2046),
        (0.9330, 0.8124, 0.2267), (0.9931, 0.6741, 0.2035),
        (0.9836, 0.4929, 0.1285), (0.9211, 0.3149, 0.0548),
        (0.8161, 0.1846, 0.0181), (0.6645, 0.0844, 0.0042),
        (0.4796, 0.0158, 0.0106)),
}


def shade(values, vmin=None, vmax=None, cmap='inferno', log=True):
    """Map a scalar field to RGBA bytes; returns ``(colors, vmin, vmax)``.

    ``log`` (the default) is a decade map: ``vmax`` is the field
    maximum and ``vmin`` sits :data:`DECADES` below it unless given.
    Zeros and negatives clamp to the bottom of the ramp rather than
    producing NaNs -- an unenergised trace is dark, not invisible.
    """
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    hi = float(vmax) if vmax is not None else (
        float(finite.max()) if finite.size and finite.max() > 0 else 1.0)
    if vmin is not None:
        lo = float(vmin)
    else:
        lo = hi/10.0**DECADES if log else 0.0
    if log:
        lo = max(lo, np.finfo(np.float64).tiny)
        t = (np.log10(np.maximum(v, lo)) - np.log10(lo)) \
            / max(np.log10(hi) - np.log10(lo), 1e-12)
    else:
        t = (v - lo)/max(hi - lo, 1e-300)
    t = np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)

    table = np.asarray(_CMAPS[cmap], dtype=np.float64)
    x = t*(len(table) - 1)
    k = np.minimum(x.astype(np.int64), len(table) - 2)
    f = (x - k)[:, None]
    rgb = table[k]*(1.0 - f) + table[k + 1]*f
    out = np.empty((v.size, 4), dtype=np.uint8)
    # sRGB encode: glTF COLOR_0 is linear, Blender shows it linear, and
    # a colourmap designed in sRGB looks washed out if handed over raw
    out[:, :3] = np.round(255.0*_to_linear(rgb)).astype(np.uint8)
    out[:, 3] = 255
    return out, lo, hi


def _to_linear(c):
    """sRGB -> linear, the transfer glTF's COLOR_0 is defined in."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c/12.92, ((c + 0.055)/1.055)**2.4)


# ----------------------------------------------------- conductor skin

def exposed_faces(struc):
    """Voxel faces where conductor meets air, per outward direction.

    Yields ``(axis, sign, cells)``: ``cells`` is an ``(n, 3)`` int
    array of the CONDUCTOR cells owning a face whose outward normal is
    ``sign`` along ``axis``. The domain boundary counts as exposed --
    a trace flush with the edge of the grid still has a visible side.
    """
    S = np.asarray(struc) > 0
    for axis in range(3):
        nb = np.zeros_like(S)
        # neighbour occupancy shifted along +axis (out of domain = air)
        sl_dst = [slice(None)]*3
        sl_src = [slice(None)]*3
        sl_dst[axis] = slice(None, -1)
        sl_src[axis] = slice(1, None)
        nb[tuple(sl_dst)] = S[tuple(sl_src)]
        yield axis, +1, np.argwhere(S & ~nb)
        nb[:] = False
        sl_dst[axis] = slice(1, None)
        sl_src[axis] = slice(None, -1)
        nb[tuple(sl_dst)] = S[tuple(sl_src)]
        yield axis, -1, np.argwhere(S & ~nb)


# corner offsets of a unit face, in the two axes tangential to `axis`,
# wound counter-clockwise seen from OUTSIDE for sign +1
_WIND = ((0, 0), (1, 0), (1, 1), (0, 1))


def surface_meshes(model, M, i, cmap='inferno', vmin=None, vmax=None,
                   J=None, parts=None):
    """Conductor skin as primitives coloured by |J_tangential|.

    Returns ``(prims, vmin, vmax)``. With ``parts`` -- the
    ``[(name, lo, hi)]`` cell ranges from
    :meth:`sppeec_input.Problem.block_cells` -- the skin is split into
    one primitive per declared block, which is what makes the export
    usable rather than merely correct: a power module's ground plane
    is a 40 x 50 mm sheet carrying real eddy current, and as a single
    welded shell it both hides the top copper from above and out-glows
    it from every other angle. Split, it is one click to hide. Blocks
    are matched in declaration order with LAST match winning, the same
    precedence the sigma painting uses, so a die on a trace is labelled
    the die.

    THE COLOUR SCALE IS GLOBAL across every part. Per-object
    normalisation would give each trace its own ramp and quietly make
    a dead signal trace look as hot as the commutation loop.

    Vertices are NOT shared between faces: each quad gets its own four,
    so the colour is flat per face and reads as one voxel = one sample
    rather than a smeared interpolation that invents crowding between
    cells. It costs 4x the vertices and buys honesty.
    """
    dims = tuple(int(d) for d in np.asarray(model.dims, dtype=int))
    spacing = np.asarray(M.e.l, dtype=float)
    struc = np.asarray(model.struc())
    if J is None:
        J = vtkout.current_density(M, i, dims)

    quads, norms, mags, cells_all = [], [], [], []
    for axis, sign, cells in exposed_faces(struc):
        if cells.size == 0:
            continue
        t0, t1 = (axis + 1) % 3, (axis + 2) % 3
        # tangential current amplitude: drop the normal component
        Jc = J[cells[:, 0], cells[:, 1], cells[:, 2]]
        mags.append(np.sqrt(np.abs(Jc[:, t0])**2 + np.abs(Jc[:, t1])**2))

        base = cells.astype(np.float64)
        base[:, axis] += 0.5*(sign + 1.0)       # -1 -> 0, +1 -> 1
        quad = np.empty((cells.shape[0], 4, 3), dtype=np.float64)
        order = _WIND if sign > 0 else _WIND[::-1]
        for c, (a, b) in enumerate(order):
            quad[:, c, :] = base
            quad[:, c, t0] += a
            quad[:, c, t1] += b
        quads.append(quad*spacing)
        n = np.zeros(3)
        n[axis] = sign
        norms.append(np.broadcast_to(n, (cells.shape[0], 3)))
        cells_all.append(cells)

    if not quads:
        raise ValueError('no conductor cells: nothing to export')
    quads = np.concatenate(quads)               # (nface, 4, 3)
    norms = np.concatenate(norms)               # (nface, 3)
    mags = np.concatenate(mags)                 # (nface,)
    cells_all = np.concatenate(cells_all)
    colors, lo, hi = shade(mags, vmin, vmax, cmap)

    # label every face with the block owning its cell (-1 = none)
    label = np.full(mags.size, -1, dtype=np.int64)
    names = []
    for k, (name, blo, bhi) in enumerate(parts or []):
        inside = np.ones(mags.size, dtype=bool)
        for a in range(3):
            inside &= ((cells_all[:, a] >= blo[a])
                       & (cells_all[:, a] < bhi[a]))
        label[inside] = k
        names.append(name)

    prims = []
    groups = [(k, names[k]) for k in range(len(names))]
    if (label < 0).any():
        groups.append((-1, 'conductor_surface' if not names
                       else 'unassigned_metal'))
    for k, name in groups:
        sel = np.flatnonzero(label == k)
        if sel.size == 0:
            continue
        nq = sel.size
        v0 = 4*np.arange(nq, dtype=np.uint32)[:, None]
        prims.append(dict(
            name=name,
            POSITION=quads[sel].reshape(-1, 3).astype(np.float32),
            NORMAL=np.repeat(norms[sel], 4, axis=0).astype(np.float32),
            COLOR_0=np.repeat(colors[sel], 4, axis=0),
            _JMAG=np.repeat(mags[sel], 4).astype(np.float32),
            indices=(v0 + np.array([0, 1, 2, 0, 2, 3],
                                   dtype=np.uint32)).ravel()))
    return prims, lo, hi


# ------------------------------------------------------------- wires

def _frame(u):
    """Any orthonormal pair perpendicular to the unit vector ``u``."""
    a = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    v = np.cross(u, a)
    v /= np.linalg.norm(v)
    return v, np.cross(u, v)


def _sphere(centre, r, nu=10, nv=6):
    """Low-poly UV sphere: the joint that closes a polyline elbow."""
    th = np.linspace(0.0, 2.0*np.pi, nu, endpoint=False)
    ph = np.linspace(0.0, np.pi, nv)
    st, cp = np.sin(ph)[:, None], np.cos(ph)[:, None]
    n = np.stack([(st*np.cos(th)), (st*np.sin(th)),
                  np.broadcast_to(cp, (nv, nu))], -1).reshape(-1, 3)
    faces = []
    for j in range(nv - 1):
        for k in range(nu):
            k2 = (k + 1) % nu
            a, b = j*nu + k, j*nu + k2
            faces += [(a, b, b + nu), (a, b + nu, a + nu)]
    return centre + r*n, n, np.asarray(faces, dtype=np.uint32).ravel()


def _foot(p, cell, r_wire, r0, l, nsides=12):
    """The flared contact foot: wire radius at the wire end, ``r0`` on
    the pad surface, plus the filled contact disc.

    ``r0`` is the model's own contact radius (``WireBondSolver
    .foot_r0``, default twice the wire radius), and it is load-bearing
    physics rather than decoration: the foot constriction the solver
    adds is ``R_disc(r0, rho) * h(r0/dx)`` from footcal, so the disc
    drawn here is exactly the disc that resistance is computed for.
    Under ``foot_model='patch'`` the same disc is resolved into
    surface cells with footcal's coverage weights; this draws the
    underlying physical contact, which is what both models share.

    The taper BETWEEN the two radii is the one drawn thing that is not
    itself a modelled surface -- the solver has a wire of one radius
    and a contact disc of another, and nothing in between. It is a
    reading of the two radii, not a third quantity.

    The surface is the outward face of the solver's own anchor cell,
    found by the same normal-axis test ``WireBondSolver._patch`` uses,
    so the foot sits on the face the current is actually injected
    through.
    """
    c = np.asarray(cell, dtype=float)
    d = np.asarray(p, dtype=float) - (c + 0.5)*l
    n = int(np.argmax(np.abs(d)))
    sgn = 1.0 if d[n] >= 0.0 else -1.0
    u = np.zeros(3)
    u[n] = sgn
    v, w = _frame(u)
    th = 2.0*np.pi*np.arange(nsides)/nsides
    rn = np.cos(th)[:, None]*v + np.sin(th)[:, None]*w
    seat = np.asarray(p, dtype=float).copy()
    # outward face of the anchor cell: the pad surface the foot lands on
    seat[n] = (c[n] + (1.0 if sgn > 0 else 0.0))*l[n]

    top = p + r_wire*rn                   # meets the last tube ring
    bot = seat + r0*rn                    # the contact circle
    pos = np.concatenate([top, bot, seat[None, :]])
    nor = np.concatenate([rn, rn, u[None, :]])
    tris = []
    for k in range(nsides):
        k2 = (k + 1) % nsides
        tris += [(k, k2, nsides + k2), (k, nsides + k2, nsides + k)]
        tris.append((2*nsides, nsides + k2, nsides + k))   # disc fan
    return pos, nor, np.asarray(tris, dtype=np.uint32)


def wire_mesh(wires, i_w, seg0, wire_of_seg, cmap='turbo', nsides=12,
              vmin=None, vmax=None, radius_scale=1.0,
              foot_cell=None, foot_r0=None, pitch=None):
    """Bond wires as swept tubes coloured by solved chain current.

    One colour band per solved segment (elements within a segment are
    parallel branches of the same piece of wire, so the chain current
    is their sum -- the same reduction :func:`vtkout.export_wires`
    makes), with a sphere at each interior polyline vertex so the
    right-angle elbows of a rise-span-descend bond close instead of
    showing a wedge of daylight.

    ``radius_scale`` draws the tubes at a multiple of the physical
    radius. A 0.13 mm bond wire on a 40 mm module is a quarter of a
    percent of the frame -- correct, and invisible next to the lit
    copper. Exaggerating it is standard practice (ParaView's Tube
    filter, and this repo's own README figure, which draws the wires
    at a 3.5 pt line width that is not physical either), so the knob
    exists; it defaults to 1.0, and any other value is recorded in the
    legend and burned into the render stamp, because an unlabelled
    exaggeration is just a wrong picture.

    Given ``foot_cell`` (``WireBondSolver.foot_cell``), ``foot_r0``
    (``.foot_r0``) and ``pitch`` (``M.e.l``), each contact also gets
    its flared FOOT -- see :func:`_foot`. Without them the wires end
    in a rounded cap, which is geometry the model does not have.

    Returns ``(prim, vmin, vmax)``, or ``(None, None, None)`` if there
    are no wires.
    """
    if not wires:
        return None, None, None
    seg0 = np.asarray(seg0, dtype=np.int64)
    wire_of_seg = np.asarray(wire_of_seg, dtype=np.int64)
    i_w = np.asarray(i_w)

    pos, nor, mag, idx = [], [], [], []
    ring = np.arange(nsides)
    ca = np.cos(2.0*np.pi*ring/nsides)[:, None]
    sa = np.sin(2.0*np.pi*ring/nsides)[:, None]
    k, base = 0, 0
    for j, w in enumerate(wires):
        ends, currents = [], []
        rad = w.radius*float(radius_scale)
        for seg in w.segments:
            f0 = seg[0]
            a, b = int(seg0[k]), int(seg0[k + 1])
            p0 = np.asarray(f0.p0, dtype=float)
            p1 = p0 + float(f0.length)*np.asarray(f0.u, dtype=float)
            I = abs(complex(np.asarray(i_w[a:b]).sum()))
            v, ww = _frame(np.asarray(f0.u, dtype=float))
            rn = ca*v + sa*ww                       # (nsides, 3) normals
            pos.append(p0 + rad*rn)
            pos.append(p1 + rad*rn)
            nor.append(rn)
            nor.append(rn)
            mag.append(np.full(2*nsides, I))
            for s in range(nsides):
                s2 = (s + 1) % nsides
                idx += [(base + s, base + s2, base + nsides + s2),
                        (base + s, base + nsides + s2, base + nsides + s)]
            base += 2*nsides
            ends.append((p0, p1))
            currents.append(I)
            k += 1
        # interior corners always; the two ENDS get a rounded cap only
        # when there is no foot to close them
        joints = []
        feet_j = None if foot_cell is None else foot_cell[j]
        if feet_j is None:
            joints += [(ends[0][0], currents[0]),
                       (ends[-1][1], currents[-1])]
        for s in range(len(ends) - 1):
            joints.append((ends[s][1],
                           0.5*(currents[s] + currents[s + 1])))
        if feet_j is not None:
            r0 = float(np.ravel(foot_r0)[j])*float(radius_scale)
            for e, (contact, I) in enumerate(
                    ((ends[0][0], currents[0]),
                     (ends[-1][1], currents[-1]))):
                fp, fn, ft = _foot(contact, feet_j[e], rad, r0,
                                   np.asarray(pitch, dtype=float),
                                   nsides=nsides)
                pos.append(fp)
                nor.append(fn)
                mag.append(np.full(fp.shape[0], I))
                idx.append((ft + base).reshape(-1, 3))
                base += fp.shape[0]
        for centre, I in joints:
            sp, sn, sf = _sphere(centre, w.radius*float(radius_scale))
            pos.append(sp)
            nor.append(sn)
            mag.append(np.full(sp.shape[0], I))
            idx.append((sf + base).reshape(-1, 3))
            base += sp.shape[0]

    pos = np.concatenate(pos)
    nor = np.concatenate(nor)
    mag = np.concatenate(mag)
    flat = np.concatenate([np.asarray(a, dtype=np.uint32).ravel()
                           for a in idx])
    # linear, auto-ranged over the DATA: bond-wire sharing spreads a
    # few percent (0.22-0.30 A on the flagship), so a ramp anchored at
    # zero paints all eight wires the same colour and hides the one
    # thing the wire model is there to show
    colors, lo, hi = shade(mag, np.min(mag) if vmin is None else vmin,
                           vmax, cmap, log=False)
    prim = dict(name='bond_wires',
                POSITION=pos.astype(np.float32),
                NORMAL=nor.astype(np.float32),
                COLOR_0=colors,
                _JMAG=mag.astype(np.float32),
                indices=flat)
    return prim, lo, hi


# --------------------------------------------------------------- glb

# glTF component types and the numpy dtypes they map to
_COMP = {np.dtype(np.float32): 5126, np.dtype(np.uint32): 5125,
         np.dtype(np.uint8): 5121, np.dtype(np.uint16): 5123}
_TYPE = {1: 'SCALAR', 2: 'VEC2', 3: 'VEC3', 4: 'VEC4'}

# glTF is Y-up; Blender's importer rotates Y-up back to its own Z-up
# with default settings, so writing (x, z, -y) here lands the model in
# Blender with the solver's axes. Getting this wrong is not an error,
# just a module lying on its side.
# glTF g = R v = (x, z, -y); Blender's importer then applies
# b = (g0, -g2, g1) = (x, y, z), landing the model in the solver's own
# axes. The TRANSPOSE of this matrix is also a valid rotation and
# silently gives (x, -y, -z) -- a module that is upside down and
# mirrored in y, which on a roughly symmetric layout still looks like
# a plausible module. It is caught by validation/validate_blendout.py,
# which round-trips a deliberately asymmetric marker through Blender.
_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)


def write_glb(path, prims, scale=1000.0, extras=None):
    """Assemble primitives into one binary glTF scene.

    Each primitive is a dict of ``name``, ``indices`` and attribute
    arrays keyed by glTF semantic (``POSITION``, ``NORMAL``,
    ``COLOR_0``, and custom ``_NAME`` scalars). Positions are metres;
    they are scaled by ``scale`` and rotated to Y-up on the way out.
    """
    blob = bytearray()
    views, accs, meshes, nodes, mats = [], [], [], [], []

    def add(arr, target):
        """One bufferView + accessor over a contiguous numpy array."""
        arr = np.ascontiguousarray(arr)
        ncomp = 1 if arr.ndim == 1 else arr.shape[1]
        while len(blob) % 4:                      # accessors need 4-byte
            blob.append(0)                        # aligned offsets
        views.append(dict(buffer=0, byteOffset=len(blob),
                          byteLength=arr.nbytes, target=target))
        blob.extend(arr.tobytes())
        acc = dict(bufferView=len(views) - 1,
                   componentType=_COMP[arr.dtype],
                   count=int(arr.shape[0]), type=_TYPE[ncomp])
        if arr.dtype == np.uint8:
            acc['normalized'] = True
        accs.append(acc)
        return len(accs) - 1

    for prim in prims:
        if prim is None:
            continue
        pos = np.asarray(prim['POSITION'], dtype=np.float64) @ _YUP.T
        pos = (pos*float(scale)).astype(np.float32)
        attrs = {'POSITION': add(pos, 34962)}
        accs[attrs['POSITION']]['min'] = [float(v) for v in pos.min(0)]
        accs[attrs['POSITION']]['max'] = [float(v) for v in pos.max(0)]
        for key, val in prim.items():
            if key in ('POSITION', 'indices', 'name'):
                continue
            arr = np.asarray(val)
            if key == 'NORMAL':
                arr = ((arr.astype(np.float64) @ _YUP.T)
                       .astype(np.float32))
            attrs[key] = add(arr, 34962)
        mats.append(dict(
            name=prim['name'] + '_mat',
            pbrMetallicRoughness=dict(baseColorFactor=[1, 1, 1, 1],
                                      metallicFactor=0.0,
                                      roughnessFactor=0.55),
            doubleSided=True))
        meshes.append(dict(name=prim['name'], primitives=[dict(
            attributes=attrs, mode=4, material=len(mats) - 1,
            indices=add(np.asarray(prim['indices'], dtype=np.uint32),
                        34963))]))
        nodes.append(dict(name=prim['name'], mesh=len(meshes) - 1))

    doc = dict(
        asset=dict(version='2.0', generator='SuperPEEC blendout',
                   extras=dict(unit_scale=float(scale),
                               source_units='m', **(extras or {}))),
        scene=0, scenes=[dict(nodes=list(range(len(nodes))))],
        nodes=nodes, meshes=meshes, materials=mats,
        accessors=accs, bufferViews=views,
        buffers=[dict(byteLength=len(blob))])

    js = json.dumps(doc, separators=(',', ':')).encode()
    js += b' '*(-len(js) % 4)
    blob.extend(b'\0'*(-len(blob) % 4))
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<III', 0x46546C67, 2,
                             12 + 8 + len(js) + 8 + len(blob)))
        fh.write(struct.pack('<II', len(js), 0x4E4F534A) + js)
        fh.write(struct.pack('<II', len(blob), 0x004E4942) + bytes(blob))
    return path


def export_scene(model, M, i, path, wires=None, i_w=None, seg0=None,
                 wire_of_seg=None, foot_cell=None, foot_r0=None,
                 scale=1000.0, cmap='inferno',
                 wire_cmap='turbo', vmin=None, vmax=None, freq=None,
                 parts=None, wire_radius_scale=1.0, verbose=False):
    """Write conductor skin (+ bond wires) as one ``.glb``.

    Returns the list of files written: the ``.glb``, and a
    ``<stem>_legend.json`` recording the colour ranges, because a
    picture of a current density with no scale on it is decoration,
    not a measurement.
    """
    prims, lo, hi = surface_meshes(model, M, i, cmap=cmap, vmin=vmin,
                                   vmax=vmax, parts=parts)
    wprim, wlo, whi = wire_mesh(
        wires, i_w, seg0, wire_of_seg, cmap=wire_cmap,
        radius_scale=wire_radius_scale, foot_cell=foot_cell,
        foot_r0=foot_r0,
        pitch=np.asarray(M.e.l, dtype=float)) if wires else \
        (None, None, None)
    nq = sum(p['_JMAG'].size for p in prims)//4
    extras = dict(surface_cmap=cmap, surface_vmin=lo, surface_vmax=hi,
                  surface_quantity='|J_tangential| (A/m^2)',
                  surface_scale='log10')
    if wprim is not None:
        extras.update(wire_cmap=wire_cmap, wire_vmin=wlo,
                      wire_vmax=whi,
                      wire_quantity='|I_chain| (A)',
                      wire_scale='linear',
                      wire_radius_scale=float(wire_radius_scale))
        if foot_cell is not None:
            extras['foot_r0_m'] = [float(v) for v in np.ravel(foot_r0)]
    if freq is not None:
        extras['frequency_Hz'] = float(freq)
    extras['parts'] = [p['name'] for p in prims]
    write_glb(path, prims + [wprim], scale=scale, extras=extras)
    legend = os.path.splitext(path)[0] + '_legend.json'
    with open(legend, 'w') as fh:
        json.dump(extras, fh, indent=2)
        fh.write('\n')
    if verbose:
        print('  %d exposed faces in %d parts, |J_tan| '
              '%.4g..%.4g A/m^2%s'
              % (nq, len(prims), lo, hi,
                 '' if wprim is None else
                 ('; %d wire tube verts, |I| %.4g..%.4g A'
                  % (wprim['POSITION'].shape[0], wlo, whi))),
              flush=True)
    return [path, legend]


# --------------------------------------------------------------- driver

def _main(argv):
    """Solve a doctrine TOML and export the first frequency as .glb.

    A convenience wrapper -- the supported route is
    ``sppeec_cli.py --export-glb``, which exports every frequency of
    the declared sweep and shares the solve with the other exports.
    """
    if not argv:
        print(__doc__)
        return 2
    import sppeec_input
    out = argv[1] if len(argv) > 1 else 'sppeec_surface.glb'
    if not os.path.dirname(out):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, out)
    prob = sppeec_input.load(argv[0])
    m = prob.model()
    parts = prob.block_cells(m)
    M = prob.tree(m)
    sw = prob.sweeper(m, M)
    f = prob.freqs[0]
    Z, info = sw.solve(f)
    sol = getattr(sw, 'sol', None)
    kw = {}
    if getattr(sol, 'wires', None) and 'i_w' in info:
        kw = dict(wires=sol.wires, i_w=info['i_w'], seg0=sol.wc.seg0,
                  wire_of_seg=sol.wire_of_seg,
                  foot_cell=getattr(sol, 'foot_cell', None),
                  foot_r0=getattr(sol, 'foot_r0', None))
    for p in export_scene(m, M, info['i_f'], out, freq=f, parts=parts,
                          verbose=True, **kw):
        print('wrote %s' % p)
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(_main(sys.argv[1:]))
