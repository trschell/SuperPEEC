# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Export SuperPEEC results to ParaView (VTK ImageData, ``.vti``).

SuperPEEC's voxel grid is uniform and Cartesian, so it maps exactly onto
VTK's ImageData: one cell per voxel, no mesh to write, no connectivity.
The only real work is getting the currents onto the cells, because they
do not live there.

THE STAGGERED LATTICE. Filament currents live on cell FACES, one leaf per
direction, and the leaf names do NOT match the axis order::

    leaf  direction   lattice           connects
    f     x           (nx-1, ny, nz)    (i,j,k) -> (i+1,j,k)
    e     y           (nx, ny-1, nz)    (i,j,k) -> (i,j+1,k)
    g     z           (nx, ny, nz-1)    (i,j,k) -> (i,j,k+1)

``e`` is Y-directed and ``f`` is X-directed -- read the resistances in
main.py (``M.e.r = l[1]/(l[0]*l[2]*sigma)``) if that looks wrong. Getting
it backwards silently transposes the vector field, which looks plausible
on a symmetric geometry.

This is the classic MAC arrangement, so the cell-centred current is the
average of the two filaments on opposite faces::

    Jx[i,j,k] = (If[i-1,j,k] + If[i,j,k]) / (2 * dy*dz)

with absent neighbours counted as zero current. At a conductor surface
that halves the value, which is correct: the current terminates there.

COMPLEX DATA. Currents are phasors and ParaView has no complex type, so
each vector is written twice (real and imaginary part) plus a magnitude
$\\sqrt{|Jx|^2+|Jy|^2+|Jz|^2}$ over complex components -- the amplitude,
independent of the phase reference. Pass ``phases=N`` to additionally
write a time series of the INSTANTANEOUS field
``Re(J e^{j\\omega t})`` over one cycle, which ParaView animates
directly and is much the clearest way to see current actually flowing.

Usage as a library::

    import vtkout
    vtkout.export_currents(model, M, i, 'out.vti', potentials=v)

Command line (solves, then exports). A bare output name lands in
``results/``; give a path with a directory to put it elsewhere::

    PYTHONPATH=. python3 vtkout.py wire_len50.0u_dia10.0u.vhr 2.5e9 out.vti
    PYTHONPATH=. PHASES=16 python3 vtkout.py <file.vhr> <freq> out.vti

With PHASES, files are written as ``out_t000.vti`` ... alongside a
``out.pvd`` collection; open the ``.pvd`` in ParaView to get the
animation with correct time values.
"""
import os
import sys
import numpy as np

import sppeec_status as _spstatus

# Where the CLI puts output when given a bare filename. The library
# functions never redirect a path the caller gave them.
RESULTS_DIR = 'results'


# ------------------------------------------------------------------ vti

def _appended(arrays):
    """Assemble the raw appended-data block and the per-array offsets."""
    blob = bytearray()
    offsets = []
    for a in arrays:
        offsets.append(len(blob))
        raw = np.ascontiguousarray(a).tobytes()
        blob += np.uint64(len(raw)).tobytes() + raw
    return bytes(blob), offsets


def write_vti(path, dims, spacing, celldata, origin=(0.0, 0.0, 0.0)):
    """Write cell data on a uniform grid as a binary ``.vti``.

    ``dims`` is the CELL count per axis; ImageData extents are given in
    points, hence the ``+1``. ``celldata`` maps name -> array, either
    shape ``dims`` (scalar) or ``dims + (3,)`` (vector). Arrays are
    written float32, except ``uint8`` ones which pass through --- which
    is what ``vtkGhostType`` requires.

    VTK's ImageData is Fortran-ordered over (x fastest), so every array
    is transposed on the way out; passing C-ordered ``[i,j,k]`` arrays
    here is correct and this function handles the rest.
    """
    nx, ny, nz = (int(d) for d in dims)
    names, arrays = [], []
    for name, a in celldata.items():
        a = np.asarray(a)
        vec = (a.ndim == 4)
        # (i,j,k[,c]) -> VTK's x-fastest ordering
        a = np.transpose(a, (2, 1, 0, 3) if vec else (2, 1, 0))
        # uint8 passes through (vtkGhostType must be UInt8); everything
        # else is float32 -- ample for visualisation and half the file.
        dt = np.uint8 if a.dtype == np.uint8 else np.float32
        names.append((name, 3 if vec else 1,
                      'UInt8' if dt is np.uint8 else 'Float32'))
        arrays.append(np.ascontiguousarray(a, dtype=dt))
    blob, offsets = _appended(arrays)

    ext = "0 %d 0 %d 0 %d" % (nx, ny, nz)
    head = ['<?xml version="1.0"?>',
            '<VTKFile type="ImageData" version="1.0" '
            'byte_order="LittleEndian" header_type="UInt64">',
            '  <ImageData WholeExtent="%s" Origin="%.17g %.17g %.17g" '
            'Spacing="%.17g %.17g %.17g">'
            % (ext, origin[0], origin[1], origin[2],
               spacing[0], spacing[1], spacing[2]),
            '    <Piece Extent="%s">' % ext,
            '      <CellData>']
    for (name, ncomp, vtype), off in zip(names, offsets):
        head.append('        <DataArray type="%s" Name="%s" '
                    'NumberOfComponents="%d" format="appended" '
                    'offset="%d"/>' % (vtype, name, ncomp, off))
    head += ['      </CellData>', '    </Piece>', '  </ImageData>',
             '  <AppendedData encoding="raw">', '   _']
    with open(path, 'wb') as fh:
        fh.write(("\n".join(head)).encode())
        fh.write(blob)
        fh.write(b"\n  </AppendedData>\n</VTKFile>\n")
    return path


# ------------------------------------------------- SuperPEEC -> cell arrays

def _coords(idx, dims):
    return np.stack([idx//(dims[1]*dims[2]), (idx//dims[2]) % dims[1],
                     idx % dims[2]], 1)


def scatter_leaf(M, leaf, values, lattice):
    """Scatter one leaf's per-filament values onto its global lattice.

    Leaf storage is (box, local index); this walks the boxes and adds the
    box origin, the same decode the far-field validators use.
    """
    out = np.zeros(tuple(int(v) for v in lattice), dtype=np.complex128)
    n = np.asarray(leaf.n, dtype=int)
    lv0 = M.lv[0]
    for g in range(np.size(leaf.idx0) - 1):
        sl = np.s_[leaf.idx0[g]:leaf.idx0[g+1]]
        if leaf.idx0[g+1] <= leaf.idx0[g]:
            continue
        c = _coords(np.asarray(leaf.idx[sl]), n)
        c[:, 0] += lv0.xidx[g]*lv0.n[0]
        c[:, 1] += lv0.yidx[g]*lv0.n[1]
        c[:, 2] += lv0.zidx[g]*lv0.n[2]
        ok = ((c >= 0).all(axis=1)
              & (c[:, 0] < lattice[0]) & (c[:, 1] < lattice[1])
              & (c[:, 2] < lattice[2]))
        c = c[ok]
        out[c[:, 0], c[:, 1], c[:, 2]] = np.asarray(values[sl])[ok]
    return out


def current_density(M, i, dims):
    """Cell-centred complex current density from filament currents.

    ``i`` is the solver's ``e|f|g`` stacked current vector. Returns a
    complex array of shape ``dims + (3,)`` in A/m^2, x/y/z order.
    """
    dims = tuple(int(d) for d in dims)
    nx, ny, nz = dims
    ne, nf = np.size(M.e.struc), np.size(M.f.struc)
    i = np.asarray(i)
    Ie = scatter_leaf(M, M.e, i[:ne], (nx, ny-1, nz))          # y-directed
    If = scatter_leaf(M, M.f, i[ne:ne+nf], (nx-1, ny, nz))     # x-directed
    Ig = scatter_leaf(M, M.g, i[ne+nf:], (nx, ny, nz-1))       # z-directed

    dx, dy, dz = (float(v) for v in np.asarray(M.e.l, dtype=float))
    J = np.zeros(dims + (3,), dtype=np.complex128)
    # average the two faces of each cell; absent neighbours are zero
    ax = np.zeros(dims, dtype=np.complex128)
    ax[1:, :, :] += If
    ax[:-1, :, :] += If
    J[..., 0] = 0.5*ax/(dy*dz)
    ay = np.zeros(dims, dtype=np.complex128)
    ay[:, 1:, :] += Ie
    ay[:, :-1, :] += Ie
    J[..., 1] = 0.5*ay/(dx*dz)
    az = np.zeros(dims, dtype=np.complex128)
    az[:, :, 1:] += Ig
    az[:, :, :-1] += Ig
    J[..., 2] = 0.5*az/(dx*dy)
    return J


def export_currents_streaming(model, M, i, path, quicklook=4,
                              slab_z=None):
    """Slab-streamed ``.vti`` export: O(one slab) memory at ANY size.

    Produces the SAME file, byte for byte, as :func:`export_currents`
    (no potentials/phases) -- gated by direct comparison -- but never
    materialises the full-domain field. The z-slowest ordering of
    VTK's appended blocks means a z-slab of cells is CONTIGUOUS in
    every array's block, so the file is written with per-slab seeks
    into precomputed offsets: one pass over the filaments, O(slab)
    transients where the old path needed the full domain (~100 GB at
    hero scale; measured at R3: 1701 -> 446 MB, byte-identical).
    Default slab thickness is one leaf-box z-span -- at 1e9 cells
    that is a few GB of transients; pass a smaller ``slab_z`` to
    shrink further at some per-slab scatter overhead.

    ``quicklook`` (the docket's REQUIRED companion): additionally
    writes ``<stem>_quicklook.vti`` with every field bin-averaged
    ``quicklook``^3 -- 1/64th the bytes at the default 4, interactive
    on any workstation, accumulated slab-by-slab in one small buffer.
    0 or 1 disables it. Returns the list of files written.
    """
    dims = tuple(int(d) for d in np.asarray(model.dims, dtype=int))
    nx, ny, nz = dims
    spacing = tuple(float(v) for v in np.asarray(M.e.l, dtype=float))
    ne, nf = np.size(M.e.struc), np.size(M.f.struc)
    i = np.asarray(i)
    vals = {'e': i[:ne], 'f': i[ne:ne + nf], 'g': i[ne + nf:]}
    dx, dy, dz = spacing
    struc = np.asarray(model.struc())

    # -- array table, SAME names/order/dtypes as export_currents -----
    spec = [('J_re', 3, np.float32, 'Float32'),
            ('J_im', 3, np.float32, 'Float32'),
            ('J_mag', 1, np.float32, 'Float32'),
            ('conductor', 1, np.float32, 'Float32'),
            ('vtkGhostType', 1, np.uint8, 'UInt8')]
    ncell = nx*ny*nz
    payload = [ncell*nc*np.dtype(dt).itemsize for _, nc, dt, _ in spec]
    offsets = np.concatenate([[0], np.cumsum([8 + p
                                              for p in payload])[:-1]])
    # header EXACTLY as write_vti builds it
    ext = "0 %d 0 %d 0 %d" % (nx, ny, nz)
    head = ['<?xml version="1.0"?>',
            '<VTKFile type="ImageData" version="1.0" '
            'byte_order="LittleEndian" header_type="UInt64">',
            '  <ImageData WholeExtent="%s" Origin="%.17g %.17g %.17g" '
            'Spacing="%.17g %.17g %.17g">'
            % (ext, 0.0, 0.0, 0.0, spacing[0], spacing[1], spacing[2]),
            '    <Piece Extent="%s">' % ext,
            '      <CellData>']
    for (name, nc, _, vt), off in zip(spec, offsets):
        head.append('        <DataArray type="%s" Name="%s" '
                    'NumberOfComponents="%d" format="appended" '
                    'offset="%d"/>' % (vt, name, nc, off))
    head += ['      </CellData>', '    </Piece>', '  </ImageData>',
             '  <AppendedData encoding="raw">', '   _']
    header = ("\n".join(head)).encode()

    # -- per-leaf slab scatter machinery -----------------------------
    lv0 = M.lv[0]
    leaves = {'e': (M.e, (nx, ny - 1, nz)),
              'f': (M.f, (nx - 1, ny, nz)),
              'g': (M.g, (nx, ny, nz - 1))}

    def scatter_slab(key, zlo, zhi):
        leaf, lattice = leaves[key]
        v = vals[key]
        out = np.zeros((lattice[0], lattice[1], zhi - zlo),
                       dtype=np.complex128)
        n = np.asarray(leaf.n, dtype=int)
        for g in range(np.size(leaf.idx0) - 1):
            gz0 = int(lv0.zidx[g])*int(n[2])
            if gz0 >= zhi or gz0 + int(n[2]) <= zlo:
                continue
            a, b = int(leaf.idx0[g]), int(leaf.idx0[g + 1])
            if b <= a:
                continue
            c = _coords(np.asarray(leaf.idx[a:b]), n)
            c[:, 0] += int(lv0.xidx[g])*int(n[0])
            c[:, 1] += int(lv0.yidx[g])*int(n[1])
            c[:, 2] += gz0
            ok = ((c[:, 0] >= 0) & (c[:, 0] < lattice[0])
                  & (c[:, 1] >= 0) & (c[:, 1] < lattice[1])
                  & (c[:, 2] >= zlo) & (c[:, 2] < zhi))
            cc = c[ok]
            out[cc[:, 0], cc[:, 1], cc[:, 2] - zlo] = \
                np.asarray(v[a:b])[ok]
        return out

    if slab_z is None:
        slab_z = max(int(np.asarray(M.e.n, dtype=int)[2]), 1)
    b = int(quicklook) if quicklook else 0
    if b > 1:
        q = [max(1, -(-d//b)) for d in dims]
        xs = np.arange(0, nx, b)
        ys = np.arange(0, ny, b)
        cx = np.add.reduceat(np.ones(nx), xs)
        cy = np.add.reduceat(np.ones(ny), ys)
        accJ = np.zeros((q[2], q[0], q[1], 3), dtype=np.complex128)
        accS = np.zeros((q[2], q[0], q[1]))
        czw = np.zeros(q[2])

    with _spstatus.task('export fields',
                        ticks=-(-nz//slab_z)) as _t, \
            open(path, 'wb') as fh:
        fh.write(header)
        base = len(header)
        for off, p in zip(offsets, payload):
            fh.seek(base + int(off))
            fh.write(np.uint64(p).tobytes())
        for z0 in range(0, nz, slab_z):
            z1 = min(z0 + slab_z, nz)
            s = z1 - z0
            Ie = scatter_slab('e', z0, z1)
            If = scatter_slab('f', z0, z1)
            glo, ghi = max(z0 - 1, 0), min(z1, nz - 1)
            Ig = scatter_slab('g', glo, ghi)
            J = np.zeros((nx, ny, s, 3), dtype=np.complex128)
            ax = np.zeros((nx, ny, s), dtype=np.complex128)
            ax[1:, :, :] += If
            ax[:-1, :, :] += If
            J[..., 0] = 0.5*ax/(dy*dz)
            ay = np.zeros((nx, ny, s), dtype=np.complex128)
            ay[:, 1:, :] += Ie
            ay[:, :-1, :] += Ie
            J[..., 1] = 0.5*ay/(dx*dz)
            az = np.zeros((nx, ny, s), dtype=np.complex128)
            a0 = max(z0, 1)               # cells with a face below
            if a0 < z1:
                az[:, :, a0 - z0:] += Ig[:, :, a0 - 1 - glo:z1 - 1 - glo]
            b0 = min(z1, nz - 1)          # cells with a face above
            if b0 > z0:
                az[:, :, :b0 - z0] += Ig[:, :, z0 - glo:b0 - glo]
            J[..., 2] = 0.5*az/(dx*dy)
            mag = np.sqrt((np.abs(J)**2).sum(axis=-1))
            st = struc[:, :, z0:z1]
            ghost = np.zeros(st.shape, dtype=np.uint8)
            ghost[st <= 0] = HIDDENCELL
            slabs = [J.real, J.imag, mag, st.astype(np.float32), ghost]
            for (name, nc, dt, _), off, p, arr in zip(spec, offsets,
                                                      payload, slabs):
                tr = (2, 1, 0, 3) if arr.ndim == 4 else (2, 1, 0)
                raw = np.ascontiguousarray(
                    np.transpose(arr, tr).astype(dt)).tobytes()
                item = np.dtype(dt).itemsize
                fh.seek(base + int(off) + 8 + z0*nx*ny*nc*item)
                fh.write(raw)
            if b > 1:
                Jx = np.add.reduceat(J, xs, axis=0)
                Jxy = np.add.reduceat(Jx, ys, axis=1)
                Sx = np.add.reduceat(st.astype(float), xs, axis=0)
                Sxy = np.add.reduceat(Sx, ys, axis=1)
                zb = (z0 + np.arange(s))//b
                np.add.at(accJ, zb, np.moveaxis(Jxy, 2, 0))
                np.add.at(accS, zb, np.moveaxis(Sxy, 2, 0))
                np.add.at(czw, zb, 1.0)
            _t.tick()
        end = base + int(offsets[-1]) + 8 + payload[-1]
        fh.seek(end)
        fh.write(b"\n  </AppendedData>\n</VTKFile>\n")
    written = [path]

    if b > 1:
        w = (cx[None, :, None]*cy[None, None, :])*czw[:, None, None]
        Jq = np.moveaxis(accJ/w[..., None], 0, 2)
        Sq = np.moveaxis(accS/w, 0, 2)
        magq = np.sqrt((np.abs(Jq)**2).sum(axis=-1))
        gq = np.zeros(Sq.shape, dtype=np.uint8)
        gq[Sq <= 0] = HIDDENCELL
        stem = path[:-4] if path.endswith('.vti') else path
        qpath = stem + '_quicklook.vti'
        write_vti(qpath, q, tuple(sp*b for sp in spacing),
                  {'J_re': Jq.real, 'J_im': Jq.imag, 'J_mag': magq,
                   'conductor': Sq.astype(np.float32),
                   'vtkGhostType': gq})
        written.append(qpath)
    return written


HIDDENCELL = 32          # vtkDataSetAttributes::HIDDENCELL


def _ghost(struc):
    """Mark empty cells HIDDEN so VTK does not render them at all.

    A voxel grid is mostly air --- 81% of square_coil, 92% of the
    circular coil --- and those cells otherwise draw as zero-valued
    blocks that fill the inside of a coil and hide its inner surface.
    Thresholding them away in ParaView works but has to be redone on
    every open; ``vtkGhostType`` is the format's own mechanism and is
    honoured automatically, so the file shows only conductor.
    """
    g = np.zeros(np.asarray(struc).shape, dtype=np.uint8)
    g[np.asarray(struc) <= 0] = HIDDENCELL
    return g


def export_currents(model, M, i, path, potentials=None, phases=0,
                    freq=None):
    """Write current density (and optionally node potential) as ``.vti``.

    ``phases > 0`` additionally writes a one-cycle animation as
    ``<stem>_tNNN.vti`` plus a ``.pvd`` collection carrying the real
    times, so ParaView shows physical seconds on the animation slider.
    Returns the list of files written.
    """
    dims = tuple(int(d) for d in np.asarray(model.dims, dtype=int))
    J = current_density(M, i, dims)
    mag = np.sqrt((np.abs(J)**2).sum(axis=-1))
    struc = np.asarray(model.struc())
    cd = {'J_re': J.real, 'J_im': J.imag, 'J_mag': mag,
          'conductor': struc.astype(np.float32),
          'vtkGhostType': _ghost(struc)}
    if potentials is not None:
        v = np.zeros(dims, dtype=np.complex128)
        vv = scatter_leaf(M, M.lv[0], np.asarray(potentials), dims)
        v += vv
        cd['V_re'] = v.real
        cd['V_im'] = v.imag
    spacing = tuple(float(v) for v in np.asarray(M.e.l, dtype=float))
    written = [write_vti(path, dims, spacing, cd)]
    if phases > 0:
        stem = path[:-4] if path.endswith('.vti') else path
        entries = []
        for k in range(int(phases)):
            th = 2.0*np.pi*k/float(phases)
            Jt = (J*np.exp(1j*th)).real
            fn = "%s_t%03d.vti" % (stem, k)
            write_vti(fn, dims, spacing,
                      {'J': Jt,
                       'J_mag': np.sqrt((Jt**2).sum(axis=-1)),
                       'conductor': struc.astype(np.float32),
                       'vtkGhostType': _ghost(struc)})
            t = (k/float(phases)/freq) if freq else float(k)
            entries.append((t, os.path.basename(fn)))
            written.append(fn)
        with open(stem + '.pvd', 'w') as fh:
            fh.write('<?xml version="1.0"?>\n<VTKFile type="Collection" '
                     'version="0.1" byte_order="LittleEndian">\n'
                     '  <Collection>\n')
            for t, fn in entries:
                fh.write('    <DataSet timestep="%.17g" file="%s"/>\n'
                         % (t, fn))
            fh.write('  </Collection>\n</VTKFile>\n')
        written.append(stem + '.pvd')
    return written


# --------------------------------------------------------------- driver

def _main(argv):
    if not argv:
        print(__doc__)
        return 2
    import vhr
    import port_impedance as pz
    name, freq = argv[0], float(argv[1]) if len(argv) > 1 else 2.5e9
    out = argv[2] if len(argv) > 2 else 'sppeec_currents.vti'
    if not os.path.dirname(out):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, out)
    path = name if os.path.exists(name) else 'VoxHenry/Input_files/' + name
    m = vhr.read_vhr(path)
    M = m.build_tree()
    m.prepare(M, freq)
    S = pz.LpRSolver(M)
    src = m.source_vector(M, 0, 1.0, 'corner')
    i, v, info = S.solve(src)
    print("solved: %d matvecs, resid %.2e" % (info['matvecs'],
                                              info['residual']))
    files = export_currents(m, M, i, out, potentials=v,
                            phases=int(os.environ.get('PHASES', '0')),
                            freq=freq)
    J = current_density(M, i, tuple(int(d) for d in np.asarray(m.dims,
                                                              dtype=int)))
    mag = np.sqrt((np.abs(J)**2).sum(axis=-1))
    print("|J|: max %.4g  mean-over-conductor %.4g A/m^2"
          % (mag.max(), mag[np.asarray(m.struc()) > 0].mean()))
    for f in files:
        print("  wrote %s" % f)
    return 0


if __name__ == '__main__':
    raise SystemExit(_main(sys.argv[1:]))


def export_wires(wires, i_w, seg0, wire_of_seg, path):
    """Write bond wires as VTK PolyData (``.vtp``): one polyline per
    segment with its solved chain current (re/im/mag) as cell data.

    ``i_w`` is the solved element-current vector; the segment total is
    the physical chain current through that piece of wire (elements
    are parallel branches). Pairs with :func:`export_currents` -- load
    both in ParaView and the wires draw over the volume current
    density (Tube filter on the wires recommended, radius from the
    ``radius`` cell array).
    """
    import numpy as np
    pts, lines, cre, cim, cmag, crad, cwire = [], [], [], [], [], [], []
    k = 0
    for j, w in enumerate(wires):
        for seg in w.segments:
            f0 = seg[0]
            a, b = int(seg0[k]), int(seg0[k + 1])
            Iseg = complex(np.asarray(i_w[a:b]).sum())
            i0 = len(pts)
            pts.append(f0.p0)
            pts.append(f0.p0 + f0.length*f0.u)
            lines.append((i0, i0 + 1))
            cre.append(Iseg.real)
            cim.append(Iseg.imag)
            cmag.append(abs(Iseg))
            crad.append(w.radius)
            cwire.append(j)
            k += 1
    pts = np.asarray(pts, dtype=float)
    n, m = pts.shape[0], len(lines)
    with open(path, 'w') as fh:
        fh.write('<?xml version="1.0"?>\n'
                 '<VTKFile type="PolyData" version="0.1" '
                 'byte_order="LittleEndian">\n<PolyData>\n'
                 '<Piece NumberOfPoints="%d" NumberOfLines="%d">\n'
                 % (n, m))
        fh.write('<Points><DataArray type="Float64" '
                 'NumberOfComponents="3" format="ascii">\n')
        for p in pts:
            fh.write('%.9g %.9g %.9g\n' % tuple(p))
        fh.write('</DataArray></Points>\n<Lines>\n'
                 '<DataArray type="Int64" Name="connectivity" '
                 'format="ascii">\n')
        for a, b in lines:
            fh.write('%d %d\n' % (a, b))
        fh.write('</DataArray>\n<DataArray type="Int64" '
                 'Name="offsets" format="ascii">\n')
        for kk in range(1, m + 1):
            fh.write('%d\n' % (2*kk))
        fh.write('</DataArray>\n</Lines>\n<CellData>\n')
        for nm, arr, typ in (('I_re', cre, 'Float64'),
                             ('I_im', cim, 'Float64'),
                             ('I_mag', cmag, 'Float64'),
                             ('radius', crad, 'Float64'),
                             ('wire', cwire, 'Int64')):
            fh.write('<DataArray type="%s" Name="%s" format="ascii">\n'
                     % (typ, nm))
            for v in arr:
                fh.write('%.9g\n' % v)
            fh.write('</DataArray>\n')
        fh.write('</CellData>\n</Piece>\n</PolyData>\n</VTKFile>\n')
    return [path]
