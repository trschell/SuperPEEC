# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Plan-view figure from exported field data -- the README image.

A worked post-processing example: reads the .vti volume export and
.vtp wire export that ``sppeec_cli.py --export-vti --export-wires``
writes (here the 51M-cell DBC half-bridge, R4), takes the
surface-following top-conductor |J| slice, and overlays the bond
wires coloured by their solved chain currents. No solve -- seconds.

Usage:  python studies/dbc_r4_plot.py [field.vti] [wires.vtp] [out.png]
        (defaults: results/dbc_r4.vti, results/dbc_r4_wires.vtp,
        results/dbc_r4_plan.png)
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')


def read_vti_array(path, name):
    """One appended-raw array from a vtkout-written .vti."""
    with open(path, 'rb') as fh:
        head = fh.read(4096).decode('latin1')
        m = re.search(r'Name="%s"[^>]*offset="(\d+)"' % name, head)
        off = int(m.group(1))
        ext = [int(v) for v in re.search(
            r'WholeExtent="([\d ]+)"', head).group(1).split()]
        nx, ny, nz = ext[1], ext[3], ext[5]
        ncomp = int((re.search(
            r'Name="%s"[^>]*NumberOfComponents="(\d+)"' % name,
            head) or [None, '1'])[1])
        start = head.index('<AppendedData encoding="raw">')
        start = head.index('_', start) + 1
        fh.seek(start + off)
        nbytes = np.frombuffer(fh.read(8), dtype=np.uint64)[0]
        data = np.frombuffer(fh.read(int(nbytes)), dtype=np.float32)
    # VTK appended order: x fastest, z slowest
    shape = (nz, ny, nx) if ncomp == 1 else (nz, ny, nx, ncomp)
    dx = float(re.search(r'Spacing="([^ "]+)', head).group(1))
    return data.reshape(shape), (nx, ny, nz), dx


def read_wires_vtp(path):
    txt = open(path).read()

    def arr(name, dtype=float):
        m = re.search(r'Name="%s"[^>]*>\n([^<]*)</DataArray>' % name,
                      txt)
        return np.array(m.group(1).split(), dtype=dtype)

    pts = arr('Points', float) if 'Name="Points"' in txt else np.array(
        re.search(r'<Points><DataArray[^>]*>\n([^<]*)</DataArray>',
                  txt).group(1).split(), dtype=float)
    pts = pts.reshape(-1, 3)
    conn = arr('connectivity', int).reshape(-1, 2)
    return pts, conn, arr('I_mag'), arr('radius')


def main():
    argv = sys.argv[1:]
    vti = argv[0] if argv else os.path.join(RES, 'dbc_r4.vti')
    vtp = argv[1] if len(argv) > 1 else os.path.join(
        RES, 'dbc_r4_wires.vtp')
    out = argv[2] if len(argv) > 2 else os.path.join(
        RES, 'dbc_r4_plan.png')
    jmag, (nx, ny, nz), dx = read_vti_array(vti, 'J_mag')
    cond, _, _ = read_vti_array(vti, 'conductor')
    # TOP-SURFACE view: for each column, |J| in the TOPMOST conductor
    # cell (surface-following). A max-over-z projection instead lets
    # the bottom plane bleed through the pad interiors and washes out
    # the contrast the original figure had.
    occ = cond > 0
    has = occ.any(axis=0)
    top = (nz - 1) - occ[::-1].argmax(axis=0)      # (ny, nx)
    plan = np.take_along_axis(jmag, top[None], axis=0)[0]
    plan = np.ma.masked_where(~has, plan)
    pts, conn, imag, rad = read_wires_vtp(vtp)

    mm = 1e3
    extent = [0, nx*dx*mm, 0, ny*dx*mm]
    vmax = float(plan.max())
    vmin = vmax/1e4

    fig = plt.figure(figsize=(12.6, 10.2))
    ax = fig.add_axes([0.07, 0.075, 0.62, 0.83])
    im = ax.imshow(plan, origin='lower', extent=extent,
                   cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax),
                   interpolation='nearest', aspect='equal')

    segs = (pts[conn]*mm)[:, :, :2]       # (nseg, 2, 2) in mm
    lc = LineCollection(segs, cmap='cool', linewidths=3.5,
                        capstyle='round')
    lc.set_array(imag)
    ax.add_collection(lc)

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('DBC half-bridge, R4 (51M cells) -- plan view, '
                 '1 MHz\ntop-surface current density + bond wires')

    # THE FIX: each colorbar gets its own axes with real separation,
    # labels on opposite outer sides so nothing can collide
    cax_j = fig.add_axes([0.66, 0.10, 0.025, 0.78])
    cb_j = fig.colorbar(im, cax=cax_j)
    cb_j.set_label(r'top-surface $|J|$  (A/m$^2$)')
    cax_w = fig.add_axes([0.755, 0.10, 0.025, 0.78])
    cb_w = fig.colorbar(lc, cax=cax_w)
    cb_w.set_label(r'bond-wire $|I|$  (A)')

    fig.savefig(out, dpi=110)
    print('wrote %s  (|J| %.3g..%.3g, wires %.3f..%.3f A, %d segs)'
          % (out, vmin, vmax, imag.min(), imag.max(), len(imag)))


if __name__ == '__main__':
    main()
