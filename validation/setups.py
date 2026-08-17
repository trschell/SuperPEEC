# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Benchmark geometries, as PURE GEOMETRY.

Extracted from main.py. These used to build the tree themselves, reading
the driver globals ``lscale``/``capacitive``/``AUTOLEAF``, which meant a
geometry could not be described without also committing to how it is
discretised. Each now returns a :class:`Geometry` and the CALLER builds
the tree, so the same definition can be written out as a VoxHenry .vhr
file (see :func:`write_vhr` in vhr.py).

The byte-regression anchors are tied to these three, so the arrays here
must not be edited casually: the byte anchors depend on them.
"""
import numpy as np
from collections import namedtuple

import stencils as st

Geometry = namedtuple('Geometry',
                      'name NT N LT nmax numlevels struc ports')

# Ports as VoxHenry voxel-FACE patches: (name, 'P'|'N', ix, iy, iz, face),
# 0-based, face one of +x/-x/+y/-y/+z/-z.
#
# Only setup3 has a port that translates. Its driver source is a two-node
# pair -- top of signal trace 1 and the ground plane directly beneath --
# which IS a pair of voxel faces. setup1 drives a diagonal node pair
# (0,0,0)->(5,5,5) and setup2 a distributed ring injection; neither is an
# electrode patch, so inventing a face set for them would describe a
# different excitation than the code actually solves. Those two export as
# geometry-only, which is honest: the file says nothing about a port
# rather than saying something wrong.
#
# The signal trace is fullstruc[:, 4, 3] -- ONE voxel in cross-section (1 in
# y, 1 in z) -- so the P terminal cannot be widened without changing the
# geometry. The return CAN: current leaving the trace spreads across the
# whole ground plane, so the N terminal spans the plane's full transverse
# width at the drive station (x=7, all 15 y). That asymmetry is the
# physical situation, not a modelling compromise. Driving a single plane
# voxel instead concentrates the answer in a spreading resistance that is
# almost entirely a discretisation artefact.
SETUP3_PORT = ((('port1', 'P', 7, 4, 3, '+z'),)
               + tuple(('port1', 'N', 7, iy, 0, '-z') for iy in range(15)))


def setup1():
    NT = np.array([7, 7, 7])
    # Single-level tree: the leaf box must span the whole NODE grid, and
    # that grid is NT under 'cell' but NT+1 under 'edge'. The literal
    # [8,8,8] this used to carry is the edge answer (and is what the
    # anchor was recorded with); under 'cell' it oversized the box by one
    # and parsesource then decoded source coordinates against the wrong
    # grid. Same class as the NT+1 -> single_level_nleaf fix already
    # applied to the five LpPR validators.
    N = st.single_level_nleaf(NT)
    LT = np.array([87.5e-6, 87.5e-6, 87.5e-6])
    nmax = 4
    numlevels = 1
    fullstruc = np.zeros((NT[0], NT[1], NT[2]), dtype=np.int8)
    fullstruc[...] = 1
    # # fullstruc[1, :, :] = 0
    # fullstruc[:, 1:, :] = 0
    # fullstruc[:, :, 1:] = 0
    # # fullstruc[0, -1, 0] = 0
    return Geometry('setup1', NT, N, LT, nmax, numlevels,
                    fullstruc, ())



def setup2():
    NT = np.array([29, 29, 29])
    N = np.array([8, 8, 8])
    LT = np.array([50e-6, 50e-6, 50e-6])
    nmax = 4
    numlevels = 3
    # NT = np.array([48, 48, 48])
    minN = min(NT[0], NT[1])
    fullstruc = np.ones((NT[0], NT[1], NT[2]), dtype=np.int8)
    for CountW in range(1, NT[0]+1):
        for CountL in range(1, NT[0]+1):
            if np.sqrt((CountW-minN/2)**2 + (CountL-minN/2)**2) < 0.3*minN:
                fullstruc[CountW-1, CountL-1, :] = 0
                if np.sqrt((CountW-minN/2)**2 + (CountL-minN/2)**2) < 0.1*minN:
                    fullstruc[CountW-1, CountL-1, :] = 1
            elif np.sqrt((CountW-minN/2)**2 + (CountL-minN/2)**2) > 0.5*minN:
                fullstruc[CountW-1, CountL-1, :] = 0
    ntxmid = int(NT[0]/2)
    ntymid = int(NT[1]/2)
    fullstruc[ntxmid-1:ntxmid+1, :ntymid, :] = 0
    fullstruc[:, :, 4:] = 0
    # fullstruc[127, 127, 127] = 1
    # fullstruc[:36, :37, :12] = 1
    # fullstruc[24:30, 24:37, :6] = 1
    #
    # fullsource = np.zeros((NT.x+1, NT.y+1, NT.z+1), dtype=np.int8)
    # fullsource[ntxmid+1, :26, :] = 1
    # fullsource[ntxmid-1, :26, :] = -1
    # A = mp.NewTree(fullstruc, N, LT, numlevels, nmax)
    return Geometry('setup2', NT, N, LT, nmax, numlevels,
                    fullstruc, ())



def setup3():
    # PCB-like structure (the IC/PCB target class): ground plane + two
    # signal traces + four corner vias, ~8% metal fill, 15^3 cells of
    # 10 um in a 16^3-padded two-level tree (4 leaf boxes per side, so
    # the near-field W is INCOMPLETE -- the realistic regime, unlike the
    # solid cube of setup1, whose homogeneity flatters every
    # preconditioner). The macro loops (trace->via->plane->via) are not
    # spanned by square plaquettes; this geometry is why
    # meshgraph.getmesh_fortran validates the Fortran plaquette basis and
    # falls back to the MST fundamental-cycle basis.
    NT = np.array([15, 15, 15])
    N = np.array([4, 4, 4])
    LT = np.array([1.6e-4, 1.6e-4, 1.6e-4])
    nmax = 4
    numlevels = 2
    fullstruc = np.zeros((NT[0], NT[1], NT[2]), dtype=np.int8)
    fullstruc[:, :, 0] = 1                     # ground plane
    fullstruc[:, 4, 3] = 1                     # signal trace 1
    fullstruc[:, 10, 3] = 1                    # signal trace 2
    for sx in (0, NT[0]-1):                    # vias tying traces to plane
        for sy in (4, 10):
            fullstruc[sx, sy, 1] = 1
            fullstruc[sx, sy, 2] = 1
    return Geometry('setup3', NT, N, LT, nmax, numlevels,
                    fullstruc, SETUP3_PORT)



GEOMETRIES = {1: setup1, 2: setup2, 3: setup3}
