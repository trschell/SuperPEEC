# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Validate stencils.py, the single definition of element existence.

The seven element-existence rules used to be written out three times --
tree.py's single-level branch, its multilevel depth-first scan, and the
Fortran ``get_node_size`` that counts them to size the arrays. They now
live once in stencils.py. (Until 2026-08-14 this file also checked the
removed corner-node/edge-filament scheme against its legacy formulas;
those parts went with the scheme.)

PART A -- BRUTE-FORCE SEMANTICS. The cell rules, checked element by
element against their definitions on random geometries: a node exists
iff its cell conducts; a filament iff BOTH cells sharing its face
conduct (and carries the weaker material tag); a panel iff occupancy
CHANGES across its face -- never inside solid metal, never in free
space.

PART B -- THE TWO ASSEMBLY PATHS AGREE. ``struc_from_cells`` (the
whole-grid single-level path) against ``struc_from_block`` assembled
box by box with the one-layer-overhang convention (the multilevel
scan's path). Any drift between them is the silent-truncation failure
class the module exists to prevent.

PART C -- THE COUNTER MATCHES THE FILLER. ``count_elements`` sizes the
multilevel arrays; its totals must equal the nonzero counts of the
strucs the scan then writes. An undercount truncates silently, the
same failure class as the OSLABIDX heap corruption fixed earlier.

Run:  PYTHONPATH=src python3 validate_stencils.py
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys

import numpy as np

import stencils as st

fails = []


def check(tag, cond, detail=''):
    if cond:
        return True
    fails.append("%s: %s" % (tag, detail))
    print("    FAIL %s  %s" % (tag, detail))
    return False


rng = np.random.default_rng(7)

print("PART A -- brute-force semantics of the cell rules")
for trial in range(40):
    nt = rng.integers(2, 7, size=3)
    fs = (rng.random(tuple(nt)) < 0.5).astype(np.int8)
    # occasional material tags (dielectric id 2)
    if trial % 3 == 0:
        fs[fs > 0] = rng.integers(1, 3, size=int((fs > 0).sum()))
    strucs = st.struc_from_cells(fs)
    shapes = st.element_shapes(nt)
    ok = True
    for o, sh, got in zip(('e', 'f', 'g'), shapes[:3], strucs[:3]):
        ax = st.FILAMENT_AXIS[o]
        ref = np.zeros(sh, fs.dtype)
        for i in range(sh[0]):
            for j in range(sh[1]):
                for k in range(sh[2]):
                    c = [i, j, k]
                    c2 = list(c)
                    c2[ax] += 1
                    ref[i, j, k] = min(fs[tuple(c)], fs[tuple(c2)])
        ok &= np.array_equal(got, np.minimum(ref, 1))
    ok &= np.array_equal(strucs[3], np.minimum(fs, 1))
    for o, sh, got in zip('xyz', shapes[4:], strucs[4:]):
        ax = st.PANEL_AXIS[o]
        ref = np.zeros(sh, fs.dtype)
        for i in range(sh[0]):
            for j in range(sh[1]):
                for k in range(sh[2]):
                    c = [i, j, k]
                    lo = list(c)
                    lo[ax] -= 1

                    def raw(cc):
                        if any(v < 0 for v in cc) or \
                           any(v >= nt[a] for a, v in enumerate(cc)):
                            return 0
                        return int(fs[tuple(cc)])
                    # a panel fires where the raw VALUES differ --
                    # material interfaces included (that is where
                    # bound charge lives), then normalised to 1
                    ref[i, j, k] = 1 if raw(c) != raw(lo) else 0
        ok &= np.array_equal(np.minimum(got, 1), ref)
    if not check("cell rules == brute force (trial %d)" % trial, ok,
                 "nt %s" % list(nt)):
        break
else:
    print("    ok   40 random geometries, filaments/nodes/panels")

print("PART B -- struc_from_cells == struc_from_block per box")
for trial in range(30):
    nleaf = rng.integers(2, 5, size=3)
    ngroups = rng.integers(1, 4, size=3)
    full = nleaf*ngroups
    # keep the outermost layer empty: the topmost panel plane (index
    # full[ax] on the panel's normal) has no box in this bare tiling --
    # the real tree pads the box lattice one cell past the grid for
    # exactly this reason (see tree.py's npad comment)
    fs = np.zeros(tuple(full), np.int8)
    inner = tuple(slice(0, max(1, int(v) - 1)) for v in full)
    fs[inner] = (rng.random(fs[inner].shape) < 0.5).astype(np.int8)
    whole = st.struc_from_cells(fs)
    shapes = st.element_shapes(full)
    ok = True
    for ii in range(7):
        asm = np.zeros(tuple(full), fs.dtype)
        pad = np.pad(fs, 1)
        for bx in range(ngroups[0]):
            for by in range(ngroups[1]):
                for bz in range(ngroups[2]):
                    o = np.array([bx, by, bz])*nleaf
                    block = pad[o[0]:o[0]+nleaf[0]+2,
                                o[1]:o[1]+nleaf[1]+2,
                                o[2]:o[2]+nleaf[2]+2]
                    got = st.struc_from_block(block, nleaf)[ii]
                    asm[o[0]:o[0]+nleaf[0], o[1]:o[1]+nleaf[1],
                        o[2]:o[2]+nleaf[2]] = got
        # crop the whole-grid struc onto the box lattice (elements
        # whose lattice is smaller than the cell grid simply have
        # trailing zeros in the assembled block)
        sh = shapes[ii]
        ref = np.zeros(tuple(full), fs.dtype)
        crop = tuple(slice(0, min(int(a), int(b)))
                     for a, b in zip(full, sh))
        ref[crop] = whole[ii][crop]
        # any element plane beyond the box tiling must be empty (the
        # empty outer layer guarantees it)
        for ax in range(3):
            if sh[ax] > full[ax]:
                sl = [slice(None)]*3
                sl[ax] = slice(int(full[ax]), None)
                ok &= not whole[ii][tuple(sl)].any()
        ok &= np.array_equal(asm, ref)
    if not check("two paths agree (trial %d)" % trial, ok,
                 "nleaf %s ngroups %s" % (list(nleaf), list(ngroups))):
        break
else:
    print("    ok   30 random partitions, all 7 element sets")

print("PART C -- count_elements matches the filler")
for trial in range(30):
    nleaf = rng.integers(2, 5, size=3)
    ngroups = rng.integers(1, 4, size=3)
    full = nleaf*ngroups
    # same empty outer layer as part B: count_elements counts within
    # the box tiling, and the real tree always pads the tiling one
    # cell past the geometry (tree.py's npad)
    fs = np.zeros(tuple(full), np.int8)
    inner = tuple(slice(0, max(1, int(v) - 1)) for v in full)
    fs[inner] = (rng.random(fs[inner].shape) < 0.4).astype(np.int8)
    counts = st.count_elements(fs, nleaf, ngroups)
    whole = st.struc_from_cells(fs)
    ok = True
    for ii in range(7):
        ok &= counts[ii, 0] == int(np.count_nonzero(whole[ii]))
    if not check("counts == filler (trial %d)" % trial, ok,
                 "%s vs %s" % (list(counts[:, 0]),
                               [int(np.count_nonzero(w))
                                for w in whole])):
        break
else:
    print("    ok   30 random partitions, counter == filler")

print("\n%d checks failed" % len(fails))
sys.exit(1 if fails else 0)
