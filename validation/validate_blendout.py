# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Gate for blendout -- the .glb actually lands right way up in Blender.

THE BUG THIS EXISTS FOR. glTF is Y-up and Blender's importer rotates
Y-up into its own Z-up, so the exporter has to pre-rotate. There are
two rotations that both "look like" the fix -- a matrix and its
transpose -- and the wrong one maps solver (x, y, z) to Blender
(x, -y, -z): a module upside down and mirrored in y. On a power module
that is a roughly symmetric slab of copper, and it renders as a
perfectly plausible picture; it was found only because bond wires that
should arc 2 mm ABOVE the traces were hidden underneath them, peeking
out at the trace edges. Nothing about the numbers is wrong, so no
value gate can catch it. Hence this one.

FOUR ANGLES, because a geometry export can be wrong in ways a single
coordinate check passes:
  A  ORIENTATION: an asymmetric marker -- distinct extents and a
     distinct sign in every axis -- round-trips through a real Blender
     import to the SAME coordinates the solver used, times the unit
     scale. Asymmetric is the whole point: a cube passes under all 48
     axis permutations, which is exactly how the bug survived.
  B  SCALE: the recorded unit_scale is the factor actually applied, so
     a 40 mm module measures 40 Blender units, not 0.04.
  C  DATA SURVIVAL: the raw _JMAG attribute arrives with the physical
     values that went in (this is the number, not the colour), and
     COLOR_0 arrives as a colour attribute.
  D  TOPOLOGY: face count and per-part split survive, so no primitive
     is silently dropped or merged.
  E  FEET: the flared contact foot seats on the outward face of the
     solver's own anchor cell at exactly the model's contact radius
     r0. A foot floating above the pad, buried inside it, or drawn at
     the wire radius would all still render as a plausible bond, and
     r0 is not cosmetic -- the constriction resistance footcal adds is
     computed for that disc.

Blender is invoked as a subprocess; the gate SKIPS (does not fail) if
no `blender` is on PATH, so it stays runnable in the solve
environment, where Blender is deliberately not a dependency.

Run: PYTHONPATH=src python3 validation/validate_blendout.py
"""
import json
import os as _op
import shutil
import subprocess
import sys as _sp
import tempfile

_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import numpy as np

import blendout

FAIL = []
SCALE = 1000.0

# Deliberately asymmetric: three different extents, and an origin
# offset that is positive in x, negative in y and positive in z, so
# every axis and every SIGN is distinguishable. A marker symmetric in
# any axis cannot detect a flip in that axis.
LO = np.array([2.0e-3, -5.0e-3, 1.0e-3])
HI = np.array([9.0e-3, -1.0e-3, 3.0e-3])
JVALS = np.array([1.5e4, 2.5e5, 3.5e6, 4.5e2], dtype=np.float64)


def check(name, ok, detail=""):
    print("  %-4s %-52s %s" % ("ok" if ok else "FAIL", name, detail),
          flush=True)
    if not ok:
        FAIL.append(name)


def marker_prims():
    """Two primitives of axis-aligned quads spanning the marker box."""
    x0, y0, z0 = LO
    x1, y1, z1 = HI
    # one quad per part, each in a different plane so an axis swap
    # changes its normal as well as its position
    faces = [[(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
             [(x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)]]
    prims = []
    for k, quad in enumerate(faces):
        pos = np.asarray(quad, dtype=np.float32)
        nrm = np.tile(np.array([[0, 0, 1 - 2*k]], np.float32), (4, 1))
        prims.append(dict(
            name='marker%d' % k, POSITION=pos, NORMAL=nrm,
            COLOR_0=np.full((4, 4), 200, np.uint8),
            _JMAG=np.full(4, JVALS[k], np.float32),
            indices=np.array([0, 1, 2, 0, 2, 3], np.uint32)))
    return prims


PROBE = r'''
import bpy, json, sys
import numpy as np
out = sys.argv[sys.argv.index('--') + 1 + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=sys.argv[sys.argv.index('--') + 1])
res = {}
for o in bpy.data.objects:
    if o.type != 'MESH':
        continue
    me = o.data
    co = np.empty(len(me.vertices)*3)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    world = np.array([list(o.matrix_world @ v.co) for v in me.vertices])
    jm = None
    if '_JMAG' in me.attributes:
        a = me.attributes['_JMAG']
        buf = np.empty(len(a.data))
        a.data.foreach_get('value', buf)
        jm = [float(buf.min()), float(buf.max())]
    res[o.name] = dict(
        vmin=world.min(0).tolist(), vmax=world.max(0).tolist(),
        nvert=len(me.vertices), npoly=len(me.polygons), jmag=jm,
        ncolattr=len(me.color_attributes))
json.dump(res, open(out, 'w'))
'''


def blender_probe(glb, workdir):
    exe = shutil.which('blender')
    if exe is None:
        return None
    probe = _op.path.join(workdir, 'probe.py')
    out = _op.path.join(workdir, 'probe.json')
    with open(probe, 'w') as fh:
        fh.write(PROBE)
    r = subprocess.run([exe, '-b', '--python', probe, '--', glb, out],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not _op.path.exists(out):
        print(r.stdout.decode()[-2000:])
        raise RuntimeError('blender probe produced no output')
    with open(out) as fh:
        return json.load(fh)


def check_feet():
    """Pure geometry: no Blender, no solve -- _foot against hand math."""
    l = np.array([0.5e-3, 0.5e-3, 0.2e-3])
    # contact 0.06 mm above the top face of cell (31, 32, 5), whose
    # outward (+z) face is at z = 6*0.2 mm = 1.2 mm
    cell = np.array([31, 32, 5])
    p = np.array([15.77e-3, 16.27e-3, 1.26e-3])
    r_wire, r0 = 0.13e-3, 0.26e-3
    pos, nor, tris = blendout._foot(p, cell, r_wire, r0, l, nsides=12)

    seat_z = pos[12:25, 2]
    check("foot seats on the anchor cell's outward face",
          np.allclose(seat_z, 1.2e-3, atol=1e-12),
          "z = %.6g mm, want 1.2" % (1e3*seat_z.mean()))
    top = pos[:12]
    check("foot meets the wire at the wire radius",
          np.allclose(np.linalg.norm(top[:, :2] - p[:2], axis=1),
                      r_wire, rtol=1e-9)
          and np.allclose(top[:, 2], p[2]),
          "%.6g mm, want %.6g" % (1e3*np.linalg.norm(top[0, :2] - p[:2]),
                                  1e3*r_wire))
    ring = pos[12:24]
    check("foot flares to the model's contact radius r0",
          np.allclose(np.linalg.norm(ring[:, :2] - p[:2], axis=1),
                      r0, rtol=1e-9),
          "%.6g mm, want %.6g (2x wire radius)"
          % (1e3*np.linalg.norm(ring[0, :2] - p[:2]), 1e3*r0))
    check("foot is a closed cone plus a disc fan",
          tris.shape == (36, 3) and int(tris.max()) == 24,
          "%s tris, max vert %d" % (tris.shape, int(tris.max())))

    # a downward-facing contact (wire landing on a ceiling) must seat
    # on the LOWER face -- the sign test, not just the axis test
    p2 = np.array([15.77e-3, 16.27e-3, 0.94e-3])
    pos2, _, _ = blendout._foot(p2, cell, r_wire, r0, l, nsides=12)
    check("foot normal follows the contact side, not just the axis",
          np.allclose(pos2[12:25, 2], 1.0e-3, atol=1e-12),
          "z = %.6g mm, want 1.0" % (1e3*pos2[12:25, 2].mean()))


def main():
    check_feet()
    work = tempfile.mkdtemp(prefix='blendout_gate_')
    glb = _op.path.join(work, 'marker.glb')
    prims = marker_prims()
    blendout.write_glb(glb, prims, scale=SCALE)
    print("wrote %s (%d bytes)" % (glb, _op.path.getsize(glb)))

    res = blender_probe(glb, work)
    if res is None:
        print("  SKIP  no `blender` on PATH -- orientation NOT gated")
        return 0

    check("both primitives imported", len(res) == 2,
          "got %s" % sorted(res))

    # -- A/B: orientation and scale, per axis and per sign -----------
    allmin = np.min([res[k]['vmin'] for k in res], axis=0)
    allmax = np.max([res[k]['vmax'] for k in res], axis=0)
    want_lo, want_hi = LO*SCALE, HI*SCALE
    elo = np.abs(allmin - want_lo).max()
    ehi = np.abs(allmax - want_hi).max()
    check("orientation: Blender coords == solver coords",
          elo < 1e-4 and ehi < 1e-4,
          "min %s want %s / max %s want %s"
          % (np.round(allmin, 5), want_lo, np.round(allmax, 5), want_hi))
    # the transposed-rotation bug specifically: y and z both negated
    flipped = (np.abs(allmin - np.array([want_lo[0], -want_hi[1],
                                         -want_hi[2]])).max() < 1e-4)
    check("not the (x,-y,-z) transposed-rotation bug", not flipped,
          "y/z sign flip" if flipped else "")
    check("scale: unit_scale applied to positions",
          abs((allmax[0] - allmin[0]) - (HI[0] - LO[0])*SCALE) < 1e-4,
          "x extent %.4f BU, want %.4f"
          % (allmax[0] - allmin[0], (HI[0] - LO[0])*SCALE))

    # -- C: the data, not the picture --------------------------------
    ok_j, det = True, []
    for k, name in enumerate(('marker0', 'marker1')):
        jm = res.get(name, {}).get('jmag')
        if jm is None or abs(jm[0] - JVALS[k]) > 1e-3*JVALS[k] \
                or abs(jm[1] - JVALS[k]) > 1e-3*JVALS[k]:
            ok_j = False
        det.append('%s %s' % (name, jm))
    check("_JMAG survives as physical values", ok_j, "; ".join(det))
    check("COLOR_0 survives as a colour attribute",
          all(res[k]['ncolattr'] >= 1 for k in res))

    # -- D: topology -------------------------------------------------
    check("no geometry dropped or merged",
          all(res[k]['nvert'] == 4 and res[k]['npoly'] == 2
              for k in res),
          "; ".join("%s %dv/%dp" % (k, res[k]['nvert'], res[k]['npoly'])
                    for k in sorted(res)))

    shutil.rmtree(work, ignore_errors=True)
    print("\n%d checks failed" % len(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
