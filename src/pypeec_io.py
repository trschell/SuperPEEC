# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""PyPEEC front-end: mesher output + problem description -> VoxelModel.

PyPEEC (Guillod, Dartmouth -- MPL-2.0, checkout in ``pypeec/``)
describes a problem as two YAML files: ``geometry.yaml`` (consumed by
its MESHER, in one of four dialects: voxel / shape / png / stl) and
``problem.yaml`` (materials, sources, sweeps). This module deliberately
ingests the mesher's OUTPUT (``voxel.json.gz``) rather than the
geometry dialects: the mesher already collapses all four into one voxel
structure with named domains, so we inherit its shape/png/stl handling
for free and never re-implement a mesher.

WHAT MAPS, WHAT DOES NOT (all rejections are loud):

* electric materials, isotropic, lumped, real rho  -> per-voxel sigma.
  ``rho_im`` != 0, anisotropic orientation and distributed values are
  REJECTED for now -- the first two have natural homes
  (``VoxelModel.impedance_density`` carries complex z(w) since the
  superconductor work; per-orientation filament resistances could
  carry anisotropy) but nothing parses them yet; distributed sources/
  materials have no SuperPEEC counterpart at all.
* magnetic / electromagnetic materials -> REJECTED (SuperPEEC has no
  magnetic-media model; 8 of PyPEEC's 23 examples use them).
* sources -> equipotential PORTS. PyPEEC drives voxel DOMAINS with
  voltage/current sources (internal impedances and drive values are
  IGNORED here: network parameters R/L do not depend on them, which is
  what SuperPEEC extracts). Each port needs a (P domain, N domain) pair:
  pass ``ports={'p1': ('src', 'sink'), ...}`` or, with exactly two
  source domains, the pair is inferred. Port faces are the domains'
  EXPOSED faces (neighbouring empty space) on ONE axis -- the axis
  with the largest exposed area, since SuperPEEC's terminals require a
  single axis per port -- mirroring how the VoxHenry corpus places
  its ports on wire-end cross-sections.
* voxel pitch: NATIVE anisotropic support (2026-08-07). Near-cubic
  pitches (within ``pitch_tol``, default 2%; PyPEEC's tutorial is 0.1%
  off) snap to the mean; everything else passes through as
  ``VoxelModel.d``, where the core is measured correct against the
  exact dense box-integral oracle (~1e-3 truncation at 2:1 aspect with
  the aspect-compensating leaves ``partition()`` now picks, ~5e-3 at
  4:1 -- an element-aspect floor; resample yourself beyond that). The
  one remaining cubic-only feature is the skin engine (``subdivide``),
  which raises a clear error on anisotropic models.
* frequencies: the union over ``sweep_solver`` entries. Per-sweep
  material values are NOT supported (the first sweep's are used;
  differing later sweeps are rejected).

Usage::

    import pypeec, pypeec_io
    pypeec.run_mesher_file('geometry.yaml', 'voxel.json.gz')
    m = pypeec_io.read_pypeec('voxel.json.gz', 'problem.yaml')
    M = m.build_tree(*m.partition())
"""

import gzip
import json

import numpy as np

from voxmodel import VoxelModel, Port

_SIDE = {(0, -1): '-x', (0, 1): '+x', (1, -1): '-y', (1, 1): '+y',
         (2, -1): '-z', (2, 1): '+z'}


def _np(enc):
    """Decode PyPEEC's serialised-ndarray dict."""
    return np.asarray(enc['data']).reshape(enc['shape'])


def _domain_masks(data):
    """Domain name -> 3-D boolean mask. PyPEEC linear indices are
    FORTRAN-ordered (x fastest; verified on the tutorial: 'src'
    decodes to a flat 21x21 pad, C order scatters it across z)."""
    n = tuple(int(v) for v in data['n'])
    out = {}
    for name, enc in data['domain_def'].items():
        idx = _np(enc).astype(np.int64).ravel()
        m3 = np.zeros(n, dtype=bool)
        x, y, z = np.unravel_index(idx, n, order='F')
        m3[x, y, z] = True
        out[name] = m3
    return out


def _exposed_faces(mask, occ):
    """(count, faces) per (axis, sign): faces of ``mask`` cells whose
    neighbour along (axis, sign) is outside the conductor."""
    out = {}
    for axis in range(3):
        for sign in (-1, 1):
            nb = np.zeros_like(occ)
            src = [slice(None)]*3
            dst = [slice(None)]*3
            if sign > 0:
                dst[axis], src[axis] = slice(0, -1), slice(1, None)
            else:
                dst[axis], src[axis] = slice(1, None), slice(0, -1)
            nb[tuple(dst)] = occ[tuple(src)]
            exp = mask & ~nb
            out[(axis, sign)] = np.argwhere(exp)
    return out


def read_pypeec(voxel_path, problem_path, ports=None, pitch_tol=0.02):
    """Build a :class:`VoxelModel` from PyPEEC mesher output + problem.

    Parameters
    ----------
    voxel_path : str
        ``voxel.json.gz`` from ``pypeec mesher``.
    problem_path : str
        The example's ``problem.yaml``.
    ports : dict, optional
        ``{port_name: (P_domain, N_domain)}``. Inferred only when the
        problem has exactly two source domains.
    pitch_tol : float, optional
        Relative tolerance for the cubic-pitch policy above.
    """
    import yaml
    data = json.load(gzip.open(voxel_path))['data']
    prob = yaml.safe_load(open(problem_path))
    dom = _domain_masks(data)
    n = list(int(v) for v in data['n'])
    d = np.asarray(data['d'], dtype=float)

    # -- pitch policy (2026-08-07: NATIVE anisotropic support) ------
    # Near-cubic pitches snap to their mean (PyPEEC's tutorial is 0.1%
    # off cubic; the snap is far below discretisation error). Anything
    # else passes through as a per-axis pitch: the core handles it
    # (measured vs the dense oracle; partition() picks aspect-
    # compensating leaves). Accuracy note: ~1e-3 truncation at 2:1
    # aspect, ~5e-3 at 4:1 (element-aspect floor); the skin engine
    # (subdivide) stays cubic-only and raises if asked.
    ratio = d/d.min()
    if np.max(np.abs(ratio - 1.0)) <= pitch_tol:
        pitch = float(d.mean())
    else:
        pitch = d.copy()

    # -- materials ---------------------------------------------------
    sweeps = prob.get('sweep_solver', {})
    matvals = [sw['param']['material_val'] for sw in sweeps.values()
               if 'material_val' in sw.get('param', {})]
    if matvals and any(mv != matvals[0] for mv in matvals[1:]):
        raise NotImplementedError(
            "%s: per-sweep material values differ -- frequency-"
            "dependent materials are not mapped" % problem_path)
    matval = matvals[0] if matvals else prob.get('material_val', {})
    sigma = np.zeros(n, dtype=np.float32)
    for mname, md in prob['material_def'].items():
        if md['material_type'] != 'electric':
            raise NotImplementedError(
                "%s: material %r is %s -- SuperPEEC has no magnetic-media "
                "model" % (problem_path, mname, md['material_type']))
        if md.get('orientation_type', 'isotropic') != 'isotropic':
            raise NotImplementedError(
                "%s: material %r is anisotropic -- expressible via "
                "SuperPEEC's per-orientation resistances, but not mapped "
                "yet" % (problem_path, mname))
        if md.get('var_type', 'lumped') != 'lumped':
            raise NotImplementedError(
                "%s: material %r uses distributed values -- no SuperPEEC "
                "counterpart" % (problem_path, mname))
        mv = matval[mname]
        if abs(float(mv.get('rho_im', 0.0))) > 0.0:
            raise NotImplementedError(
                "%s: material %r has rho_im != 0 -- complex resistivity "
                "belongs in VoxelModel.impedance_density (the "
                "superconductor machinery) but is not mapped yet"
                % (problem_path, mname))
        s = 1.0/float(mv['rho_re'])
        for dn in md['domain_list']:
            sigma[dom[dn]] = s

    # -- ports from source domains ----------------------------------
    for sname, sd in prob['source_def'].items():
        if sd.get('var_type', 'lumped') != 'lumped':
            raise NotImplementedError(
                "%s: source %r is distributed -- no SuperPEEC counterpart"
                % (problem_path, sname))
    srcs = list(prob['source_def'])
    if ports is None:
        if len(srcs) != 2:
            raise ValueError(
                "%s: %d source domains -- pass ports={name: (P, N)} to "
                "pair them" % (problem_path, len(srcs)))
        ports = {'p1': (srcs[0], srcs[1])}
    occ = sigma != 0.0
    plist = []
    for pname, (dp, dn_) in ports.items():
        expP = _exposed_faces(dom[dp], occ)
        expN = _exposed_faces(dom[dn_], occ)
        best = max(range(3), key=lambda a: sum(
            len(e[(a, s)]) for e in (expP, expN) for s in (-1, 1)))
        port = Port(pname)
        for exp, tag in ((expP, 'P'), (expN, 'N')):
            for sign in (-1, 1):
                for ix, iy, iz in exp[(best, sign)]:
                    port._add(tag, (int(ix), int(iy), int(iz),
                                    best, sign))
        port._freeze()
        if len(port.pos) == 0 or len(port.neg) == 0:
            raise ValueError(
                "port %r: no exposed faces on axis %d for one side -- "
                "pick the pairing/axis by hand" % (pname, best))
        plist.append(port)

    # -- assemble ----------------------------------------------------
    m = VoxelModel(str(voxel_path))
    m.dims = tuple(n)
    m.d = pitch
    m.sigma = sigma
    m.ports = plist
    freqs = sorted({float(sw['param']['freq']) for sw in sweeps.values()
                    if 'freq' in sw.get('param', {})})
    m.freq = np.array(freqs if freqs else [0.0], dtype=float)
    return m
