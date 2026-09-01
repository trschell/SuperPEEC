# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""SuperPEEC's native input file: one TOML file = one solvable problem.

WHY A DOCTRINE. The repo grew three ways to define a problem (.vhr
files, PyPEEC mesher output, hand-built VoxelModels in studies) and no
way at all to define a bond wire. This module fixes the CONVENTIONS in
one place and gives every element -- voxel geometry, wires, port,
solve options -- a declarative home. See docs/input_doctrine.md for
the prose version; this docstring is the format's contract.

THE RULES (the doctrine part):
  * SI units throughout: metres, S/m, Hz. No unit fields, no implicit
    micrometres. A pitch of 200 um is written 200e-6.
  * ONE frame: the world origin is the corner of voxel (0,0,0); cell
    c occupies [c, c+1)*pitch per axis; a cell's centre is (c+0.5)*
    pitch. Wire polylines live in this same frame -- the convention
    every kernel and coupler in the wire program already uses.
  * Wires are POLYLINES: physical centreline points, first and last
    point are the bond CONTACTS (keep them a fraction of a cell above
    the metal surface; the foot cell underneath is FOUND, not stated,
    and the constriction model owns the remaining gap).
  * Unknown keys are ERRORS, not warnings. A typo like "radiius" must
    not silently produce a default-radius wire.
  * The skin-shape frequency policy is explicit: wire cross-section
    shapes retune per solve frequency (delta = sqrt(rho/(pi f mu0))),
    which rebuilds the wire coupling blocks -- build cost is part of
    the sweep, not hidden.

FILE SHAPE::

    [grid]                      # inline voxel geometry ...
    dims  = [24, 8, 3]
    pitch = 1e-6                # scalar or [px, py, pz]

    [[block]]                   # half-open cell ranges, like slicing
    from  = [0, 0, 0]
    to    = [10, 8, 2]
    sigma = 5.8e7

    [model]                     # ... OR a reference instead of [grid]
    vhr = "path/to/model.vhr"

    [[wire]]
    name   = "A"
    points = [[9.3e-6, 1.5e-6, 2.4e-6], [12e-6, 1.5e-6, 4e-6], ...]
    radius = 0.25e-6
    sigma  = 3.77e7
    # optional: max_seglen, nring, nsect, foot_r0

    [port]
    p_cells = [[0, 2, 1], [0, 3, 1]]
    n_cells = [[23, 2, 1], [23, 3, 1]]

    [solve]
    freq = [1e6, 1e7, 1e8]
    # optional: rtol, current
"""
import os
import tomllib

import numpy as np

import sppeec_status as _status

MU0 = 4e-7*np.pi

_SCHEMA = {
    'grid': {'dims', 'pitch', 'subpixel'},
    'cylinder': {'axis', 'center', 'radius', 'from', 'to',
                 'from_m', 'to_m', 'sigma', 'name'},
    'block': {'from', 'to', 'from_m', 'to_m', 'sigma', 'name',
              'epsilon', 'loss_tangent', 'lambda_l',
              'dispersion', 'f_ref', 'f1', 'f2', 'film'},
    'model': {'vhr'},
    'wire': {'name', 'points', 'radius', 'sigma', 'max_seglen',
             'nring', 'nsect', 'foot_r0', 'shape', 'start_vec',
             'end_vec', 'sagitta'},
    'port': {'p_cells', 'n_cells', 'p_box', 'n_box',
             'p_faces', 'n_faces', 'name', 'equipotential'},
    'solve': {'freq', 'rtol', 'current', 'foot_model', 'basis',
              'amg_cycles', 'method', 'gram_solver', 'formulation',
              'skin', 'maxiter'},
}

_FACE = {'+x': (0, 1), '-x': (0, -1), '+y': (1, 1), '-y': (1, -1),
         '+z': (2, 1), '-z': (2, -1)}


def _auto_rc(occ, axis):
    """Width-scaled mode-coupling truncation radii (2026-08-20).

    The skin engine's mode couplings correlate over the CROSS-SECTION
    WIDTH (the flat-section sibling of the wire-class rc ~ 2D law,
    doctrine rule 13): measured on the straight-bar ladder, rc = (3,4)
    is fine at 2 cells across but silently truncates ~20 delivered
    points at 4 across, where (6,8) -- 1.5-2x the width -- recovers
    +14 of them at unchanged apply cost. So: estimate the section
    width as the per-transverse-axis MEDIAN run length of occupied
    cells (median so one wide pour does not inflate rc everywhere),
    and take rc = (ceil(1.5 W), ceil(2 W)) off the THIN dimension.

    TWO GUARDS, both measured:
    * cost cap (12, 16): table setup grows with rc^3 (clipped to the
      grid extent);
    * the DAMAGE ZONE: on wide sections a hard cutoff at 0.5-1.5x the
      width lands in the rc-ladder's measured non-monotonic region
      (20x20 bar: rc 12-20 WORSE than (3,4) -- the cutoff breaks the
      cancellation of the net-zero dipole tails mid-shell). When the
      scaled rc cannot clear 1.5x the WIDE dimension under the cap,
      fall back to the small-(3,4) regime (locally-1-D flat-face
      physics) instead of stopping mid-shell.
    """
    occ = np.asarray(occ).astype(bool)
    meds = []
    for t in (a for a in range(3) if a != int(axis)):
        o = np.moveaxis(occ, t, -1)
        n = o.shape[-1]
        flat = o.reshape(-1, n)
        pad = np.zeros((flat.shape[0], n + 2), dtype=np.int8)
        pad[:, 1:-1] = flat
        d = np.diff(pad, axis=1)
        starts = np.argwhere(d == 1)
        ends = np.argwhere(d == -1)
        if starts.size == 0:
            continue
        meds.append(float(np.median(ends[:, 1] - starts[:, 1])))
    if not meds:
        return 3, 4
    w_thin, w_wide = min(meds), max(meds)
    ru = max(3, int(np.ceil(1.5*w_thin)))
    rc = max(4, int(np.ceil(2.0*w_thin)))
    if ru > 12 or rc > 16:
        return 3, 4                       # cap: fall back, never mid-shell
    if rc > 0.5*w_wide and ru < 1.5*w_wide:
        return 3, 4                       # would cut the wide axis mid-shell
    return ru, rc
_TOP = {'grid', 'block', 'model', 'wire', 'port',
        'cylinder', 'solve'}


def _reject_unknown(doc):
    for key, val in doc.items():
        if key not in _TOP:
            raise ValueError("unknown top-level table %r -- the input "
                             "doctrine rejects unknown keys" % key)
        entries = val if isinstance(val, list) else [val]
        for ent in entries:
            bad = set(ent) - _SCHEMA[key]
            if bad:
                raise ValueError("unknown key(s) %s in [%s] -- typo? "
                                 "allowed: %s"
                                 % (sorted(bad), key,
                                    sorted(_SCHEMA[key])))


class Problem:
    """A parsed input file: build the model, wires and solver from it.

    ``wires(freq)`` rebuilds the Wire objects with the skin shape
    retuned to ``freq`` -- by doctrine the shape follows the solve
    frequency, so a sweep rebuilds the wire coupling per point.
    """

    def __init__(self, doc, path='<inline>'):
        _reject_unknown(doc)
        self.path = path
        if ('grid' in doc) == ('model' in doc):
            raise ValueError("give exactly one of [grid] (+[[block]]) "
                             "or [model]")
        self._doc = doc
        self.wire_specs = doc.get('wire', [])
        # [[cylinder]]: the first SUBPIXEL primitive (2026-08-18) --
        # a round conductor voxelized WITH per-cell fill fractions,
        # folded to sigma_eff = sigma*fill so the per-cell-
        # conductivity machinery carries the partial-cell resistance
        # exactly. v1 scope: conductors only; Lp stays full-cell
        # (stage B of the subpixel program owns the inductance
        # correction).
        if doc.get('cylinder') and self.wire_specs:
            raise ValueError(
                "[[cylinder]] and [[wire]] do not combine in subpixel "
                "v1 -- the wire path has no partial-cell corrections")
        for k, cy in enumerate(doc.get('cylinder', [])):
            for req in ('axis', 'center', 'radius', 'sigma'):
                if req not in cy:
                    raise ValueError("cylinder %d is missing %r"
                                     % (k, req))
            if str(cy['axis']) not in ('x', 'y', 'z'):
                raise ValueError("cylinder %d: axis must be 'x', 'y' "
                                 "or 'z'" % k)
            if float(cy['radius']) <= 0:
                raise ValueError("cylinder %d: radius must be > 0" % k)
            if len(cy['center']) != 2:
                raise ValueError(
                    "cylinder %d: center is the TWO transverse "
                    "coordinates (metres), in axis order with the "
                    "cylinder axis removed" % k)
        for k, w in enumerate(self.wire_specs):
            for req in ('points', 'radius', 'sigma'):
                if req not in w:
                    raise ValueError("wire %d is missing %r"
                                     % (k, req))
            nm = w.get('name', k)
            shape = str(w.get('shape', 'polyline'))
            if shape not in ('polyline', 'spline'):
                raise ValueError("wire %r: shape must be 'polyline' "
                                 "(default) or 'spline', not %r"
                                 % (nm, shape))
            ends = [key for key in ('start_vec', 'end_vec') if key in w]
            if shape == 'spline':
                # a spline is UNDER-determined without both takeoff
                # vectors: with no middle points the two feet alone
                # fix nothing but a straight line
                missing = sorted({'start_vec', 'end_vec'} - set(ends))
                if missing:
                    raise ValueError(
                        "wire %r: shape='spline' needs %s -- the "
                        "takeoff vector at each foot, pointing AWAY "
                        "from its own pad (both +z for an ordinary "
                        "bond); its magnitude is the control handle "
                        "and sets the loop height"
                        % (nm, " and ".join(repr(v) for v in missing)))
                for key in ends:
                    v = np.asarray(w[key], dtype=float)
                    if v.shape != (3,) or not np.linalg.norm(v) > 0:
                        raise ValueError(
                            "wire %r: %s must be a non-zero [x, y, z] "
                            "vector in metres" % (nm, key))
                if float(w.get('sagitta', 0.1)) <= 0:
                    raise ValueError("wire %r: sagitta must be > 0"
                                     % (nm,))
            elif ends:
                # silently ignoring these would let a typo'd shape key
                # produce square wires that look deliberate
                raise ValueError(
                    "wire %r: %s only apply to shape='spline' (this "
                    "wire is a %s)" % (nm, " and ".join(ends), shape))
        # [port] (one) or [[port]] (many, Z-matrix order = declaration
        # order). Each is face-style (p_faces/n_faces -- the LpPR
        # injection port, entries [ix, iy, iz, "+z"] naming a conductor
        # cell and which face carries the terminal current) or
        # cell-style (p_cells/n_cells/p_box/n_box -- the LpR terminal
        # port); one style per file.
        port_raw = doc.get('port', {})
        self.port_specs = port_raw if isinstance(port_raw, list) \
            else ([port_raw] if port_raw else [])
        styles = set()
        self.ports_faces = []
        for k, port in enumerate(self.port_specs):
            pname = str(port.get('name', 'port%d' % (k + 1)))
            for side in ('p', 'n'):
                if (side + '_cells' in port) and (side + '_box' in port):
                    raise ValueError("port %r: give %s_cells OR %s_box,"
                                     " not both" % (pname, side, side))
            pf = self._parse_faces(port.get('p_faces'))
            nf = self._parse_faces(port.get('n_faces'))
            if (pf is None) != (nf is None):
                raise ValueError("port %r: p_faces and n_faces come "
                                 "together" % pname)
            has_cells = any(key in port for key in
                            ('p_cells', 'n_cells', 'p_box', 'n_box'))
            if (pf is not None) and has_cells:
                raise ValueError(
                    "port %r is EITHER face-style (p_faces/n_faces -- "
                    "the LpPR injection port) OR cell-style (p_cells/"
                    "n_cells/p_box/n_box -- the LpR terminal port), "
                    "not both" % pname)
            if pf is not None:
                styles.add('faces')
                self.ports_faces.append((pname, pf, nf))
            elif has_cells:
                styles.add('cells')
        if len(styles) > 1:
            raise ValueError("all ports must share one style -- got "
                             "both face-style and cell-style tables")
        faces_port = 'faces' in styles
        cells_port = 'cells' in styles
        if cells_port and len(self.port_specs) > 1:
            raise ValueError(
                "multiple ports need face-style tables (the LpPR "
                "Z-matrix path) -- the LpR terminal port is "
                "two-terminal by construction")
        # legacy single-port attributes (the LpR path reads these)
        first = self.port_specs[0] if self.port_specs else {}
        self.port_p = [tuple(int(v) for v in c)
                       for c in first.get('p_cells', [])]
        self.port_n = [tuple(int(v) for v in c)
                       for c in first.get('n_cells', [])]
        # physical port boxes (metres): resolved against the built
        # model, so a refinement ladder keeps the SAME physical port
        # (the doctrine's own rule)
        self.port_p_box = first.get('p_box')
        self.port_n_box = first.get('n_box')
        # equipotential port: the terminal current SPLIT is solved,
        # not prescribed (EquiTerminalSolver). Face-style, single
        # port, LpR-class -- the LpPR injection port has no
        # equipotential terminal.
        self.equipotential = any(bool(p.get('equipotential', False))
                                 for p in self.port_specs)
        if self.equipotential:
            if not faces_port:
                raise ValueError(
                    "equipotential = true needs a face-style port "
                    "(p_faces/n_faces): the equipotential terminal "
                    "lives on conductor faces")
            if len(self.port_specs) > 1:
                raise ValueError(
                    "equipotential ports are single-port for now -- "
                    "the Z-matrix path uses prescribed injections")
        solve = doc.get('solve', {})
        fr = solve.get('freq', [])
        if isinstance(fr, dict):
            # sweep expression: freq = {from, to, points, spacing}
            bad = set(fr) - {'from', 'to', 'points', 'spacing'}
            if bad:
                raise ValueError("freq table: unknown key(s) %s -- "
                                 "allowed: from, to, points, spacing"
                                 % sorted(bad))
            for req in ('from', 'to', 'points'):
                if req not in fr:
                    raise ValueError("freq table needs %r" % req)
            f0, f1 = float(fr['from']), float(fr['to'])
            npts = int(fr['points'])
            sp = str(fr.get('spacing', 'log'))
            if sp not in ('log', 'lin'):
                raise ValueError("freq.spacing must be 'log' or "
                                 "'lin', got %r" % sp)
            if not (0 < f0 < f1) or npts < 2:
                raise ValueError(
                    "freq table needs 0 < from < to and points >= 2 "
                    "(a single point is a literal list: freq = [%g])"
                    % f0)
            self.freqs = [float(v) for v in (
                np.logspace(np.log10(f0), np.log10(f1), npts)
                if sp == 'log' else np.linspace(f0, f1, npts))]
        else:
            self.freqs = [float(f) for f in fr]

        # -- formulation: what physics the file asks for ------------
        # 'auto' derives it from the materials (the doctrine's
        # feet-are-found rule applied to physics): any dielectric
        # block or face-style port means charge matters -> LpPR;
        # otherwise the galvanic wire path -> LpR. The explicit key
        # exists for the case auto cannot know: conductor-only
        # capacitance near the wL/R crossover.
        has_eps = any('epsilon' in b for b in doc.get('block', []))
        has_sc = any('lambda_l' in b for b in doc.get('block', []))
        if has_eps and has_sc:
            raise ValueError(
                "dielectric (epsilon) and superconducting (lambda_l) "
                "blocks cannot combine in one model -- "
                "impedance_density does not compose them yet")
        form = str(solve.get('formulation', 'auto'))
        if form not in ('auto', 'LpR', 'LpPR'):
            raise ValueError("solve.formulation must be 'auto', 'LpR' "
                             "or 'LpPR', got %r" % (form,))
        if form == 'auto':
            if self.equipotential:
                form = 'LpR'         # solved-split terminal, inductive
            else:
                form = 'LpPR' if (has_eps or has_sc or faces_port) \
                    else 'LpR'
        if self.equipotential:
            if has_eps:
                raise ValueError(
                    "equipotential ports cannot combine with "
                    "dielectric blocks -- charge needs the LpPR path, "
                    "whose port is a prescribed injection")
            if form == 'LpPR':
                raise ValueError(
                    "formulation LpPR has no equipotential terminal "
                    "(prescribed injection only) -- drop "
                    "equipotential = true or the formulation key")
            if self.wire_specs:
                raise ValueError(
                    "equipotential ports and [[wire]] tables do not "
                    "combine -- the wire path derives its own "
                    "terminals from cell-style ports")
        if form == 'LpPR':
            if self.wire_specs:
                raise ValueError(
                    "formulation LpPR with [[wire]] tables is not "
                    "supported (wire capacitance is excluded by "
                    "doctrine v1) -- model the bond as voxel metal, "
                    "or drop the dielectric/superconducting blocks "
                    "and run LpR")
            if 'grid' in doc and not faces_port:
                raise ValueError(
                    "formulation LpPR needs a face-style port "
                    "(p_faces/n_faces): the LpPR port is a nodal "
                    "current injection on conductor faces")
        else:
            if has_eps:
                raise ValueError(
                    "dielectric blocks (epsilon) need formulation "
                    "LpPR -- the LpR path has no charge unknowns")
            if has_sc and not self.equipotential:
                raise ValueError(
                    "superconducting blocks (lambda_l) need "
                    "formulation LpPR or an equipotential port -- "
                    "the wire path's foot calibration assumes ohmic "
                    "pads")
            if faces_port and not self.equipotential:
                raise ValueError(
                    "face-style ports are the LpPR injection port; "
                    "the LpR path uses p_cells/n_cells or p_box/n_box "
                    "(or add equipotential = true for the solved-"
                    "split terminal)")
        self.formulation = form
        # engineering default per formulation: 1e-4 buys 5-digit Z on
        # the LpR path (the iteration-growth law); every recorded LpPR
        # anchor (validate_dielectric, the pdn capstone/sweeps) was
        # taken at 1e-10, so that stays its default.
        self.rtol = float(solve['rtol']) if 'rtol' in solve else \
            (1e-4 if form == 'LpR' else 1e-10)
        self.current = float(solve.get('current', 1.0))
        # 'patch' = the calibrated footprint foot (default); 'point' =
        # legacy single-cell + analytic disc, kept for comparison
        self.foot_model = str(solve.get('foot_model', 'patch'))
        # 'auto' (size-evaluated, default), 'selected', or
        # 'overcomplete' (BlockAMG + Schur, memory-flat)
        self.solver_basis = str(solve.get('basis', 'auto'))
        self.amg_cycles = int(solve.get('amg_cycles', 4))
        # outer Krylov: 'lgmres' (default, small basis inner_m=10 --
        # the efficient memory/time point) or 'bicgstab' (leanest at
        # ~8 work vectors, for a hard memory ceiling)
        self.method = str(solve.get('method', 'lgmres'))
        # outer Krylov cycle cap (matvec budget = maxiter * inner_m);
        # None keeps each solver's own default (30). Exists for large
        # runs that hit the cap (2026-08-27: the 6.8M-cell RSFQ JTL
        # rung stopped at 331 matvecs / resid 6e-2 where the
        # overcomplete N^0.66 law wants ~560) -- with SPPEEC_CHECKPOINT
        # a capped run resumes, but the cap itself must be settable.
        self.maxiter = solve.get('maxiter')
        if self.maxiter is not None:
            if isinstance(self.maxiter, bool) or int(self.maxiter) < 1:
                raise ValueError("solve.maxiter must be an integer >= 1 "
                                 "(outer Krylov cycles)")
            self.maxiter = int(self.maxiter)
        if self.method not in ('bicgstab', 'lgmres'):
            raise ValueError("solve.method must be 'bicgstab' or "
                             "'lgmres', got %r" % (self.method,))
        # loop-Gram preconditioner under basis='overcomplete':
        # 'geo' (geometric MG, default) or 'amg' (smoothed aggregation)
        self.gram_solver = str(solve.get('gram_solver', 'geo'))
        if self.gram_solver not in ('geo', 'amg'):
            raise ValueError("solve.gram_solver must be 'geo' or "
                             "'amg', got %r" % (self.gram_solver,))

        # -- sub-cell skin engine (equipotential path only) ----------
        # Default policy: engage AUTOMATICALLY with the conduction
        # basis, but only when the cell size justifies it -- the
        # engine's own recommend_subdivision returns k = 1 (off) when
        # the mesh already resolves the skin depth at the sweep's
        # highest frequency, so it costs nothing where it buys
        # nothing. Deliberately NOT exposed: use_fft, csr_max_gb,
        # chol_*, split_axis (solver internals).
        skin = solve.get('skin', {})
        if not isinstance(skin, dict):
            raise ValueError("solve.skin is a table, e.g. "
                             "skin = { mode = \"off\" }")
        bad = set(skin) - {'mode', 'basis', 'k', 'f_ref', 'rc_uu',
                           'rc_cross', 'boundary_only'}
        if bad:
            raise ValueError("solve.skin: unknown key(s) %s -- "
                             "allowed: mode, basis, k, f_ref, rc_uu, "
                             "rc_cross, boundary_only" % sorted(bad))
        if skin and not self.equipotential:
            raise ValueError(
                "solve.skin configures the sub-cell skin engine, "
                "which lives on the equipotential-terminal path -- "
                "add equipotential = true to the port (the wire and "
                "LpPR paths carry their own skin models)")
        self.skin = dict(
            mode=str(skin.get('mode', 'auto')),
            basis=str(skin.get('basis', 'conduction')),
            k=skin.get('k'),
            f_ref=skin.get('f_ref'),
            rc_uu=skin.get('rc_uu'),
            rc_cross=skin.get('rc_cross'),
            boundary_only=bool(skin.get('boundary_only', True)))
        if self.skin['mode'] not in ('auto', 'on', 'off'):
            raise ValueError("skin.mode must be 'auto' (engage when "
                             "the cell size justifies it), 'on' or "
                             "'off', got %r" % self.skin['mode'])
        if self.skin['basis'] not in ('conduction', 'linear', 'diff'):
            raise ValueError("skin.basis must be 'conduction' (the "
                             "measured best), 'linear' or 'diff', "
                             "got %r" % self.skin['basis'])
        if self.skin['k'] is not None:
            kv = int(self.skin['k'])
            if kv == 2:
                raise ValueError(
                    "skin.k = 2 is BLIND to axially symmetric "
                    "neighbourhoods (all four quadrants equivalent, "
                    "cross-couplings cancel exactly -- measured at "
                    "machine zero); use 3 or higher, odd preferred")
            if kv < 3:
                raise ValueError("skin.k must be >= 3 (or omit it "
                                 "for the automatic choice)")
            self.skin['k'] = kv
        if self.skin['f_ref'] is not None \
                and float(self.skin['f_ref']) <= 0:
            raise ValueError("skin.f_ref must be > 0")
        for key in ('rc_uu', 'rc_cross'):
            if self.skin[key] is not None:
                self.skin[key] = int(self.skin[key])
                if self.skin[key] < 1:
                    raise ValueError("skin.%s must be >= 1 (or omit "
                                     "it for the width-scaled "
                                     "automatic choice)" % key)

    def model(self):
        """Build the VoxelModel (inline grid or .vhr reference)."""
        import voxmodel
        if 'model' in self._doc:
            # READ the file, do not just name it. `VoxelModel(path)` is
            # an empty shell -- dims is None until something parses the
            # geometry -- so this branch used to hand `tree()` a model
            # with no dimensions and raise
            #     TypeError: int() argument must be ... not 'NoneType'
            # `[model] vhr` is a first-class part of the format (schema
            # line 72, and mutually exclusive with `[grid]`), so this was
            # THE documented route for running a VoxHenry file, and it
            # could not run one. Nothing in examples/, validation/ or
            # docs/input_doctrine.md exercised it, which is why it
            # survived. `vhr.read_vhr` returns a fully-populated
            # VoxelModel (VhrModel IS VoxelModel), which is what every
            # other caller in the tree already uses.
            import vhr as _vhr
            return _vhr.read_vhr(self._doc['model']['vhr'])
        g = self._doc['grid']
        m = voxmodel.VoxelModel(self.path)
        m.dims = tuple(int(v) for v in g['dims'])
        m.d = np.asarray(g['pitch'], dtype=float)
        if m.d.ndim == 0:
            m.d = np.full(3, float(m.d))
        # f32 storage: conductivity is ~3-digit material data and a
        # full-grid array; resistances() upcasts to f64 for compute
        m.sigma = np.zeros(m.dims, dtype=np.float32)
        eps_blocks = []
        disp_blocks = []
        # SUBPIXEL BLOCKS. `[grid] subpixel = true` stops rounding a
        # block's physical bounds to the nearest cell boundary and
        # represents the boundary cells for what they are: partial. A
        # cell cut by an axis-aligned plane is a LAMINATE, whose exact
        # effective conductivity is anisotropic (arithmetic along the
        # layers, harmonic across), so this is not an approximation of
        # the material law -- it is the material law the staircase was
        # approximating. See VoxelModel.laminate_sigma.
        subpix = bool(g.get('subpixel', False))
        # coverage ACCUMULATES across blocks and is capped at 1: two
        # abutting layers routinely share a boundary cell (in the SFQ5ee
        # stack at 100 nm, M5 covers 0.35 of cell 21 and J5 the other
        # 0.65), and that cell is FULL, of two conductors. Taking a
        # minimum instead would have called it 35% filled.
        cover = np.zeros(m.dims, dtype=np.float64) if subpix else None
        cut_axis = None
        # THIN-FILM DECLARATION (2026-09-01). `film = "z"` on a block is
        # a HINT about the stiff axis, not a formulation change: in-plane
        # current through the block varies on the lambda (or skin-depth)
        # scale along that normal, so the sub-cell mode engine should
        # spend its budget there -- a 1-D split along the normal with the
        # normal-axis shape family, instead of the k x k cross-section
        # palette (see EquiTerminalSolver). Amplitudes are always SOLVED;
        # if the field disagrees with the profile ansatz near an edge or
        # via mouth, the solve wins. One normal per model in v1 (a layer
        # stack shares it); mixed normals raise.
        film_normal = None
        for b in self._doc.get('block', []):
            fa = b.get('film')
            if fa is not None:
                if fa not in ('x', 'y', 'z'):
                    raise ValueError("block film = %r: must be 'x', 'y' "
                                     "or 'z' (the film NORMAL)" % (fa,))
                fa = 'xyz'.index(fa)
                if film_normal is not None and film_normal != fa:
                    raise ValueError(
                        "blocks declare different film normals (%s vs "
                        "%s) -- one normal per model in v1"
                        % ('xyz'[film_normal], 'xyz'[fa]))
                film_normal = fa
            has_cells = ('from' in b) or ('to' in b)
            has_m = ('from_m' in b) or ('to_m' in b)
            if has_cells == has_m:
                raise ValueError("block %r: give from/to (cells) OR "
                                 "from_m/to_m (metres)"
                                 % b.get('name', '?'))
            frac_axis = None
            if has_m and subpix:
                # which axes are NOT commensurate with the pitch?
                nm = b.get('name', '?')
                bad = []
                for k in range(3):
                    for v in (b['from_m'][k], b['to_m'][k]):
                        c = float(v)/float(m.d[k])
                        if abs(c - round(c)) > 1e-9:
                            bad.append(k)
                            break
                bad = sorted(set(bad))
                if len(bad) > 1:
                    raise ValueError(
                        "block %r is off-grid on axes %s. Subpixel v1 "
                        "cuts ONE axis per model: a cell cut on two "
                        "axes at once is not a laminate, and neither "
                        "the arithmetic nor the harmonic mean applies "
                        "to it. Make the other axis commensurate, or "
                        "set [grid] subpixel = false to snap."
                        % (nm, bad))
                if bad:
                    frac_axis = bad[0]
                    if cut_axis is not None and cut_axis != frac_axis:
                        raise ValueError(
                            "block %r is off-grid on axis %d but an "
                            "earlier block cut axis %d. Subpixel v1 "
                            "carries ONE cut axis per model (the same "
                            "restriction the [[cylinder]] path has)."
                            % (nm, frac_axis, cut_axis))
                    cut_axis = frac_axis
                # the CELL RANGE is the outer hull of the coverage, so
                # a partial cell is included and then weighted below
                lo, hi = [], []
                for k in range(3):
                    a = float(b['from_m'][k])/float(m.d[k])
                    z = float(b['to_m'][k])/float(m.d[k])
                    lo.append(int(np.floor(a + 1e-9)))
                    hi.append(int(np.ceil(z - 1e-9)))
            elif has_m:
                lo = self._snap(b['from_m'], m.d, b.get('name', '?'))
                hi = self._snap(b['to_m'], m.d, b.get('name', '?'))
            else:
                lo = [int(v) for v in b['from']]
                hi = [int(v) for v in b['to']]
            if any(h <= l for l, h in zip(lo, hi)) or \
               any(l < 0 for l in lo) or \
               any(h > d for h, d in zip(hi, m.dims)):
                raise ValueError("block %s..%s does not fit the grid "
                                 "%s (half-open cell ranges)"
                                 % (lo, hi, m.dims))
            if not any(k in b for k in ('sigma', 'epsilon',
                                        'lambda_l')):
                raise ValueError("block %r: give sigma (conductor), "
                                 "epsilon (dielectric), lambda_l "
                                 "(superconductor) or a combination"
                                 % b.get('name', '?'))
            for k in ('loss_tangent', 'dispersion'):
                if k in b and 'epsilon' not in b:
                    raise ValueError("block %r: %s needs epsilon"
                                     % (b.get('name', '?'), k))
            for k in ('f_ref', 'f1', 'f2'):
                if k in b and 'dispersion' not in b:
                    raise ValueError("block %r: %s only means "
                                     "something under dispersion"
                                     % (b.get('name', '?'), k))
            if subpix and has_m:
                # per-cell coverage along the cut axis, clipped to this
                # block's cell range; other axes are commensurate so
                # their coverage is exactly 1
                # every block contributes, on-grid ones included: a
                # commensurate block covers its cells exactly 1.0, and
                # leaving it out would leave its cells at zero
                ax = frac_axis if frac_axis is not None else (
                    cut_axis if cut_axis is not None else 2)
                cv = self._cover(b['from_m'][ax], b['to_m'][ax],
                                 m.d[ax], m.dims[ax])
                shape = [1, 1, 1]
                shape[ax] = m.dims[ax]
                blk = [slice(lo[k], hi[k]) for k in range(3)]
                cover[tuple(blk)] += cv.reshape(shape)[tuple(
                    blk[k] if k == ax else slice(None)
                    for k in range(3))]
            if 'sigma' in b:
                m.sigma[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = \
                    float(b['sigma'])
            if 'lambda_l' in b:
                # two-fluid London cell: lambda_l is the London depth
                # (metres), sigma (if given) the normal channel.
                # VoxHenry's convention is honoured downstream:
                # lambda_l == 0 on a conductor cell means NORMAL metal
                # (the infinite-lambda limit), so the default zeros
                # leave every plain block ohmic.
                lam = float(b['lambda_l'])
                if lam <= 0:
                    raise ValueError("block %r: lambda_l must be > 0 "
                                     "(metres)" % b.get('name', '?'))
                if m.lambdaL is None:
                    m.lambdaL = np.zeros(m.dims, dtype=np.float32)
                m.lambdaL[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = lam
                m.superconductor = True
            if 'epsilon' in b:
                er, df = float(b['epsilon']), \
                    float(b.get('loss_tangent', 0.0))
                if er <= 0 or df < 0:
                    raise ValueError("block %r: epsilon must be > 0 "
                                     "and loss_tangent >= 0"
                                     % b.get('name', '?'))
                # a lossy dielectric's eps_r*(1 - j*df) rides the same
                # Ruehli material law, so its conduction loss enters
                # Re(Z) for free (the pdn_sweep convention)
                eps_blocks.append((lo, hi,
                                   er*(1.0 - 1j*df) if df else er))
                if 'dispersion' in b:
                    # Djordjevic-Sarkar wideband Debye: (epsilon,
                    # loss_tangent) are the values AT f_ref; fit
                    # (eps_inf, deps) so eps(f_ref) == er*(1 - j*df)
                    # exactly, then the model supplies the causal
                    # eps(f) everywhere else (voxmodel.ds_epsilon).
                    if str(b['dispersion']) != 'djordjevic':
                        raise ValueError(
                            "block %r: dispersion must be "
                            "'djordjevic' (the wideband-Debye "
                            "model), got %r"
                            % (b.get('name', '?'), b['dispersion']))
                    if df <= 0:
                        raise ValueError(
                            "block %r: dispersion needs "
                            "loss_tangent > 0 -- a lossless "
                            "dielectric does not disperse"
                            % b.get('name', '?'))
                    if 'f_ref' not in b:
                        raise ValueError(
                            "block %r: dispersion needs f_ref (the "
                            "frequency epsilon/loss_tangent are "
                            "stated at)" % b.get('name', '?'))
                    fr_ = float(b['f_ref'])
                    fl = float(b.get('f1', 1e3))
                    fh = float(b.get('f2', 1e12))
                    if not (0 < fl < fr_ < fh):
                        raise ValueError(
                            "block %r: dispersion needs "
                            "0 < f1 < f_ref < f2 (got f1 %g, f_ref "
                            "%g, f2 %g)" % (b.get('name', '?'),
                                            fl, fr_, fh))
                    L = np.log((fh + 1j*fr_)/(fl + 1j*fr_)) \
                        / np.log(fh/fl)
                    deps = -er*df/L.imag
                    eps_inf = er - deps*L.real
                    if eps_inf < 1.0:
                        raise ValueError(
                            "block %r: loss_tangent %g is too large "
                            "for the (f1, f2) band -- the fitted "
                            "eps_inf %.3g < 1 is unphysical; widen "
                            "the band (f1 %g, f2 %g)"
                            % (b.get('name', '?'), df, eps_inf,
                               fl, fh))
                    disp_blocks.append((lo, hi, float(eps_inf),
                                        float(deps), fl, fh))
        if subpix and cut_axis is not None:
            np.clip(cover, 0.0, 1.0, out=cover)
            # ONE pass, after every block has painted. Z_SCALE rather
            # than laminate_sigma: this model may be ohmic (z = 1/sigma)
            # or London (z = j w mu lambda^2), and in BOTH the partial
            # cell's impedance density scales as 1/f in the plane of the
            # cut. One multiplier serves both, where a sigma array
            # cannot describe a lossless superconductor at all -- its
            # sigma is legitimately zero everywhere.
            occ = np.asarray(m.struc()) > 0
            if np.any((cover > 0.0) & (cover < 1.0) & ~occ):
                raise ValueError(
                    "a partial cell carries no material -- subpixel "
                    "coverage and the painted blocks disagree")
            zs = np.ones(tuple(m.dims) + (3,), dtype=np.float64)
            inv = np.where(cover > 0.0,
                           1.0/np.maximum(cover, 1e-300), 1.0)
            for a in range(3):
                if a != cut_axis:
                    zs[..., a] = np.where(occ, inv, 1.0)
            # The CUT AXIS keeps the bulk value. The laminate's harmonic
            # mean is the through-plane figure for a path passing all
            # the way THROUGH a cell, and no filament does that: the
            # half-pair rule gives a filament the top half of one cell
            # and the bottom half of the next, so the through-plane
            # value is a HALF-CELL quantity this array cannot express
            # (see VoxelModel.laminate_sigma). In-plane is exact and is
            # what is measured; the cut axis is left alone rather than
            # given a value that was never tested.
            m.z_scale = zs
            m.slab_fill = dict(fill=cover, axis=int(cut_axis))
        for k, cy in enumerate(self._doc.get('cylinder', [])):
            axis = 'xyz'.index(str(cy['axis']))
            t1, t2 = [ax for ax in range(3) if ax != axis]
            if ('from' in cy) or ('to' in cy):
                a0 = int(cy.get('from', 0))
                a1 = int(cy.get('to', m.dims[axis]))
            elif ('from_m' in cy) or ('to_m' in cy):
                a0 = int(round(float(cy.get('from_m', 0.0))
                               / m.d[axis]))
                a1 = int(round(float(cy.get('to_m',
                                            m.dims[axis]*m.d[axis]))
                               / m.d[axis]))
            else:
                a0, a1 = 0, m.dims[axis]
            if not (0 <= a0 < a1 <= m.dims[axis]):
                raise ValueError("cylinder %d: axial span [%d, %d) "
                                 "does not fit the grid" % (k, a0, a1))
            c1, c2 = float(cy['center'][0]), float(cy['center'][1])
            R = float(cy['radius'])
            sig = float(cy['sigma'])
            # per-cell fill fraction by sub-sampling (s^2 points per
            # transverse cell; error ~ chord/s of a cell, well under
            # the 1e-3 sliver threshold below)
            s = 64
            o = (np.arange(s) + 0.5)/s
            p1, p2 = float(m.d[t1]), float(m.d[t2])
            x = (np.arange(m.dims[t1])[:, None] + o[None, :])*p1 - c1
            y = (np.arange(m.dims[t2])[:, None] + o[None, :])*p2 - c2
            inside = (x[:, None, :, None]**2
                      + y[None, :, None, :]**2) < R*R
            fill = inside.mean(axis=(2, 3))
            # k=4 sub-fill bins per cell (stage B consumes these as
            # sub-prism current weights; 64 samples/bin)
            # k = 4 sub-cells: measured NOT the accuracy limiter --
            # k = 8 left the Kelvin-gate errors unchanged-to-worse;
            # the high-dx/delta residual is cell-level discretization
            # of the crowding envelope (stage C.2's charter)
            ks = 4
            sub = inside.reshape(inside.shape[0], inside.shape[1],
                                 ks, 64//ks, ks, 64//ks
                                 ).mean(axis=(3, 5))
            if m.fill_frac is None:
                m.fill_frac = (m.sigma != 0).astype(np.float32)
            if not hasattr(m, 'subpixel') or m.subpixel is None:
                m.subpixel = dict(axis=axis, k=ks, cells={}, geom={})
            elif m.subpixel['axis'] != axis:
                raise ValueError(
                    "cylinder %d: mixed cylinder axes in one model "
                    "are not supported in subpixel v1" % k)
            span = [slice(None)]*3
            span[axis] = slice(a0, a1)
            for i1, i2 in zip(*np.nonzero(fill >= 1e-3)):
                pos = [None]*3
                pos[axis] = slice(a0, a1)
                pos[t1], pos[t2] = int(i1), int(i2)
                pos = tuple(pos)
                if np.any(m.sigma[pos] != 0.0):
                    raise ValueError(
                        "cylinder %d overlaps existing conductor "
                        "cells -- subpixel v1 keeps primitives "
                        "disjoint (transverse cell %d,%d)"
                        % (k, i1, i2))
                m.sigma[pos] = np.float32(sig*fill[i1, i2])
                m.fill_frac[pos] = np.float32(fill[i1, i2])
                m.subpixel['cells'][(int(i1), int(i2))] = \
                    sub[i1, i2].astype(np.float64)
                # stage C needs the resolved surface: each cell
                # remembers its cylinder's (center, R, sigma)
                m.subpixel['geom'][(int(i1), int(i2))] = \
                    (c1, c2, R, sig)
        if eps_blocks:
            cplx = any(np.iscomplexobj(ec) for _, _, ec in eps_blocks)
            eps = np.ones(m.dims,
                          dtype=np.complex128 if cplx else np.float64)
            m.epsilon_dispersion = disp_blocks
            for lo, hi, ec in eps_blocks:
                sl = np.s_[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
                # dielectric applies to the block's NON-conductor
                # cells only: a conductor's charge lives on its
                # surface panels, not in a polarization branch
                eps[sl] = np.where(m.sigma[sl] == 0.0, ec, eps[sl])
            m.epsilon = eps
        if self.ports_faces:
            plist = []
            for pname, pfaces, nfaces in self.ports_faces:
                p = voxmodel.Port(pname)
                for term, faces in (('P', pfaces), ('N', nfaces)):
                    for (ix, iy, iz, ax, sg) in faces:
                        if not all(0 <= c < d for c, d
                                   in zip((ix, iy, iz), m.dims)):
                            raise ValueError(
                                "port %r face cell (%d,%d,%d) outside "
                                "the grid %s"
                                % (pname, ix, iy, iz, m.dims))
                        if m.sigma[ix, iy, iz] == 0.0 and not (
                                m.lambdaL is not None
                                and m.lambdaL[ix, iy, iz] != 0.0):
                            # a pure London cell conducts through the
                            # superfluid (sigma == 0, lambda_l > 0)
                            raise ValueError(
                                "port %r face cell (%d,%d,%d) is not "
                                "a conductor -- the injection port "
                                "lives on conductor (or "
                                "superconductor) faces"
                                % (pname, ix, iy, iz))
                        p._add(term, (ix, iy, iz, ax, sg))
                p._freeze()
                plist.append(p)
            m.ports = plist
        m.freq = np.array(self.freqs if self.freqs else [0.0])
        m.film_normal = film_normal
        return m

    @staticmethod
    def _parse_faces(entries):
        """[[ix, iy, iz, "+z"], ...] -> [(ix, iy, iz, axis, sign)]."""
        if entries is None:
            return None
        out = []
        for e in entries:
            if len(e) != 4 or str(e[3]) not in _FACE:
                raise ValueError(
                    "port face %r: expected [ix, iy, iz, face] with "
                    "face one of %s" % (e, sorted(_FACE)))
            ax, sg = _FACE[str(e[3])]
            out.append((int(e[0]), int(e[1]), int(e[2]), ax, sg))
        return out

    @staticmethod
    def _cover(lo_m, hi_m, pitch, n):
        """Per-cell coverage of ``[lo_m, hi_m]`` on one axis, in [0, 1].

        Exact 1-D clip: cell ``k`` spans ``[k*p, (k+1)*p]``, so its
        coverage is the overlap over the pitch. Whole cells give 1.0 and
        the two ends give the fraction the staircase would otherwise
        round away.
        """
        edges = np.arange(n + 1)*float(pitch)
        ov = np.minimum(edges[1:], float(hi_m)) - np.maximum(edges[:-1],
                                                             float(lo_m))
        return np.clip(ov/float(pitch), 0.0, 1.0)

    @staticmethod
    def _snap(coords, pitch, name):
        """Physical coordinates -> cell boundaries, with a warning when
        the snap moves a face by more than 5% of a cell (a refinement
        ladder should use dimensions commensurate with every pitch)."""
        out = []
        for k, v in enumerate(coords):
            c = float(v)/float(pitch[k])
            r = int(round(c))
            if abs(c - r) > 0.05:
                import warnings
                warnings.warn(
                    "block %r: coordinate %g m snapped to cell "
                    "boundary %d (moved %.0f%% of a cell at pitch "
                    "%g)" % (name, v, r, 100*abs(c - r), pitch[k]),
                    RuntimeWarning, stacklevel=3)
            out.append(r)
        return out

    def block_cells(self, m):
        """``[(name, lo, hi)]`` cell ranges of the declared blocks.

        The VoxelModel keeps only a painted ``sigma`` grid -- block
        identity does not survive ``model()``, because nothing in the
        solve needs it. Visualisation does: a 40 x 50 mm ground plane
        and a 3 mm die are one undifferentiated shell otherwise, and
        the plane hides the module. This re-derives the ranges from
        the same document and the same ``_snap`` the painting used,
        so the labels cannot drift from the geometry.
        """
        out = []
        for k, b in enumerate(self._doc.get('block', [])):
            if ('from_m' in b) or ('to_m' in b):
                lo = self._snap(b['from_m'], m.d, b.get('name', '?'))
                hi = self._snap(b['to_m'], m.d, b.get('name', '?'))
            else:
                lo = [int(v) for v in b['from']]
                hi = [int(v) for v in b['to']]
            out.append((str(b.get('name', 'block%d' % k)),
                        tuple(int(v) for v in lo),
                        tuple(int(v) for v in hi)))
        return out

    def _box_cells(self, box, m):
        """Occupied cells whose centres lie inside a physical box."""
        lo = np.asarray(box[0], dtype=float)
        hi = np.asarray(box[1], dtype=float)
        occ = np.argwhere(m.struc() > 0)
        ctr = (occ + 0.5)*m.d[None, :]
        sel = np.all((ctr >= lo[None, :]) & (ctr <= hi[None, :]),
                     axis=1)
        cells = [tuple(int(v) for v in c) for c in occ[sel]]
        if not cells:
            raise ValueError("port box %s..%s contains no occupied "
                             "cell centres" % (list(lo), list(hi)))
        return cells

    def wires(self, freq, seg_cap=None):
        """Wire objects with skin shapes retuned to ``freq``.

        ``seg_cap`` bounds the segment length in addition to any
        per-wire ``max_seglen``: the coupler requires segments no
        longer than a leaf box, and leaf boxes SHRINK under
        refinement, so the cap is a per-rung property the solver
        derives from the tree -- wire discretisation refines with the
        mesh, by design."""
        from wireassembly import Wire, spline_points
        out = []
        for w in self.wire_specs:
            sigma = float(w['sigma'])
            delta = (np.sqrt(1.0/(np.pi*freq*MU0*sigma))
                     if freq and freq > 0 else None)
            ms = float(w['max_seglen']) if 'max_seglen' in w else None
            if seg_cap is not None:
                ms = seg_cap if ms is None else min(ms, seg_cap)
            pts = np.asarray(w['points'], dtype=float)
            if str(w.get('shape', 'polyline')) == 'spline':
                # sample the curve to chords the coupler will accept:
                # curvature-adaptive, and never longer than the leaf
                # box cap the tree just handed us
                pts = spline_points(
                    pts, w['start_vec'], w['end_vec'],
                    float(w['radius']), max_seglen=ms,
                    sagitta=float(w.get('sagitta', 0.1)))
            out.append(Wire(pts,
                            float(w['radius']), sigma,
                            nring=int(w.get('nring', 3)),
                            nsect=tuple(w.get('nsect', (4, 8, 12))),
                            delta=delta, max_seglen=ms))
        return out

    def foot_r0(self):
        """Per-wire foot radius, or None for the 2x-radius default."""
        r = [w.get('foot_r0') for w in self.wire_specs]
        if all(v is None for v in r):
            return None
        return np.array([(2.0*float(w['radius']) if v is None
                          else float(v))
                         for v, w in zip(r, self.wire_specs)])

    def tree(self, m):
        """Formulation-appropriate tree for this problem's model.

        LpPR: capacitive tree, LEAN when multilevel (fftnear near
        field + no stored n2n -- the pdn_sweep recipe, 33 -> 0.2 GB
        on the 320^2 board). LpR with wires: the coupler's near/far
        split needs a real box partition (single-level e/f/g leaves
        disagree on it by construction), so single-level partitions
        bump to the shipped example's shape rule -- INTERIM POLICY
        pending a wire-aware partition() or a [solve] override.
        """
        leaf, levels = m.partition()
        if self.formulation == 'LpPR':
            if levels > 1:
                return m.build_tree(leaf, levels, capacitive=True,
                                    fftnear=True, keep_n2n=False)
            return m.build_tree(leaf, levels, capacitive=True)
        if levels == 1 and self.wire_specs:
            leaf, levels = [min(int(d), 5) for d in m.dims], 2
        return m.build_tree(leaf, levels)

    def sweeper(self, m, M, verbose=False):
        """The uniform per-frequency solve interface for both
        formulations: ``solve(freq) -> (Z, info)`` with ``info['i_f']``
        the filament currents (for field export). LpR constructs a
        WireBondSolver per point (skin shapes follow the frequency);
        LpPR sets up once and warm-starts each point from the last
        (the pdn_sweep pattern)."""
        if self.formulation == 'LpPR':
            return _LpPRSweep(self, m, M, verbose)
        if self.equipotential:
            return _EquiSweep(self, m, M, verbose)
        return _LpRSweep(self, m, M, verbose)

    def solver(self, M, freq, model=None, **kw):
        """A WireBondSolver for one frequency (wires retuned)."""
        from wireassembly import WireBondSolver
        model = model if model is not None else self.model()
        kw.setdefault('foot_model', self.foot_model)
        kw.setdefault('basis', self.solver_basis)
        kw.setdefault('amg_cycles', self.amg_cycles)
        kw.setdefault('gram_solver', self.gram_solver)
        pp, pn = self.port_p, self.port_n
        if self.port_p_box is not None:
            pp = self._box_cells(self.port_p_box, model)
        if self.port_n_box is not None:
            pn = self._box_cells(self.port_n_box, model)
        # segment cap from THIS tree's leaf boxes (they shrink under
        # refinement; 0.95 keeps a margin off the coupler's hard limit)
        if M.numlevels > 1:
            n = np.asarray(M.e.n, dtype=float)
            l = np.asarray(M.e.l, dtype=float)
            seg_cap = 0.95*float((n*l).min())
        else:
            seg_cap = None
        return WireBondSolver(model, M, self.wires(freq, seg_cap),
                              pp, pn, foot_r0=self.foot_r0(), **kw)


def _status_meta(prob, m, M, **params):
    """Publish run identity + RESOLVED solver parameters to the status
    API (sppeec_status; no-op unless enabled). Lives here, not in the
    CLI, so library users -- a study driving a sweeper directly under
    SPPEEC_STATUS=... -- report the same metadata."""
    if not _status.enabled():
        return
    model = dict(
        name=os.path.splitext(os.path.basename(prob.path))[0],
        input=prob.path, formulation=prob.formulation)
    try:
        # two cell counts, both true: OCCUPIED is what the unknowns
        # scale with (filaments/loops live only on metal); LATTICE is
        # the bounding grid -- the count if the whole envelope were
        # metal -- which sizes the FFT/Toeplitz tables and is what the
        # 567 B/cell memory law and the rung names (R4 = 51M, R5 =
        # 200M, R6 = 1e9) refer to. `cells` stays as a legacy alias
        # for occupied (schema 1 additivity).
        occ = int(np.asarray(m.struc()).sum())
        lat = int(np.prod(np.asarray(m.dims, dtype=np.int64)))
        model.update(dims=[int(v) for v in m.dims],
                     cells=occ, cells_occupied=occ, cells_lattice=lat,
                     fill_pct=float(m.fill()),
                     nports=len(getattr(m, 'ports', []) or []),
                     tree_levels=int(getattr(M, 'numlevels', 0)))
    except Exception:                 # metadata must never break setup
        pass
    _status.run_meta(model=model,
                     params=dict(method=prob.method, rtol=prob.rtol,
                                 current=prob.current,
                                 basis=prob.solver_basis, **params))
    _status.sweep_meta(prob.freqs)


def _status_result(freq, Z, info):
    """Record one completed frequency point (scalar Z only -- the
    multi-port matrix keeps counters but no single R/L)."""
    if not _status.enabled():
        return
    if np.ndim(Z) != 0:
        _status.record_result(freq, matvecs=info.get('matvecs'),
                              time_s=info.get('time'))
        return
    w = 2*np.pi*float(freq)
    _status.record_result(
        freq, R=np.real(Z), imZ=np.imag(Z),
        L=(abs(np.imag(Z))/w if w > 0 else None),
        matvecs=info.get('matvecs'),
        residual=info.get('residual', info.get('true_residual')),
        time_s=info.get('time'))


class _LpRSweep:
    """Per-frequency WireBondSolver behind the sweeper interface."""
    formulation = 'LpR'

    def __init__(self, prob, m, M, verbose=False):
        self.prob, self.m, self.M = prob, m, M
        self.verbose = verbose
        self.sol = None          # last WireBondSolver (wire export)
        _status_meta(prob, m, M, wires=True)

    def solve(self, freq):
        with _status.freq_task(freq):
            # the wire path rebuilds its solver per point (skin shapes
            # follow the frequency), so the build is solve-time work
            # worth showing, not setup
            with _status.task('build wire solver'):
                self.m.prepare(self.M, freq)
                self.sol = self.prob.solver(self.M, freq, model=self.m,
                                            verbose=self.verbose)
            Z, info = self.sol.solve(freq, current=self.prob.current,
                                     rtol=self.prob.rtol,
                                     **_maxiter_kw(self.prob))
        _status_result(freq, Z, info)
        return Z, info


def _maxiter_kw(prob):
    """``{'maxiter': n}`` when the file sets one, else ``{}`` so the
    solver's own default stays in force (and the gate bit-identical)."""
    return {} if prob.maxiter is None else {'maxiter': prob.maxiter}


class _EquiSweep:
    """One EquiTerminalSolver across the sweep (its solve() reruns
    prepare/set_frequency per point, so setup is reused)."""
    formulation = 'LpR'

    @staticmethod
    def _skin_unsupported(m):
        """Why auto skin subdivision cannot engage on this model, or
        None. Mirrors the engine's own loud guards -- auto degrades
        gracefully where mode = "on" would raise."""
        import numpy as np
        d = np.asarray(m.d, dtype=float)
        if not np.allclose(d, d[0]):
            return 'anisotropic cells'
        if getattr(m, 'superconductor', False):
            # AUTO stays off here on purpose. The conduction palette can
            # now serve a uniform-lambda superconductor (it takes the
            # London rate 1/lambda directly, since 2026-08-29), but
            # engaging it by DEFAULT would change every existing
            # superconductor solve, so it is opt-in: an explicit
            # skin = { mode = "on", basis = "conduction" } passes this
            # reason by and lets the engine's own guards decide.
            #
            # The reason this used to give -- "the two-fluid z(w)
            # already carries the current profile" -- was only true
            # where the mesh resolves it. At two cells across a film
            # z(w) delivers the BULK kinetic value and nothing else:
            # studies/london_crowding.py measures 0.9999 of bulk at two
            # cells against 1.44 at twelve.
            return ('superconductor -- sub-cell London modes are '
                    'opt-in; set skin = { mode = "on", basis = '
                    '"conduction" }')
        if getattr(m, 'subpixel', None) is not None:
            # fill models have mixed sigma_eff by construction, but the
            # surface-anchored SubpixelModes engine handles exactly
            # that -- supported, not degraded
            return None
        try:
            m.uniform_sigma()
        except ValueError:
            return 'mixed conductivities'
        return None

    def __init__(self, prob, m, M, verbose=False):
        from equiterminal import EquiTerminalSolver
        if getattr(m, 'fill_frac', None) is not None:
            # equipotential terminals on subpixel models need every
            # port face on a FULL cell: partial rim cells carry
            # distinct sigma_eff values, and the terminal machinery
            # wants one port conductivity (and full terminal
            # cross-sections)
            bad = []
            for pname, pf, nf in prob.ports_faces:
                for (ix, iy, iz, ax, sg) in pf + nf:
                    if m.fill_frac[ix, iy, iz] < 1.0:
                        bad.append((ix, iy, iz))
            if bad:
                raise ValueError(
                    "equipotential port faces must sit on FULL cells "
                    "of a subpixel model (fill == 1) -- move these "
                    "off the partial rim: %s" % bad[:6])
        m.prepare(M, prob.freqs[0] if prob.freqs else 1e6)
        kw = dict(amg_cycles=prob.amg_cycles,
                  gram_solver=prob.gram_solver,
                  basis=prob.solver_basis,
                  nsolves=max(1, len(prob.freqs)))
        # 'auto' is size- and topology-evaluated in EquiTerminalSolver
        # too (2026-08-26): the exact Cholesky where the plaquettes
        # span and the model is small, the over-complete frame where
        # they do not (moated planes) or it is not -- doctrine 6b
        # sub-cell skin engine: conduction by default, engaged only
        # when the cell size justifies it (the engine's
        # recommend_subdivision at the sweep's highest frequency)
        sk = prob.skin
        if sk['mode'] == 'off':
            self.skin_kwargs = dict(subdivide=False)
        else:
            sub = sk['k'] if sk['k'] else 'auto'
            why = self._skin_unsupported(m)
            if why is not None:
                if sk['mode'] == 'on' or sk['k']:
                    # explicit request: let the engine's guards speak
                    pass
                else:
                    if verbose:
                        print('  skin: off (%s)' % why, flush=True)
                    sub = False
            elif sub == 'auto' and sk['basis'] == 'conduction':
                # conduction's k is pure quadrature (mode count is
                # fixed), and the engine's own guidance is k ~ 6-8 to
                # resolve the exponentials -- the generic
                # recommend_subdivision cap of 3 exists for bases
                # whose DOF grow with k and undersells conduction by
                # ~19 points of gap (measured on equibar at
                # dx/delta = 4.8: k=3 57.6%, k=7 76.2%). Engagement
                # is still the engine's own justification rule.
                from equiterminal import recommend_subdivision, \
                    skin_depth
                fref = (float(sk['f_ref']) if sk['f_ref'] is not None
                        else max(prob.freqs) if prob.freqs else 0.0)
                spx = getattr(m, 'subpixel', None)
                sig0 = (next(iter(spx['geom'].values()))[3] if spx
                        else m.uniform_sigma())
                if recommend_subdivision(m.dx, sig0, fref) > 1:
                    # RESOLUTION-AWARE k (2026-08-20, palette_ablation
                    # + xsection_tabulated): once the shape family is
                    # not the binder, the piecewise-constant sub-bar
                    # grid is -- pick k so a sub-bar is <= delta/2.
                    # Measured: dx/delta = 6 needs k = 12 (+4
                    # delivered points over k = 7 at unchanged
                    # matvecs; km, not k, drives solve cost) while
                    # dx/delta = 3 is already resolved at 7. Floor 7
                    # keeps the previous default; cap 12 is the
                    # measured point and bounds the k^2 setup growth.
                    delta = skin_depth(sig0, fref)
                    sub = int(min(12, max(7, np.ceil(2*m.dx/delta))))
                else:
                    sub = False
            rcu, rcc = sk['rc_uu'], sk['rc_cross']
            if rcu is None or rcc is None:
                # width-scaled automatic radii (see _auto_rc); the
                # port axis is the mode-carrying axis, the width is
                # transverse to it. Explicit values pass through.
                # SCOPE: coarse-engine (FFT-path) models only --
                # subpixel models run SubpixelModes' SPARSE path,
                # whose cost grows as (2rc+1)^3 and whose Kelvin
                # bands are validated at (3,4); they keep the small
                # radii unless set explicitly.
                if getattr(m, 'film_normal', None) is not None:
                    # declared thin films: Stage 0 (studies/
                    # london_film.py, 2026-09-01) measured the film's
                    # mode dipoles as coherently ALIGNED, so recovery
                    # keeps climbing with the coupling radius (68.7 ->
                    # 83.0% over rc (3,4) -> (12,16) at k = 7) and falls
                    # with film extent at small rc. The film palette's
                    # 1-D split makes these radii affordable (tables
                    # cost rc^3 * kz, not rc^3 * kz^2-squared).
                    au, ac = 12, 16
                elif getattr(m, 'subpixel', None) is None:
                    pax = int(prob.ports_faces[0][1][0][3])
                    au, ac = _auto_rc(m.struc(), pax)
                else:
                    au, ac = 3, 4
                rcu = au if rcu is None else int(rcu)
                rcc = ac if rcc is None else int(rcc)
            self.skin_kwargs = dict(
                subdivide=sub, mode_basis=sk['basis'],
                rc_uu=rcu, rc_cross=rcc,
                boundary_only=sk['boundary_only'])
            if sk['f_ref'] is not None:
                self.skin_kwargs['skin_freq'] = float(sk['f_ref'])
        kw.update(self.skin_kwargs)
        self.prob = prob
        self.S = EquiTerminalSolver(m, M, 0, verbose=verbose, **kw)
        if verbose and self.skin_kwargs.get('subdivide') is not False:
            print('  skin: %s' % (
                'k=%d, basis=%s, f_ref %.3g Hz'
                % (self.S.skin_k, sk['basis'], self.S.skin_freq)
                if self.S.skin_k > 1 else
                'off (mesh resolves the skin depth)'), flush=True)
        self.sol = None          # no wire solver on this path
        self.efg = self.S.efg
        _status_meta(prob, m, M, skin=dict(self.skin_kwargs))

    def solve(self, freq):
        with _status.freq_task(freq):
            Z, i, info = self.S.solve(float(freq),
                                      current=self.prob.current,
                                      rtol=self.prob.rtol,
                                      method=self.prob.method,
                                      **_maxiter_kw(self.prob))
        info['i_f'] = np.asarray(i[:self.efg])
        _status_result(freq, Z, info)
        return Z, info


class _LpPRSweep:
    """One LpPRSolver, frequency-tracked and warm-started.

    Multi-port models return the full open-circuit Z matrix per
    frequency (one drive per port; ``info`` aggregates the drives:
    matvecs summed, true_residual the worst, ``i_f`` the PORT-0-driven
    currents for field export). Warm starting is single-port only for
    now -- each drive would need its own chain.
    """
    formulation = 'LpPR'

    def __init__(self, prob, m, M, verbose=False):
        from port_impedance import LpPRSolver
        m.prepare(M, prob.freqs[0] if prob.freqs else 1e6)
        self.prob = prob
        self.verbose = verbose
        self.S = LpPRSolver(m, M)
        self.nports = len(m.ports)
        self.sol = None          # no wire solver on this path
        self._x = self._f = None
        _status_meta(prob, m, M)

    def solve(self, freq):
        if self.nports > 1:
            with _status.freq_task(freq):
                Z, infos = self.S.impedance_matrix(
                    float(freq), current=self.prob.current,
                    tol=self.prob.rtol, keep_drive=0,
                    verbose=self.verbose)
            info = dict(
                matvecs=sum(i['matvecs'] for i in infos),
                flag=max(i['flag'] for i in infos),
                residual=max(i['residual'] for i in infos),
                true_residual=max(i['true_residual'] for i in infos),
                i_f=infos[0]['i_f'])
            _status_result(freq, Z, info)
            return Z, info
        with _status.freq_task(freq):
            z, x, info = self.S.solve(float(freq),
                                      current=self.prob.current,
                                      tol=self.prob.rtol, x0=self._x,
                                      x0_freq=self._f,
                                      verbose=self.verbose)
        self._x, self._f = x, float(freq)
        info['i_f'] = np.asarray(x[:self.S.S.efgsize])
        _status_result(freq, z, info)
        return z, info


def load(path):
    """Parse an SuperPEEC input file."""
    with open(path, 'rb') as fh:
        doc = tomllib.load(fh)
    return Problem(doc, path=path)


def loads(text):
    """Parse from a string (tests, generated inputs)."""
    return Problem(tomllib.loads(text))
