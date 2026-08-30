# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Format-neutral internal representation of a voxel PEEC problem.

Split out of vhr.py 2026-08-06: everything here is what a problem IS
-- a voxel grid with per-cell materials (conductivity, London depth,
and through :meth:`VoxelModel.impedance_density` a complex z(w)),
equipotential ports as face sets, and a frequency list -- plus the
SuperPEEC-facing plumbing (partition/build_tree/prepare/resistances/
source_vector). Nothing here knows about any FILE FORMAT: vhr.py
(the VoxHenry .vhr dialect) and pypeec_io.py (PyPEEC's mesher output
+ problem description) are front-ends that construct this model, and
new formats should do the same rather than extending a format they
merely resemble. ``VhrModel`` remains as a backward-compatible alias.
"""

# Thread defaults -- see sppeec_threads.py. The library defaults cost
# 6.3x on the FMM path (OpenBLAS spawning a dozen threads for each of
# many small gemv calls) and 1.8x even on the dense LpPR path, and the
# penalty is LARGEST on the coarse meshes the skin/corner studies use.
# The runtime call works whatever the import order; the environment
# block inside the module covers OMP and FFTW for anyone importing
# early.
try:
    import sppeec_threads as _spthreads
    _spthreads.enforce_blas()
except Exception:                    # tuning must never break a solve
    _spthreads = None

import numpy as np
import sppeec_status as _spstatus
import stencils as st

MU0 = 4e-7*np.pi


def allocate(M, reuse=True):
    """Allocate the tree's solution buffer and alias the leaf views.

    ``Tree.__init__`` sizes every element set but leaves ``data``
    unallocated (0-d); main.py's ``SystemMat.__init__`` is what creates
    the single contiguous complex buffer and points ``e``, ``f``, ``g``
    and ``lv[0]`` at consecutive slices of it. ``parsesource`` shapes its
    output from ``lv[0].data``, so it fails on a freshly built tree until
    this has run.

    This mirrors that layout exactly -- ``e | f | g | nodes`` -- so the
    returned buffer is interchangeable with ``SystemMat.wholedata`` and
    the block offsets below match ``esize``/``efsize``/``efgsize``.
    main.py is a driver script rather than an importable module, so the
    allocation is repeated here rather than shared.

    Parameters
    ----------
    M : multipole.Tree
    reuse : bool, optional
        Zero and re-alias the buffer already attached to ``M`` rather
        than making a new one. Default True, so that repeated
        :meth:`VhrModel.prepare` calls across a frequency sweep keep the
        SAME buffer -- anything holding a reference to it (a solver, a
        cached view) stays valid. Pass False to force a fresh array.

    Returns
    -------
    whole : ndarray of complex
        The contiguous buffer; ``M.e.data`` etc. are views into it.
    offsets : tuple of int
        ``(esize, efsize, efgsize, wholesize)``.
    """
    sizes = [np.size(M.e.struc), np.size(M.f.struc), np.size(M.g.struc),
             np.size(M.lv[0].struc)]
    total = int(sum(sizes))
    cur = getattr(M, '_vhr_whole', None)
    if reuse and cur is not None and cur.size == total:
        whole = cur
        whole[:] = 0
    else:
        whole = np.zeros((total,), dtype=np.complex128)
        M._vhr_whole = whole
    off = 0
    for leaf, n in zip((M.e, M.f, M.g, M.lv[0]), sizes):
        leaf.data = whole[off:off+n]
        off += n
    e, f, g, _ = sizes
    return whole, (e, e+f, e+f+g, off)


class Port:
    """One current port: a set of positive and negative voxel faces.

    Attributes
    ----------
    name : str
        Port name as written in the file.
    pos, neg : ndarray of int, shape (k, 5)
        Columns ``(ix, iy, iz, axis, sign)``; voxel indices are 0-BASED
        (converted from the file's 1-based convention), ``axis`` is
        0/1/2 for x/y/z and ``sign`` is -1 or +1.
    """

    def __init__(self, name):
        self.name = name
        self._pos = []
        self._neg = []

    def _add(self, terminal, entry):
        (self._pos if terminal == 'P' else self._neg).append(entry)

    def _freeze(self):
        self.pos = np.array(self._pos, dtype=int).reshape(-1, 5)
        self.neg = np.array(self._neg, dtype=int).reshape(-1, 5)
        del self._pos, self._neg

    def __repr__(self):
        return "<Port %r: %d P faces, %d N faces>" % (
            self.name, len(self.pos), len(self.neg))


def filament_cells(M, leaf):
    """Lower cell index ``(i,j,k)`` of every filament, in leaf order.

    Under the ``cell`` scheme a filament joins two adjacent cell CENTRES,
    so it is identified by the lower of the two; the upper is that plus
    one step along the filament's own axis. Leaf storage is
    ``(box, local index)``, so this walks the boxes and adds the box
    origin -- the same decode the far-field validators and ``vtkout`` use.

    Returns an ``(nfil, 3)`` int array.
    """
    n = np.asarray(leaf.n, dtype=int)
    lv0 = M.lv[0]
    out = np.zeros((int(np.size(leaf.idx)), 3), dtype=int)
    for g in range(np.size(leaf.idx0) - 1):
        if leaf.idx0[g+1] <= leaf.idx0[g]:
            continue
        sl = np.s_[leaf.idx0[g]:leaf.idx0[g+1]]
        idx = np.asarray(leaf.idx[sl])
        c = np.stack([idx//(n[1]*n[2]), (idx//n[2]) % n[1], idx % n[2]], 1)
        c[:, 0] += lv0.xidx[g]*lv0.n[0]
        c[:, 1] += lv0.yidx[g]*lv0.n[1]
        c[:, 2] += lv0.zidx[g]*lv0.n[2]
        out[sl] = c
    return out


class VoxelModel:
    """A parsed ``.vhr`` model plus the adapters SuperPEEC needs.

    Attributes
    ----------
    path, name : str
        Source file path and basename.
    dx : float
        Voxel side length in metres (cubic voxels).
    dims : tuple of int
        Grid dimensions ``(L, M, N)`` in voxels.
    sigma : ndarray, shape ``dims``
        Per-voxel conductivity in S/m; 0 marks an empty voxel (unless
        the model is a superconductor, where ``lambdaL`` decides).
    lambdaL : ndarray or None
        Per-voxel London penetration depth, or None for normal metal.
    freq : ndarray
        Sorted unique frequency list in Hz.
    ports : list of Port
    grounds : ndarray, shape (k, 5)
        Grounded nodes, same column layout as :attr:`Port.pos`.
    superconductor : bool
    """

    def __init__(self, path):
        self.path = path
        self.name = path.rsplit('/', 1)[-1]
        self.d = 1e-6
        self.dims = None
        self.sigma = None
        # per-cell fill fraction (subpixel program): None = all-full;
        # else 1.0 on full cells, the covered fraction on partial
        # (cylinder-boundary) cells. Resistance carries it via
        # sigma_eff = sigma*fill; Lp is full-cell until stage B.
        # (named fill_frac: .fill() is the percent-occupancy METHOD)
        self.fill_frac = None
        # subpixel stage-B data: dict(axis=, k=, cells={(t1,t2): (k,k)
        # sub-fill array}) for cylinder-boundary cells
        self.subpixel = None
        self.lambdaL = None
        self.freq = np.zeros((0,))
        # dispersive dielectric blocks: (lo, hi, eps_inf, deps, f1,
        # f2) tuples re-evaluated into self.epsilon per solve
        # frequency (see _apply_dispersion)
        self.epsilon_dispersion = []
        self.ports = []
        self.grounds = np.zeros((0, 5), dtype=int)
        self.superconductor = False
        # Homogeneous background relative permittivity (dielectric
        # support phase 1, 2026-08-07): scales the whole capacitive
        # path's coefficient of potential by 1/eps_bg at tree build.
        # Exact for a uniform dielectric filling all space; per-cell
        # dielectrics are phase 2. No file format carries this yet --
        # it is set programmatically.
        self.eps_bg = 1.0
        # Per-cell RELATIVE permittivity (dielectric phase 2): None
        # means vacuum everywhere. Cells with epsilon != 1 and sigma
        # == 0 are PURE DIELECTRICS: they join the mesh as Ruehli
        # excess-capacitance cells (polarization current == conduction
        # current with sigma_eff = j*w*eps0*(eps_r - 1), carried by
        # impedance_density), and the material-ID occupancy makes
        # panels -- hence charge -- appear at their buried interfaces
        # with conductors. Programmatic only; no file format carries
        # permittivity.
        self.epsilon = None

    # -- pitch: per-axis, with a scalar compatibility view ----------
    #
    # The FMM core is per-axis throughout and MEASURED correct on
    # anisotropic cells (2026-08-07, dense box-integral oracle:
    # ~1e-3 at 2:1 aspect with aspect-compensating leaves, 3e-4
    # cubic; element aspect floors 4:1 at ~5e-3). `d` is the truth;
    # `dx` remains for the many cubic callers and RAISES on an
    # anisotropic model rather than silently averaging.

    @property
    def d(self):
        """Per-axis voxel pitch, metres, shape (3,)."""
        return self._d

    @d.setter
    def d(self, value):
        v = np.asarray(value, dtype=float)
        self._d = np.array([float(v)]*3) if v.ndim == 0 else v.copy()
        if self._d.shape != (3,) or np.any(self._d <= 0):
            raise ValueError("pitch must be a positive scalar or "
                             "3-vector, got %r" % (value,))

    @property
    def dx(self):
        """The single cubic pitch; raises if the model is anisotropic."""
        if self._d[0] != self._d[1] or self._d[0] != self._d[2]:
            raise ValueError(
                "%s: anisotropic pitch %s has no single dx -- use .d "
                "(and note the skin engine and a few studies are still "
                "cubic-only)" % (self.name, list(self._d)))
        return float(self._d[0])

    @dx.setter
    def dx(self, value):
        self.d = float(value)

    @property
    def anisotropic(self):
        return bool(self._d[0] != self._d[1] or self._d[0] != self._d[2])

    # -- geometry ---------------------------------------------------

    def release_lattice_arrays(self):
        """Free the per-cell lattice arrays (sigma, epsilon, lambdaL).

        For a run that will do no further ``prepare()`` (no more
        frequency points) and no field export, these arrays are dead
        weight -- ~5 GB at the 1e9-cell hero rung. This is an
        EXPLICIT opt-in for callers that know their run shape (a
        single-point extraction after the solver is built); nothing
        calls it automatically, because ``struc()`` feeds the field
        exporter and ``resistances()`` every retune. Afterwards both
        raise loudly rather than compute garbage. Returns bytes
        freed. NOTE the measured RSS high-water sits in the BUILD
        phase, so this lowers solve-phase residency (co-tenancy,
        head-room for exports elsewhere), not the OOM-critical peak.
        """
        freed = 0
        for name in ('sigma', 'epsilon', 'lambdaL'):
            a = getattr(self, name, None)
            if isinstance(a, np.ndarray):
                freed += a.nbytes
                setattr(self, name, None)
        self._lattice_released = True
        return freed

    def _lattice_guard(self, what):
        if getattr(self, '_lattice_released', False):
            raise RuntimeError(
                "%s needs the per-cell lattice arrays, but "
                "release_lattice_arrays() has freed them -- do not "
                "release before the last prepare()/export" % what)

    def struc(self):
        """Occupancy grid for ``Tree``: int8, 1 = conductor voxel.

        A voxel counts as occupied if it has a nonzero conductivity, or
        -- for superconductors, where sigma may legitimately be 0 -- a
        nonzero London depth.
        """
        self._lattice_guard('struc()')
        occ = self.sigma != 0.0
        if self.lambdaL is not None:
            occ |= self.lambdaL != 0.0
        if self.epsilon is not None:
            occ |= np.asarray(self.epsilon) != 1.0
        return occ.astype(np.int8)

    def material_struc(self):
        """Material-ID occupancy for ``Tree``: 0 empty, 1 conductor,
        2 pure dielectric.

        The panel rule fires wherever the ID CHANGES, so buried
        conductor-dielectric interfaces get panels -- and through
        ``node2panel`` their nodes become charge-carrying ``external``
        nodes -- exactly where bound charge belongs. Conductor cells
        that also carry epsilon != 1 stay id 1: interfaces WITHIN the
        conductor family are not distinguished (documented limit).
        """
        ids = self.struc().astype(np.int8)
        if self.epsilon is not None:
            pure = (np.asarray(self.epsilon) != 1.0) & (self.sigma == 0.0)
            if self.lambdaL is not None:
                pure &= self.lambdaL == 0.0
            ids[pure] = 2
        return ids

    def fill(self):
        """Occupied fraction of the bounding box, as a percentage."""
        return 100.0*float(self.struc().sum())/float(np.prod(self.dims))

    def sigma_values(self):
        """Sorted distinct nonzero conductivities present, in S/m."""
        return np.unique(self.sigma[self.sigma != 0.0])

    def uniform_sigma(self):
        """The model's single conductivity, in S/m.

        Raises
        ------
        ValueError
            If the model mixes materials -- SuperPEEC folds resistance into
            a translation-invariant Toeplitz diagonal (``e/f/g.r`` is a
            scalar), so there is no place to put a second value.
        """
        vals = self.sigma_values()
        if vals.size == 0:
            raise ValueError("%s: no conductor voxels" % self.name)
        if vals.size > 1:
            raise ValueError(
                "%s: %d distinct conductivities [%s] -- SuperPEEC carries a "
                "single scalar e/f/g.r folded into the translation-"
                "invariant Toeplitz diagonal, so mixed materials are not "
                "representable"
                % (self.name, vals.size, ", ".join("%g" % v for v in vals)))
        return float(vals[0])

    def impedance_density(self, freq):
        """Per-cell series impedance density ``z(w)`` in ohm*m.

        The quantity that replaces ``1/sigma`` in every filament
        resistance. Normal cells: ``1/sigma``, real and frequency
        independent. Superconducting cells (nonzero ``lambdaL``):
        VoxHenry's two-fluid model (VoxHenry_executer.m lines 237-254)
        -- the London channel ``1/(j w mu lambda^2)`` in PARALLEL with
        the normal channel ``sigma``:

            z = (sigma*(w mu lam^2)**2 + j*w mu lam^2)
                / ((sigma*w mu lam^2)**2 + 1)

        Limits pin the formula: ``sigma = 0`` gives the pure London
        kinetic reactance ``j w mu lam^2``; ``lam -> inf`` gives
        ``1/sigma``. VoxHenry ENCODES a normal-metal cell as
        ``lambdaL == 0`` (meaning that infinite-``lam`` limit, NOT the
        formula's ``lam = 0``), and an empty cell as both zero; both
        conventions are honoured here, so ``z == 0`` only on empty
        cells. Note ``z -> 0`` as ``w -> 0`` on every superconducting
        cell -- the superfluid shorts DC -- so a superconductor model
        needs ``freq > 0`` (enforced by :meth:`resistances`).
        """
        sig = np.asarray(self.sigma, dtype=np.float64)
        if not self.superconductor and self.epsilon is None:
            z = np.zeros(sig.shape)
            nz = sig != 0.0
            z[nz] = 1.0/sig[nz]
            return z
        if self.epsilon is not None:
            if self.superconductor:
                raise NotImplementedError(
                    "%s: superconductor + dielectric in one model is "
                    "not composed yet" % self.name)
            if self.epsilon_dispersion:
                self._apply_dispersion(freq)
            # Ruehli excess-capacitance material law: polarization
            # current is conduction current with
            #   sigma_eff = sigma + j*w*eps0*(eps_r - 1),
            # so a lossy conductor, a lossy dielectric (complex eps_r)
            # and a pure dielectric are ONE formula. z -> infinity on
            # pure dielectrics as w -> 0 (they do not conduct DC);
            # resistances() guards freq > 0 when any exist.
            eps0 = 1.0/(MU0*299792458.0**2)
            w = 2*np.pi*float(freq)
            epsr = np.asarray(self.epsilon)
            sig_eff = sig + 1j*w*eps0*(epsr - 1.0)
            z = np.zeros(sig.shape, dtype=np.complex128)
            nz = sig_eff != 0.0
            z[nz] = 1.0/sig_eff[nz]
            return z
        lam = np.asarray(self.lambdaL, dtype=np.float64)
        w = 2*np.pi*float(freq)
        wml2 = w*MU0*lam*lam
        den = (sig*wml2)**2 + 1.0
        z = (sig*wml2*wml2 + 1j*wml2)/den
        nrm = (lam == 0.0) & (sig != 0.0)
        z[nrm] = 1.0/sig[nrm]
        return z

    @staticmethod
    def ds_epsilon(freq, eps_inf, deps, f1, f2):
        """Djordjevic-Sarkar wideband-Debye ``eps_r(f)``.

        ``eps_inf + deps*ln((f2 + jf)/(f1 + jf))/ln(f2/f1)`` under the
        e^{+jwt} convention (Im < 0 is loss): ``eps_inf + deps`` at
        DC, ``eps_inf`` far above ``f2``, and a NEAR-CONSTANT loss
        tangent for ``f1 << f << f2`` -- the causal (Kramers-Kronig-
        consistent) replacement for a frequency-independent loss
        tangent, which is non-causal over wide bands.
        """
        f = float(freq)
        return complex(eps_inf
                       + deps*np.log((f2 + 1j*f)/(f1 + 1j*f))
                       / np.log(f2/f1))

    def _apply_dispersion(self, freq):
        """Re-evaluate dispersive blocks into ``self.epsilon``.

        In place, per solve frequency. The MASK (a block's cells with
        ``sigma == 0``) is frequency independent and eps never
        reaches exactly 1 inside the band, so occupancy and material
        ids -- decided once at build -- stay valid.
        """
        for (lo, hi, eps_inf, deps, f1, f2) in self.epsilon_dispersion:
            sl = np.s_[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            val = self.ds_epsilon(freq, eps_inf, deps, f1, f2)
            self.epsilon[sl] = np.where(self.sigma[sl] == 0.0, val,
                                        self.epsilon[sl])

    def port_sigma(self, port=0):
        """Conductivity at a port's terminal cells.

        :func:`port_impedance.terminal_impedance` needs the sigma of the
        metal the terminal half-filaments actually sit in, which on a
        mixed-material model is NOT the (nonexistent) global one --
        ``*_Cu_Al.vhr`` has one port on copper and one on aluminium.
        Raises if a single port straddles two materials, where the
        terminal resistance would be ambiguous.
        """
        p = self.port(port)
        vals = set()
        for arr in (p.pos, p.neg):
            for e in arr:
                ix, iy, iz = int(e[0]), int(e[1]), int(e[2])
                vals.add(float(self.sigma[ix, iy, iz]))
        vals = sorted(v for v in vals if v != 0.0)
        if not vals:
            raise ValueError("%s: port %s touches no conductor"
                             % (self.name, port))
        if len(vals) > 1:
            raise ValueError(
                "%s: port %s spans %d conductivities [%s] -- the terminal "
                "half-filament resistance is ambiguous there. Use "
                "port_sigma_faces() for the per-face array instead."
                % (self.name, port, len(vals),
                   ", ".join("%g" % v for v in vals)))
        return vals[0]

    def port_sigma_faces(self, port=0):
        """Conductivity at EACH of a port's faces, in face order.

        The scalar :meth:`port_sigma` cannot describe a port whose faces
        sit in different metals -- or, once boundary cells carry
        ``sigma_eff = sigma*fill``, a port that touches a PARTIAL CELL,
        which on real layout geometry it routinely will. Each terminal
        half-filament sits in exactly one cell, so there is no ambiguity
        to resolve: the scalar was simply the wrong shape. This is the
        same move ``resistances()`` already makes for interior filaments
        on a mixed model, and the superconductor branch of
        ``Terminals.set_frequency`` already indexes ``z`` per face.

        Order matches ``Terminals.faces`` (pos entries then neg), so the
        result can be used directly as a per-face ``R``.
        """
        p = self.port(port)
        out = []
        for arr in (p.pos, p.neg):
            for e in arr:
                out.append(float(self.sigma[int(e[0]), int(e[1]),
                                            int(e[2])]))
        out = np.asarray(out, dtype=float)
        if not np.all(out > 0.0):
            raise ValueError("%s: port %s has a face on a cell with zero "
                             "conductivity" % (self.name, port))
        return out

    # -- octree tree construction -----------------------------------

    def partition(self):
        """Pick ``(nleaf, numlevels)`` for this geometry.

        Applies the fill-dependent leaf rule measured for this repo (see
        ``main.recommend_leaf``): leaf 5 above 50% fill, 8 between 5% and
        50%, 16 below -- the optimum moves strongly with occupancy
        because in dense geometry the near-field pair count grows as
        ``leaf**3`` per node, while in sparse geometry the neighbouring
        boxes are mostly empty and larger boxes simply cut box count.

        Then chooses the depth. A multilevel tree needs at least 3 leaf
        boxes per axis (``traverseP3`` returns NaN at ``ng == 2``, a
        pre-existing far-field defect), and the leaf is shrunk to try to
        reach 4 boxes per axis before giving up. If even that fails the
        model is too small to partition and this returns the single-level
        convention: ONE box spanning the whole node grid, which requires
        ``nleaf == dims + 1`` exactly.

        FLAT AND ELONGATED bounding boxes take an ANISOTROPIC escape
        instead: axes thinner than the fill-rule leaf are spanned by ONE
        box (measured exact -- the per-axis ``nmid``/``nmidlev`` level
        construction handles a flattened axis by design, giving an
        effectively 2-D FMM above the leaf), because clamping the
        whole leaf to the thinnest axis makes the bounding-box-
        proportional top-level FFT explode (circular_coil: 16x the
        matvec cost). The escape triggers only when the clamped choice
        would exceed 32 boxes on some axis, so compact geometries keep
        their historical trees bit-for-bit.

        Returns
        -------
        nleaf : ndarray of int, shape (3,)
        numlevels : int
        """
        dims = np.asarray(self.dims, dtype=int)
        ntotal = dims + 1
        fill = self.fill()/100.0
        leaf0 = 5 if fill > 0.5 else (8 if fill > 0.05 else 16)
        if self.anisotropic:
            # ASPECT-COMPENSATING per-axis leaf: multipole truncation
            # degrades over stretched boxes (measured 2026-08-07:
            # 2:1 cells at 2.5-5e-3 with cubic-count leaves, ~1e-3
            # when the leaf BOXES are made physically cubic), so pick
            # nleaf per axis with nleaf[a]*d[a] ~ leaf0*min(d). ng==2
            # is promoted to a spanned axis as in the pancake escape.
            dmin = float(self._d.min())
            nleaf = np.maximum(1, np.rint(
                leaf0*dmin/self._d).astype(int))
            nleaf = np.minimum(nleaf, ntotal)
            ng = np.ceil(ntotal/nleaf).astype(int)
            nleaf[ng == 2] = ntotal[ng == 2]
            ng = np.ceil(ntotal/nleaf).astype(int)
            fat = ng >= 3
            if not fat.any():
                return st.single_level_nleaf(dims), 1
            numlevels = 3 if int(ng[fat].min()) >= 8 else 2
            return nleaf.astype(int), numlevels
        leaf = min(leaf0, max(2, int(dims.min())//3))
        # PANCAKE/NEEDLE ESCAPE (2026-08-06). The min-axis clamp above
        # is right for compact boxes but ruinous for flat or elongated
        # ones: circular_coil (310x310x10) gets clamped to leaf 3, and
        # the top-level M2L FFT -- whose cost is BOUNDING-BOX
        # proportional, not occupancy proportional -- then covers a
        # ~104^2 top grid: 2722 vs 165 ms/matvec against leaf 12.
        # Measured 2026-08-06 (raw traverseRL, operator invariants to 9
        # digits): thin-axis-spanning and ANISOTROPIC leaves are exact
        # and fast; a spanned axis (ONE box) is fine, only ng == 2 is
        # forbidden (traverseP3's NaN defect). Trigger ONLY when the
        # clamped choice is clearly pathological, so every compact
        # geometry -- and every recorded anchor -- keeps its old tree.
        # A COLLAPSED clamp (leaf < 3 -- e.g. a 4-cell-thick plane
        # pair, pdn_planes 320x320x4, 2026-08-07) is the same pancake
        # one step further: it used to fall through to a single-level
        # tree, so judge those by the fill-rule leaf instead.
        if (int(np.ceil(ntotal/leaf).max()) >= 32 if leaf >= 3
                else int(np.ceil(ntotal/leaf0).max()) >= 32):
            nleaf = np.minimum(leaf0, ntotal)      # span thin axes
            ng = np.ceil(ntotal/nleaf).astype(int)
            nleaf[ng == 2] = ntotal[ng == 2]       # 2 boxes -> span
            ng = np.ceil(ntotal/nleaf).astype(int)
            fat = ng >= 3
            if fat.any():
                # Spanned axes (ng == 1) are fully supported at any
                # depth since the MidLevel per-axis generalisation
                # (2026-08-06): an unsplit axis rides through the mid
                # levels with nmidlev = 1 -- the 2-D FMM ladder.
                # Verified on circular_coil: [8,8,11] lv3 matches the
                # lv2 consensus to 1e-5 at 171 vs 383 ms/matvec.
                numlevels = 3 if int(ng[fat].min()) >= 8 else 2
                return nleaf.astype(int), numlevels
        while leaf >= 3 and int(np.ceil(ntotal/leaf).min()) < 4:
            leaf -= 1
        if leaf < 3 or int(np.ceil(ntotal/leaf).min()) < 3:
            return st.single_level_nleaf(dims), 1
        ngroups = int(np.ceil(ntotal/leaf).min())
        numlevels = 3 if ngroups >= 8 else 2
        return np.array([leaf]*3, dtype=int), numlevels

    def build_tree(self, nleaf=None, numlevels=None, nmax=4,
                   capacitive=False,
                   lscale=1.0, **kwargs):
        """Build an octree ``Tree`` with the cell pitch pinned to ``dx``.

        ``Tree`` sets the cell pitch to ``ltop/ntotalfull`` where
        ``ntotalfull`` is the grid padded up to a whole number of leaf
        boxes -- not ``ltop/dims``. Passing the physical extent directly
        would therefore shrink the cells. This probes the realised pitch
        on a minimal-occupancy tree of the same shape (``l`` depends only
        on shape, ``nleaf``, ``numlevels`` and ``ltop``, so one occupied
        voxel is enough and the probe is cheap), rescales ``ltop``
        accordingly, and asserts the result.

        Parameters
        ----------
        nleaf : array_like of int, optional
            Cells per leaf box per axis. For ``numlevels == 1`` this must
            be ``dims + 1`` -- the single-level tree is one box holding
            the entire node grid. Defaults to :meth:`partition`.
        numlevels : int, optional
            Defaults to :meth:`partition`.
        nmax : int or None
            Multipole order; ignored when ``numlevels == 1`` (``Tree``
            forces it to None there).
        capacitive : bool, optional
            Build the panel/potential machinery. Default False (the LpR
            inductance-only case, which is what VoxHenry itself solves).
        lscale : float, optional
        **kwargs
            Forwarded to ``Tree`` (``circulant``, ``fftnear``).

        Returns
        -------
        multipole.Tree
        """
        import multipole as mp
        # Mixed conductivity and superconductors ARE supported (see
        # resistances / impedance_density); only the empty case is fatal
        # here. Occupancy comes from struc(), which counts lambdaL-only
        # cells -- a lossless superconductor legitimately has sigma == 0
        # everywhere.
        fullstruc = self.material_struc()
        if not bool(fullstruc.any()):
            raise ValueError("%s: no conductor voxels" % self.name)
        if nleaf is None or numlevels is None:
            auto_leaf, auto_levels = self.partition()
            nleaf = auto_leaf if nleaf is None else nleaf
            numlevels = auto_levels if numlevels is None else numlevels
        nleaf = np.asarray(nleaf, dtype=int)
        LT0 = np.asarray(self.dims, dtype=float)*self.d
        if numlevels == 1:
            want = st.single_level_nleaf(self.dims)
            if not np.array_equal(nleaf, want):
                raise ValueError(
                    "%s: a single-level tree is one box spanning the whole "
                    "element lattice, so nleaf must be %s, not %s"
                    % (self.name, list(want), list(nleaf)))
            # single level takes lf = ltop/nt directly; no padding
            with _spstatus.task('build tree'):
                return mp.Tree(fullstruc, nleaf, LT0, 1, lscale, nmax,
                               capacitive=capacitive, eps_r=self.eps_bg,
                               **kwargs)
        with _spstatus.task('build tree'):
            probe_struc = np.zeros(self.dims, dtype=np.int8)
            probe_struc[0, 0, 0] = 1
            probe = mp.Tree(probe_struc, nleaf, LT0, numlevels, lscale,
                            nmax, capacitive=False)
            # PER-AXIS rescale. The padding ntotalfull/ntotal differs
            # from axis to axis whenever the grid is not cubic (60x20x20
            # with leaf 5 pads to 65x25x25: 1.083 on x but 1.25 on y and
            # z), so the single scalar factor used elsewhere in this
            # repo -- correct for the cubic test geometries it was
            # written for -- would leave anisotropic cells here.
            # Anisotropy is not loudly wrong: it shows up only as the
            # x-directed filament resistance drifting from the y- and
            # z-directed ones.
            fac = self.d/np.asarray(probe.e.l, dtype=float)
            del probe
            M = mp.Tree(fullstruc, nleaf, LT0*fac, numlevels, lscale,
                        nmax, capacitive=capacitive, eps_r=self.eps_bg,
                        **kwargs)
        got = np.asarray(M.e.l, dtype=float)
        if np.any(np.abs(got/self.d - 1.0) > 1e-9):
            raise AssertionError(
                "cell pitch pinning failed: wanted %s m per axis, "
                "got %s (ratios %s)"
                % (list(self.d), list(got), list(got/self.d)))
        return M

    def prepare(self, M, freq, alpha=0.0, rscale=1.0):
        """Set the frequency-dependent scalars SuperPEEC needs to solve.

        Does everything main.py's preamble does between building the
        tree and forming the system: allocates the solution buffer
        (:func:`allocate`), assigns the filament resistances from this
        model's conductivity, sets ``jomega``, and runs ``RDFinit`` --
        which is what publishes ``lv[0].beta``, the incidence scaling
        every ``node2*`` operator and ``parsesource`` read. A freshly
        built ``Tree`` has neither ``data`` nor ``beta``, so call this
        before :meth:`source_vector`.

        Parameters
        ----------
        M : multipole.Tree
        freq : float
            Frequency in Hz (not rad/s).
        alpha : float, optional
            RDF continuity selector; 0 matches main.py.
        rscale : float, optional
            Resistance scaling divisor, as in main.py.

        Returns
        -------
        whole : ndarray of complex
            The solution buffer, as :func:`allocate` returns it.
        offsets : tuple of int
            ``(esize, efsize, efgsize, wholesize)``.
        """
        with _spstatus.task('prepare'):
            whole, offsets = allocate(M)
            re, rf, rg = self.resistances(M, freq)
            M.e.r = re/rscale
            M.f.r = rf/rscale
            M.g.r = rg/rscale
            M.jomega = 1j*2*np.pi*freq
            M.alpha = alpha
            M.RDFinit()
            return whole, offsets

    def resistances(self, M, freq=None):
        """Per-orientation filament resistance for this model's material.

        Returns ``(re, rf, rg)`` to be assigned to ``M.e.r``, ``M.f.r``
        and ``M.g.r``. ``e`` is y-directed, ``f`` x-directed and ``g``
        z-directed, matching main.py.

        ``freq`` matters only for superconductors, whose series
        impedance density is complex and frequency dependent
        (:meth:`impedance_density`); the returned "resistances" are
        then COMPLEX, carrying the kinetic inductance in their
        imaginary part. R stays diagonal either way, so nothing in the
        Toeplitz/FMM structure is touched -- the same argument that
        admitted per-cell conductivity.
        """
        self._lattice_guard('resistances()')
        if self.superconductor or self.epsilon is not None:
            return self._complex_resistances(M, freq)
        vals = self.sigma_values()
        if vals.size == 1:
            # Uniform: keep the scalar. An array of one repeated value is
            # bit-identical through traverseRL (verified), but the scalar
            # costs nothing and leaves every existing result untouched.
            sigma = float(vals[0])
            re = M.e.l[1]/(M.e.l[0]*M.e.l[2]*sigma)
            rf = M.f.l[0]/(M.f.l[1]*M.f.l[2]*sigma)
            rg = M.g.l[2]/(M.g.l[0]*M.g.l[1]*sigma)
            return re, rf, rg
        # Mixed: R is DIAGONAL and never enters the Toeplitz/FMM structure
        # (that carries Lp, which is geometry only), so a per-filament
        # array is all that is needed -- traverseRL applies the resistive
        # drop as `leaf.r*leaf.data`, which broadcasts.
        #
        # A filament joins two adjacent cell CENTRES, so it spans half of
        # each and its resistance is that series pair:
        #     (l_d/2)/(sigma_A*A_d) + (l_d/2)/(sigma_B*A_d).
        sig = np.asarray(self.sigma, dtype=np.float64)
        l = [float(v) for v in np.asarray(M.e.l, dtype=float)]
        area = (l[1]*l[2], l[0]*l[2], l[0]*l[1])       # perp to x, y, z
        out = []
        for leaf, axis in ((M.e, 1), (M.f, 0), (M.g, 2)):
            c = filament_cells(M, leaf)
            up = c.copy()
            up[:, axis] += 1
            sa = sig[c[:, 0], c[:, 1], c[:, 2]]
            sb = sig[up[:, 0], up[:, 1], up[:, 2]]
            if not (np.all(sa > 0) and np.all(sb > 0)):
                raise RuntimeError(
                    "%s: a filament spans a cell with zero conductivity -- "
                    "the occupancy stencil and the sigma array disagree"
                    % self.name)
            out.append(0.5*l[axis]/area[axis]*(1.0/sa + 1.0/sb))
        return out[0], out[1], out[2]

    def _complex_resistances(self, M, freq):
        """Complex filament impedance densities folded into ``r``.

        Same series-half-pair rule as the mixed-conductivity branch of
        :meth:`resistances`, with ``z(w)`` in place of ``1/sigma``.
        Serves superconductors (kinetic inductance) AND dielectrics
        (excess capacitance) -- both are just material laws inside
        :meth:`impedance_density`. A UNIFORM material keeps the
        scalar, for the same reason the uniform-sigma branch does.
        """
        if freq is None or not freq > 0.0:
            raise ValueError(
                "%s: complex material laws need freq > 0. A superfluid "
                "channel SHORTS DC (operator singular) and a pure "
                "dielectric OPENS at DC (branch impedance infinite) -- "
                "neither solve is meaningful at f = 0." % self.name)
        z = self.impedance_density(freq)
        occ = self.struc().astype(bool)
        l = [float(v) for v in np.asarray(M.e.l, dtype=float)]
        area = (l[1]*l[2], l[0]*l[2], l[0]*l[1])       # perp to x, y, z
        vals = np.unique(z[occ])
        if vals.size == 1:
            zz = complex(vals[0])
            re = l[1]*zz/area[1]
            rf = l[0]*zz/area[0]
            rg = l[2]*zz/area[2]
            return re, rf, rg
        out = []
        for leaf, axis in ((M.e, 1), (M.f, 0), (M.g, 2)):
            c = filament_cells(M, leaf)
            up = c.copy()
            up[:, axis] += 1
            za = z[c[:, 0], c[:, 1], c[:, 2]]
            zb = z[up[:, 0], up[:, 1], up[:, 2]]
            if np.any(za == 0.0) or np.any(zb == 0.0):
                raise RuntimeError(
                    "%s: a filament spans a cell with zero impedance "
                    "density -- the occupancy stencil and the "
                    "sigma/lambdaL arrays disagree" % self.name)
            out.append(0.5*l[axis]/area[axis]*(za + zb))
        return out[0], out[1], out[2]

    # -- port excitation --------------------------------------------

    def _face_nodes(self, entry):
        """The octree node coordinates a voxel face's current enters at.

One node per cell, at its centre. The current crossing a
        face enters the cell behind that face, so the face maps to that
        ONE node -- the voxel itself. Note this puts the port half a
        cell inside the conductor; closing that gap is what the
        terminal filaments of ``terminal.py`` are for, and until they
        are wired the model is ``dx/2`` short at each port face.
        """
        ix, iy, iz, axis, sign = entry
        return [(int(ix), int(iy), int(iz))]

    def port_nodes(self, port=0, current=1.0, weight='corner'):
        """Node coordinates and injected currents for one port.

        Each declared port face contributes its share of the terminal
        current to the four octree nodes at its corners -- the consistent
        lumping of a uniform face current density onto a bilinear node
        set. Contributions at shared nodes are SUMMED here, because
        ``Tree.parsesource`` assigns rather than accumulates and would
        otherwise keep only the last write at every shared node.

        Parameters
        ----------
        port : int or str
            Port index or name.
        current : complex
            Total port current; ``+current`` is injected across the P
            terminal and ``-current`` extracted across the N terminal.
        weight : {'corner', 'uniform', 'rim'}
            ``'corner'`` (default) splits each face's current equally
            over its four corner nodes, which weights the interior of a
            terminal face above its rim -- the area-consistent choice.
            ``'uniform'`` instead spreads the terminal current equally
            over the distinct nodes it touches.

            ``'rim'`` weights each face by ``1 + (missing in-plane
            neighbours)``, so a face in the interior of a terminal block
            scores 1, an edge face 2 and a corner face 3.

            WHY 'rim' EXISTS (measured 2026-08-09). Under the CELL
            scheme ``_face_nodes`` returns exactly ONE node per face, so
            whenever the port's faces map to distinct cells 'corner' and
            'uniform' are IDENTICAL BY CONSTRUCTION -- both hand every
            face 1/N. The profile-sensitivity diagnostic built on that
            pair (:func:`port_impedance.profile_sensitivity`) therefore
            reports a spread of exactly 0.000e+00 on every cell-scheme
            model -- measured on a 9-face pdn port and a 4-face washer
            port -- which reads as "the extraction is insensitive to the
            port profile" when it actually means "the two options are
            the same vector". Since 'cell' is the default scheme and the
            only one dielectrics support, that diagnostic was vacuous
            exactly where it was needed. 'rim' is a genuinely different,
            physically motivated profile (current crowds toward a
            terminal's rim), so uniform-vs-rim BRACKETS the prescribed
            profile assumption instead of confirming it.

        Returns
        -------
        snx, sny, snz : ndarray of int
            Node coordinates on the ``(L+1, M+1, N+1)`` grid.
        val : ndarray of complex
            Injected current at each node; sums to zero over the port.
        """
        p = self.port(port)
        if weight not in ('corner', 'uniform', 'rim'):
            raise ValueError("weight must be 'corner', 'uniform' or 'rim'")
        acc = {}
        for faces, sgn in ((p.pos, 1.0), (p.neg, -1.0)):
            if len(faces) == 0:
                continue
            local = {}
            if weight == 'rim':
                # in-plane neighbours within THIS end's face set; a face
                # scores 1 + (how many of its 4 are absent), so interior
                # 1, edge 2, corner 3
                present = {(int(e[0]), int(e[1]), int(e[2])) for e in faces}
                for entry in faces:
                    cell = [int(entry[0]), int(entry[1]), int(entry[2])]
                    axis = int(entry[3])
                    others = [c for c in range(3) if c != axis]
                    miss = 0
                    for oc in others:
                        for step in (-1, 1):
                            nb = list(cell)
                            nb[oc] += step
                            if tuple(nb) not in present:
                                miss += 1
                    fw = 1.0 + miss
                    for node in self._face_nodes(entry):
                        local[node] = local.get(node, 0.0) + fw
            else:
                for entry in faces:
                    for node in self._face_nodes(entry):
                        local[node] = local.get(node, 0.0) + 1.0
            if weight == 'uniform':
                share = sgn*current/len(local)
                for node in local:
                    acc[node] = acc.get(node, 0j) + share
            else:
                tot = sum(local.values())
                for node, w in local.items():
                    acc[node] = acc.get(node, 0j) + sgn*current*w/tot
        nodes = sorted(acc)
        snx = np.array([n[0] for n in nodes], dtype=int)
        sny = np.array([n[1] for n in nodes], dtype=int)
        snz = np.array([n[2] for n in nodes], dtype=int)
        val = np.array([acc[n] for n in nodes], dtype=np.complex128)
        return snx, sny, snz, val

    def source_vector(self, M, port=0, current=1.0, weight='corner'):
        """Node excitation vector in SuperPEEC's compressed node ordering.

        Thin wrapper over ``M.parsesource(..., 'node')`` that passes a
        copy of the value array, because parsesource scales it in place
        by ``lv[0].beta``.

        Requires :meth:`prepare` to have run -- ``beta`` is published by
        ``RDFinit``, not by the ``Tree`` constructor.
        """
        if not hasattr(M.lv[0], 'beta'):
            raise RuntimeError(
                "tree has no lv[0].beta -- call VhrModel.prepare(M, freq) "
                "before source_vector() (beta is published by RDFinit)")
        snx, sny, snz, val = self.port_nodes(port, current, weight)
        return M.parsesource(snx, sny, snz, val.copy(), 'node')

    def ground_nodes(self):
        """Node coordinates of grounded (``well conductor``) nodes.

        Parsed for completeness; SuperPEEC's inductive formulation has no
        place to apply them (node potential is defined up to a constant).
        None of the input files shipped with VoxHenry declare any.
        """
        out = set()
        for entry in self.grounds:
            out.update(self._face_nodes(entry))
        return sorted(out)

    def port(self, key):
        """Look a port up by index or by name."""
        if isinstance(key, str):
            for p in self.ports:
                if p.name == key:
                    return p
            raise KeyError("%s: no port named %r (have %s)"
                           % (self.name, key, [p.name for p in self.ports]))
        return self.ports[key]

    # -- reporting --------------------------------------------------

    def summary(self):
        """Multi-line human-readable description of the model."""
        s = self.struc()
        lines = ["%s" % self.name,
                 "  grid          %d x %d x %d = %d voxels"
                 % (self.dims[0], self.dims[1], self.dims[2],
                    int(np.prod(self.dims))),
                 "  occupied      %d (%.1f%% fill)"
                 % (int(s.sum()), self.fill()),
                 "  voxel size    %s m" % (
                     "%g" % self._d[0] if not self.anisotropic
                     else list(self._d)),
                 "  extent        %g x %g x %g m"
                 % tuple(np.asarray(self.dims)*self.d)]
        vals = self.sigma_values()
        if vals.size == 1:
            lines.append("  conductivity  %g S/m" % vals[0])
        elif vals.size == 0:
            lines.append("  conductivity  none (lossless superconductor)")
        else:
            lines.append("  conductivity  %d distinct: %s S/m  "
                         "(NOT representable in SuperPEEC)"
                         % (vals.size, ", ".join("%g" % v for v in vals)))
        if self.superconductor:
            lam = self.lambdaL[self.lambdaL != 0.0]
            lines.append("  lambda_L      %g .. %g m  (superconductor -- "
                         "NOT representable in SuperPEEC)"
                         % (lam.min(), lam.max()))
        if self.freq.size == 1:
            lines.append("  frequency     %g Hz" % self.freq[0])
        else:
            lines.append("  frequencies   %d points, %g .. %g Hz"
                         % (self.freq.size, self.freq[0], self.freq[-1]))
        for i, p in enumerate(self.ports):
            lines.append("  port %d %-8s %d P faces, %d N faces"
                         % (i, p.name, len(p.pos), len(p.neg)))
        if len(self.grounds):
            lines.append("  grounded      %d faces (ignored by SuperPEEC)"
                         % len(self.grounds))
        return "\n".join(lines)

    def __repr__(self):
        return "<VhrModel %r %s, %d ports>" % (
            self.name, 'x'.join(str(d) for d in self.dims), len(self.ports))


# Backward-compatible alias: the class was born as vhr.VhrModel.
VhrModel = VoxelModel
