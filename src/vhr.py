# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""VoxHenry ``.vhr`` reader and SuperPEEC adapter.

The ``.vhr`` format (VoxHenry, Yucel/White MIT, as extended by
FastFieldSolvers) describes a voxelised inductance-extraction problem:
a uniform cubic voxel grid, a per-voxel conductivity, a frequency list
and one or more current ports declared as sets of voxel *faces*. That
maps almost directly onto SuperPEEC's own inputs -- ``Tree`` takes exactly
such an occupancy grid -- so these files give us a corpus of externally
authored geometries plus reference results to check against.

Grammar (mirrors ``VoxHenry/src_pre_process/pre_input_file.m``; keywords
are case-insensitive, and ``=`` and ``,`` are token separators just like
whitespace, so ``dx=1e-6`` and ``dx 1e-6`` are the same line)::

    * ... comment (any line whose first token is not a keyword)
    freq= f1 f2 ...          may repeat; all values are appended
    dx=<metres>              voxel side (cubic voxels only)
    LMN=<L>,<M>,<N>          grid dimensions in voxels
    Superconductor           if present, voxel lines carry a 5th field
    StartVoxelList
    V <ix> <iy> <iz> <sigma> [<lambdaL>]      indices are 1-BASED
    EndVoxelList
    N <port> <P|N|other> <ix> <iy> <iz> <side>    side in +-x/+-y/+-z

A ``P`` node is a positive (excitation) terminal, ``N`` a negative one,
and any other tag marks a grounded node. ``<side>`` names which face of
voxel ``(ix, iy, iz)`` carries the terminal.

THREE THINGS THIS ADAPTER HANDLES THAT A NAIVE ONE GETS WRONG:

1. CELL-PITCH PINNING, PER AXIS. ``Tree`` pads the grid up to
   ``ntotalfull``, a multiple of the leaf size, and divides the *top
   box* by that padded count -- so passing ``ltop = dims*dx`` silently
   gives cells SMALLER than ``dx`` (measured elsewhere in this repo:
   down to 0.75x). That is a different physical problem and it
   masquerades as discretisation error. Worse for this corpus, the
   padding ratio differs BETWEEN AXES on a non-cubic grid (60x20x20 at
   leaf 5 pads to 65x25x25: 1.083 on x but 1.25 on y and z), so the
   single scalar rescale used elsewhere in this repo -- correct for the
   cubic test geometries it was written against, and every VoxHenry
   model is non-cubic -- would leave anisotropic cells. That failure is
   quiet: it shows only as the x-directed filament resistance drifting
   from the y- and z-directed ones. :meth:`VhrModel.build_tree`
   rescales each axis independently and asserts all three afterwards.

2. PORT DEDUPLICATION. ``Tree.parsesource`` ASSIGNS (``source[i] =
   value``), it does not accumulate. Adjacent port faces share corner
   nodes, so feeding it raw per-face contributions would silently drop
   all but the last one at every shared node. :meth:`VhrModel.port_nodes`
   sums duplicates before returning.

3. IN-PLACE MUTATION. ``parsesource`` does ``value[i] *= beta`` on the
   caller's array for node-orientation sources. :meth:`source_vector`
   passes a copy so the returned excitation stays reusable.

Limitations, all detected and reported rather than silently ignored:

* mixed conductivity and superconductors need the CELL scheme: R is
  diagonal there (per-filament arrays broadcast through ``traverseRL``),
  folded into the translation-invariant Toeplitz diagonal.
  :meth:`uniform_sigma` still raises on mixed files -- callers that
  need ONE sigma (skin-depth estimates) have no meaningful answer there.
* superconductors (``Superconductor`` files, :attr:`superconductor`)
  carry VoxHenry's two-fluid London model via
  :meth:`impedance_density`: ``r`` becomes COMPLEX and frequency
  dependent (kinetic inductance in its imaginary part), so
  :meth:`resistances` needs ``freq > 0`` -- the superfluid shorts DC.
* grounded nodes are parsed into :attr:`grounds` but not applied; the
  inductive formulation fixes node potential only up to a constant.

Usage::

    import vhr
    m = vhr.read_vhr('VoxHenry/Input_files/wire_len2.0u_dia4.0u.vhr')
    print(m.summary())
    M = m.build_tree(nleaf=[4, 4, 4], numlevels=2, nmax=4)
    src = m.source_vector(M, port=0, current=1e-6)

Command line::

    python3 vhr.py <file.vhr> [<file.vhr> ...]      summarise
    python3 vhr.py --build <file.vhr>               summarise and build
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
import stencils as st


# The model itself lives in voxmodel.py (format-neutral); this module
# is the .vhr FORMAT layer. Names below are re-exported so every
# existing `import vhr` caller keeps working unchanged.
from voxmodel import (MU0, VoxelModel, VhrModel, Port, allocate,  # noqa: F401
                      filament_cells)

# side name -> (axis, sign); sign +1 selects the face at the voxel's
# upper node index on that axis, -1 the lower one.
SIDES = {'-x': (0, -1), '+x': (0, 1),
         '-y': (1, -1), '+y': (1, 1),
         '-z': (2, -1), '+z': (2, 1)}

# token separators, per pre_input_file.m
_SEPS = ' \f\n\r\t\v=,'


def _split(line):
    """Tokenise one input line the way ``strsplit`` does in VoxHenry."""
    out = []
    tok = ''
    for ch in line:
        if ch in _SEPS:
            if tok:
                out.append(tok)
                tok = ''
        else:
            tok += ch
    if tok:
        out.append(tok)
    return out


def write_vhr(path, struc, dx, sigma=5.8e7, freq=(1e9,), ports=(),
              comment=None, lambdaL=None):
    """Write a voxel structure out as a VoxHenry ``.vhr`` file.

    Parameters
    ----------
    path : str
    struc : (L, M, N) array
        Nonzero entries are conductor voxels.
    dx : float
        PHYSICAL cell pitch in metres. Note this is NOT ``LT/NT`` for a
        tree built by :func:`main.build_tree`: ``mp.Tree`` pads the grid
        up to ``ntotalfull``, so the pitch is ``LT/ntotalfull``. Pass the
        pitch actually realised by the tree (``M.e.l``), or the file will
        describe a different physical object than the code solved.
    sigma : float or array
        Conductivity, scalar (uniform) or per-voxel.
    freq, ports : sequence
        Frequencies and ``(name, tag, ix, iy, iz, face)`` port entries,
        both 0-based in ix/iy/iz; written 1-based as the format requires.
    lambdaL : float or array, optional
        London penetration depth; writing it makes the file a
        ``Superconductor`` model with the 5-field voxel lines (VoxHenry
        convention: ``lambdaL = 0`` on a voxel means normal metal).

    Round-trips through :func:`read_vhr` to the same occupancy and dx.
    """
    struc = np.asarray(struc)
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), struc.shape)
    lam = None
    if lambdaL is not None:
        lam = np.broadcast_to(np.asarray(lambdaL, dtype=float), struc.shape)
    with open(path, 'w') as fh:
        if comment:
            for ln in str(comment).splitlines():
                fh.write("* %s\n" % ln)
        if freq:
            fh.write("freq " + " ".join("%.10g" % f for f in freq) + "\n")
        fh.write("dx %.12g\n" % dx)
        fh.write("LMN %d %d %d\n" % tuple(struc.shape))
        if lam is not None:
            fh.write("Superconductor\n")
        # StartVoxelList/EndVoxelList are REQUIRED by VoxHenry's own
        # pre_input_file.m: it only consumes V lines between them. Our
        # read_vhr is lenient and accepts V lines anywhere, so omitting
        # these still round-trips through this module while VoxHenry
        # silently reads ZERO voxels (nnz(sigma_e) == 0) and reports no
        # error. Do not drop them.
        fh.write("StartVoxelList\n")
        nz = np.nonzero(struc)
        for ix, iy, iz in zip(*nz):
            if lam is None:
                fh.write("V %d %d %d %.10g\n"
                         % (ix + 1, iy + 1, iz + 1, sig[ix, iy, iz]))
            else:
                fh.write("V %d %d %d %.10g %.10g\n"
                         % (ix + 1, iy + 1, iz + 1, sig[ix, iy, iz],
                            lam[ix, iy, iz]))
        fh.write("EndVoxelList\n")
        for (nm, tag, ix, iy, iz, face) in ports:
            fh.write("N %s %s %d %d %d %s\n"
                     % (nm, tag, ix + 1, iy + 1, iz + 1, face))
    return path


def read_vhr(path):
    """Parse a VoxHenry ``.vhr`` file.

    Parameters
    ----------
    path : str

    Returns
    -------
    VhrModel

    Raises
    ------
    ValueError
        On a missing ``LMN`` header, an out-of-range voxel index, a
        malformed port line or an unknown face name.
    """
    m = VhrModel(path)
    freq = []
    ports = {}
    order = []
    grounds = []
    nvox = 0
    with open(path, 'r') as fh:
        for lineno, line in enumerate(fh, 1):
            tok = _split(line)
            if not tok:
                continue
            key = tok[0].lower()
            if key == 'freq':
                freq.extend(float(t) for t in tok[1:])
            elif key == 'dx':
                if len(tok) >= 2:
                    m.dx = float(tok[1])
            elif key == 'lmn':
                if len(tok) >= 4:
                    m.dims = (int(tok[1]), int(tok[2]), int(tok[3]))
                    # f64 ON PURPOSE (unlike sppeec_input/pypeec_io):
                    # validate_vhr asserts sum(sigma) EXACTLY against
                    # VoxHenry's Octave dump -- parse fidelity, not memory,
                    # is this adapter's contract
                    m.sigma = np.zeros(m.dims, dtype=np.float64)
            elif key == 'superconductor':
                m.superconductor = True
            elif key == 'v':
                if m.sigma is None:
                    raise ValueError(
                        "%s:%d: voxel line before LMN header" % (path, lineno))
                if m.superconductor and m.lambdaL is None:
                    m.lambdaL = np.zeros(m.dims, dtype=np.float64)
                if len(tok) < 5:
                    raise ValueError("%s:%d: short voxel line: %r"
                                     % (path, lineno, line.strip()))
                ix = int(tok[1]) - 1
                iy = int(tok[2]) - 1
                iz = int(tok[3]) - 1
                if not (0 <= ix < m.dims[0] and 0 <= iy < m.dims[1]
                        and 0 <= iz < m.dims[2]):
                    raise ValueError(
                        "%s:%d: voxel (%d,%d,%d) outside LMN grid %s"
                        % (path, lineno, ix+1, iy+1, iz+1, m.dims))
                m.sigma[ix, iy, iz] = float(tok[4])
                if m.superconductor and len(tok) >= 6:
                    m.lambdaL[ix, iy, iz] = float(tok[5])
                nvox += 1
            elif key == 'n':
                if len(tok) < 7:
                    continue          # matches VoxHenry: short N lines ignored
                pname, tag = tok[1], tok[2].upper()
                side = tok[6].lower()
                if side not in SIDES:
                    raise ValueError("%s:%d: unknown face %r (expected %s)"
                                     % (path, lineno, tok[6],
                                        '/'.join(sorted(SIDES))))
                axis, sign = SIDES[side]
                entry = (int(tok[3]) - 1, int(tok[4]) - 1, int(tok[5]) - 1,
                         axis, sign)
                if tag in ('P', 'N'):
                    if pname not in ports:
                        ports[pname] = Port(pname)
                        order.append(pname)
                    ports[pname]._add(tag, entry)
                else:
                    grounds.append(entry)
    if m.dims is None:
        raise ValueError("%s: no LMN header found" % path)
    if nvox == 0:
        raise ValueError("%s: no voxels found" % path)
    for pname in order:
        ports[pname]._freeze()
        m.ports.append(ports[pname])
    m.grounds = np.array(grounds, dtype=int).reshape(-1, 5)
    m.freq = np.unique(np.array(freq, dtype=float)) if freq \
        else np.array([1.0])
    if not freq:
        print("%s: no frequencies specified, defaulting to 1 Hz" % m.name)
    return m


def _main(argv):
    build = '--build' in argv
    paths = [a for a in argv if not a.startswith('--')]
    if not paths:
        print(__doc__.split('Command line::')[-1].strip())
        return 1
    for path in paths:
        m = read_vhr(path)
        print(m.summary())
        if build:
            leaf, levels = m.partition()
            try:
                M = m.build_tree(leaf, levels)
            except ValueError as exc:
                print("  build        skipped: %s" % exc)
                print()
                continue
            m.prepare(M, m.freq[-1])
            snx, sny, snz, val = m.port_nodes(0, 1e-6)
            src = m.source_vector(M, 0, 1e-6)
            print("  built        leaf %s, %d levels, pitch %g m (dx %g)"
                  % ('x'.join(str(v) for v in leaf), levels,
                     float(M.e.l[0]), m.dx))
            print("  unknowns     %d e + %d f + %d g + %d nodes"
                  % (np.size(M.e.struc), np.size(M.f.struc),
                     np.size(M.g.struc), np.size(M.lv[0].struc)))
            print("  resistance   e %.4g  f %.4g  g %.4g ohm/filament"
                  % (M.e.r, M.f.r, M.g.r))
            print("  port 0       %d nodes, sum %.3g (should be ~0), "
                  "%d nonzero in source vector"
                  % (val.size, abs(val.sum()), int(np.count_nonzero(src))))
        print()
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(_main(sys.argv[1:]))
