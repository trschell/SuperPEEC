# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""A-priori (leaf, depth) argmin -- the tree cost model, calibrated.

The recorded lever: matvec cost ~ c1*P2P + c2*M2L + c3*top-volume,
countable from the occupancy grid before building anything (docketed
since the circular_coil 16.5x incident; the depth heuristic's
"min fat ng >= 8" proxy put 51% of the DBC matvec into the top FFT).

CALIBRATION (profile_dbc_leaf.py, R2 rung, 2026-08-12, quiet box):
    config        topfft/Vtop   mid/occbox   p2p/(fil*gdim)  measured
    [3,3,8]lv2     41.6 us/box      --        0.186 us       1803 ms
    [3,3,8]lv3     34.5             32.5      0.172          1714
    [4,4,10]lv2    47.5             --        0.207          1222
    [4,4,10]lv3    ~44              45        0.198          1337
    [6,6,15]lv2    45.0             --        0.152          1033
  * top FFT is LINEAR in the top-grid volume (log absorbed): ~45
    us/top-box per matvec (all three orientations).
  * mid M2L (depth >= 3): ~110 us per occupied leaf box.
  * p2p is SUBLINEAR IN PAIRS (the index-pack/Toeplitz effect): the
    usable fit is ~0.17 us per filament per geometric-mean leaf
    dimension. Naive pair counting over-predicts big leaves badly.
  * precond: leaf-independent (~1.5 us/fil CPU, ~0.1 us/fil GPU) --
    included for the time estimate, irrelevant to the argmin.
  * iteration count rises mildly with leaf volume: mv_factor =
    (leafvol/72)^0.05 fits 300 -> 334 over 7.5x. Crude, labeled.

The OBJECTIVE is predicted time-to-solution per matvec x mv_factor.
Constants are one geometry class's calibration -- recalibrate from a
sweep row if a new class disagrees by more than ~30%.
"""
import numpy as np

C_TOP = 45e-6        # s per top-grid box per matvec (GPU-RESIDENT
#                      spectra -- every calibration row fit on the
#                      12 GB card; see the vram term below for the
#                      streamed regime)
PCIE_BW = 20e9       # B/s, effective host->device streaming rate for
#                      the per-channel ftrans pass when the top
#                      spectra exceed the VRAM budget. ESTIMATE, NOT A
#                      MEASUREMENT: the streamed path has never run on
#                      hardware where it engages (the local card fits
#                      every rung's spectra). Analytic model: one full
#                      pass over the spectra per matvec, hidden
#                      imperfectly under the MAC. Recalibrate on the
#                      first machine where the streamed path actually
#                      runs.
NMAX_CH = 81         # (2*nmax+1)^2 combined-harmonic channels at the
#                      shipped nmax = 4
C_MID = 110e-6       # s per occupied leaf box per matvec (depth >= 3)
#                      (recalibrated: 651ms/6517 boxes and 397/3200
#                       from the lv3 sweep rows; first cut used a
#                       triple-counted box number and sat 40% low)
C_P2P = 0.17e-6      # s per filament per geometric-mean leaf dim
C_PRE_GPU = 0.10e-6  # s per filament (BlockAMG apply, GPU)
C_PRE_CPU = 1.5e-6
MV_EXP = 0.05        # mv ~ (leafvol/72)^MV_EXP, crude
DEPTH3_PENALTY = 1.4  # MEASURED (2026-08-12): three lv3 configs ran
#                       1.16/1.38/1.65x their staged prediction while
#                       lv2 rows fit to ~15%; the [5,5,12]lv3 argmin
#                       pick lost the check run 248 vs 150 s against
#                       [6,6,15]lv2 at identical mv and Z. Depth 3
#                       carries per-level overheads the stage model
#                       misses; penalize it empirically rather than
#                       model it -- no measurement campaign wanted.


def counts(model):
    """Exact filament count and the occupancy grid, once."""
    occ = model.struc() > 0
    nf = 0
    for ax in range(3):
        a = occ & np.roll(occ, -1, axis=ax)
        sl = [slice(None)]*3
        sl[ax] = slice(0, -1)
        nf += int(a[tuple(sl)].sum())
    return occ, nf


def _smooth5(k):
    """Smallest 5-smooth integer >= k (mirrors levels._m2l_gpu_init)."""
    def ok(x):
        for p in (2, 3, 5):
            while x % p == 0:
                x //= p
        return x == 1
    while not ok(k):
        k += 1
    return k


def top_spectra_bytes(ng, numlevels):
    """Device bytes of the GPU top-M2L spectra for this tree shape --
    the quantity the residency decision in levels._m2l_gpu_init
    compares against free VRAM (85% budget, complex128, 5-smooth
    padded grid)."""
    ng = np.asarray(ng, dtype=int)
    if numlevels >= 3:
        ng = np.maximum(1, np.ceil(ng/2).astype(int))
    S = [(_smooth5(2*int(v) - 1) if v > 1 else 1) for v in ng]
    return NMAX_CH*int(np.prod(S))*16


def evaluate(model, nleaf, numlevels, occ=None, nfils=None, gpu=True,
             vram_gb=None):
    """Predicted per-matvec seconds and its breakdown, or None if the
    configuration is invalid.

    ``vram_gb`` -- VRAM-AWARENESS (docketed since the R4 OOM, built
    2026-08-14): when given (or when a local GPU is queryable), each
    candidate's top-spectra size is compared against the same 85%
    budget levels uses; candidates whose spectra DON'T fit carry an
    analytic streaming penalty of spectra_bytes/PCIE_BW per matvec.
    This is what flips the (leaf, depth) argmin at the scale where
    residency is lost -- e.g. a deeper tree with a smaller top can
    beat a shallow one purely by staying resident. Pass vram_gb
    EXPLICITLY to plan for target hardware from a different box
    (vram_gb=96 for an RTX PRO 6000 host, 24 for an A10). None ->
    query the local device; no device -> no penalty term (CPU top)."""
    d = np.asarray(model.d, dtype=float)
    dims = np.asarray(model.dims, dtype=int)
    ntotal = dims.copy()
    nleaf = np.asarray(nleaf, dtype=int).copy()
    ng = np.ceil(ntotal/nleaf).astype(int)
    nleaf[ng == 2] = ntotal[ng == 2]          # span, never ng == 2
    ng = np.ceil(ntotal/nleaf).astype(int)
    fat = ng >= 3
    if not fat.any():
        return None
    if numlevels >= 3 and int(ng[fat].max()) < 6:
        return None                            # nothing to coarsen
    if occ is None or nfils is None:
        occ, nfils = counts(model)
    # occupied leaf boxes (all orientations share the partition)
    cells = np.argwhere(occ)
    boxes = cells//nleaf[None, :]
    nocc = np.unique(
        (boxes[:, 0]*int(1e6) + boxes[:, 1])*int(1e6)
        + boxes[:, 2]).size
    vtop = int(np.prod(ng))
    if numlevels >= 3:
        vtop = int(np.prod(np.maximum(1, np.ceil(ng/2))))
    gdim = float(np.prod(nleaf)**(1/3))
    stream = 0.0
    resident = True
    if gpu:
        if vram_gb is None:
            vram_gb = _local_vram_gb()
        if vram_gb is not None:
            sb = top_spectra_bytes(ng, numlevels)
            resident = sb < 0.85*vram_gb*1e9
            if not resident:
                stream = sb/PCIE_BW
    t = dict(
        topfft=C_TOP*vtop + stream,
        mid=(C_MID*nocc if numlevels >= 3 else 0.0),
        p2p=C_P2P*nfils*gdim,
        precond=(C_PRE_GPU if gpu else C_PRE_CPU)*nfils,
    )
    mvfac = (float(np.prod(nleaf))/72.0)**MV_EXP
    if numlevels >= 3:
        mvfac *= DEPTH3_PENALTY
    t['permv'] = sum(t.values())
    t['objective'] = t['permv']*mvfac
    t['mvfac'] = mvfac
    t['nleaf'] = list(int(v) for v in nleaf)
    t['numlevels'] = numlevels
    t['nocc'] = nocc
    t['vtop'] = vtop
    t['resident'] = resident
    return t


def _local_vram_gb():
    """Total VRAM of the local device in GB, or None without one."""
    try:
        import cupy as cp
        _free, total = cp.cuda.runtime.memGetInfo()
        return total/1e9
    except Exception:
        return None


def recommend(model, gpu=True, box_sizes_m=None, verbose=False,
              vram_gb=None):
    """(nleaf, numlevels) minimizing predicted time-to-solution.

    Candidates: physically near-cubic boxes over a geometric ladder of
    box sizes, at depths 2 and 3. Returns (nleaf, numlevels, table).
    ``vram_gb``: see :func:`evaluate` -- pass the TARGET machine's
    VRAM to plan trees for hardware you are not running on.
    """
    d = np.asarray(model.d, dtype=float)
    occ, nfils = counts(model)
    if vram_gb is None:
        vram_gb = _local_vram_gb()
    if box_sizes_m is None:
        base = float(np.min(d))*3
        box_sizes_m = [base*1.2**k for k in range(16)]
    seen = set()
    rows = []
    for s in box_sizes_m:
        nleaf = np.maximum(1, np.round(s/d)).astype(int)
        key = tuple(nleaf)
        if key in seen:
            continue
        seen.add(key)
        for nlv in (2, 3):
            t = evaluate(model, nleaf, nlv, occ=occ, nfils=nfils,
                         gpu=gpu, vram_gb=vram_gb)
            if t is not None:
                rows.append(t)
    rows.sort(key=lambda r: r['objective'])
    if verbose:
        print("  nleaf        lv  Vtop   occbox   permv(ms) obj(ms) res")
        for r in rows[:8]:
            print("  %-12s %d  %6d %7d  %8.0f %8.0f  %s"
                  % (r['nleaf'], r['numlevels'], r['vtop'], r['nocc'],
                     1e3*r['permv'], 1e3*r['objective'],
                     'y' if r['resident'] else 'STREAM'))
    best = rows[0]
    return best['nleaf'], best['numlevels'], rows
