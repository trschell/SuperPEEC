# R4 memory census — every item above 1% of peak

Measured 2026-08-26 on `examples/dbc_halfbridge_r4.toml` (wire path,
640×800×100 lattice = 51.2M cells, 4.62M occupied, 12.92M filaments,
12.00M mesh unknowns), CPU-only (`SPPEEC_GPU=0`), current defaults
(overcomplete basis, GeoMG with the reordered stencil engaged, int8
hierarchy, fp32 Krylov at rtol 1e-4). Three instruments, all
reproducible:

* **resident attribution**: `memcensus.census()` over the built
  solver (`{sol, M, m}` roots), grouped by owner;
* **phase timeline**: `VmRSS`/`ru_maxrss` sampled at phase
  boundaries, plus monkeypatched high-water markers inside the
  solver build;
* **transients**: high-water deltas across the marked sub-phases.

Headline numbers (before → after the 2026-08-26/27 transient fixes;
"after" is the shipped state):

    peak (HWM, build phase)  11.73 -> 10.46 -> 10.13 -> 9.45 GB  <- what OOM-kills
    post-build resident (RSS)        7.97 ->  6.79 GB
    census-reachable arrays          6.40 GB
    unattributed resident      1.5 -> ~0.6 GB  (FFTW plan buffers x18,
                                    cholmod factor, allocator overhead;
                                    the other ~0.9 GB was the
                                    node_of_cell python dict, now a
                                    75 MB CellIndex)

The peak sits **in the build, not the solve**: model 0.41 GB → tree
0.99 (HWM 1.41) → solver build 7.97 resident with the 11.73 HWM
inside it. Solve-phase additions (Krylov basis) fit under that peak.
1% of peak = 117 MB; every resident item above it is documented
below.

> **CORRECTION (2026-08-28): the sentence above is wrong, and this
> census structurally could not see why.** It walks the BUILT solver,
> before any matvec. The FMM leaf gather buffers are allocated
> LAZILY on first use (`levels.py`, `_ynmr_g` / `_mfil_g`:
> `np.ascontiguousarray(self.ynmr[self.idx, :])`, a contiguous
> gather that makes the P2M/L2P products BLAS-friendly), so at census
> time they do not exist yet. Measured on this file, CPU-only, via
> the CLI's own build path:
>
>     tree + prepare                rss  0.97   hwm  1.38 GB
>     solver build DONE             rss  6.60   hwm  9.38 GB   <- census sees this
>     after ONE matvec              rss 17.05   hwm 18.15 GB   <- true peak
>
> The build reproduces this census (9.38 vs 9.45 GB HWM). The first
> matvec then adds **10.45 GB resident**, all of it census-reachable
> once it exists — six complex128 buffers, one pair per leaf
> direction, 25 harmonics per filament:
>
>     M.e._ynmr_g / _mfil_g   1840.9 MB each
>     M.f._ynmr_g / _mfil_g   1840.4 MB each
>     M.g._ynmr_g / _mfil_g   1487.3 MB each   = 10.09 GB
>
> So the honest R4 peak is **~18 GB CPU-only, not 9.45**, and a
> figure taken from this document alone understates it about 2x.
> Full-run peaks measured the same day, `/usr/bin/time -v`:
> >=21.0 GB (CPU-only CLI, no export), 29.3 GB (GPU + `--export-glb`;
> the export is +3.97 GB of that, and is the non-streaming
> `current_density` in `blendout`), 30.6 GB (same, spline wires).
>
> **DONE (2026-08-29), fp32 campaign phase 4.** The six buffers are
> now stored complex64 with the magnitude factored into an fp64
> scalar, and built chunkwise straight into the single-precision
> output (`levels._gather_single`; `SPPEEC_LEAF_FP64=1` restores
> fp64). Measured on this file, CPU-only:
>
>     leaf gather buffers   10.34 -> 5.17 GB
>     peak after one matvec 18.16 -> 13.34 GB   (-4.81 GB, -27%)
>     steady matvec         11.84 -> 11.58 s    (unchanged)
>
> A plain `.astype(complex64)` would have been WRONG: these tables
> carry `r**n` in SI metres and `ynmr` also `m0 = mu0/(4 pi) l**2`,
> leaving the smallest non-zero entry only 7.6 decades above float32's
> smallest normal at R4, 4.6 at R6, and UNDERFLOWING at a 1 um pitch —
> silently zeroing the high-order harmonics. Normalising removes the
> scale dependence entirely. Building the fp64 gather first would
> also have handed most of the saving back in transients (measured
> 3.8x the final buffer before the chunking was sized by elements
> rather than rows). Gated in `validation/validate_leaf_fp32.py`;
> R3 reproduces the fp64 answer to every printed digit (R 0.00504333,
> L 2.0089e-08, 143 matvecs, identical wire shares).
>
> **Re-census after a matvec, not before.**

## Resident items (largest first)

### 1. Wire coupler blocks and tables — 2574 MB (27% of the
shipped 9.45 GB peak, 22% of the 11.73 GB peak this census
started from, 32% of resident)

`sol.wc` (`wirecoupler.WireCoupler`): complex64 2266 MB, float32
142 MB, int64 92 MB, sparse 70 MB.

* **Purpose**: everything wire-related in the operator — the 27-box
  near blocks (C and Lww), the attachment-free far-field tables
  (M2P read / P2L inject), and the cached *point-resolved* kernels
  that let `set_frequency()` retune wire skin shapes by reweighting
  instead of requadraturing (seconds instead of minutes per point).
* **Lifecycle**: built during the wire-solver constructor (the
  sub-phase that raises HWM to 3.6 GB); read on **every matvec**
  (`pre_l2p` far-field inject, near-block multiplies) and at every
  frequency retune; never freed.
* **Precision**: complex64 dominant (deliberate — the tables are
  coupling coefficients, fp32-adequate by the wirekernel gates).
* **Attribution (done 2026-08-26, and it closes the big idea)**:
  the 2.27 GB of complex64 is `wc._far` at 2.36 GB — the per-leaf
  far-field tables read on **every matvec**. The retune-only kernel
  caches (`_kern_ww` etc.) measure only ~130 MB. A release knob for
  the cache is therefore worth ~1% of peak, not 20% — deprioritised.
  Any real reduction here means shrinking `_far` itself (per-leaf
  list-of-arrays layout, dedup/packing) — unscoped.

### 2. GeoMG preconditioner (hierarchy + stencil + Schur) — 1666 MB (14%)

`sol.chol` closure: float32 765 MB, int64 549 MB, sparse 352 MB.

* **Purpose**: the v-cycle. Float32 = the stencil tile images (three
  164.6 MB working buffers + the tiled `wdi`, 3511 tiles × 3 normals
  × 16³) plus dinv vectors; int64 = the stencil pack maps
  (`tile_of` + 4 `loc` arrays + a closure duplicate, 6 × 91.5 MB);
  sparse = coarse levels (int8 data), prolongators and their CSR
  transposes. Level-0 CSR (once 720 MB) is **dropped** — the
  reordered stencil replaced it.
* **Lifecycle**: built at solver construction (GeoMG init + stencil
  certification probes); applied once per matvec (the `_precond`
  call); never freed.
* **Opportunity (open, easy)**: the pack maps are five int64 arrays
  used only as fancy-index vectors — a single precomputed flat
  `intp` index replaces all five: **−450 MB at R4 (~−2 GB at R5)**
  and a faster pack. Recorded as the next GeoMG morsel.

### 3. Stacked basis `Bmat` — 597 MB (5.1%)

* **Purpose**: the mesh basis [Y | S | T] (plaquettes, wire
  distribution, sharing chords) mapping mesh unknowns to filament +
  wire currents; applied twice per matvec (`Bmat @ x`, `Bmatᵀ @ v`).
* **Lifecycle**: assembled mid-build (`sp.bmat`, with a COO
  transient of its own size); never freed.
* **Precision**: float64 — and it **stays** float64: the
  `shrink_exact_f32` pass refuses it because the S/T blocks carry
  wire quadrature weights that don't store exactly in fp32 (the Y
  block alone is ±1s). **Opportunity (open, medium)**: split the
  apply into a f32 Y-block and a small f64 S/T block, ~−300 MB at
  R4 — only worth bundling with other wire-path work.

### 4. FMM transfer/Toeplitz tables — 333 MB (2.8%)

`M.pz.*`: complex64 182 MB (the top-level `ftrans`), int tables,
39 MB complex128.

* **Purpose**: the far-field translation operators (top and mid
  levels) — the heart of `apply_Z`.
* **Lifecycle**: built at tree/prepare time; read on every matvec;
  never freed. (The mid-level fp32 variant `SPPEEC_MID_FP32=1`
  exists for the biggest tables; default off.)
* **Opportunity**: none worth taking at this share.

### 5. Solution buffer `whole` — 268 MB (2.3%)

* **Purpose**: the tree-resident filament data buffer `traverseRL`
  reads and writes each matvec; also the field the exporter reads.
* **Lifecycle**: allocated at `prepare()`; live for the run.
* **Precision**: complex128, and should remain so — it accumulates
  the FMM sweep, and halving it would put fp32 in the one place the
  field's precision is actually set. **Not recommended** to shrink.

### 6. Filament incidence `B` — 264 MB (2.3%)

* **Purpose**: filament→cell incidence (KCL structure, potential
  recovery).
* **Lifecycle**: built early in the solver constructor; read in
  assembly and post-processing; never freed.
* **Precision**: **float32 since the shrink pass** (±1 entries store
  exactly; every product bit-unchanged). Was 340 MB as float64.

### 7. Lattice conductivity `m.sigma` — 195 MB (1.7%)

* **Purpose**: per-cell σ; feeds `resistances()` at every
  `prepare()` (each frequency on the wire path) and derives
  `struc()` for the exporter.
* **Lifecycle**: created at `model()`; consumed at prepare/export;
  never freed automatically. `VoxelModel.release_lattice_arrays()`
  (explicit opt-in) frees it — legitimate only after the final
  prepare on a run that will not export. Note this lowers
  solve-phase residency, **not** the build-phase peak.

### 8. Filament geometry maps — 148 + 49 MB (1.7% together)

`sol.fil_cell` (int32, 12.9M×3) and `fil_axis`: filament→cell/axis
lookup used by the terminal machinery, plaquette geometry, stencil
build and exports. Built once, kept. Already int32; no opportunity.

### 9. Tree leaf/node arrays — 119 MB (1.0%)

`M.e/f/g` index structures (mostly int64 box indices; the `r`
resistance entries are scalars at uniform σ). Kept for the run; no
opportunity at this share.

Below 1%: `ihat_f` (99 MB, the cached topological particular
solution), the spanning-forest arrays (4 × 35 MB), adjacency tables
(2 × 49 MB), `psign`.

## Solve-phase additions (fit under the build peak)

* **Krylov basis**: scipy lgmres allocates ~(inner_m 10 + augmentation
  + workspace) ≈ 16–19 vectors × 12.0M unknowns × 8 B (complex64 at
  rtol ≥ 1e-5) ≈ **1.5–1.8 GB**, freed when the solve returns.
  *Computed, not measured — the bounded-solve measurement run was
  interrupted; the formula is the one the lgmres(10) decision table
  in `krylov_solve` was built from.*
* **Checkpointing** (`SPPEEC_CHECKPOINT`): one transient
  iterate+rhs copy (~190 MB here) per dump.
* Per-solve temporaries: `Baugᵀ` complex cast for the lsqr
  projections, rhs/ihat vectors — a few hundred MB, freed per solve.

## Build transients — corrected attribution, and the fixes

The census's first draft blamed `getmesh_full`; finer pre/post
brackets acquitted it (its Fortran buffers are int32/int8, ~0.5 GB,
and its whole transient fits under the peak already set). The real
owners, with the shipped fixes:

| sub-phase | transient (before) | after | fix |
|---|---:|---:|---|
| **`_forest` spanning-forest build** | **+4.0 GB** | ~+1.1 GB | the defaultdict of ~26M python tuples replaced by sorted adjacency arrays + a list-based DFS with the identical traversal order — **bit-identical forest**, 6 s wall |
| **`sparse_incidence`** | +2.6 GB | ~+1.7 GB | direct CSR assembly (2 entries/row ⇒ no COO sort; int32 indices, f32 ±1 data = the canonical stored form; per-row column swap keeps the old summation order — bit-identical) |
| `node_of_cell` dict | 0.9 GB *resident* | 75 MB | `CellIndex` (sorted-key searchsorted, dict-compatible `[]` + vectorised `lookup`) |
| Gram/hierarchy formation (`_GeoMGFactor`) | +1.3 GB | +1.3 GB | eager `del G`/`del A` shipped earlier; the remaining live-set is the Galerkin products — open |
| stencil extraction | +0.6 GB | +0.6 GB | already trimmed (500k-row sample) |
| `getmesh_full`, `sp.bmat`, coupler quadrature | fit under peak | — | innocent / no action |

**Tier 3 (2026-08-27) closed the plateau's three faces at once**, and
the investigation exposed a fourth hidden one:

* level-0 Gram **never materialised** — its stencil is extracted from
  a thin sampled product and certified matrix-free
  (`basis @ (basisᵀ x)` probes);
* level 1 assembled by **colour probing** (81 exact-integer probe
  vectors through the certified stencil; entries verified EQUAL to
  the SpGEMM's, ~150 MB transient where the `W1 = Yᵀ P` route
  measured a peak-neutral 1.8 GB — the first tier-3 attempt that
  taught us replacement ≠ reduction);
* the flat-intp pack maps (−0.36 GB resident);
* `_aggregate`'s `np.unique(axis=0)` — **~1.5 GB of void-record sort
  copies on every hierarchy build, the shared transient every
  Gram-formation variant kept hitting** (three different level-1
  constructions all peaked at ~10.13 GB before this was found) —
  replaced by an order-preserving scalar-key unique, bit-identical.

Post-tier-3 peak: **9.45 GB** (resident 6.60), the factor phase
+1.1 GB (probe assembly, P0 construction, Schur columns). The
trade-off bought with RAM: the probing assembly costs ~55 s more
build time at R4 than the SpGEMM it replaces (~+4 min at R5,
~+20 min at R6); batching the 81 probes into multi-vector stencil
applies is the recorded follow-up if that matters. Tier 3 is
CPU-path only (`SPPEEC_GPU=0`) until the GPU stencil (stage B)
uploads tiles instead of the level-0 csr.

## Scaling notes

Resident items scale ~linearly in occupied cells (×3.9 R4→R5,
measured 83–85 GB solve-resident at R5) except `sigma`/lattice
arrays (×3.9 in lattice) and the wire coupler (scales with wire
count × tree depth, not cells). The build-peak overhang
(getmesh + transients) measured ~11% over solve-resident at R5.

## Opportunity register (status at 2026-08-26)

| action | scope | saving @R4 | status |
|---|---|---:|---|
| `_forest` array/list rewrite | peak | ~2.9 GB | **done** (bit-identical) |
| `sparse_incidence` direct CSR | peak | ~0.9 GB | **done** (bit-identical) |
| `node_of_cell` → `CellIndex` | resident | ~0.9 GB | **done** |
| wc retune-cache release | resident | ~130 MB (measured — not the hoped 2.3 GB; `_far` is hot) | deprioritised |
| ~~chunk `getmesh_full`~~ | — | — | **withdrawn** (acquitted by measurement) |
| shrink `_far` layout (dedup/pack per-leaf lists) | resident | unscoped | open |
| blockwise Gram + astype copy=False | peak | 0.33 GB measured (hoped ~1.5: the SpGEMM's own internals and the level-1 Galerkin products immediately claim the lowered ceiling) | **done** — csr/wire path blockwise (bit-identical, verified); csc/equi path keeps the historical construction because a csc product stores sorted indices where a csr product does not |
| flat-intp stencil pack maps | resident | ~450 MB | open, easy |
| Bmat Y/ST block split (f32/f64) | resident | ~300 MB | open, medium |
| int8 hierarchy, level-0 stencil drop, B/Y/YT/Baug f32, fp32 Krylov, eager G frees | — | shipped | done |
| `release_lattice_arrays()` (sigma etc.) | solve resident | 195 MB | done (explicit opt-in) |
| GPU stencil (device VRAM) | VRAM | — | parked (docket; treecost trigger) |
| `whole` buffer to complex64 | resident | 134 MB | **rejected** (field precision) |

Projected to the larger rungs (these transients scale with
filaments): R5 peak ~93 → **~83–85 GB**; R6 projected peak ~490 →
**~430–445 GB** — the 624 GB hero node's margin widens accordingly.
