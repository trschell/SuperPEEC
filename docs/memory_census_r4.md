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

Headline numbers:

    peak (HWM, build phase)         11.73 GB   <- what OOM-kills
    post-build resident (RSS)        7.97 GB
    census-reachable arrays          6.40 GB
    unattributed resident            ~1.5 GB   (FFTW plan buffers x18,
                                                cholmod factor,
                                                allocator overhead)

The peak sits **in the build, not the solve**: model 0.41 GB → tree
0.99 (HWM 1.41) → solver build 7.97 resident with the 11.73 HWM
inside it. Solve-phase additions (Krylov basis) fit under that peak.
1% of peak = 117 MB; every resident item above it is documented
below.

## Resident items (largest first)

### 1. Wire coupler blocks and tables — 2574 MB (22% of peak, 32% of resident)

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
* **Opportunity (open, the largest single one)**: the retune cache
  is dead weight on a single-frequency run, but the census does not
  yet separate "cache" (retune-only) from "tables" (read per
  matvec) inside the 2.27 GB of complex64 — **attribute first,
  then add a release knob** for the retune-only share. Do not
  release blind: the far tables are hot.

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

## Build transients (the ~3.8 GB between resident 7.97 and peak 11.73)

Attribution by high-water markers, in build order:

| sub-phase | HWM after | owns |
|---|---:|---|
| wire coupler build | 3.6 GB | quadrature temporaries |
| **`getmesh_full` (plaquette enumeration)** | **10.2 GB** | **+6.6 GB — the peak owner**: the overcomplete basis's COO intermediates |
| stencil extraction + certification | ~+0.6 GB | sampled COO/key arrays (was +2.5 GB before the 500k-row trim) |
| Gram formation / factor / cholS | no further growth | freed eagerly since the `del G` pass |

**Opportunity (open, the #1 peak reduction)**: `getmesh_full`'s
+6.6 GB transient — chunked or streaming plaquette enumeration in
`meshgraph` would lower the number that actually OOM-kills runs.
This is real Fortran/scipy surgery and is recorded on the docket.

## Scaling notes

Resident items scale ~linearly in occupied cells (×3.9 R4→R5,
measured 83–85 GB solve-resident at R5) except `sigma`/lattice
arrays (×3.9 in lattice) and the wire coupler (scales with wire
count × tree depth, not cells). The build-peak overhang
(getmesh + transients) measured ~11% over solve-resident at R5.

## Opportunity register (status at 2026-08-26)

| action | scope | saving @R4 | status |
|---|---|---:|---|
| separate + release wc retune cache | resident | up to ~2.3 GB (attribute first) | **open, #1 resident** |
| chunk `getmesh_full` enumeration | peak | up to ~6 GB | **open, #1 peak** |
| flat-intp stencil pack maps | resident | ~450 MB | open, easy |
| Bmat Y/ST block split (f32/f64) | resident | ~300 MB | open, medium |
| int8 hierarchy, level-0 stencil drop, B/Y/YT/Baug f32, fp32 Krylov, eager G frees | — | shipped | done |
| `release_lattice_arrays()` (sigma etc.) | solve resident | 195 MB | done (explicit opt-in) |
| GPU stencil (device VRAM) | VRAM | — | parked (docket; treecost trigger) |
| `whole` buffer to complex64 | resident | 134 MB | **rejected** (field precision) |
