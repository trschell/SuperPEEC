# Enrichment unification plan

Working document for collapsing every beyond-voxel accuracy mechanism
in SuperPEEC into one framework. Started 2026-09-03 at commit 9e672ff.
Each phase lands as one commit; the ledger at the bottom is updated in
the same commit. The failure threshold is stated in the ledger and is
not negotiable: if `src/` does not shrink by at least 1200 lines by the
end of phase 5, the program has failed.

## 1. What exists today

Eight mechanisms in five code homes, all correcting sub-cell physics the
voxel basis cannot resolve. Every one of them except the wire path is
the same construction written separately.

| # | Mechanism | Home | Corrects | Enters the system as | Scope and guards |
|---|---|---|---|---|---|
| A | `Redistribution` (coarse skin engine) | `equiterminal.py` | skin and proximity in straight sections; London screening at rate 1/lambda | net-zero mode DOFs after the loop basis; blocks Ru, Zuu, Zcross, Zt; Toeplitz tables + FFT apply, or CSR | port-axis filaments only; palettes diff/linear/conduction; `boundary_only`; anisotropic OK; uniform sigma or uniform lambda |
| A' | film palette | same class, `film_kk=(1,kz)` | through-film London profile on declared films | same blocks; corner columns dropped under a 1-D split | needs `film=` on a block and an in-plane port; rc (12,16) |
| B | `SubpixelModes` | `equiterminal.py` (subclass of A) | skin on subpixel cylinders, shapes anchored to the true circle | per-cell weights, fill-weighted Ru, aggregate side = fill shares; sparse only | cubic only; cylinder axis = port axis; one sigma; `bnd_reach` ring |
| C | `CornerModes` | `cornermode.py` | current turning at 90-degree bends (AC only) | tabulated stream-function fields, in-plane sub-strips, dense per-patch blocks | opt-in, no TOML knob; cubic; LpPR not wired; both in-plane orientations |
| D | `ModeStack` | `cornermode.py` | composition of exactly A + C with the Zec cross block | duck-types the `redist` slot | refuses B; pairwise; reads engine privates |
| E | subpixel stage A, cylinder | `sppeec_input.py` | partial-cell resistance | `sigma *= fill`, `fill_frac`, `subpixel` dict | via `resistances()` per-cell branch |
| F | subpixel stage A, slab | `sppeec_input.py` | partial-cell resistance for layer stacks | `z_scale` multiplier, `slab_fill` | cut axis left at bulk; `laminate_sigma`/`sigma_axis` is a third representation with no production caller |
| G | subpixel stage B | `subpixel.py` | partial-cell mutual footprint, dL = w'Tw - u'Tu | `dL_near` (equiterminal), `dZ_near` (systemmat, LpPR) | two implementations of one identity; cylinder path unvectorised |
| H | wire path | `wirekernel.py`, `wireassembly.py`, `footcal.py` | round-wire skin (imposed shape per sector) + solved zero-sum S columns; foot constriction | own kernels and coupler | left alone by this plan |

Material laws underneath: `VoxelModel.impedance_density` (per cell) and
`equiterminal.material_response` (engine-facing, uniform only). Two
accessors for one physics.

Duplication, measured by reading:

* three sub-prism box builders (`_sub_boxes`, the corner box loop, stage
  B `table()`/`getB()`), all feeding `terminal.box_mutual_matrix` or
  `greens.box_pair_stencil_pairs`;
* three separation-keyed table caches plus `ModeStack._Lec_raw`;
* three net-zero weight constructions with pivoted-QR pruning and three
  net-zero checks at three tolerances;
* three block-Jacobi mode preconditioners;
* four representations of "this cell is partly filled" and three
  branches in `resistances()` to read them;
* three mode-placement rules and five frequency policies;
* the k rule and the engagement guards duplicated between
  `sppeec_input._EquiSweep` and `EquiTerminalSolver.__init__`, with
  `_skin_unsupported` still refusing anisotropic cells that the engine
  has accepted since 296bcdb;
* an undocumented `redist` protocol (`nmode, sel, Ru, Zuu, Zcross, Zt,
  use_fft, apply_fft, set_frequency, mode_precond, kk, nnz, ntable`);
* `Redistribution.__init__` accepts `split_axis` and never reads it.

## 2. The target: one `Enrichment` framework

One module `src/enrich.py` replaces `cornermode.py`, `subpixel.py` and
the engine half of `equiterminal.py`. Five parts.

1. **Split** (geometry). A filament of orientation `axis` is cut into
   sub-prisms by a per-axis count triple. One vectorised
   `sub_boxes(cells, axis, n3, d3)` returns (lo, hi). The transverse
   k x k split, the 1-D film split, the 1-D slab split, the in-plane
   corner split and the whole cell are all count triples.
2. **Tables** (kernel). One separation-keyed cache
   `PairTables(splitA, splitB)` of raw sub-prism mutual tables from
   `greens.box_pair_stencil_pairs`. Serves mode tables, terminal
   tables, stage B and every cross-family block.
3. **Weights** (basis). Per filament: one aggregate column (uniform or
   fill share) plus net-zero mode columns. Storage is `shared` (Toeplitz
   tables, FFT apply) or `per_cell` (CSR). Generators: conduction
   (normal or London rate), surface-anchored, corner tabulated, slab
   fill, cylinder fill. One `netzero_prune`.
4. **Fold**. Tables x weights -> Ru, Zuu, Zcross, Zt and the aggregate
   correction dL = G'TG - u'Tu. Stage B is the aggregate-aggregate block
   of the same fold. `build_fft`/`apply_fft` survive as the shared-weight
   apply path.
5. **Family and stack**. One `Enrichment` class with a documented
   protocol. The coarse engine, film palette, subpixel modes and corner
   modes are four configurations of it (split, weight generator,
   placement, frequency policy). `ModeStack` becomes n-ary with cross
   blocks from `PairTables`. Placement is one rule: a filament carries
   modes if a supported sub-prism lies within `reach` cells of the
   conductor surface (reach 0 = today's `boundary_only`). Frequency
   policy is one method: recompute (p, z); regenerate weights if the
   generator depends on p and p moved; rescale Ru if only z moved.

Supporting merges: `material_response` folds into `VoxelModel` as
per-cell (p, z); one `model.fill` array plus one `model.cut` record
replaces `fill_frac`, the scaled `sigma`, `z_scale`, `slab_fill`, the
`subpixel` dict, `sigma_axis` and `laminate_sigma`; `resistances` and
`_complex_resistances` merge into one per-filament path (the scalar
fast path goes; the repeated-value array was measured bit-identical).

Left alone by decision: the wire kernels, `footcal`, `TerminalCoupler`,
the FMM. The corner tabulation (`_tabulate`, `find_corners`) is science,
not plumbing, and keeps its own local solver.

## 3. Interface: one `enrich` table, no compatibility

Principle: geometry declares what a cell contains; `[solve]` declares
what enrichment to spend on it; one resolver turns both into an engine
configuration for the TOML path and the direct API alike.

Geometry side:

* Partial cells are always exact. Off-grid metre bounds on a block or
  cylinder produce fill fractions; there is no `[grid] subpixel` switch
  and no snapping (`_snap` goes). A cell cut on two axes is an error.
* `[[block]]` and `[[cylinder]]` keep their keys. `film = "z"` stays as
  the stiff-axis hint on the block.

Solve side, replacing `[solve] skin` entirely:

```toml
[solve]
enrich = "auto"            # "off" | "auto" | a table

[solve.enrich]
families = ["section", "corner"]   # auto = ["section"], + film palette when a block declares film
k        = 7                       # sub-prisms per split axis; auto = sub-bar <= delta/2 (or lambda/2), floor 7, cap 12
reach    = 0                       # cells from the conductor surface that carry modes; 0 = exposed face
rc       = [12, 16]                # coupling radii (mode-mode, mode-aggregate); auto = width / film / per-cell rule
f_ref    = 1e9                     # shape reference frequency; auto = sweep maximum
```

Dropped: `basis` (the `diff` and `linear` research palettes; conduction
measured complete for straight sections and six points better). This is
the one deliberate feature removal; it can be reinstated as a `shapes`
key for about fifteen lines. Dropped from the direct API: `use_fft`,
`csr_max_gb`, `split_axis`, `boundary_only`, `corner_modes`, `subdivide`,
`mode_basis`, `rc_uu`, `rc_cross`, `skin_freq`; the solver takes
`enrich=` (a resolved config) instead. The CLI gains nothing. The
converter loses `--subpixel` and always emits metre bounds. Six example
TOMLs migrate. Doctrine rules 13, 14 and 14b collapse into one rule.

## 4. Validators

A validator survives if it pins an invariant of the new framework or a
physics headline; it goes if it pins a deleted function or duplicates a
generic check.

| Validator | Verdict | Reason |
|---|---|---|
| `validate_equiterminal` A-E, H | keep | terminal and topology |
| `validate_equiterminal` F | prune | palettes' identity/span; identity moves to `validate_enrich`, palettes gone |
| `validate_equiterminal` G | move | retune-equals-fresh is generic |
| `validate_subpixel` + `validate_aniso_sigma` | merge -> `validate_partial` | both gate the fill record; `aniso_sigma` A and B die with `laminate_sigma` and the scalar fast path; C, D, F survive |
| `validate_corner` | keep, shrink | detection, DC decoupling, AC bands stay; net-zero, symmetry, reciprocity move |
| `validate_superconductor` A-D | keep | material law and London screening |
| `validate_superconductor` E | trim | the conduction-only guard no longer exists |
| `validate_aniso` D | invert | engine now accepts anisotropic cells |
| `validate_input_lppr` skin section | rewrite | gates the `enrich` resolver's rules |
| `validate_cli`, `validate_footcal`, `validate_dielectric`, `validate_port_impedance` | keep | unaffected, must stay green |
| new `validate_enrich` | add | tables vs `greens` (absorbs `zuudiag`, `zcross`), net-zero + aggregate identity, reciprocity, frequency policy, DC decoupling O(w^2), stack composition, London bar gain, film point vs the 2-D oracle |

## 5. Studies

Keep, framework instruments (re-run by phase gates): `london_film`,
`london1d`, `london_oracle2d`, `london_crowding`, `xsection`,
`xsection_tabulated`, `modebasis2d`, `corner_referee`, `mode_referee`,
`ltrace_ladder`, `palette_ablation`, `slabfill`, `hrefine`, `stripedge`
(open question, never run to a verdict).

Keep, deliverables and tools: `rsfq_gds2toml`, `rsfq_jtl`, `rsfq_xnor`,
`sfq5ee_measured`, `sfq5ee_lc`, `dbc_*`, `pdn_*`, `squid_washer`,
`module_wires`, `blender_view`, `threadtune`.

Keep, wire and dielectric evidence: `wire_bessel`, `wire_converge`,
`wire_proximity`, `bond_foot`, `bem_dielectric`, `proto_surface_nodes`.

Prune, superseded skin-story chapters: `skinconv`, `barconv2`,
`wireconv`, `shortwire`, `wirerc`, `wirefull`, `wirebnd`, `basis4wire`,
`thirdwire`, `linbnd`, `frozenstair` (keep `frozenwire`), `cmpeq`;
`zuudiag` and `zcross` after absorption into `validate_enrich`.

Prune, narrative duplicates: `skinnarr`, `skinnarr_report` (keep the
`_default` pair and `skinnarr_profile`).

Prune, done measurements: `p3point`, `p3pinned`, `memsolve`, `onefac2`,
`ordering`, `vhrsurvey`.

Triage separately (loop-basis / preconditioner cluster, 18 files):
`amg_as_precond`, `cyclebasis`, `deflate`, `gram`, `gram_amg`,
`hx_nullspace`, `inertia`, `symbasis`, `yfull`, `oc_solve`,
`overcomplete`, `geomg`, `iterlaw`, `krylov_probe`, `mvsplit`,
`holerank`, `moatcount`, `bigcoil`.

Scratch to delete and gitignore: `studies/_narr/`, `studies/*.vhr`,
`studies/*.log`. The `*_results.json` files stay (quoted in docstrings).
The exact prune list is frozen only after phases 1-5 confirm which
instruments the gates used.

## 6. Phases

| Phase | Work | Validators touched | Gate |
|---|---|---|---|
| 0 | baseline verdicts, anchors, line counts; line report in `run_baseline.sh` | none | full suite |
| 1 | `sub_boxes` + `PairTables`; stage B as a fold output | new `validate_enrich` (tables part) | tables bit-equal to old; `validate_subpixel` green |
| 2 | `Enrichment` class, shared and per-cell weights; FFT apply fed from the fold; `reach` replaces both placement rules (re-measure the Kelvin gate at reach 0) | `validate_enrich` (identity, retune, London bar, film point); `validate_equiterminal` F pruned, G moved; `validate_superconductor` E trimmed | film bench numbers unchanged |
| 3 | corner as a generator; n-ary stack | `validate_corner` shrunk; generic parts to `validate_enrich` | corner bands unchanged; corner+subpixel composes |
| 4 | fill record and material merge; one `resistances` | `validate_partial` created; `validate_aniso` D inverted | `validate_dielectric`, `validate_superconductor` green |
| 5 | `enrich` resolver and schema; snapping removed; API collapsed; converter flag removed; examples migrated; doctrine merged | `validate_input_lppr` rewritten; `validate_cli` on migrated examples | full suite |
| 6 | study prune | none | full suite |

Rules for the whole run: anchors that shift by rounding dust are
re-based only with the mechanism named in the commit; a physics gate is
never re-based; the full gate is about 40 minutes and runs alone on the
box, so one phase per session at most; feature gains that fall out
(composable corner+subpixel, vectorised cylinder stage B, per-cell
material in the engine) are recorded, not extended.

## 7. Ledger

Line counts are `wc -l` over `*.py`. Threshold: `src/` total at the end
of phase 5 must be <= 25927 (baseline minus 1200). Estimated target for
the scoped code is ~1300 lines from 3334, i.e. about -2000 on `src/`.

| Phase | commit | `src/` | `validation/` | `studies/` | scoped src files | suite | anchors |
|---|---|---|---|---|---|---|---|
| 0 baseline | 9e672ff | 27127 | 12054 | 11054 | equiterminal 3398, cornermode 615, subpixel 444, voxmodel 1179, sppeec_input 1399 | 44 pass / 0 skip / 1 fail (`validate_aniso`, stale guard; fixed, 9/9 standalone) | setup1 1.0187104968887117e-30, setup2 3.740159228711359e-30, setup3 2.34153831468213e-31 |
| 1 geometry + tables | cea4b11 | 26826 | 12294 | 11054 | equiterminal 3165, cornermode 607, enrich 385 (new), subpixel 0 (deleted), voxmodel 1179, sppeec_input 1399 | 46 pass / 0 skip / 0 fail (incl. new `validate_enrich`) | all three BIT-IDENTICAL to phase 0 |
| 2 the Enrichment class | d306965 | 26278 | 12251 | 11050 | equiterminal 1943, cornermode 607, enrich 1060, voxmodel 1179, sppeec_input 1398 | 46 pass / 0 skip / 0 fail | all three BIT-IDENTICAL to phase 0 |
| 3 corner generator + n-ary stack | 4fe8028 | 26235 | 12275 | 11050 | equiterminal 1938, cornermode 379, enrich 1250, voxmodel 1179, sppeec_input 1398 | 45 pass / 0 skip / 1 fail in the gate (`validate_subpixel` read a validator attribute moved in this phase; fixed, green standalone; no src change) | all three BIT-IDENTICAL to phase 0 |
| 4 fill record + material merge | 171e28f | 26018 | 12161 | 11050 | equiterminal 1929, cornermode 379, enrich 1188, voxmodel 1057, sppeec_input 1374 | 44 pass / 0 skip / 1 fail in the gate (`validate_vhr` compared the filament resistance as a scalar; validator-only fix, green standalone) | all three BIT-IDENTICAL to phase 0 |
| 5 the enrich interface | 5f39c39 | 25819 | 12151 | 11037 | equiterminal 1755, cornermode 379, enrich 1399, voxmodel 1057, sppeec_input 1139 | 45 pass / 0 skip / 0 fail | all three BIT-IDENTICAL to phase 0 |
| 6 study prune + dormant code | (next commit) | 24675 | 12150 | 9593 | tree 1310, leaf_induct 410, enrich 1399, equiterminal 1755, sppeec_input 1139 | 45 pass / 0 skip / 0 fail | all three BIT-IDENTICAL to phase 0 |

Scoped validators at baseline: equiterminal 407, subpixel 380, corner
202, superconductor 249, aniso 189, aniso_sigma 220, input_lppr 520.

## 8. Phase log

### Phase 6 (2026-09-04): the study prune, and 1000 dormant lines

* Studies removed (21 files, 1444 lines), per section 5: the
  superseded skin chapters `skinconv`, `barconv2`, `wireconv`,
  `shortwire`, `wirerc`, `wirefull`, `wirebnd`, `basis4wire`,
  `thirdwire`, `linbnd`, `frozenstair`, `cmpeq`; `zuudiag` and `zcross`
  (their block checks are validate_enrich B); the narrative duplicate
  `skinnarr_report`; the done measurements `p3point`, `p3pinned`,
  `memsolve`, `onefac2`, `ordering`, `vhrsurvey`. KEPT against the
  plan: `skinnarr.py` -- it is the model library the three kept
  narrative scripts import, not a duplicate. The loop-basis /
  preconditioner cluster (18 files) is untouched, as planned.
* Scratch deleted and ignored: `studies/_narr/` (16 MB of meshes),
  `studies/*.vhr`, `*.log`, `validation/studies/`, `.ipynb_checkpoints`.
  The `*_results.json` files stay.
* Kept studies moved to the current API where they still read old
  attributes (`mode_referee`: `split.boxes`, `_rc`, the palette's
  per-cell geometry, the whole-sub-bar impedance; `skinnarr.solve_sp`:
  `enrich=`). Smoke-run on the new API: `london1d` (100.0% contin),
  `london_crowding` (1.3009 at two cells, as in phase 2),
  `mode_referee` (below). `studies/README.md` rewritten for the
  remaining set.
* DORMANT CODE: the SPAI/RDF preconditioner -- `spaiinit`,
  `spaiapply`, `spaiapply2` in tree.py and `spaiinit`, `spaistruc`,
  `spaiapply` in leaf_induct.py (1017 lines, "implemented but unused",
  one commented-out caller), plus the `RDFapply*` trio that called
  them -- is gone. `RDFinit` STAYS: it publishes alpha/beta to the
  leaves and `VoxelModel.prepare` calls it (my pattern cut took it
  along and the import check caught it; restored from HEAD).
* Ledger: src 25819 -> 24675 (-1144; -2452 total, 9.0% of the
  baseline). validation 12150, studies 11037 -> 9593.

### Phase 5 (2026-09-04): one `enrich` interface, one resolver

* `enrich.resolve(model, request, port_axis)` is the ONE place the
  engagement rules live (`'off' | 'auto' | table{families, k, reach,
  rc, f_ref, use_fft, csr_max_gb}`), and `enrich.build()` the one
  place the families are constructed. They replace: the seven-key
  `skin` table parser (60 lines), `_skin_unsupported`, the sweeper's
  duplicated k rule and rc dispatch (~90 lines), the solver's
  engagement block (~100 lines incl. the film palette selection and
  the corner branch), `skin_depth` / `recommend_subdivision` in
  equiterminal (the resolver's rule subsumes both: engage when 2 dx >
  the length the current varies on, k = min(12, max(7, ceil(2 dx /
  length))), k = 2 refused). `_auto_rc` moved to enrich.py. The stale
  guard is gone: `auto` now engages on anisotropic cells.
* `EquiTerminalSolver(..., enrich=None)`: the ten engine kwargs
  (`subdivide, rc_uu, rc_cross, skin_freq, use_fft, csr_max_gb, reach,
  corner_modes, ...`) are one argument; the direct API and the TOML
  path resolve identically (validate_input_lppr's TOML == direct
  check). Consequence, by design: a direct-API caller that gave no
  radii now gets the width-scaled defaults the TOML path always had
  (A/B case B: rc (3,4) -> (6,8) on the 4x4 bar).
* Partial cells are always exact: `[grid] subpixel` and `_snap` are
  gone, off-grid metre bounds produce the fill record, a two-axis cut
  is an error. Every example's metre bounds were verified on-grid to
  1e-13 before removing the snap. The converter's `--subpixel` flag is
  gone (physical z bounds always).
* Docs: doctrine rules 13, 14 and 14b are two rules (enrichment;
  partial cells); README and the status example follow.
* Validators: `validate_input_lppr`'s section gates the resolver's
  rules (k rule, width-scaled rc, override, off, k = 2 refused, table
  without equipotential refused, unknown key refused, TOML == direct,
  reach 0, wide-bar band, reach "all" overshoot, superconductor auto
  off); `validate_partial` builds its staircase reference with cell
  bounds now that snapping is gone; keeper studies and validate_corner
  / enrich / superconductor / aniso use `enrich=`.
* A/B vs the phase-4 snapshot: TOML, cylinder, slab and corner paths
  identical to 1e-19; direct-API case B differs only by its new radii.
* The TOML parser validates the table at load time through the same
  `check_request` the resolver uses (doctrine errors fire at parse or
  build; `validate_input_lppr` expects `k = 2` and unknown keys to be
  refused before any solve).
* Ledger: src 26018 -> 25819 (-199; -1308 total). THRESHOLD 25927 MET
  with 108 lines to spare.

### Phase 4 (2026-09-04): one fill record, one material accessor

* `VoxelModel.fill` (covered fraction per cell) + `VoxelModel.cut`
  (`{'kind': 'slab', 'axis'}` or `{'kind': 'cylinder', 'axis', 'k',
  'cells', 'geom'}`) replace `fill_frac`, the fill-scaled `sigma`,
  `z_scale`, `slab_fill`, the `subpixel` dict, `sigma_axis` and
  `laminate_sigma`. `sigma` is the BULK metal everywhere now; the
  per-orientation consequence is one method, `impedance_scale()`
  (1/fill along the layers of a slab cut, bulk across it; every
  orientation for a cylinder), read by one `resistances()` (the scalar
  fast path and `_complex_resistances` are gone: every model goes
  through the per-filament `(l/2)/A (z_a + z_b)` with the cell's
  impedance density), by `sigma_along()` / `impedance_along()` for the
  terminals (per face, effective along the port axis -- the slab path
  used bulk at a partial port cell before, the cylinder path
  sigma*fill; now both use the effective value), and by the mode
  engine. `london_rate` and `material_response` are `VoxelModel`
  methods; `base_sigma` is gone (a fill model is uniform-sigma now).
  `fill()` the percent-occupancy method became `fill_pct()`.
* Front end: the cylinder painter and the slab coverage pass write the
  one record; the equipotential port-on-whole-cell rule applies to
  cylinder cuts only (a slab port MAY touch a partial cell, as before);
  near-whole coverage (1 - 7e-16 from the coverage arithmetic) snaps to
  exactly 1.
* Validators: `validate_partial` (488) replaces `validate_subpixel`
  (380) and `validate_aniso_sigma` (220): the cylinder sections as
  they were, plus the slab laminate rule per orientation (in-plane R
  doubles in a half cell, the cut axis keeps bulk, the London path
  scales the same way) and the 75 nm film end-to-end headline.
  `laminate_sigma`'s own checks (two means, crack, open_floor) died
  with it.
* A/B vs the phase-3 snapshot: every block bit-identical; matvecs
  within 7e-11 (the cylinder path no longer stores sigma*fill as
  float32).
* Gate: `validate_vhr` asserted the filament resistance as ONE scalar
  (the removed fast path); it now checks every filament. Also found:
  `spaiinit` in tree.py still adds a scalar `r` -- dead code (one
  commented-out caller), left for the study prune.
* Ledger: src 26235 -> 26018 (-217; -1109 total). Threshold 25927:
  phase 5 must deliver >= 91 net.

### Phase 3 (2026-09-03): corner modes as a palette, n-ary stack

* The corner modes are PATCH-level unknowns (3 per corner) over BOTH
  in-plane orientations with overlapping patches, which the per-
  filament, single-orientation `Enrichment` could not hold. It grew
  three things, each generic: ENTRIES (a filament under one palette;
  any orientations, repeats allowed; pairs are built per orientation
  and perpendicular pairs vanish), a PROLONGATION `P` that ties
  per-entry columns into patch modes (blocks fold as `P' Z P`;
  block-Jacobi groups come from the palette), and a PALETTE object
  (`ConductionPalette`, `SurfacePalette`, `FixedPalette`) replacing
  the if/else on weights. `cornermode.py` keeps `_tabulate` and
  `find_corners` and becomes `corner_palette()`: 607 -> 379 lines;
  `CornerModes` and the pairwise `ModeStack` are gone. `ModeStack` in
  `enrich.py` is n-ary: union of aggregates, generic sparse cross
  blocks between any two families over parallel entry pairs (radius =
  the larger `rc_uu`), block-diagonal preconditioner. Corner + surface
  palette now composes (the phase-2 ValueError is gone).
* THE TRAP: the generic family coupled its modes only to its OWN
  entries. The engine never noticed (its entries are all parallel
  filaments) but a patch-subset family lost the drive from the
  filaments just outside its patch and the corner bands collapsed
  (x0.35 vs x0.66; composed over-corrected). Measured on the Z-trace
  that neither the terminal coupling nor the cross radius nor the
  aggregate radius mattered -- only the aggregate SET. The family now
  carries `agg`: every filament of its orientations within rc_cross
  of an entry. Engine-off bands back to exactly the old x0.66 / x0.74;
  composed x0.54 / x0.46 (old ~0.53 / ~0.53; the corner's aggregate
  radius is now the engine's own rc_cross rather than a window around
  the vertex; at 2W + rc_cross it reads x0.41, still in band).
* Validators: `validate_corner` keeps detection, DC decoupling and the
  AC bands (202 -> 171); net-zero, Zuu symmetry, augmented reciprocity
  (one family and the two-family stack) are `validate_enrich` J.
* A/B vs the phase-1 snapshot: engine paths unchanged (<= 1.5e-11);
  the corner stack's matvec differs by 2e-3 (the wider aggregate set
  and the corner entries' new terminal coupling), its Ru identical to
  1.4e-15 -- the tabulated weights are reproduced.
* Gate slip: `validate_subpixel` read the per-cell geometry as engine
  attributes (`_tkey`, `_percell`, `_rfac`) that this phase moved onto
  the palette, and I had not re-run it after the rework; it failed in
  the full gate. Validator-only fix (`rt.palette.*`, and the per-entry
  impedance is now an array), green standalone; no `src/` change, so
  the gate's other 45 verdicts and the anchors stand.
* LEDGER WARNING: before the prose trim this phase was +16 lines on
  `src/` -- the generalisation (entries, `P`, palettes, `agg`, the
  n-ary stack) cost what deleting the two classes saved. After moving
  duplicated prose to the history document: 26278 -> 26235 (-43;
  -892 total). Phases 4 and 5 must deliver >= 308 net to meet the
  25927 threshold; their scoped code is ~640 lines.

### Phase 2 (2026-09-03): one Enrichment class

* `enrich.Enrichment` replaces `Redistribution` and `SubpixelModes`:
  one constructor, one weight setter, one assembly, one frequency
  policy, one preconditioner, with the shared-weights (Toeplitz/FFT)
  and per-cell (surface palette, CSR) configurations as a flag on the
  same fold. `material_response`, `london_rate`, `conduction_weights`
  moved into `enrich.py`; `netzero_prune` is the one net-zero /
  normalise / pivoted-QR prune (was three); `surface_weights` and
  `_surface_geometry` are the per-cell palette; `_near_surface` is the
  one placement rule (`reach` cells beyond the exposed layer; the
  surface palette's ring rule is the same rule on the resolved
  surface -- reach 0 reproduces both old defaults).
* DELETED: the `diff` and `linear` palettes (the one deliberate feature
  removal; TOML `skin.basis` accepts only `conduction` until phase 5
  drops the key), `split_axis`, `mode_basis`, `boundary_only`
  (`reach` on the solver), the runtime aggregate self-check, the
  superconductor `mode_basis` guard. The deleted measurement prose
  lives in docs/enrichment_history.md.
* Validators: `validate_equiterminal` parts F and G removed (F's
  palettes are gone; G's retune-equals-fresh is `validate_enrich` H,
  on TOML-built bars so it no longer needs the VoxHenry corpus);
  `validate_superconductor` E no longer asserts that generic palettes
  are refused; `validate_subpixel` reads `shared`, `_rc`, `_rfac`;
  `validate_enrich` gains H (frequency policy: km-change rebuild, FFT
  Z retune == fresh, CSR blocks bitwise, London Ru scales exactly with
  w and the rate does not move) and I (the London bar: plain exactly
  bulk, modes lift kinetic/bulk to 1.32 against a 1e-8 reference,
  band 1.30..1.55).
* A/B against a worktree of cea4b11 (same five models): every block,
  spectrum and matvec agrees to <= 1.5e-11; the residual is last-bit
  rounding of the rate (1e-15 in W, amplified to 1e-11 in the
  spectra). One trap found and fixed: with 16 candidate columns on 9
  sub-bars (k = 3) symmetric shapes tie in norm and the pivoted QR's
  choice depends on the LAST BIT of the column mean and norm -- a 2-D
  reduction rounds differently from the old per-column loop and picked
  a different (equal-span) basis. The prune keeps the per-column
  arithmetic on purpose.
* Pre-existing defect found: `studies/london_film.py` could not print
  its result line (a bare trailing `%`) since 4ca6684; fixed in place.
  Benches, phase-1 worktree vs this tree (nt = 2, per-square
  normalisation): modes off 54.1% / 54.1%; full palette 76.3% /
  76.3% (geometric term 0.2323 vs 0.2324 pH, a 4e-4 shift from the
  preconditioner's self block now coming from the D = 0 table, which
  moves the Krylov path within the solve tolerance); film palette
  83.6% / 83.6%. London bar with the 1e-8 reference: 1.3009 at two
  cells, 1.3517 at four (no "before" exists: the old bench crashed on
  its 1e-9 reference, every mode column pruned at that rate).
* Ledger: src 26826 -> 26278 (-548 this phase, -849 total). NOTE the
  phase-1 row first recorded the pre-amend hash; a phase's hash is now
  stamped by the NEXT phase's commit.

### Phase 1 (2026-09-03): one geometry, one table cache

* `src/enrich.py` (new): `Split` (one vectorised sub-prism box builder:
  k x k transverse, 1-D film/slab, in-plane corner, whole cell,
  terminal bar), `PairTables` (raw sub-prism mutual tables between any
  two parallel splits, cached by separation set, chunked), the pure
  `unique_separations` / `neighbour_pairs` helpers, and `partial_dL`:
  subpixel stage B for cylinders AND slabs as one function -- the
  aggregate-aggregate block of the fold, values computed once per
  distinct (offset, w_i, w_j) and gathered, emitted exactly symmetric.
* DELETED: `src/subpixel.py` (444 lines: `slab_dL`, `_build`,
  `build_dZ`/`_profile_weights` -- the C.1 imposed-profile null is now
  one paragraph in `partial_dL`'s docstring); from `Redistribution`:
  `_sub_boxes`, `_full_boxes`, `_term_boxes`, `_terminal_boxes`,
  `_raw_tables`, `_raw_terminal_table` (rebuilt as 5 lines on the
  tables), `_uniq_sep`, `_check_aggregate` (its identity is now part D
  of `validate_enrich`), the inline separation keying in
  `_build_truncated`; from `cornermode.py`: the two box loops and
  `ModeStack`'s private `_Lec_raw` kernel evaluation (now the same
  tables, gathered).
* The engine's transverse-axes list is `Redistribution.tr` now; `split`
  is the `Split` object. `EquiTerminalSolver` and the LpPR sweeper call
  one `partial_dL`; the LpPR path thereby gains slab stage B for free.
* A/B GATE (scratchpad abdump/abcmp, baseline worktree at 5242dd1, five
  models: equibar FFT conduction, equibar CSR diff, round-wire cylinder
  dL + SubpixelModes, two-film slab dL, corner Z-trace ModeStack):
  every mode block, FFT spectrum, Ru and Zt BIT-IDENTICAL; `apply_Z` on
  a random vector identical to 1e-16..1e-19 on the engine paths. Not
  bit-identical, all explained: corner<->engine cross block Zec and the
  corner-stack matvec at 1.2e-11 (tables are evaluated at the origin
  and translated; the closed-form kernel's cancellation amplifies
  coordinate ulps -- validate_corner's physics bands are the gate);
  slab dL at 3e-13 (einsum vs per-entry dots); cylinder dL drops 22670
  entries that were all below 3e-29, rounding dust the old loop emitted
  on full-cell pairs, and agrees to 5e-11 on every real entry.
* `validate_enrich.py` (new, 27 checks): boxes tile, tables == explicit
  `box_mutual_matrix`, reciprocity, the aggregate identity, terminal
  bars, `partial_dL` symmetry/first-principles pair/whole-whole zero,
  neighbour pairs. `validate_equiterminal` F drops the `aggregate_err`
  check (moved here) and reads `Split.boxes`; `validate_subpixel` reads
  `Split.boxes`.
* Ledger: src 27127 -> 26826 (-301). validation +247, of which 242 is
  the new validator.

### Phase 0 (2026-09-03)

* Plan written. Baseline line counts recorded above.
* `run_baseline.sh` gains a line-count report (physics-free change).
* Full suite run on the idle box (~100 min incl. anchors, corpus links
  in place so nothing skipped): 44 pass / 0 skip / 1 fail. Anchors:
  setup1 bit-identical to the runner's 2026-08-25 value; setup2 and
  setup3 shifted in the 1e-30 rounding dust with NO src/ change (setup3
  reproduces bit-identically standalone). Runner's expected values
  re-based to the measured ones with the lineage recorded there.
* PRE-EXISTING RED found at HEAD: `validate_aniso` part D asserted that
  `EquiTerminalSolver(subdivide=3)` raises on anisotropic cells. That
  guard was lifted by 296bcdb (2026-09-01) and the validator was not
  updated. Inverted in this phase (the check now requires the engine to
  engage with per-axis `dt`); standalone 9/9 ok. This is the "invert"
  item the plan had scheduled for phase 4, pulled forward because a
  baseline with a known-stale red is not a baseline.
