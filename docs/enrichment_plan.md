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
| 1 geometry + tables | 2f0e495 | 26826 | 12294 | 11054 | equiterminal 3165, cornermode 607, enrich 385 (new), subpixel 0 (deleted), voxmodel 1179, sppeec_input 1399 | 46 pass / 0 skip / 0 fail (incl. new `validate_enrich`) | all three BIT-IDENTICAL to phase 0 |

Scoped validators at baseline: equiterminal 407, subpixel 380, corner
202, superconductor 249, aniso 189, aniso_sigma 220, input_lppr 520.

## 8. Phase log

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
