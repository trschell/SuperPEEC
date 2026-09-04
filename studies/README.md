# studies/ — measurement scripts (2026-08-02)

Ad-hoc measurement scripts, kept because their *results* are quoted in
docstrings and memory and someone will want to re-run or extend them.
These are NOT validators: they print tables, they do not assert, and
several are memory-bound. Run from the repo root inside the toolbox:

    SPPEEC_SCHEME=cell PYTHONPATH=src python3 studies/<name>.py

They write scratch `.vhr` files next to themselves.

## The skin / enrichment story

The 2026-08 investigation of the redistribution engine (flat bars, the
24 um and 50 um wires, rc ladders, boundary-vs-interior placement,
block-assembly checks) reversed twice; its scripts are gone (2026-09-04)
and its conclusions live in `docs/enrichment_history.md`, the block
checks in `validation/validate_enrich.py`. What remains here are the
INSTRUMENTS the enrichment framework is gated on and the studies that
established the current design:

| script | what it measures |
|---|---|
| `london_film.py` | London kinetic sheet inductance vs an exact per-square reference (relative comparisons; the film-palette bench) |
| `london1d.py` | the 1-D formulation bench: the engine's own palette against exact 1-D kernels |
| `london_oracle2d.py` | 2-D cross-section oracle: the TRUE kinetic term of a finite strip |
| `london_crowding.py` | does the voxel model develop the London profile, and at what mesh |
| `xsection.py`, `xsection_tabulated.py` | 2-D Galerkin cross-section basis studies (conduction family complete for straight sections) |
| `modebasis2d.py` | which net-zero basis represents skin effect (6.6x for conduction) |
| `mode_referee.py` | zero-truncation Galerkin referee for the surface (cylinder) palette |
| `corner_referee.py`, `ltrace_ladder.py` | corner-mode subspace referee and the bend refinement ladder |
| `palette_ablation.py` | P0 vs P1 palette ablation (individual corner columns) |
| `frozenwire.py` | freeze the staircase, refine the mesh: the staircase-vs-circle gap |
| `slabfill.py` | fill-fraction slabs vs a commensurate reference |
| `stripedge.py` | is the RSFQ gap in-plane edge crowding? (open, never run to a verdict) |
| `skinnarr.py`, `skinnarr_default.py`, `skinnarr_profile.py`, `skinnarr_default_report.py` | the current-flow narrative vs VoxHenry (defaults through the TOML path; profiles) |

## Scaling / preconditioner

| script | what it measures |
|---|---|

Result: `simplicial` + `metis` beats the previously hard-coded
`supernodal` + `amd` by 38% factor memory and 3× time (single core) —
now the default in `port_impedance.LpRSolver`. Caveat: supernodal is the
mode that can use threaded BLAS, so the TIME ranking may reverse at 16
threads; the memory result is structural.

## VoxHenry corpus

| script | what it measures |
|---|---|

`validate_port_impedance.py` (repo root) exercises the
prescribed-current port instead — NOT the equipotential terminal; the
two give materially different answers above ~1 GHz.

## FMM diagnostics

`p3point.py` / `p3pinned.py` (the point-charge far-field check that
localised the `leafinit` panel-offset sign bug) are superseded by
`validate_leafinit_geometry.py` and were removed 2026-09-04.
