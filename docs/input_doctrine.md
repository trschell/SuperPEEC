# The SuperPEEC input doctrine

One TOML file describes one solvable problem. `sppeec_input.load()`
parses it; `Problem.model()` / `.wires(freq)` / `.solver(M, freq)`
build the pieces. This page is the prose contract; the format grammar
lives in `sppeec_input.py`'s module docstring and is enforced there.

## Why

The repo accumulated three ways to define a problem — `.vhr` files
(VoxHenry compatibility), PyPEEC mesher output, and hand-built
`VoxelModel`s inside studies — and none of them could describe a bond
wire, a foot, or a drive in one place. Every convention below already
existed implicitly somewhere in the code; the doctrine's job is to
make each one *load-bearing in exactly one place* so a model file
written today still means the same thing after the next refactor.

## The rules

1. **SI units, no exceptions.** Metres, S/m, Hz. There are no unit
   suffixes and no per-file unit declarations: `radius = 0.25e-6`,
   never `radius_um`. A format with switchable units eventually feeds
   someone's micrometres to someone else's metres.

2. **One frame.** The world origin is the corner of voxel `(0,0,0)`.
   Cell `c` occupies the half-open box `[c, c+1)·pitch` per axis; its
   centre is `(c+0.5)·pitch`; the filament at low cell `c` runs from
   the centre of `c` to the centre of `c+1`. Wire polylines, port
   cells and blocks all live in this frame. (This is the convention
   `wirecoupler`, `vtkout` and the kernels already share — the
   doctrine just names it.)

3. **Wires are polylines.** A `[[wire]]` is the physical centreline:
   a list of 3-D points, a radius, a conductivity. Discretisation
   (segment subdivision, ring/sector counts) is solver policy with
   file-level overrides, not geometry. The validated defaults are the
   1-4-8-12 cross-section and `max_seglen` at or below the tree's
   leaf-box extent.

4. **Feet are found, not stated.** The first and last polyline points
   are the bond contacts. The attachment is a coverage-weighted PATCH
   of surface cells under the contact disc (radius `foot_r0`, default
   2× the wire radius — the flattened-bond rule), plus a CALIBRATED
   deficit resistance: the model term supplies exactly what the
   lattice cannot resolve of the contact-edge singularity
   (`footcal.py`, exact lattice-Green's-function calibration, no user
   action needed — `foot_model = "point"` in `[solve]` restores the
   legacy single-cell foot for comparison). Keep the contact a
   fraction of a cell *above* the metal surface — the wire's
   cross-section must stay clear of the voxel metal (the overlap
   rule) — and over a cell CENTRE: a contact on a cell boundary gets
   its footprint tie-broken half a cell sideways (measured as a 9e-4
   sharing bias; the solver warns).

5. **Half-open ranges.** Inline geometry blocks use `from`/`to` cell
   ranges that behave exactly like Python slices: `from = [0,0,0]`,
   `to = [10,8,2]` is a 10×8×2-cell pad. Inclusive ranges breed
   off-by-ones; there is one convention and it is the language's.

6. **Unknown keys are errors.** A misspelled `radiius` must kill the
   run, not produce a default-radius wire. The parser rejects any
   table or key it does not know.

6b. **Defaults serve end users; validators pin their own rigor.**
   `[solve]` defaults are the engineering ones: `rtol = 1e-4` (the
   residual tail below that moves extracted Z by < 1e-3 %; FEKO
   ships 3e-3), `basis = "auto"` (small cycle spaces get the exact
   Cholesky, large ones the memory-flat overcomplete + BlockAMG
   path), and the GPU preconditioner apply engages automatically
   when the hardware works (`SPPEEC_GPU=0` opts out). Oracle-grade
   comparisons pass `rtol = 1e-10` explicitly — the validators do —
   rather than the defaults carrying validation's burden.

7. **Frequency-dependent state is rebuilt, not patched.** Wire skin
   shapes follow the solve frequency (`delta = sqrt(rho/(pi f mu0))`),
   and by doctrine a sweep rebuilds the wires — and therefore the
   wire coupling blocks — per point. The cost is visible and
   attributable rather than hidden behind a stale-shape answer.

8. **Ports are strips, held physically fixed.** A port is a list of
   cells per polarity with an equal prescribed split. Single-cell
   ports have a dx-dependent port-local term (the recorded
   single-node divergence); use a strip of several cells, and when
   refining the mesh keep the *physical* port region fixed, never a
   fixed cell count.

9. **The formulation is derived, not stated** (added 2026-08-15).
   `[[block]]` tables may carry `epsilon` (and `loss_tangent`) for
   dielectric cells — applied where the block has no conductor; a
   lossy `eps_r*(1 - j*df)` rides the same Ruehli material law, so
   dielectric loss enters Re(Z) with no extra machinery. A constant
   loss tangent is non-causal over wide bands: add
   `dispersion = "djordjevic"` + `f_ref` (and optionally `f1`/`f2`,
   default 1e3/1e12 Hz) to make `epsilon`/`loss_tangent` the values
   *at* `f_ref` of the causal wideband-Debye `eps_r(f)` — eps' falls
   logarithmically, tan-delta stays near-flat inside the band, and
   the fit reproduces the stated values at `f_ref` exactly (gated to
   1e-12, and the solve to 4e-13 against the constant-df model at
   `f_ref`). They may
   instead carry `lambda_l` (the London depth, metres): alone it is
   a pure lossless London superconductor, with `sigma` a lossy
   two-fluid one (the VoxHenry material law; `lambda_l` and
   `epsilon` do not compose, and the superfluid shorts DC so every
   solve frequency must be positive). Any epsilon or lambda_l block
   (or a face-style port) resolves `formulation = "auto"` to the
   LpPR mixed-row path; wires-only inputs resolve to LpR. The
   explicit `[solve] formulation` key exists for the one case auto
   cannot know: conductor-only capacitance near the ωL/R crossover.
   Impossible combinations are errors, not fallbacks: epsilon or
   lambda_l under LpR, wires under LpPR (wire capacitance is
   excluded by doctrine, rule "What v1 leaves out"), a cell-style
   port under LpPR.

10. **Two port styles, one per formulation.** The LpR terminal port
    is cell-style (`p_cells`/`n_cells`/`p_box`/`n_box`, rule 8). The
    LpPR port is a *nodal current injection* on conductor faces:
    `p_faces`/`n_faces` entries `[ix, iy, iz, "+z"]` naming a
    conductor cell and which face carries the terminal current
    (corner-consistent lumping, the pdn convention). Faces must land
    on conductor cells — a face on a dielectric or empty cell is an
    error. `rtol` defaults per formulation: 1e-4 (LpR, the
    iteration-growth law's engineering point) and 1e-10 (LpPR, the
    tolerance every recorded dielectric anchor was taken at).

11. **Multi-port is `[[port]]`, LpPR-only** (added 2026-08-15).
    Several `[[port]]` tables (optionally named) give the open-circuit
    Z matrix in declaration order: one solve per driven port, other
    ports open, off-diagonal voltages read as the work-conjugate
    pairing of each port's injection profile with the driven
    solution. Reciprocity `Z_ij == Z_ji` is a *solved* property the
    validator gates (measured 1e-13 on the coupled-plates example),
    not an imposed symmetry. The LpR terminal port stays
    two-terminal; declaring several cell-style ports is an error.

12. **Sweeps are expressions or literals; equipotential terminals
    are a port flag** (added 2026-08-15). `[solve] freq` takes a
    literal list or `{ from, to, points, spacing = "log"|"lin" }`
    (log default; `0 < from < to`, `points >= 2`). A face-style
    `[port]` may set `equipotential = true`: the terminal current
    *split* becomes an unknown (the equiterminal machinery) instead
    of a prescribed profile — the port model whose terminal
    treatment recovers the full physical length (gated: R equals
    `l/(sigma*A)` analytically on the equibar example). LpR-class,
    single-port; combines with `lambda_l` (the validated
    superconductor route) but not with dielectrics (charge needs
    LpPR, whose port is a prescribed injection) or wires.

13. **The sub-cell skin engine is on by default, where it counts**
    (added 2026-08-17; equipotential path only). `[solve]
    skin = { ... }` controls the cross-section redistribution modes:
    `mode = "auto"` (default) engages the engine with the
    `conduction` basis — the measured-best basis, 93% of the
    skin-depth correction delivered, geometry-independent — but ONLY
    when the cell size justifies it: the engine's own
    `recommend_subdivision` at the sweep's highest frequency returns
    k = 1 when the mesh already resolves the skin depth, so the
    default costs nothing where it buys nothing. When it does engage
    with the conduction basis, the default is k = 7: conduction's k
    is pure quadrature (the mode count is fixed), and k = 7 measured
    19 points of skin-correction gap better than the generic cap of
    3 -- at zero additional unknowns, though the stiffer mode system
    does cost Krylov iterations at the highest frequencies. Auto degrades
    gracefully (to off, with a verbose note) on models where
    subdivision is undefined — anisotropic cells, superconductors,
    mixed conductivities — while `mode = "on"` or an explicit `k`
    lets the solver's loud guards fire instead. Exposed knobs:
    `basis` (`conduction`/`linear`/`diff`), `k` (>= 3; k = 2 is
    rejected — a 2x2 split is provably blind to axially symmetric
    neighbourhoods), `f_ref` (skin-shape reference frequency;
    defaults to the sweep maximum, and the shapes retune per solve
    point regardless), `rc_uu`/`rc_cross` (mode-coupling truncation
    radii — READ THIS before touching them or trusting wire skin
    numbers: the defaults (3, 4) are measured-correct for
    RECTANGULAR cross-sections, but on round/staircase sections
    they deliver only 5–36% of the skin correction; full delivery
    there needs rc ≈ 2x the section diameter, at close to
    untruncated cost, because a net-zero mode's distant couplings
    nearly cancel and axial coupling matters out to ~2 diameters.
    Do NOT split the difference: intermediate rc on wide sections
    is NON-MONOTONICALLY WRONG — measured worse than the small
    default, stable under solve tolerance — a hard cutoff mid-shell
    leaves an unbalanced residue rather than dropping negligible
    terms), `boundary_only` (default
    TRUE: modes live only on filaments with an exposed face in the
    split plane, which is where the physics is — measured on a
    20x20-cell bar at dx/delta = 4.8, modes-everywhere overshoots
    the physical skin limit rho*L/(P*delta) by ~2.8x through
    spurious interior-mode excitation, while boundary-only lands
    within ~15%; on few-cell cross-sections the two agree and
    boundary-only is simply cheaper. Set false only to reproduce
    the historical volume placement). Solver
    internals (FFT toggles, memory guards, factorization modes) are
    deliberately not exposed. HONEST LIMIT: the engine delivers ~93%
    of the correction; the residual is staircase voxelization of
    curved cross-sections, not the mode basis.

14. **Subpixel geometry starts with `[[cylinder]]`** (added
    2026-08-18; stage A of the subpixel program). A round conductor
    is declared by `axis`, `center` (the two transverse coordinates,
    metres), `radius`, `sigma` and an optional axial span
    (`from`/`to` cells or `from_m`/`to_m`). Boundary cells get a
    computed FILL FRACTION and carry `sigma_eff = sigma*fill`, so
    the partial-cell resistance is exact through the per-cell-
    conductivity machinery with no solver changes — measured on the
    round-wire example: the center-in staircase reads DC resistance
    11.6% low, the fill-corrected model lands within ~1% of
    L/(sigma*pi*R^2). Stage B (same date) adds the matching
    partial-cell INDUCTANCE: a sparse near-field correction
    dL = w^T T w - u^T T u over exact sub-prism mutual tables
    (4x4 sub-cells, 2-cell window; entries verified against first
    principles to machine zero), applied on the branch rows with the
    Toeplitz far field untouched. Measured on the round-wire
    example against a 2x-refined reference, the inductance error
    improves strictly through the stages: staircase 2.4% -> fill
    1.2% -> fill+dL 0.9%. Honest scope: corrections cover filaments
    ALONG the cylinder axis (the dominant current direction);
    transverse-filament and cross-orientation corrections belong to
    stage C. The sub-cell skin engine auto-disables on fill models
    (mixed effective conductivity), equipotential ports are
    unsupported there, [[cylinder]] does not combine with [[wire]],
    primitives must not overlap other conductors, slivers below
    fill 1e-3 are dropped, and port faces belong on solid-ish
    cells.

## What v1 deliberately leaves out

* Wire capacitance — wires are chargeless inductors by decision
  (2026-08-11); the LpPR composition point is recorded in the program
  memory.
* (Equipotential ports graduated into rule 12 on 2026-08-15.)
* Superconducting *wires* and grounds — compose them in code for
  now; the tables are reserved. (Per-cell epsilon and superconducting
  *blocks* graduated into rule 9, multi-port into rule 11, all
  2026-08-15.)

## Examples

See `examples/module3wire.toml` — the three-wire power-module
capstone: two pads, three arced bond wires, a port strip on each far
edge, solved over a frequency sweep for loop R, L and per-wire
current sharing — and `examples/plate_pair.toml`, the smallest
dielectric capstone: two copper plates, an FR4 gap, a face-style
port, solved by the LpPR path (C above the parallel-plate formula by
its fringing margin).
