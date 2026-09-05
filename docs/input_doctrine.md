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

3. **Wires are centrelines.** A `[[wire]]` is the physical
   centreline: a list of 3-D points, a radius, a conductivity.
   Discretisation (segment subdivision, ring/sector counts) is solver
   policy with file-level overrides, not geometry. The validated
   defaults are the 1-4-8-12 cross-section and `max_seglen` at or
   below the tree's leaf-box extent.

   The default `shape = "polyline"` joins the points with straight
   runs. `shape = "spline"` instead fits a clamped cubic through
   them and requires `start_vec` and `end_vec`: the takeoff vectors
   at the two feet, each pointing AWAY from its own pad (both `+z`
   for an ordinary bond), so a mirrored pair reads as mirrored in the
   file. Their MAGNITUDE is read as well as their direction — each is
   the cubic's control handle, and a plain two-foot loop passes
   `0.75*|v|` above the midpoint of its feet. Middle points are
   optional (zero is normal and fully determined) and unlimited; the
   sampler pins every declared point as an actual vertex, because a
   middle point is usually a clearance the loop has to make and a
   chord that merely passes near it cuts inside.

   Sampling is curvature-adaptive: chord length is whatever holds the
   sagitta under `sagitta` (default 0.1) times the wire radius,
   capped by `max_seglen` and the leaf box. THIS IS A COST KNOB, not
   only an accuracy one — every wire-path store is linear in segment
   count, and a slanted segment couples into two or three lattice
   orientations where an axis-aligned one couples into a single one
   (measured: ~2x the coupling entries and ~3x the wire setup on the
   flagship bonds). A spline tight enough to run away is refused
   rather than silently inflating the far-field cache.

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
   path — on BOTH the LpR and the equipotential solvers since
   2026-08-26, and either one also takes the frame when the
   plaquettes cannot span the cycle space, e.g. moated ground
   planes, detected by a free enumerator probe rather than paid for
   in the MST fallback: the RSFQ JTL went 694 → 78 s of setup for
   the same answer), and the GPU preconditioner apply engages automatically
   when the hardware works (`SPPEEC_GPU=0` opts out). Oracle-grade
   comparisons pass `rtol = 1e-10` explicitly — the validators do —
   rather than the defaults carrying validation's burden. `maxiter`
   (outer Krylov cycles; the matvec budget is `maxiter × inner_m`,
   solver default 30) exists for large runs that hit the cap — the
   6.8M-cell RSFQ JTL rung stopped at 331 matvecs where the
   overcomplete N^0.66 iteration law wants ~560; a capped run resumes
   under `SPPEEC_CHECKPOINT`, but the cap itself must be settable.

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
    LpPR, whose port is a prescribed injection) or wires. Since
    2026-08-27 its particular solution is a spanning-tree route of
    the port current (KCL exact, O(N)) and the port voltage is read
    through the work-conjugate identity `V·I = ihat·(Z i)`, corrected
    for the finite Krylov residual by a Gram solve (right-preconditioned
    lgmres with the loop preconditioner, to 1e-8; the tree route has a
    large loop-space component that the minimum-norm lsqr solution
    never had — uncorrected it read 4e-3 off the equibar razor gate at
    rtol 1e-4 and 0.12% off at 1.1M cells; corrected, 2.7e-6 on the
    razor gate and 5.5e-5 off the rtol-1e-6 lsqr reference at 1.1M
    cells. A single preconditioner apply is not enough at that size,
    plain refinement diverges, and cg fails because the GeoMG/Schur
    apply is not symmetric) — the two
    `lsqr` projections it used before cost 25 s each at 147k cells
    and dominated the per-frequency time at 18M filaments (the JTL
    200 nm solve went 117 → 24 s; the readouts agree to 1.8e-8 at
    rtol 1e-8). `EquiTerminalSolver.solve(readout='lsqr')` keeps the
    old path for comparison.

13. **Sub-cell enrichment: one `[solve] enrich` table, on by default
    where it counts** (equipotential path only; rewritten 2026-09-04,
    no compatibility with the former `skin` table). `enrich = "auto"`
    (the default) engages the SECTION family -- net-zero conduction
    modes on every filament along the port axis, the measured-best
    palette (93% of the skin-depth correction, geometry-independent)
    -- only when a transverse cell exceeds half the skin depth at the
    sweep's highest frequency, so it costs nothing where it buys
    nothing; it never engages by itself on a superconductor (London
    modes are opt-in). `enrich = "off"` disables it. A table is
    explicit and engages per the same rules, raising where the model
    cannot be served (several London depths, mixed conductivities):

        [solve.enrich]
        families = ["section", "corner"]   # default ["section"]
        k        = 7        # sub-bars per split axis; auto = min(12, max(7, ceil(2 dx/delta)))
        reach    = 0        # cells beyond the exposed layer that carry modes; "all"
        rc       = [6, 8]   # coupling radii (mode-mode, mode-aggregate); auto = width-scaled
        f_ref    = 1e9      # shape reference frequency; auto = sweep maximum

    `k` is quadrature, not unknowns (the mode count is fixed by the
    palette), and k = 2 is refused -- a 2x2 split is blind to axially
    symmetric neighbourhoods. `reach` defaults to the exposed layer:
    modes-everywhere overshoots the physical skin limit ~2.8x on a wide
    section through spurious interior excitation. `rc` defaults to
    (ceil 1.5W, ceil 2W) off the section width, floored (3,4), capped
    (12,16), with the mid-shell damage-zone fallback; (12,16) on
    declared films; (3,4) on cylinder fills. A block with `film = "z"`
    (the film normal) gets the 1-D film palette along the normal when
    the port is in-plane. The CORNER family adds tabulated circulation
    modes at 90-degree bends (3 solved amplitudes per corner, both
    in-plane orientations; AC-only by the DC-decoupling theorem). The
    shapes retune per solve frequency; the evidence for every rule is
    in docs/enrichment_history.md.

14. **Partial cells are always exact.** A `[[block]]` whose `from_m` /
    `to_m` fall off-grid is never snapped: the boundary cells keep
    their exact coverage (`model.fill`) and carry the LAMINATE
    effective conductivity -- `sigma*fill` along the layers (exact for
    a layered medium, not a bound), bulk across them (no filament
    passes all the way through a cell, so the through-plane value is a
    half-cell quantity a per-cell array cannot express) -- together
    with the matching partial-cell inductance correction
    (`enrich.partial_dL`, dL = w'Tw - u'Tu over exact sub-prism
    tables, raised automatically). Measured on a 75 nm film at 30 nm
    pitch against the same film at 15 nm: R error 16.67% -> 0.00%, L
    error 2.42% -> 0.30%. ONE cut axis per model; a cell cut on two
    axes is an error. The SECTION cut is the other record: geometry
    invariant along one lattice axis and resolved below the cell in
    the plane across it -- `[[cylinder]]` (`axis`, `center`, `radius`,
    `sigma`, optional span) and `[[trace]]` (`path_m` polyline in xy,
    `width_m`, `z_m` whole cells, `sigma`, optional `film = "z"`; any
    angle; bends mitred) are its two shapes, painted as one union:
    fill fractions sampled from the shape union, in-plane filaments
    at the conductance of the FACE they cross (a tilted cut joins
    cells of unequal fill, and the per-cell rule is first order
    there; the face rule leaves a 45-degree bar's DC R within 0.1% of
    the axis-aligned solve at 8 cells across), the same inductance
    correction (round-wire example vs a 2x reference: staircase 2.4% ->
    fill 1.2% -> fill+dL 0.9% in L), and under enrichment the
    SURFACE palette -- per-cell modes anchored to the true surface,
    R_AC/R_DC within ~1% of the exact Kelvin solution at dx/delta 1-2,
    usable to 3-4. Union rule: a cell is metal if any primitive claims
    it, a later primitive's sigma wins, and a cell a `[[block]]` fills
    whole is never carved -- so a trace ends inside a pad and the port
    goes on the pad's axis-aligned face; a commensurate trace IS a
    block. Equipotential ports on a section cut must sit on whole
    cells; a slab port may touch a partial cell. Section and slab cuts
    do not combine in one model, and neither combines with `[[wire]]`.

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
