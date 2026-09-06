# Diagonal traces: the section-cut program

Status: PHASES 0-4 COMPLETE, 2026-09-05 (phase 5 deferred). Phases ran
one at a time on the user's go, each closed by the validator gate and
a ledger entry, as docs/enrichment_plan.md was. Sections 1-3 are the
design as planned; section 3.4 and 3.6 carry the phase-1 and phase-3
revisions; section 10 is the record.

## 1. The problem, measured

A voxel lattice staircases a trace that is not axis-aligned. The
45-degree copper bar ladder (100 x 50 um x 1.6 mm, cubic pitch W/nw,
prescribed-current face ports, plain basis; scratch/diagonal_bar.py,
numbers in docs/enrichment_plan.md) against the same bar axis-aligned,
which is the rotation-invariant reference and reproduces the DC closed
form to six digits:

    cells across           4        8       16     observed order
    DC   R diag/aligned  1.420    1.132    1.029   1.7 / 2.2
    DC   L diag/aligned  1.061    1.021    1.005   1.6 / 2.1
    1e8  R               1.632    1.171    1.044   1.9 / 2.0
    1e9  R (deep skin)   1.690    1.385    1.204   0.84 / 0.91
    1e9  L               1.060    1.017    1.003

Mechanism. A square resistor network is isotropic in the continuum
limit, so DC converges at second order despite every Manhattan path
being sqrt(2) too long. The staircase bites only in the layer where
current cannot average over neighbouring cells, and that layer is one
skin depth thick: the deep-skin error is a function of h/delta in the
EDGE cells, not a fixed geometric penalty. Fixable to the extent the
edge cells' sub-cell description follows the true tilted edge; beyond
h/delta of roughly 5 no per-cell basis will, and that regime is the
strip-element program (section 8), not this one.

These five rows are the program's gate. Every phase re-runs the ladder
through the TOML path and reports the same table.

## 2. What exists, and what each phase reuses

* `VoxelModel.fill` + `VoxelModel.cut`: ONE cut record per model,
  `slab` (axis-aligned layer boundary, fill = covered fraction, the
  laminate rule in `impedance_scale`) or `cylinder` (axis, k=4 sub-fill
  bins per transverse cell, `geom` = (c1, c2, R, sigma) per cell).
  Resistance on a cut cell is the 1/fill rule on every orientation
  (cylinder) or all but the cut axis (slab). The painters live in
  `sppeec_input.Problem.model()`; the cylinder painter samples 64x64
  points per transverse cell and reduces them to fill and bins.
* `enrich.Split`: sub-prisms of a filament on all three axes,
  INCLUDING the axial one (a filament spans centre to centre, half of
  each end cell). `PairTables`: box-mutual tables between sub-prism
  sets keyed by separation. `partial_dL`: the stage-B correction
  `w'Tw - u'Tu` over near pairs; cylinder branch corrects the
  filaments ALONG the cylinder over the k x k bins, slab branch the two
  in-plane orientations over a 1-D split.
* `enrich._surface_geometry` / `surface_weights` / `SurfacePalette`:
  per-cell mode weights `exp(-p d)` with d the signed distance to the
  resolved circle, resampled at the engine's k from the cylinder
  `geom`, fill-weighted, net-zero pruned, placed within `reach` of the
  surface. Cylinder-specific in two places only: the distance/fill
  resampling reads (c1, c2, R), and the palette is applied to the
  filaments along the cylinder axis with a purely transverse split.
* `Enrichment`: entries of any orientation (`sel`), explicit aggregate
  set `agg`, shared weights on the FFT path or per-cell weights on the
  CSR path; `ModeStack`: n-ary composition with generic cross blocks.
  The corner family is the existing example of a per-cell, subset,
  mixed-orientation family stacked on a shared one. `build()` today
  constructs the shared section family on the PORT-AXIS orientation
  only.
* Ports: face-style `[ix, iy, iz, "+x"]` on conductor cells; the
  equipotential terminal requires one face axis and, on a cylinder
  model, full cells (fill == 1) under every face. The prescribed-current
  face port accepts mixed orientations (commit 6039f9b).
* `studies/`-grade instrument: scratch/diagonal_bar.py (hand-built
  model, centre-inside rasteriser, staircased end-cut port).

## 3. The design

### 3.1 One section cut, two shapes

The cylinder cut and the trace cut are the same thing: a geometry that
is INVARIANT along one lattice axis and resolved below the cell in the
section perpendicular to it. The record becomes

    cut = dict(kind='section', axis=a,        # the invariance axis
               shapes=[...],                   # circles and polygons
               k=ks, cells={(t1, t2): bins},   # ks x ks sub-fill bins
               )

with `shapes` the list of section shapes in metres: `('circle', c1,
c2, R)` for a `[[cylinder]]`, `('polygon', vertices)` for a
`[[trace]]`. Everything downstream asks the shape list two questions,
"is this point inside the union" and "signed distance to the union
boundary", so cylinder and trace share the painter, the fill, the bins
and the palette. The `cylinder` kind is deleted, not kept alongside.

The "list of planes per cell" of the discussion is the mental model;
the implementation stores the polygon and evaluates distance and
inside-ness on demand, which handles bends, ends, mitres and unions
with no per-cell case analysis.

### 3.2 The `[[trace]]` primitive

    [[trace]]
    path_m  = [[0.0, 0.0], [1.0e-3, 0.0], [1.5e-3, 0.5e-3], [2.5e-3, 0.5e-3]]
    width_m = 100e-6
    z_m     = [0.0, 35e-6]     # extent along the invariance axis
    sigma   = 5.8e7
    film    = "z"              # optional, as on a block
    name    = "sig1"           # optional

The invariance axis is the trace's normal, fixed to z in v1 (a PCB
trace is a film). The path is a polyline in the xy plane; each segment
is a rectangle of the given width, bends are mitred (sharp outer and
inner corners), the union is the trace polygon. A cell whose centre
lies inside the polygon is metal; a cell the boundary crosses is a cut
cell with sampled fill and bins. `z_m` must be commensurate with the
pitch in v1 (section 3.6). Rounded or chamfered bends are more path
points; the painter does not know about bends.

### 3.3 Union rule (traces, blocks, cylinders, ports)

A cell is metal if any primitive claims it. A section cut applies to a
cell only if no primitive claims the WHOLE cell: a trace ending inside
a pad block never carves the pad, two traces crossing inside shared
metal stay solid, and an axis-aligned run of a trace is cell-for-cell
identical to a block (the bit-identity check of phase 1). Two section
shapes in one cell (a bend, two traces meeting, a trace over a
cylinder end) are the union at the sampling level, so nothing special
happens. Conductivity: one sigma per trace; where primitives of
different sigma overlap the later one wins, as blocks do today.

Ports go on axis-aligned faces, which for a diagonal trace means on the
pad or block the trace ends in. That keeps the equipotential terminal
single-axis and the "full cells under every port face" rule as it is.
Trace ends that carry no pad are closed by the end plane and get no
port; a user who wants a port on a bare diagonal end has the mixed-
orientation face port, which phase 4 decides to keep or drop.

### 3.4 Resistance (stage A): the FACE rule (revised in phase 1)

The plan's first version kept the cylinder's 1/fill-per-cell rule on
every orientation and asserted DC would converge at second order
under it. Measured in phase 1, it does not: on a 45-degree edge every
in-plane link joins two cells of UNEQUAL fill, the half-cell series
rule charges each such link an O(1) excess, and the ladder's DC ratio
went 1.144 / 1.065 / 1.034 at 4/8/16 across -- first order, and worse
than the staircase at 16. The cylinder and the slab never see this
because their fills are constant along the current.

The rule that is right for a tilted cut: a filament takes the
conductance of the FACE it crosses, the metal fraction of the shared
cell face (sampled from the shape union at paint time,
`cut['faces'][axis]`), with bulk impedance density on both sides. For
a uniform flow at any angle the link currents are the face fluxes and
the dissipation sums to the exact area to O(h^2). Measured: DC R
ratio 1.0093 / 1.0008 / 1.0007, DC L 0.9997 / 0.9971 / 0.9992.
Filaments along the invariance axis keep the cell rule (the face IS
the cell's section there, so the cylinder is unchanged); a face with
a whole cell on either side is whole; a face with no metal is capped
at 1e-3 of bulk.

### 3.5 Inductance (stage B) through the cut

The cylinder corrects filaments ALONG the invariance axis over a
transverse split. A trace carries current ACROSS the cut, on x and y
filaments, and the plane slices those filaments obliquely, so the
weights must resolve the AXIAL direction too: a split `(kx, ky, 1)` for
an x filament (kx along its length, over the two half-cells it spans,
ky across), weights = metal fraction of each sub-prism from the shape
union, normalised. `Split` already supports axial subdivision;
`_pair_correction` gets per-pair weight vectors instead of a few
distinct ones (the cylinder's memo by weight triple is replaced by
tables per separation and a k^2 product per pair; for a 10 mm trace at
6 um pitch that is ~4e5 pairs, seconds).

### 3.6 Skin (the edge family) -- as built in phase 3

`EdgePalette` (enrich.py): per-entry weights on the x and y filaments
whose end cell lies within `reach` section-plane steps of a partial
cell, on a `(k, k)` transverse split (in-plane x through-thickness),
columns `exp(-p d)` and its two tangential partners with d the signed
distance to the section boundary at the sub-prism centroid (the
filament's midpoint), fill-weighted and net-zero pruned as the
cylinder's surface palette. Built as a subset `Enrichment` (explicit
`sel`, aggregates found by neighbour search) and stacked by
`ModeStack` on the shared section family, which `build()` now
constructs for BOTH in-plane orientations on a trace model (the port
axis only before: half the current carried no modes).

The two design points the plan left open, both measured (phase 3
log): the edge family REPLACES the face-anchored shared modes on its
entries rather than adding to them (`Enrichment(exclude=...)`), and it
carries no column for the exposed faces along the section axis (the
shared families own those). `enrich = "auto"` includes the edge family
on a trace model whenever the section family engages;
`families = ["section", "edge"]` asks for it; `"edge"` on a model with
no trace raises. The distance along the filament is not resolved
(an axial split would be the next refinement; not needed at h/delta
<= 4 by the measurements).

Not done: the plan's 2-D Galerkin referee. The direct ladder (the
dogleg against its own converged value) was cheaper and answered the
same question with the real operator; the referee pattern stays
available if the axial question is ever opened.

### 3.7 What v1 refuses

* A section cut and a slab cut in one model (a trace whose z extent is
  off-grid). Both cuts in one cell need a 3-D bin pattern and a
  laminate-times-section resistance rule; deferred to section 8.
* Invariance axis other than z for traces.
* A trace and a cylinder with different invariance axes.
* Equipotential port faces on cut cells (existing rule, kept).

## 4. Interface changes

* `[[trace]]` as in 3.2; keys `path_m`, `width_m`, `z_m`, `sigma`,
  `film`, `name`; `_TOP` and the key whitelist updated; the doctrine
  gains a rule.
* `[[cylinder]]` keeps its keys; its record changes kind (internal).
* `[solve] enrich` unchanged: the edge family engages with the section
  family (auto) or by `families = ["section", "edge"]`; `"edge"` is
  refused on a model with no section cut.
* No compatibility shims: the `cylinder` cut kind is gone from every
  reader (`impedance_scale`, `partial_dL`, `Enrichment`, `_EquiSweep`,
  validators).

## 5. Validators

* NEW `validate_trace.py` (the ladder as a gate): (A) an axis-aligned
  trace is bit-identical to the equivalent block, R, L and every
  matrix; (B) sampled fill sums to w*L within 1e-3 at 45 and 30
  degrees; (C) the 45-degree ladder at nw = 4, 8, 16 through the TOML
  path with thresholds set per phase (section 7); (D) the dogleg with
  pads, both port paths, R within the aligned bar's DC bound; (E) union:
  a trace ending inside a pad leaves the pad's cells whole.
* `validate_partial.py`: the cylinder parts move to the section-cut
  record (geometry, bins, stage B, Kelvin razor, surface palette);
  no numbers change, the record does.
* `validate_enrich.py`: one part for the edge palette (net-zero,
  support mask, weights at two frequencies) and the two-orientation
  shared family (bit-identical when the model is a plain bar).
* `validate_port_impedance.py`: mixed-orientation port part stays or
  goes with phase 4's decision.

## 6. Phases

Each phase: build, run the touched validators, run the full gate before
the commit that closes the phase (~100 min with the corpus linked; one
heavy job at a time, detached launch, monitor by PID), append the
ledger and the phase log here.

### Phase 0: baseline and instrument

* Record src/validation/studies line counts; full gate green.
* Fold scratch/diagonal_bar.py's rasteriser and ladder into the
  skeleton of `validate_trace.py` part C, reading a TOML that does not
  parse yet (the part is marked pending until phase 1). Nothing else.

### Phase 1: geometry

* `cut kind='section'` with the shape list; the cylinder painter
  rewritten on the shared sampler; `[[trace]]` painter (polygon from
  path, mitred; centre-inside; sampled fill and bins on boundary
  cells; union rule); input validation of 3.7.
* `impedance_scale`, `partial_dL` (cylinder branch), `Enrichment`,
  `_EquiSweep` read the new kind. `SurfacePalette` distance from the
  shape union (circle unchanged in value).
* Gate: validate_partial bit-identical on the cylinder parts;
  validate_trace A, B, E; C at DC only (thresholds: R ratio at 16
  across <= 1.015, i.e. the area term gone).
* Expected ledger: roughly flat (the cylinder painter shrinks, the
  trace painter grows).

### Phase 2: inductance through the cut

* Axial-split stage B on x and y filaments of cut cells; per-pair
  weights; pair cap semantics unchanged.
* Gate: dL exactly symmetric, zero on whole-whole pairs, one near pair
  against first principles (as validate_partial's cylinder part does);
  validate_trace C at DC: L ratio at 8 across <= 1.005, 16 across
  <= 1.002.

### Phase 3: skin

* Referee first: a zero-truncation Galerkin referee on the 45-degree
  bar's edge cells (the pattern of the retired studies/mode_referee.py)
  to measure how far the anchored exponentials track a fine sub-bar
  truth as h/delta grows; the number sets phase 3's threshold and is
  the go/no-go for the edge family. Half a day. If the referee says the
  modes do not carry past h/delta ~2, the phase ships the two-
  orientation shared family only and records the finding.
* `EdgePalette`; `build()` stacks it; two-orientation shared family;
  `resolve()` radii.
* Gate: validate_enrich new part; validate_trace C at 1e8 and 1e9 with
  the referee-set thresholds (target from the discussion: 1e9 R ratio
  at 16 across from 1.20 to below 1.05, order >= 1.5).

### Phase 4: connections

* The dogleg example `examples/diagonal_trace.toml` (x pad, 45-degree
  run, x pad, ports on the pads, both port paths) and a 30-degree
  variant in the validator.
* Decide the mixed-orientation face port by measurement: if a bare
  diagonal end port and a pad port agree on the dogleg within the DC
  bound, the bare-end port has no user and its code (commit 6039f9b's
  +18 lines) goes; else it stays as the bare-end port and gets a
  doctrine line.
* Doctrine and examples/README updated; the study in scratch/ retired
  in favour of validate_trace.

### Phase 5 (deferred, not scheduled)

* Section + slab coexistence (3-D bins, laminate x section rule).
* Sub-bar series-parallel resistance on cut cells.
* Traces with normal x or y (vertical traces in a stack).

## 7. Gate thresholds by phase (validate_trace part C, 45 degrees)

    phase   metric                 nw=8      nw=16
    1       DC R ratio             <= 1.01   <= 1.005   (face rule; was 1.06 / 1.015)
    2       DC L ratio             <= 1.005  <= 1.002
    3       dogleg R at 100 MHz vs its converged value (validate_trace G,
            equipotential, enrich auto): within 4% at 4 and 8 across
            (h/delta 3.8 and 1.9), solve converged. The 1e9 rows of
            part C run on the LpR path, which carries no modes, and
            are informational.

## 8. Beyond this program

Strip-shaped elements on the wire path (arbitrary-orientation prisms,
sub-filaments across width and thickness in the local frame, a trace
primitive that feet into voxel pads) are the fix for h/delta >> 5 and
for the rotated square loop. They compose with the section cut: the
pads and junctions of a strip trace are voxels with tilted edges.
Separate plan when this one closes.

## 9. Ledger

    phase   date         src     validation   studies   note
    base    2026-09-04   24707   12150        9593      45 validators; src includes 6039f9b's mixed-orientation port
    0       2026-09-05   24707   12471        9593      validate_trace.py skeleton (+321)
    1       2026-09-05   24985   12486        9593      section.py 260 (pieces, field, painter, face fills); the cylinder painter left sppeec_input; enrich/voxmodel/port_impedance small
    2       2026-09-05   25043   12529        9593      _plane_weights + per-pair contraction (+50), stage B on the LpR path (+8); validate_trace F (+43)
    3       2026-09-05   25178   12575        9593      EdgePalette (+75), two-orientation shared family + exclude + the edge hook in build (+45), resolve rules; validate_trace G (+46)
    4       2026-09-05   25188   12582        9593      engagement threshold (comments), validate_trace D at two angles; examples/diagonal_trace.toml

## 10. Phase log

### Phase 0 (2026-09-04/05)

* Baseline recorded (ledger). Full gate on the idle box: 45 pass / 0
  skip / 0 fail; anchors setup1/2/3 bit-identical to the runner's
  2026-09-03 values.
* `validation/validate_trace.py` written in full (parts A-E, the
  per-phase gate table of section 7 as `GATE`, `PHASE = 0`). It SKIPs
  with "[[trace]] not parsed yet" until phase 1, which the runner
  counts as not tested. Its TOML docs are generated (a numpy-float
  repr bug in them was caught by loading all three through tomllib).
  Part C attaches the staircased end-cut port programmatically, the
  scratch ladder's rule, so the phase-1+ numbers are comparable with
  the baseline table row for row. validation 12150 -> 12471 (+321,
  the new validator).

### Phase 1 (2026-09-05)

* `src/section.py`: convex pieces (`trace_pieces`: rectangles per
  segment, mitre quad or bevel per bend), `field` (signed distance +
  outward-normal angle of the union), `lattice` (row-shifted sample
  lattice), `paint` (SDF classification of every cell, sampled fill and
  bins on boundary cells, union rule, block-whole cells untouched,
  record dropped when nothing is partial, FACE fills). The cylinder
  painter in `sppeec_input` is gone; both primitives go through
  `paint`. `cut['kind'] == 'section'` everywhere (`partial_dL`,
  `_surface_geometry` now reads the shape union, `Enrichment`,
  `resolve`, `_EquiSweep`).
* `[[trace]]` (`path_m`, `width_m`, `z_m`, `sigma`, `film`, `name`);
  refusals of 3.7 in place (off-grid z, section + slab, mixed axes,
  wires).
* Three findings, all measured on the ladder (validate_trace C):
  1. The unshifted 64 x 64 sample lattice over-counts a 45-degree edge
     by +1/(4s) per cell (every lattice diagonal flips at once): fill
     area 1.0025 / 1.0013 of w*L at 4/8 across. Rows shifted by the
     golden fraction (`section.lattice`) -> 1.0000 at both angles.
  2. A prescribed-current port on sliver faces: equal shares through
     1/fill terminal resistances took DC from 1.03 to 1.09 and deep
     skin from 1.20 to 2.13. The terminal now takes per-face sigma
     (`port_sigma_faces`, `port_impedance.terminal_impedance`), and
     the ladder's end-cut port sits on the staircase's own cells
     (fill >= 1/2); slivers beyond it are dead-end stubs.
  3. THE ONE THAT CHANGED THE DESIGN (section 3.4): the per-cell 1/fill
     rule is first order on a tilted cut. Face rule shipped:

         cells across        4        8       16
         DC R, staircase   1.420    1.132    1.029
         DC R, cell fills  1.144    1.065    1.034
         DC R, face rule   1.0093   1.0008   1.0007
         DC L, face rule   0.9997   0.9971   0.9992
         1e9 R, face rule  1.556    1.735    1.967   (phase 3's problem)
         1e9 L, face rule  0.984    0.983    0.991

     Deep skin is WORSE than the staircase (1.20 at 16) under any
     fill rule without modes: the lattice's outermost layer is now the
     partial cells, the skin current crowds into them, and their
     conductance is a fraction of bulk. That is the lossy-shell
     artefact the edge palette exists to remove (phase 3); the
     cylinder shows the same base behaviour and its surface palette
     fixes it (validate_partial: Kelvin within 2.5% at dx/delta 2).
* The dogleg with pads (validate_trace D): DC R within 0.09%
  (prescribed) and 0.72% (equipotential) of the aligned bound; the
  union rule (E) holds.
* Gate: validate_trace A-E green with the phase-1 DC thresholds
  (1.01 / 1.005); validate_partial green after its stage-B strip was
  taught to keep the face fills; full gate 46 pass / 0 skip / 0 fail, anchors setup1/2/3 bit-identical.
* Ledger: src +278 over phase 0 (section.py 260, less the painter it
  replaced); the plan's "roughly flat" was optimistic by ~200 lines,
  the face rule and the SDF classification being the additions.

### Phase 2 (2026-09-05)

* `partial_dL` on a section cut now corrects the in-plane orientations
  too: each filament through a partial cell gets its own sub-prism
  fills on a `(k, k, 1)` split (k along its length, over the two
  half-cells it spans; k across in the plane; the section axis whole),
  sampled 8 x 8 per sub-prism from the shape union (`_plane_weights`).
  `_pair_correction` contracts per pair when the weight rows are many
  (a tilted cut: one row per filament), and keeps the matrix-over-
  unique-rows memo when they are few (cylinder, slab).
* FOUND: stage B was attached on the LpPR and equipotential paths
  only. The plain `LpRSolver` + `impedance_matrix` path -- the one the
  ladder and the scratch study drive -- never carried it, for
  cylinders either. `impedance_matrix` now builds it once per solver
  and `_apply_Z` adds `jw dL i`. validate_partial's LpR-path numbers
  moved at the 1e-4 level (Kelvin razor 3.1845 -> unchanged to 4
  digits).
* MEASURED, 45-degree bar at 8 across (validate_trace F and the
  diagnostic), diagonal/aligned:

         f        with B: R        L      | without: R        L
         1e3      1.000808   0.997192     | 1.000808   0.997131
         1e8      1.218540   0.990248     | 1.222532   0.989980
         1e9      1.729056   0.983738     | 1.734824   0.983415

  Stage B through the cut is real but SMALL on a trace: +0.006% in L
  at DC, +0.03% at 1 GHz, -0.4% in R at 100 MHz. With the face rule
  the edge links carry current in proportion to their face fill, so
  the I^2 dL of the partial cells is second order in the fill; the
  cylinder needed stage B (2.4% -> 0.9% in L) because its rim cells
  carry the full axial current. The remaining -0.3% in L at 8 across
  (-0.08% at 16) is not a partial-cell inductance effect; the
  terminal's neglected mutual to the interior (port_impedance
  docstring) on a staircased end cut is the likely owner, and it
  converges with the pitch.
* Gate: validate_trace A-F green at the phase-2 thresholds (DC L
  <= 1.005 / 1.002, met at 0.9971 / 0.9992); validate_partial green;
  full gate 46 pass / 0 skip / 0 fail, anchors bit-identical.
* Ledger: src +58. Kept, flagged: the in-plane stage B costs ~50 lines
  for a 1e-4 effect on traces; it is the consistent physics and the
  cylinder's transverse filaments now get it too, but it is the first
  candidate to strip if the count must come down.

### Phase 3 (2026-09-05)

The instrument. The bare bar's mixed-orientation end cut lives on the
prescribed-current LpR path, which carries no modes, so deep skin
was measured on the DOGLEG (x pad 75 um, 45-degree run ~1.6 mm, x
pad) on the equipotential path, against the dogleg's own converged
value: the plain basis at 48 cells across (h/delta = 0.32) at 100 MHz
(delta = 6.6 um). The straight bar is not the target here -- the two
bends cost real resistance in the skin regime, and the shared family
on the straight bar is itself unreliable above h/delta ~6 (its 1 GHz
R moved 50% between 4 and 8 across). The plain ladder 16 / 24 / 32 /
48 across read 2.341 / 2.278 / 2.318 / 2.306 e-2 ohm (a +-1%
staircase-parity wobble), L 1.153 nH. An earlier ladder with pads of
6 CELLS was not one geometry (the pads shrank with the pitch, 2.5% of
R between 16 and 48 across); `validate_trace.PAD_M` fixes the pad in
metres.

Steps, dogleg R at 100 MHz as a fraction of the converged 2.306e-2:

    cells across (h/delta)     4 (3.8)    8 (1.9)    16 (0.95)
    plain basis                 0.609      0.960      1.015
    shared family, x only       (raises before phase 3)
    shared families x + y       1.094      1.057      stall (331 mv)
    + edge family, stacked      1.121      1.068
    + edge family, REPLACING    1.012      1.029      stall (331 mv)
    + z-face column (replacing) 1.143      1.058

* The face-anchored shared modes converge to the STAIRCASE answer
  (the perimeter sqrt(2) too long): +9% / +6%, worse with refinement
  in the wrong direction. Stacking true-edge modes on top of them
  does not help (+12% / +7%): the solve keeps both. Replacing them on
  the edge cells is the design: +1.2% / +2.9% at h/delta 3.8 / 1.9,
  where the plain basis is -39% / -4%. A column for the exposed z
  faces in the edge palette is worse again; the shared families own
  those faces.
* ENGINE FINDING, out of scope, recorded: at h/delta ~ 1 (16 across
  at 100 MHz) the shared section family stalls at the 331-matvec cap
  on the straight bar (one family; R came out NEGATIVE) and on the
  dogleg alike. The engagement rule (2 dx / delta > 1) engages there
  with k = 7. The plain basis is within 1.5% at that pitch, so the
  fix is a rule, not a basis; it belongs to the enrichment engine's
  docket, with this measurement.
* `EdgePalette`, `Enrichment(exclude=)`, the two-orientation shared
  family, `resolve` rules (`edge` auto on traces, refused without a
  trace, dropped with the section family below engagement),
  `_surface_geometry` unchanged; `SPPEEC_EDGE_*` experiment flags
  removed after the measurement.
* validate_trace G: dogleg plain and auto at 4 and 8 across against
  the recorded reference, 4% band, convergence checked; the 1e9 rows
  of C are informational (LpR path). Gate: full gate 46 pass / 0 skip / 0 fail, anchors bit-identical.
* Ledger: src +135 (EdgePalette 75).

### Phase 4 (2026-09-05)

* `examples/diagonal_trace.toml`: the dogleg at 12.5 um pitch (8
  across), equipotential port on the pads, `enrich = "auto"`, sweep
  1e5..1e8. 107 x 103 x 4 cells; wall 8:16, peak RSS 7.4 GB on the
  12-core box; R 6.108e-3 / 6.161e-3 / 8.531e-3 / 2.372e-2 ohm, L
  1.256 / 1.254 / 1.219 / 1.158 nH at 1e5 / 1e6 / 1e7 / 1e8, all
  points converged (30 / 37 / 56 / 170 matvecs) -- after the rule
  change below; before it the 1e7 point stalled.
* validate_trace D now runs the dogleg at 45 AND ~30 degrees (the
  run's projections snapped to whole cells), both port paths: DC R
  within 0.09% / 0.72% (45) and 0.08% / 0.76% (30) of the aligned
  bound.
* THE MIXED-ORIENTATION FACE PORT STAYS. The plan's default was to
  drop it if a pad port made it redundant; the bare-end port and the
  pad port agree at DC (1.0008 vs 1.0009 of the aligned bound at 8
  across), so it IS redundant for users -- but validate_trace C, the
  rotation-invariance ladder with an exact reference, is built on it
  (the pad geometry has bends and no exact reference), and that ladder
  is the program's cleanest gate. 18 lines, validated by
  validate_port_impedance; kept for the instrument, not for users. The
  doctrine does not advertise it.
* THE ENGINE STALL, FIXED (a doctrine amendment, flagged): the
  example's 10 MHz point stalled at the 331-matvec cap (residual
  0.98) -- dx/delta = 0.6 there, the same stall as the 16-across bar
  at 0.95 in phase 3 and the equibar's low points before the
  enrichment plan's retune fix. The engagement threshold moves from
  2 dx > length to dx > length, for both the build (`resolve`) and
  the per-frequency retune (`Enrichment.set_frequency`); k is
  unchanged. Evidence: between dx/length 0.5 and 1 the pruned face
  exponentials are nearly degenerate (the enrichment plan's own
  finding at 0.5) and the plain basis is within ~2% there
  (validate_partial's Kelvin razor at dx/delta 1: 1.1%), so the modes
  buy nothing and cost a stall. Verified: validate_superconductor,
  validate_equiterminal, validate_enrich, validate_aniso,
  validate_input_lppr green; equibar converges at every point (13 /
  23 / 28 / 112 / 202 matvecs, the 1e8 and 1e9 points still engaged
  and unchanged); the example's 1e7 point 331 -> 56 matvecs. The
  16-across straight bar of phase 3 now takes the plain basis at
  dx/delta 0.95 by the same rule.
  The scratch instruments (scratch/diagonal_bar.py, trace_skin.py)
  are superseded by validate_trace C and G and stay local.
* Gate: full gate 46 pass / 0 skip / 0 fail, anchors bit-identical.
* Phases 0-4 complete; phase 5 (section + slab in one model, the
  sub-bar edge resistance, traces with normal x or y) stays deferred.
