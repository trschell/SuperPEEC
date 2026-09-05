# Diagonal traces: the section-cut program

Status: PLAN, 2026-09-04. Nothing built. Phases run one at a time on
the user's go, each closed by the validator gate and a ledger entry,
as docs/enrichment_plan.md was.

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

### 3.4 Resistance (stage A)

The 1/fill rule on every orientation, exactly the cylinder's, in v1.
It is direction-blind (charges a filament along a thin sliver the same
as one across it); DC already converges at second order under it, and
the sub-bar series-parallel network is the contained refinement if
phase 1 shows DC on edge cells limiting anything. Not planned.

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

### 3.6 Skin (the edge family)

An `EdgePalette` beside `SurfacePalette`: per-cell weights on the x and
y filaments of cut cells (and within `reach`), sub-prism split
`(kx, ky, kz)` = axial x transverse-in-plane x film-normal, weight
columns `exp(-p d)` with d the signed distance to the section boundary
at the sub-prism centre, times the film profile along z when the trace
declares `film`, fill-weighted, net-zero pruned as today. Built as a
subset `Enrichment` (explicit `sel`, `agg` found by neighbour search as
the corner family does) and stacked by `ModeStack` on the shared
section family. Two changes to the shared family fall out and are
needed regardless of traces: `build()` constructs it for BOTH in-plane
orientations when the film normal is declared (today: port axis only,
so on a diagonal half the current carries no modes), and `resolve()`
picks the section-cut radii (3, 4) for the new kind as it did for
`cylinder`.

Whether the modes carry the tilted layer beyond the round wire's
h/delta of 3 to 4 is the program's open question, and phase 3 measures
it with a referee before assembling anything.

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
    1       DC R ratio             <= 1.06   <= 1.015
    2       DC L ratio             <= 1.005  <= 1.002
    3       1e9 R ratio            referee-set; target <= 1.10, <= 1.05
    3       1e9 L ratio            <= 1.005  <= 1.002

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
