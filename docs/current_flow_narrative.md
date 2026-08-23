# Where the current goes: SuperPEEC and VoxHenry at high frequency

An engineering comparison of two voxel PEEC codes, judged on whether
they put the current in the right place when the mesh is too coarse to
resolve the skin depth — and on what each one's extra machinery is
actually for.

Measured 2026-08-23. Harness `studies/skinnarr.py`, tables
`studies/skinnarr_report.py`, current maps `studies/skinnarr_profile.py`,
raw data `studies/skinnarr_results.json`.

---

## How to read this

**The question is engineering accuracy, not digits.** Getting within a
few percent of the truth on a mesh that cannot possibly resolve the
physics is a real achievement, and that is the bar used throughout. No
result here is quoted past three significant figures, and every
"truth" carries an uncertainty from its own convergence ladder.

**Ground truth** comes from mesh refinement in SuperPEEC's *plain*
piecewise-constant basis — deliberately the dumbest option, so nothing
under test is assumed. Each problem was solved at 3–5 resolutions and
the truth is the finest rung, with the last inter-rung step quoted as
the uncertainty. Where that uncertainty exceeds a few percent the row is
marked and no verdict is drawn from it.

**The controlling parameter is dx/δ** — cell size over skin depth. Below
1 the mesh resolves the current layer and everything works. Above 1 the
mesh cannot, and whatever the code gets right has to come from its basis
functions. That is where the two codes separate.

**The three arms**, all on the *same* mesh and the same input file:

| arm | what it is |
|---|---|
| `plain` | SuperPEEC, one current unknown per cell per direction |
| `VoxHenry` | VoxHenry as shipped: **five** unknowns per voxel — Jx, Jy, Jz, **J2d, J3d** |
| `engine` | SuperPEEC + conduction modes: exponential skin profiles anchored to cell faces and corners |

**All SuperPEEC numbers here are DEFAULT settings**, taken through the
documented user route: a TOML naming the `.vhr`, solved via
`sppeec_input` -- the same path the CLI runs -- with every skin knob
left unset so it takes its default. Those defaults resolve to

    skin.mode = auto   skin.basis = conduction   boundary_only = true
    skin.k    = auto, min(12, max(7, 2*dx/delta))   rc = width-scaled

**An earlier revision of this document reported something else.** Its
SuperPEEC column came from a hand-built `EquiTerminalSolver`, which
silently overrode three of those defaults -- most importantly
`boundary_only`, which the low-level API defaulted to `false` while the
user-facing path had defaulted it `true` since 2026-08-17. On that
configuration the engine appeared to overshoot by +70.9% at 100 GHz;
with the defaults it is -3.4%. Every SuperPEEC figure below has been
re-measured. A measurement of a hand-assembled configuration is not a
measurement of the product and should not have been written up as one.

Two code defects surfaced in the process and are fixed: the
`boundary_only` API default now matches the user-facing one, and
`[model] vhr = "..."` -- the documented, schema-blessed way to run a
VoxHenry file -- previously raised `TypeError` because it handed the
solver an unread model. That second bug is why studies hand-built
solvers at all.

**One methodological trap, paid for during this study.** Comparing
VoxHenry against a truth ladder built with a *different port model*
produced a spurious 2.6% "disagreement" on numex1. SuperPEEC has two
port treatments — prescribed-current (`LpRSolver`) and equipotential
terminal (`EquiTerminalSolver`) — and they differ by a few percent,
which is the same size as the effects being measured. Every number in
this document uses the equipotential terminal throughout, on files both
codes read identically. Re-run consistently, VoxHenry and SuperPEEC-plain
agree on numex1 at 2.5 GHz to six digits (0.011236 both). **A
cross-code comparison is only worth the port model it holds fixed.**

---

## The headline

At **dx/δ ≈ 1.5** (10 GHz on a 1 µm mesh), error in R against converged truth:

| problem | truth ± | plain | VoxHenry | SuperPEEC (defaults) |
|---|---:|---:|---:|---:|
| bar, 2 cells across | ±0.5% | −12.5% | **−12.5%** | **−0.7%** |
| bar, 4 cells across | ±0.6% | −9.9% | **−9.9%** | **+0.5%** |
| round wire (numex2) | ±0.0% | −5.5% | **−5.5%** | **+7.2%** |
| hairpin (turning) | ±1.7% | −10.0% | −10.6% | **−0.5%** |
| numex1, 20 across | ±0.6% | −2.8% | **−2.8%** | **−2.0%** |

Two findings dominate everything below.

**1. On straight conductors VoxHenry's answer is indistinguishable from
a plain piecewise-constant solver** — not close, *identical to four or
five significant figures*, at every mesh and every frequency tested.

**2. SuperPEEC's conduction modes are the only thing in this study that
converts a coarse-mesh answer from unusable to engineering-grade.** On
rectangular sections they hold **within about 2% out to dx/delta ~ 2.4**,
where the plain basis is 24-36% low, on the SAME cells. The striking
case is the 2-cell bar at 100 GHz: -63.4% plain against **-0.5%** with
the defaults, from two cells across the section. The one geometry where
they genuinely misfire is the staircase round wire, +7 to +16%.

---

## Problem 1 — the straight rectangular bar, 2 cells across

A 10 µm copper bar of 2×2 µm section on a 1 µm mesh: 40 voxels, two
cells across. This is squarely inside VoxHenry's own design point (its
shipped examples run 1–4 cells across).

The plain basis returns **exactly the DC resistance at every frequency
up to 100 GHz** — 0.0431034 Ω at 1 Hz and 0.0431034 Ω at 1e11 Hz, to
every digit. This is not a convergence failure, it is a structural one:
with 2×2 cells all four are equivalent by symmetry, so a piecewise
constant current has no freedom to crowd anywhere. The basis cannot
represent skin effect at all. By 10 GHz the true resistance has risen
15% above DC and the plain answer has not moved.

**VoxHenry returns the same number.** Its five-unknown basis produces
0.0431034 Ω at 100 GHz too. Reading its saved solution vector confirms
why: the J2d and J3d coefficient blocks have norms of 6.6e-14 and
4.3e-14 against 0.57 for Jx — they are numerically *zero*. On this
geometry the extra unknowns are not merely ineffective, they are never
excited.

SuperPEEC's conduction modes do what the mesh cannot. They place
exponential skin profiles *inside* each cell, so the current can crowd
toward a face without the mesh knowing. That takes the error from
−12.5% to **−0.7%** at 10 GHz, −35.8% to **−1.7%** at 25 GHz, and
−63.4% to **−0.5%** at 100 GHz, on the identical 40-voxel mesh. In
engineering terms the same model goes from useless to good enough to
design with — and it stays there as frequency rises, which is the part
worth noticing: the plain error grows without bound while the enriched
error does not.

A subtlety worth stating because it will trip up anyone who plots
fields: the cell-averaged current map for the mode-enriched solve is
*still perfectly uniform* — crowding factor 1.000, same as plain. The
redistribution happens below cell resolution, so a per-cell current plot
cannot see it. Only R reveals it.

## Problem 2 — the straight bar, 4 cells across

Same story with the symmetry constraint relaxed. At 4 cells across the
plain basis can now put more current in the outer ring than the inner
one, so it captures part of the effect on its own — the cell-averaged
crowding factor reaches 1.15 against the converged 1.71, and the error
at 10 GHz is −9.9% rather than −12.5%.

VoxHenry again lands on −9.9%, matching plain to four digits across the
whole ladder (m1 through m4, seven frequencies). Its enrichment blocks
are now non-zero but small: (J2d+J3d)/Jx = 0.033, about 3%.

The conduction modes hold at **+0.5%** at 10 GHz and **+0.2%** at
25 GHz, degrading to +6.1% only at 100 GHz (dx/δ = 4.79). Notice that
the mode-enriched error is essentially the *same* at 2 and 4 cells
across while the plain error improves with resolution — the modes are
doing the work the mesh would otherwise have to do, so they are least
dispensable exactly where meshing is most expensive.

## Problem 3 — the round wire, VoxHenry's own numex2

A 50 µm long, 10 µm diameter copper wire on a 1 µm mesh: 4000 voxels,
staircase-discretised circle. This is VoxHenry's shipped example, run
from its own generator (`gen2_param.m`, verified to reproduce the
shipped file's occupancy exactly).

Here the mesh is genuinely helpful — ten cells across a round section
lets a piecewise-constant basis form a real surface layer, and the
crowding factor reaches 2.12 against the converged 3.28. Plain and
VoxHenry both land at **−5.5%** at 10 GHz, which for a design-stage
number is defensible. Both are −18.4% by 25 GHz. Once again the two
agree to four digits at every rung.

**This is the one geometry where SuperPEEC's conduction modes make
things worse.** At defaults they read +3.4% at 2.5 GHz, **+7.2% at
10 GHz**, +7.6% at 25 GHz and +15.8% at 100 GHz — an overshoot
throughout, where plain is 0.5–48% low. The modes are exponentials
anchored to *cell faces*, exactly right for an axis-aligned rectangular
boundary and wrong for a staircase approximation of a circle, where the
true surface cuts diagonally through cells; the engine crowds current
toward the wrong surfaces.

Two things make this the most interesting failure in the study. It is a
property of the METHOD, not of a setting — it survives every default
being correct, unlike the rectangular-section overshoots an earlier
revision reported, which were a configuration error. And it errs HIGH,
overstating loss, where the plain basis errs low and can be bounded. At
10 GHz an engineer choosing between them would be better served by the
plain basis (−5.5%) than by the modes (+7.2%), which is not true
anywhere else in this study.

One honest caveat about this problem's ground truth: refining a
staircase circle *changes the geometry*, because the set of cells whose
centres fall inside the circle is re-chosen at each resolution. The DC
resistance therefore wobbles ±0.7% between rungs rather than converging
monotonically, and the extrapolation is unreliable even where the value
is stable. At 10 GHz the two finest rungs agree to four digits so the
truth is solid; at 100 GHz the uncertainty is ±12% and no verdict is
drawn.

## Problem 4 — the hairpin, where the current has to turn

Two parallel arms joined at the far end, forcing the current through two
right angles. This was added specifically to test what VoxHenry's extra
unknowns are *for*, after the straight-conductor results showed them
idle.

**They are for turning current, not for skin depth.** The enrichment
ratio (J2d+J3d)/Jx jumps from 1.9e-13 on the straight 2-cell bar and
0.017–0.033 on every other straight run (numex1 0.017, the round wire
0.018, the 4-cell bar 0.033) to **0.369** on the hairpin — a factor of
twenty, and the only geometry in the study where it moves. The J2d block alone goes from numerically zero to
0.156 against Jx's 0.450.

And where they engage, they buy something real: at DC, VoxHenry's error
is **+0.2%** against plain's +0.9%. That is a genuine improvement in
corner resistance, in the direction its design intends, and it is the
only place in this study where VoxHenry's basis measurably beats a plain
solver. It is worth about 0.7 percentage points.

At high frequency the picture reverses slightly — VoxHenry is −10.6% at
10 GHz against plain's −10.0%, i.e. marginally worse — because the
enrichment does nothing for skin crowding while adding its own
discretisation. The conduction modes, meanwhile, give **−0.5%** at
10 GHz, the best result on any problem in the study.

---

## Problem 5 — VoxHenry's flagship numex1, on its own shipped mesh

The 30 µm straight conductor of 10×10 µm section on VoxHenry's own
0.5 µm mesh — 24 000 voxels, twenty cells across. This is the file
VoxHenry leads with, and the fair question is whether the numbers it
publishes are right. Truth here is this campaign's own refinement to
0.25 µm (192 000 voxels) with the port model held fixed throughout.

| f | dx/δ | R_true | plain | VoxHenry | **SuperPEEC (defaults)** | R/R_dc |
|---|---:|---:|---:|---:|---:|---:|
| 1 GHz | 0.24 | 0.00780181 | −0.2% | −0.2% | −0.2% | 1.51 |
| 2.5 GHz | 0.38 | 0.0113059 | −0.6% | −0.6% | −0.6% | 2.19 |
| 10 GHz | 0.76 | 0.0205872 | −2.8% | −2.8% | **−2.0%** | 3.98 |
| 25 GHz | 1.20 | 0.0310657 | −7.0% | −7.0% | **−5.1%** | 6.01 |
| 100 GHz | 2.39 | 0.0559391 | −23.4% | −23.3% | **−3.4%** | 10.81 |

All three arms use the **same 0.5 µm mesh and the same 24 000 voxels** —
the only difference is what each puts inside a cell. (At DC and 100 MHz
all three are exact to the digits shown; those rows are dropped.)

The modes are worth little across VoxHenry's advertised band, because
there the mesh is already doing the job: at dx/δ = 0.76 they improve
−2.8% to −2.0%, which is inside the truth's own ±0.6%. **Their value
appears only where the mesh gives out** — at 100 GHz, dx/δ = 2.39, they
turn −23.4% into **−3.4%**.

Note the non-monotonicity: −2.0%, then −5.1%, then −3.4%. The engine is
better at 100 GHz than at 25 GHz on this problem. Whatever sets its
accuracy is not a simple function of dx/δ, and this study does not
resolve what it is.

**Across VoxHenry's whole advertised band its published numbers are
sound**: exact at DC, −0.6% at 2.5 GHz, −2.8% at 10 GHz. For an
extraction that is a good result, and the reason is that the shipped
mesh is matched to the shipped frequencies — dx/δ never exceeds 0.76, so
the mesh resolves the current layer and the basis is never asked for
anything difficult.

The rows past 10 GHz are ones VoxHenry does not advertise and are shown
to locate the cliff: −7.0% at dx/δ = 1.2 and −23% at 2.4. That is the
same degradation curve as every other straight conductor here.

**And on this file the two codes are the same solver.** Across all seven
frequencies the largest relative difference between VoxHenry and the
plain arm is **2.2e-4**. On its own flagship example, VoxHenry's five
unknowns per voxel produce the answer of three.

## How much does SuperPEEC's enrichment actually buy?

Every row below is default settings on the coarse mesh, against the
plain-basis refinement ladder.

| problem | f | dx/δ | R_true | plain | VoxHenry | SuperPEEC (defaults) |
|---|---:|---:|---:|---:|---:|---:|
| bar2 | 1e+09 Hz | 0.48 | 0.043173 | -0.2% | -0.2% | **-0.2%** |
| bar2 | 2.5e+09 Hz | 0.76 | 0.043537 | -1.0% | -1.0% | **-0.1%** |
| bar2 | 1e+10 Hz | 1.51 | 0.049239 | -12.5% | -12.5% | **-0.7%** |
| bar2 | 2.5e+10 Hz | 2.39 | 0.067108 | -35.8% | -35.8% | **-1.7%** |
| bar2 | 1e+11 Hz | 4.79 | 0.11764 | -63.4% | -63.4% | **-0.5%** |
| bar4 | 1e+09 Hz | 0.48 | 0.011003 | -0.6% | -0.6% | **-0.6%** |
| bar4 | 2.5e+09 Hz | 0.76 | 0.012056 | -2.9% | -2.9% | **-0.1%** |
| bar4 | 1e+10 Hz | 1.51 | 0.019085 | -9.9% | -9.9% | **+0.5%** |
| bar4 | 2.5e+10 Hz | 2.39 | 0.027375 | -23.6% | -23.6% | **+0.2%** |
| bar4 | 1e+11 Hz | 4.79 | 0.047272 | -52.2% | -52.2% | **+6.1%** |
| bend4 | 1e+09 Hz | 0.48 | 0.046449 | -0.1% | -0.7% | **-0.1%** |
| bend4 | 2.5e+09 Hz | 0.76 | 0.056352 | -2.7% | -3.2% | **+0.2%** |
| bend4 | 1e+10 Hz | 1.51 | 0.10096 | -10.0% | -10.6% | **-0.5%** |
| bend4 | 2.5e+10 Hz | 2.39 | 0.14971 | -24.1% | -24.7% | **-0.4%** |
| bend4 | 1e+11 Hz | 4.79 | 0.25505 | -50.7% | -51.1% | **+26.5%** |
| wire10 | 1e+09 Hz | 0.48 | 0.015073 | -0.5% | -0.5% | **-0.5%** |
| wire10 | 2.5e+09 Hz | 0.76 | 0.021974 | -0.5% | -0.5% | **+3.4%** |
| wire10 | 1e+10 Hz | 1.51 | 0.040226 | -5.5% | -5.5% | **+7.2%** |
| wire10 | 2.5e+10 Hz | 2.39 | 0.061204 | -18.4% | -18.4% | **+7.6%** |
| wire10 | 1e+11 Hz | 4.79 | 0.11069 | -48.0% | -48.0% | **+15.8%** |
| numex1 | 1e+09 Hz | 0.24 | 0.0078018 | -0.2% | -0.2% | **-0.2%** |
| numex1 | 2.5e+09 Hz | 0.38 | 0.011306 | -0.6% | -0.6% | **-0.6%** |
| numex1 | 1e+10 Hz | 0.76 | 0.020587 | -2.8% | -2.8% | **-2.0%** |
| numex1 | 2.5e+10 Hz | 1.20 | 0.031066 | -7.0% | -7.0% | **-5.1%** |
| numex1 | 1e+11 Hz | 2.39 | 0.055939 | -23.4% | -23.3% | **-3.4%** |

**On rectangular sections the modes hold to a couple of percent well
past the point where the mesh has given up.** `bar2` at dx/delta = 4.79
-- two cells across a section nearly ten skin depths wide, at 100 GHz --
is -0.5% where plain is -63.4%. `bar4` and `bend4` behave the same way
out to dx/delta ~ 2.4. This is the regime the engine was built for and
it holds it comfortably.

**Three places it degrades, all visible above:**

* **The staircase round wire, +3 to +16%.** The modes are exponentials
  anchored to CELL FACES, exactly right for an axis-aligned rectangular
  boundary and wrong where the true surface cuts diagonally through
  cells. This is the one failure that is a property of the method
  rather than of a setting, and it errs HIGH -- overstating loss.
* **dx/delta ~ 4.8 on the widest sections**, where `bar4` reaches +6.1%
  and `bend4` +26.5%. There the sub-bar grid is trying to resolve a
  skin depth thinner than a fifth of a cell.
* **numex1's mid-band, -5.1% at dx/delta = 1.2**, slightly worse than
  its own -3.4% at 2.39. The engine is not monotone in frequency.

**The cost.** Roughly 1.5-4.4x the plain wall clock, the multiplier
shrinking as the problem grows (1.5x at 24 000 voxels). Against buying
the same accuracy with cells -- numex1 would need its 192 000-voxel
rung -- the modes are about an order of magnitude cheaper.

## What each code's extra machinery is actually for

This is the substantive conclusion, and it reframes the comparison.

VoxHenry carries five unknowns per voxel where a plain solver carries
three. That is a 67% larger unknown count, and on every straight
conductor tested it purchases **no measurable change in R at any
frequency**. The extra unknowns are not wasted, though — they are
solving a different problem. They represent 2-D and 3-D current flow
*within* a voxel, which is what a corner or a bend demands and what a
straight run never asks for. Measured against the incidence matrix, the
J2d and J3d blocks carry net current between nodes, so they are genuine
current-carrying basis functions rather than net-zero shape corrections;
they simply are not shaped for skin crowding.

SuperPEEC's conduction modes attack precisely the other problem. They
are exponential profiles in the skin depth, net-zero within a cell, and
they exist to let current crowd toward a surface at a resolution the
mesh cannot express. They are worth roughly 5–10× in accuracy on
rectangular sections at dx/δ ≥ 1.5, and they are actively harmful on
staircase-curved ones.

**Neither code does both.** A voxel solver that handled turning *and*
crowding would need both families, and nothing in this study prevents
that — SuperPEEC already composes the two (`ModeStack`), and the corner
work has the tabulated bend shapes; they have simply never been measured
together against VoxHenry on the same file.

## Where each excels, where each falls short

**VoxHenry excels** at being unfussy. It ran every file first time, at
every resolution, with no configuration and no failure modes — 0.7 s on
40 voxels, 20 s on 19 000. Its answers are stable and it never
overshoots. For a straight-conductor extraction where you can afford
dx ≈ δ, it is a perfectly sound tool and its answer will be within a few
percent.

**VoxHenry falls short** in that its headline feature does not do what
its presence suggests on the geometries most people extract. On a
straight conductor you are paying for five unknowns per voxel and
getting the accuracy of three. When the mesh is too coarse for the skin
depth, VoxHenry has no answer — its error at dx/δ = 2.4 is −18% to −36%
depending on the section, and the only fix available is a finer mesh.

**SuperPEEC excels** where the mesh is coarse and the section is
rectangular. -0.5% against -63.4% on the same 40 voxels at 100 GHz is
the largest effect measured here, and it lands exactly where refining
the mesh is most expensive: a two-cell-across model returning an
engineering-grade answer in a regime the plain basis cannot represent at
all. It also converges reliably -- the plain-basis ladder underpins
every truth value in this document.

**SuperPEEC falls short** on the staircase-discretised round wire, where
it overshoots 7-8% through the mid-band and +15.8% at 100 GHz while the
plain basis runs 5-48% low. The direction matters: an overshoot reports
more loss than exists, whereas an under-resolved plain solve is always
low and can be bounded. Nothing in the code warns that a curved section
sits outside what face-anchored modes represent well. It is also the
slowest arm on every problem here, and not monotone in frequency.

## Where the plain error goes as the mesh loses the skin depth

For anyone sizing a mesh, this is the practically useful table. Error in
R for a piecewise-constant voxel basis (which, per the above, is also
VoxHenry's error on a straight conductor):

| dx/δ | rectangular section | round wire |
|---:|---:|---:|
| 0.5 | −0.2% to −0.6% | −0.5% |
| 0.8 | −1% to −3% | −0.5% |
| 1.5 | −10% to −13% | −5.5% |
| 2.4 | −24% to −36% | −18% |
| 4.8 | −50% to −63% | −48% |

**Mesh to dx ≤ δ and a plain voxel solver is an engineering tool.
Beyond dx ≈ 1.5 δ it is not, unless something in the basis is doing the
work.**

## What this study does not establish

* Only R was compared in depth. L agreed closely everywhere and was not
  the interesting axis, but it is not audited here.
* One material (copper), one topology family per problem, single runs.
* Truth above dx/δ ≈ 2.4 is soft (±5–17%); the 100 GHz rows are
  indicative and no verdict rests on them.
* Multi-conductor proximity was measured only indirectly. The hairpin
  is a go-return pair and so carries proximity as well as turning, but
  the two effects are not separated, and no two-port conductor pair was
  run. That is the case most likely to show VoxHenry's enrichment in a
  better light and it remains the clearest gap.
* numex1's truth rests on two rungs (24k and 192k cells), so it carries
  no independent convergence check the way the three- to five-rung
  problems do; the 100 GHz row in particular should be read as
  indicative.
* The crowding-factor figures come from the earlier profile run, before
  the configuration was corrected. They are unaffected: two of the three
  are plain-basis (no modes involved), and the third is bar2's engine,
  where every cell is a boundary cell so `boundary_only` cannot change
  anything. The profile run has NOT been repeated for the other models.
* The SuperPEEC column measures the DEFAULTS and nothing else. No knob
  was swept, so this says what a user gets, not what the engine could do
  if tuned. The one probe that was run (an rc ladder on numex1) is not
  reported here because it was taken on the mis-configured path and
  would need redoing.
* Why the engine is non-monotone in frequency on numex1, and why the
  round wire overshoots by a roughly constant 7-8% rather than growing
  with dx/delta, are both unexplained.
* VoxHenry's J2d/J3d shapes were characterised by their *effect* and by
  their coefficient norms, not by reading their definitions out of the
  Green's-function tensor. The conclusion "they are for turning, not
  crowding" is an empirical inference from the enrichment ratio jumping
  20× at a bend, well supported but not a proof from first principles.
