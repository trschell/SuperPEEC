# Enrichment: measurement history

The numbers that used to live in the engine's docstrings and comments,
moved here when the classes were merged (docs/enrichment_plan.md,
phases 1-2). Each entry names the study that produced it. The code
keeps one-line pointers; this file keeps the evidence.

## Placement: modes on exposed cells only (`reach = 0`)

20-cell-wide numex1 bar, dx/delta = 4.8, error against a converged
refinement ladder (`studies/wirebnd.py`, 2026-08-17/23):

    placement          10 GHz   25 GHz   100 GHz    cost
    everywhere          -0.2%    +4.6%    +70.9%    43.5 s
    exposed faces only  -2.1%    -5.4%     -7.7%    36.9 s

Modes-everywhere overshoots the physical skin limit rho*L/(P*delta)
by ~2.8x through spurious interior-mode excitation: a cell with metal
on all sides has no surface to crowd against, and because the modes
are net-zero their leading far field is a dipole that the coupling
truncation mishandles. Exposed-only is cheaper and errs low, the same
direction as the plain basis. Default since 2026-08-17 (TOML) and
2026-08-23 (direct API). The surface palette's ring rule (any
supported sub-prism within one cell of the resolved surface) is the
same rule on a resolved geometry.

## Coupling radii (`rc = (rc_uu, rc_cross)`)

Uniform bar at dx/delta = 4.8, error vs untruncated:

    rc_uu, rc_cross    1,1     2,2     1,2     2,3     3,4
    error              9.1e-2  2.7e-2  1.2e-2  5.2e-3  4.4e-4

(1,1) -> (1,2) cuts the error nearly 8x while (1,2) -> (2,2) makes it
WORSE: the mode-aggregate block (dipole-monopole, 1/r^2) needs reach,
the mode-mode block (dipole-dipole, 1/r^3) does not. `rc_uu` and
`rc_cross` act in opposite directions when swept alone. The (3,4)
default is chosen for the FFT apply, where a radius costs only
padding; under the CSR path pairs grow as (2rc+1)^3 (27 -> 343
neighbours for Zuu, 125 -> 729 for Zcross) and a large problem will
not fit -- hence `csr_max_gb`. Width-scaled radii (`_auto_rc`,
2026-08-20): rc = (ceil 1.5W, ceil 2W) floored (3,4), capped (12,16),
with the mid-shell damage-zone fallback; fixed (3,4) silently cut ~20
delivered points at 4 cells across. Round/staircase sections need
rc ~ 2x the diameter (studies/wirerc.py, wirefull.py); intermediate
rc on a wide section is non-monotonically wrong.

## The palette

Daniel / Sangiovanni-Vincentelli / White (EPEP 2000) face and corner
exponentials; `studies/modebasis2d.py` (2-D Galerkin, 2026-08-03):
conduction misses 6.6x less of the skin correction than the
consecutive-difference basis at matched mode count, which was the
root cause of the 201% "overshoot" on the 50 um wire. Linear modes on
boundary cells (98% of the gap at 136 DOF) beat linear-everywhere
(91% at 240). `studies/xsection_tabulated.py` (2026-08-20): the smooth
conduction family is COMPLETE for straight sections
(-0.0002..0.006% at every aspect ratio); tabulated per-section
profiles are a measured dead end. `studies/palette_ablation.py`
(2026-08-20): individual corner columns (P1) over the symmetric sum
(P0) measured +6 delivered points at dx/delta 3-6 (83->89 / 75->82 /
69->75% at 2/3/4 cells across), neutral at ~2; every knob's delta
matches P0's to 0.1-0.3 points. `diff` and `linear` palettes were
removed 2026-09-03 (enrichment phase 2): no production path selected
them.

k is quadrature, not unknowns: conduction-auto k = min(12, max(7,
ceil(2 dx/delta))) (sub-bar <= delta/2), measured dx/delta = 6 needs
k = 12 (+4 delivered points over 7 at unchanged matvecs), dx/delta =
3 is resolved at 7. Delivered error budget on the 2-across bar at
1e10: shipped-old 83.1 -> P1 88.7 -> +k12 92.7 -> +rc 96.1%.

`recommend_subdivision` never returns 2: a 2x2 split is blind to an
axially symmetric neighbourhood (measured |Zcross|max = 1.4e-27 on
setup3's collinear vias at k=2 against 2.2e-13 at k=3).

## London superconductors

Same Helmholtz equation, different constant: normal `grad^2 J = j w mu
sigma J` (rate (1+j)/delta), London `grad^2 J = J/lambda^2` (rate
1/lambda, REAL, frequency independent). Passing delta = lambda into
the complex rate spans exp(-x/lam)cos/sin, which does NOT contain
cosh((x-t/2)/lambda): residual 1.2e-2 vs 2.2e-15 for the real-rate
pair. The London palette prunes to 8 real columns (imaginary parts
vanish at a real rate); the discarded oscillatory columns are right
to drop. Ru = j w mu lambda^2 * l/a is linear in w, so the mode
equation's frequency cancels and the profile is frequency independent
-- validate_superconductor C's flat L(f). `studies/london_crowding.py`
(360 nm bar, lambda 90 nm): plain 0.9999 / 1.2764 / 1.3575 / 1.4442
at 2/4/6/12 cells across, modes 1.4157 / 1.4505 / 1.5163 at 2/4/6
(cylinder analog 1.534). At two cells the plain mesh is symmetry-pinned
to exactly bulk. On the RSFQ XNOR the modes are LOCALISED to vias (top
1% of filaments hold 94% of |u|), so the aggregate effect is ~0.5%.

## Thin films (`kk = (1, kz)`)

`studies/london_film.py`, 200 nm films / 200 nm gap / lambda 90 nm,
per-square normalisation (RELATIVE comparisons only; the true finite-
strip kinetic term is 69-79% of per-square, `studies/london_oracle2d.py`):
modes off 54.1 / 70.7 / 81.5 / 83.9% at 2/4/8/10 cells per film, error
~ dz^0.65; full palette +26 points at 2 cells; film palette 83-85% at
nt=2 for ~2 min per configuration against 32-73 min for the equal-
quality tuned full palette. The corner columns under a 1-D split make
the mode operator nearly singular (symbol condition 2.7e12; XNOR
2527 mv / 3.6 h -> 153 mv / 676 s without them, commit e26f1f2).
Against the 2-D oracle the film palette is ~94% of truth at nt=2, k=7
and ~98% at nt=4, k=12.

## Surface-anchored modes on resolved cylinders

`studies/mode_referee.py` (zero-truncation Galerkin referee): the
solved mode subspace tracks the fine sub-bar truth to ~2% through
dx/delta = 8. validate_subpixel: R_AC/R_DC within ~1% of the exact
Kelvin solution at dx/delta 1-2, usable to 3-4; deeper skin carries
two separate effects (a finite core-fed wire is not the infinite
Kelvin wire, +7% at dx/delta 6; the coarse transverse paths freeze an
under-crowded profile, ~20% high). An imposed Bessel profile (stage
C.1) measured WORSE than the geometry-only dL: the lattice already
carries the between-cell phase evolution and imposing the intra-cell
phase double-counts it. Without the per-cell block-Jacobi mode
preconditioner the deep solve ran 2078 matvecs without converging.

## Mode preconditioning

The mesh preconditioner's mode block is the identity. Without
`mode_precond` the engine-only ladder rungs at 1e10 hit the 311-matvec
cap; with the shared Kronecker inverse matvecs went 95 -> 11 and 63 ->
15 with answers bit-identical (commit c2f1a03). Three further
preconditioner variants for the film stall (column-block, head-to-mode
Gauss-Seidel, coarse space) were dead ends: the stall was the basis
(above), not the preconditioner. Lesson: a Krylov stall that resists
three preconditioners is a spectrum question -- probe the symbol.

## Partial cells (subpixel stage B)

dL = w'Tw - u'Tu is local by construction: on a z-cut at fill 0.5,
-21.8% of the pair at one cell, -12.1% at two, -8.2% at three, with
the absolute dL falling 2.14 -> 0.60 -> 0.27 e-15 H (a 2-cell window is
enough). Pairing only partial-partial recovered ~10%: every pair with
at least one partial end is emitted. Round-wire example against a 2x
reference: staircase 2.4% -> fill 1.2% -> fill+dL 0.9% in L. Slab (75
nm film at 30 nm pitch vs the same film at 15 nm): R error 16.67% ->
0.00%, L error 2.42% -> 0.30%. The RSFQ XNOR at pz = 67.5 nm has 13
distinct fills (all n/27) and 706254 partial cells; the memoised
correction takes 15.9 s and 1.74 GB CSR.

## Traces: the section cut and the edge family (docs/trace_plan.md)

A tilted edge on the voxel lattice, 2026-09-05. The 45-degree bar
ladder against the same bar axis-aligned (rotation invariance is the
reference; validate_trace C, prescribed-current path, no modes):
staircase DC R 1.420 / 1.132 / 1.029 at 4 / 8 / 16 cells across;
exact cell fills with the per-cell 1/fill rule 1.144 / 1.065 / 1.034
(FIRST order: every in-plane link joins cells of unequal fill); the
FACE rule (a filament takes the conductance of the face it crosses)
1.0093 / 1.0008 / 1.0007. Stage B through the cut is +0.006% in L.

Deep skin on the 45-degree dogleg (equipotential path, 100 MHz,
against the dogleg's converged plain-basis value at 48 across):

    cells across (h/delta)      4 (3.8)    8 (1.9)
    plain basis                  0.609      0.960
    face-anchored shared modes   1.094      1.057   (the staircase answer)
    + true-edge modes stacked    1.121      1.068
    + true-edge modes REPLACING  1.012      1.029   (shipped)
    + a z-face column            1.143      1.058

Face-anchored modes converge to the staircase perimeter; the edge
family must replace them on the edge cells, not join them. At
h/delta ~ 1 (16 across) the shared section family stalls at the
matvec cap, straight bar or dogleg alike -- an engagement-rule item
on the enrichment docket.

## Cross blocks between families: the per-pair table copy (2026-09-07)

Memory profile of examples/diagonal_trace.toml at 100 MHz (RSS
sampled at 0.5 s against the status events): the run's peak, 7.23 GB,
sat in `ModeStack._restack`'s cross-block fold for the edge family --
180k parallel pairs at k = 49 sub-prisms -- while every other phase,
setup and solve alike, stayed under 2.4 GB. `T[inv]` materialised a
per-pair copy of the 49 x 49 table (3.5 GB) and the three-operand
einsum ran as nested loops (71 s per block, twice). Grouping the
pairs by separation and folding each group with two matmuls is
bit-level identical (max |old - new| 2e-25 against entries of 2e-11)
and measured: peak 7.23 -> 3.10 GB, the two folds 71 -> 0.6 s each,
the single-frequency run 5:15 -> 3:04. Only stacked families with a
per-cell member (the edge family, the corner family) ever ran this
fold; a single shared family, the XNOR and the wire-bond path are
untouched. The post-Krylov readout on the equipotential path (the
Gram correction) costs no memory at all; the R3 wire-bond run's peak
is inside its Krylov (basis vectors on the mesh unknown) and its
first FMM sweep, as the R4 census correction recorded.

## The RSFQ XNOR memory profile and the Gram readout (2026-09-07)

RSS sampled at 1 s against the status events, examples/rsfq_xnor.toml
at 10 GHz (film modes on M0-M7, k = 7, rc (12, 16)), GPU on:

    phase                        dur s   RSS@start   max    RSS@end   (GB)
    setup (spectra, precond)      1171      0.75    10.83     9.05
    krylov, 137 matvecs           1912     15.78    24.35    22.86
    readout: gram correction      1044     16.36    26.87    17.20   <- the peak

The post-Krylov Gram correction, lgmres on Y^T Y over the WHOLE
basis, set the run's high-water mark (2.5 GB over the Krylov's own)
and took a third of the solve. Two facts make it unnecessary at that
size: the mode columns are unit vectors orthogonal to every loop
column, so the Gram is block-diagonal with an IDENTITY mode block
(the correction's mode part is d itself), and the mode
preconditioner is the mesh system's inverse, not the Gram's (its
single apply left a Gram residual of 744% on the trace example that
lgmres then iterated down to 3e-7). Solving the loop block only, with
the mode part exact: loop-sized Krylov vectors (24M of the XNOR's 32M
columns are modes), first-guess residual 3e-8 on the trace example
with lgmres returning at once, Z identical to 12 digits. XNOR
measurement with the loop-block readout, loop columns and their
nonzero rows sliced out of the basis once (`_Yl`), the Cholesky alone
as the preconditioner: readout 1044 -> 667 s, its high-water mark
26.87 -> 24.11 GB, the run's peak 26.87 -> 24.36 GB (now the
Krylov's own), L 1.66421 pH unchanged. Two more steps the same day:
(1) the correction is a . d with a = G^-1 Y^T ihat, and a is
GEOMETRIC (the tree route scales with the current), so it is solved
ONCE per solver and every frequency's readout is a dot product -- on
one and the same solution vector the a-form and the c-form agree to
all 12 printed digits; (2) that one solve targets 1e-4: the readout
error is |r| x (Gram defect), so 1e-4 already sits four decades under
the mesh residual, where the old per-frequency target of 1e-2 x rtol
chased accuracy the solution does not have. XNOR with both:
readout 1044 -> 28 s, its high-water mark 26.87 -> 18.90 GB, the
solve phase 3004 -> 2190 s, L 1.66421 pH and 137 matvecs unchanged;
the run's peak is the Krylov's 24.4 GB.

Without modes the XNOR's peak was the same readout (15.2 -> 21.1 GB,
153 s; the mode-less run exists because the phase-4 engagement
threshold had switched the London film modes off at dz/lambda 0.75 --
a real-rate palette has no re/im degeneracy and never stalled, so
London models keep the half-length threshold; the XNOR's L moved
1.66421 -> 1.67061 pH without its modes, 0.4%, the localised-to-vias
effect recorded in the film program).

## The XNOR census, and the symmetric half of the spectra (2026-09-07)

memcensus over the BUILT equipotential solver of examples/rsfq_xnor
(7.16 GB resident, 7.51 GB walked): mode spectra Fu + Fc 1.74 GB (the
London film palette prunes to km = 2, so the "19.8 GB" the build_fft
docstring once quoted belonged to an earlier palette), loop basis
1.30, Gram/GeoMG hierarchy 1.24, M2L top spectra 0.66, tree and model
1.1. The Krylov-phase peak (24.4 GB) is therefore ~17 GB ABOVE the
built solver: the lazily allocated FMM leaf buffers and the Krylov
basis, the R4 mechanism (docs/memory_census_r4.md). Census AFTER one
matvec: +7.6 GB resident (7.17 -> 14.81), of which the six leaf gather
buffers (`_ynmr_g` / `_mfil_g`, complex64 already) are 5.5 GB and
allocator retention ~2; the Krylov phase then adds ~9.6 GB more. That
last block is NOT mostly the basis (complex64 at rtol 1e-4, inner_m
10): tracemalloc on the trace example puts the mode-FFT apply's
transient at ~30 padded-grid slabs, and on the XNOR one complex128
slab is 0.58 GB -- U (km slabs), F, acc, accf, the scatter buffer and
the ifft outputs, all complex128 while the spectra they multiply are
complex64. Built the next day (2026-09-08): `apply_fft` scatters by
3-D index into km + 3 slabs allocated per call and reused within it
by in-place products (a reshape of the padded slab's sub-block
copies -- the first version wrote into that copy and moved Z by 5%).
The slabs stay complex128: single-precision FFTs of the Krylov
vectors measured +23% / +11% matvecs on the trace example at 4 / 8
across for Z within 3e-6 (`SPPEEC_MODE_APPLY_FP32=1` opts in for
A/B). A version that CACHED the slabs across matvecs measured +2.9 GB
resident through the XNOR's Krylov for a peak of 24.15 GB against
24.08 -- more memory for nothing, withdrawn before commit. XNOR with
the per-call version: peak 24.08 -> 23.47 GB, the Krylov's starting
residency 16.5 -> 15.5, the solve phase 2059 -> 1957 s, L and
matvecs unchanged -- the grid-sized copies the old scatter and
gather made on every call, and the in-place products. The
extrapolation was wrong: on the trace example the
padded slab is 2 MB and the transient tracemalloc saw was vector-
sized arrays, not slabs, so "30 slabs" scaled the wrong thing. The
Krylov-phase excess on the XNOR (~8 GB over the built solver plus
its leaf buffers) was then measured directly (scratch/
xnor_matvec_profile.py: tracemalloc plus a 20 ms RSS sampler around
each operation, from a resident 17.8 GB after two matvecs): the full
matvec peaks +0.7 GB, the mode-block apply +0.4, the preconditioner
+0.55, the bare FMM sweep +0.15 -- every operation's transient is
under a gigabyte. The Krylov phase's remaining ~6 GB is therefore the
Krylov's OWN storage: at rtol 1e-4 the vectors are complex64, 155 MB
each on the XNOR, and lgmres at inner_m 10 with outer_k 3 keeps
about thirty of them plus the solver's working vectors. That knob is
doctrine (inner_m 10 was decided with the user as the efficient
point), not a bug: inner_m 5 would return ~2 GB for more matvecs.
The apply change is kept: cleaner, no slower, and its prediction is
recorded as a miss.

The mode-mode spectra are now stored as their upper triangle:
reciprocity gives Bu[d, m, n] = Bu[-d, n, m] and the kernels are
real, so the spectrum of block (n, m) is the conjugate of block
(m, n); km(km+1)/2 + km spectra in place of km^2 + km (on the trace
example's km = 16 family: 136 + 16 spectra for 256 + 16, Fu 0.13 GB
for 0.26). Z moves at 3e-7, the complex64 rounding of the spectra
themselves. XNOR: Fu 1.16 -> 0.87 GB, run peak 24.4 -> 24.1 GB, L
1.66421 pH unchanged -- small there because km = 2; on a normal-metal
film family (km 16) it is the difference between 272 and 152 padded
slabs.

## Duplicates in the built equipotential solver (2026-09-08)

Three copies the XNOR census showed that need not exist, universal
to every run of this path with or without a GPU: `Y^T` was a
separate CSC copy of the loop basis (0.49 GB on the XNOR) and is now
a CSR view sharing Y's arrays (scipy transposes CSC to CSR without a
copy; re-taken after the float32 shrink, which replaces Y's data);
the incidence matrix `B` (0.26 GB) is the first efg rows of `Baug`
and is dropped after assembly, the spanning tree reading it there;
and the terminal coupler kept the previous matvec's WHOLE vector
alive through a 126-entry view (`i_t`, 0.31 GB) -- it now copies the
slice and drops both slices after the sweep. One thing this must NOT change: the float32
copy handed to the Gram factor stays CSC. Handed CSR (the view's
format), `_GeoMGFactor` takes its row-slice branch, and the XNOR's
assembly went from 1050-1210 s to over 2400 s before the run was
stopped -- that branch's cost on this path is a separate question. Two
things the km-change REBUILD path taught (validate_enrich H caught
the first): the rebuild reads B back from Baug's first rows, and it
resets the readout's cached loop block and the once-per-solver Gram
correction, which are sized by the basis (a latent bug of the same
day's readout change, not of the dedup); and validate_spmv forms
its own Gram from `S.YT`, which it now converts to CSC first, as the
solver does for its factor. Dogleg Z bit-identical;
XNOR: peak 21.04 -> 20.34 GB, the Krylov's starting residency 13.2 ->
12.4, assembly 1195 s and the solve 2087 s (both in their usual
bands), L 1.66421 pH and 137 matvecs unchanged.

## Allocator tuning: a null result (2026-09-08)

The XNOR under three allocators, RSS sampled per phase: glibc default
/ glibc with MALLOC_MMAP_THRESHOLD_ and MALLOC_TRIM_THRESHOLD_ pinned
at 128 kB and MALLOC_ARENA_MAX 2 / jemalloc 5 via LD_PRELOAD. Setup
maxima 10.18 / 10.09 / 10.05 GB -- identical, so the ~2 GB the
census cannot attribute after the first matvec is LIVE memory the
walker does not see (the eighteen opaque pyfftw plans and their
aligned buffers are the lead), not reclaimable fragmentation. Krylov
maxima 20.34 / 22.06 / 20.40: the pinned threshold RAISED the peak by
1.7 GB (mechanism not established; the per-matvec churn of mmapped
temporaries changes when the kernel sees freed pages, not the live
set), jemalloc matched the default's peak with higher residency
between spikes (its decay window keeps freed extents ~10 s). Both
runs were stopped before their solves ended; nothing to adopt.

## The unattributed residency, closed (2026-09-09)

scratch/xnor_unaccounted.py on the XNOR after two matvecs (13.2 GB
resident): the census walks 9.13 GB of arrays; all eighteen pyfftw
plans' buffers are arrays already in that walk (0.00 GB extra -- the
top-level M2L's FFTW workspaces are lazy and CPU-path only, never
allocated on a GPU box); the CUDA context is 0.1-0.3 GB; Python's
own objects ~0.7 GB. glibc's ledger (mallinfo2): arena 8.63 GB with
1.64 GB FREE-RETAINED, 3.02 GB in separately mmapped blocks;
malloc_trim(0) released 1.52 GB (RSS 13.31 -> 11.78). So the
'unaccounted' memory is the setup's freed transients kept inside the
arena -- dead weight under the solve's peak, because the Krylov's
155 MB vectors are far above the mmap threshold and never reuse those
chunks. That is also why swapping allocators did nothing: the memory
is freed inside one process, not fragmented across it.
A malloc_trim(0) at the entry of every Krylov solve was then
measured and WITHDRAWN: XNOR peak 20.34 -> 20.25 GB, R3 5.29 -> 5.28
-- the freed chunks are reused by the solve's sub-threshold
temporaries, so at the peak instant they are live again; the trim
only lowers residency at quiet moments. The accounting is closed:
nothing in the unattributed part is a lever on the peak.