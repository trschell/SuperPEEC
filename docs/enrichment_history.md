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
