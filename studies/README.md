# studies/ — measurement scripts (2026-08-02)

Ad-hoc measurement scripts, kept because their *results* are quoted in
docstrings and memory and someone will want to re-run or extend them.
These are NOT validators: they print tables, they do not assert, and
several are memory-bound. Run from the repo root inside the toolbox:

    SPPEEC_SCHEME=cell PYTHONPATH=src python3 studies/<name>.py

They write scratch `.vhr` files next to themselves.

## The open question: does the redistribution (skin-effect) engine work?

Read these in order — the story reverses twice, so partial reading
misleads.

| script | what it measures |
|---|---|
| `skinconv.py` | flat bar, dx/δ = 0.757, mesh refinement vs k=3 |
| `barconv2.py` | flat bar, dx/δ = **1.513** — the control that removed the regime confound |
| `wireconv.py` | 50 µm round wire, refinement + **analytic** Bessel round-wire reference |
| `shortwire.py` | 24 µm wire: refinement AND an rc ladder toward untruncated |
| `wirerc.py`, `wirefull.py` | rc sweeps on the 50 µm wire, incl. lopsided `rc_uu` vs `rc_cross` |
| `wirebnd.py` | modes restricted to boundary-only / interior-only cells |
| `zuudiag.py`, `zcross.py` | verify the assembled blocks against `greens.box_pair_stencil_pairs` |

**Where it stands.** `zuudiag.py` + `zcross.py` prove the assembly is
*correct*: `Ru`, the `Zuu` diagonal, the `Zuu` off-diagonals and `Zcross`
all match independent computation (exact to ~1e-16, `Zcross` bit-for-bit).
So there is no assembly bug.

On the 24 µm wire the rc ladder converges monotonically to ≈ 0.03223
(untruncated), against a refined-mesh value of ≈ 0.0340 — the engine
delivers **67%** of the needed correction, an UNDERSHOOT. The flat bar
gives 86%. **The engine under-corrects and does not overshoot.**

An earlier conclusion that it *overshoots* on staircase boundaries
(+255% on the 50 µm wire) came from a BAD REFERENCE: that wire's dx=0.5
and dx=0.333 rows agreed to six digits and were taken as converged. The
24 µm wire's equivalent rows differ by 0.7% and dx=0.25 moves again, so
that agreement was anomalous, not convergence. Do not trust the 50 µm
wire reference or anything derived from it until it is re-run refined.

## Scaling / preconditioner

| script | what it measures |
|---|---|
| `memsolve.py` | stage-by-stage RSS for the LpR port solve |
| `ordering.py` | CHOLMOD ordering × mode, fill and time |
| `onefac2.py` | ONE clean factorization, peak RSS (no `F.L()` materialisation) |

`onefac2.py` exists because measuring four factorizations in one process
*and* calling `F.L()` (which materialises the whole factor just to count
nonzeros) inflated peak RSS enough to OOM. Measure one at a time.

Result: `simplicial` + `metis` beats the previously hard-coded
`supernodal` + `amd` by 38% factor memory and 3× time (single core) —
now the default in `port_impedance.LpRSolver`. Caveat: supernodal is the
mode that can use threaded BLAS, so the TIME ranking may reverse at 16
threads; the memory result is structural.

## VoxHenry corpus

| script | what it measures |
|---|---|
| `vhrsurvey.py` | which of the 17 `.vhr` files are buildable, and why not |
| `cmpeq.py` | the corpus through `EquiTerminalSolver` (`argv[1] == 'skin'` enables subdivision) |

`validate_port_impedance.py` (repo root) exercises the
prescribed-current port instead — NOT the equipotential terminal; the
two give materially different answers above ~1 GHz.

## FMM diagnostics (resolved, kept for technique)

`p3point.py` builds the far field as an explicit POINT-CHARGE sum from
`leafinit`'s own positions. Agreement to 2e-5 proved the FMM ladder was
correctly wired and left the *positions* as the only suspect — which is
how the `leafinit` panel-offset sign bug was localised. `p3pinned.py` is
the pitch-pinned nleaf sweep. Both are superseded as tests by
`validate_leafinit_geometry.py`, but the point-source technique is
worth remembering.
