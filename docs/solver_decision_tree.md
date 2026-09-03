# SuperPEEC solver decision tree

Every supported problem class, the path it takes, and every setting
that changes along the way. Grouping is deliberate: configurations
listed together are handled identically. Every bifurcation below is a
real code-path or settings difference. Status: 2026-08-08, after the
dielectric program, the hole-augmented basis, the band-W rhs fix, and
the Schur-ordering study.

Dimensions covered: formulation (LpR / LpPR), scheme (cell / edge),
materials (homogeneous / mixed sigma, superconductor, dielectric),
cell shape (cubic / anisotropic), geometry (compact / flat / needle,
perforated or not, multiply-connected or not, multi-conductor),
size (small / medium / large), frequency band (vs the wL/R crossover
and vs cavity resonance), skin-effect resolution, and hardware
(memory ceiling, GPU).

---

## 0. Root: which formulation?

```mermaid
flowchart TD
    A[problem] --> B{charge / capacitive\nphysics needed?}
    B -- "dielectric cells present\n(model.epsilon)" --> LPPR[LpPR]
    B -- "port has NO galvanic\nreturn (plane pair, PDN:\nreturn = displacement)" --> LPPR
    B -- "C extraction, resonance,\nfull-wave-quasistatic Z(f)" --> LPPR
    B -- "R / L / skin / current\ndistribution only,\ngalvanic loop exists" --> LPR[LpR]
```

Hard rules, enforced by guards:

* Pure-dielectric cells -> **LpPR only** (`EquiTerminalSolver`
  raises: an excess-capacitance branch without its bound charge is
  not a dielectric).
* Port P and N on galvanically separate conductors with no stitching
  -> **LpPR only** (LpR raises `no return path`). Adding decap
  stitching vias makes the LpR loop measurement valid again
  (`build_pdn(stitch=N)` pattern).
* Magnetic materials: **not supported** (relegated; PyPEEC wins
  there for now). Multi-dielectric eps-contrast interfaces (two
  different eps_r touching): **not supported** (`|2-2| = 0` fires no
  panel) — single dielectric + vacuum + conductor only.
* Scheme: `SPPEEC_SCHEME=cell` for everything below. The edge scheme
  is deprecated, refuses material IDs (mixed sigma, dielectrics),
  and exists only for the legacy byte anchors.

---

## 1. LpR branch (`equiterminal.EquiTerminalSolver`)

```mermaid
flowchart TD
    L[LpR] --> M{materials}
    M -- "homogeneous sigma" --> M1[scalar r - default]
    M -- "mixed sigma (Cu/Al...)" --> M2[per-filament r,\ncell scheme required]
    M -- "superconductor\n(lambdaL set)" --> M3[complex two-fluid r,\nset_frequency per solve,\nskin subdivision REFUSED]
    M1 --> S{skin effect\nresolved?}
    M2 --> S
    M3 --> SZ[skip subdivision]
    S -- "cells < skin depth\nor DC/low f" --> SZ2[subdivide=False]
    S -- "need in-cell profile\n(d > delta at fmax)" --> SK[subdivide='auto'\nconduction palette,\nper-axis cells]
    SZ2 --> BAS{cycle basis}
    SZ --> BAS
    SK --> BAS
    BAS -- "default" --> B1["basis='auto'\n(overcomplete+AMG,\nfalls back to selected)"]
    BAS -- "memory-rich, fewest\nmatvecs wanted" --> B2[basis='selected'\n+ CHOLMOD Cholesky]
    B1 --> GPU{large + GPU present?}
    B2 --> GPU
    GPU -- yes --> G1[SPPEEC_GPU=1:\nresident m2l_top +\ndevice AMG apply]
    GPU -- no --> G2[CPU: OPENBLAS=1 OMP=4\nFFTW_THREADS_TOP=6]
```

Settings detail:

| decision | grouped configurations | setting / consequence |
|---|---|---|
| conductivity | homogeneous vs mixed | automatic (`resistances()`); mixed needs cell scheme (guard) |
| superconductor | any lambdaL | complex r via `impedance_density`; kinetic L exact; `subdivide` refused |
| skin resolution | wires/conductors thicker than delta | `subdivide='auto'` (conduction palette, 93% delivered); frequency-retuning automatic; anisotropic cells supported; use generous `rc` on staircased wires |
| anisotropic pitch | `VoxelModel.d` per-axis | native; aspect-compensating leaves automatic; accuracy ~1e-3 at 2:1, ~5e-3 floor at 4:1; skin engine refused |
| multiply-connected conductor (holes: antipads, slots) | any | automatic since 2026-08-08: tree-cotree hole generators complete the overcomplete basis (`nholes` reported); `selected` always worked |
| multi-conductor | separate components | spanning FOREST automatic; per-port component guard |
| flat / needle geometry | thin boards, coils | `partition()` pancake escape automatic (incl. collapsed-clamp boards < 9 cells thick); 2-D FMM, top level cheap |
| size: small (< ~50k cells) | — | defaults fine; single/2-level tree |
| size: medium (50k–1M) | — | 3-level tree automatic; CPU knobs `OPENBLAS_NUM_THREADS=1` in solve, `OMP=4`, `FFTW_THREADS_TOP=6` |
| size: large (1M–10M+) | — | matvec scales (3.1M cells: 9 s); **setup (loop Cholesky) dominates** -> `basis='auto'`/overcomplete (AMG; ~10x less memory, ~1.5–2x matvecs); `SPPEEC_GPU=1` for AMG apply (34x) and m2l_top (22x) on compact trees; on pancake trees GPU gains are small (~8%) |

---

## 2. LpPR branch (`SystemMat` + `port_impedance.LpPRSolver`)

### 2a. Tree / memory (choose FIRST — this is the binding constraint)

```mermaid
flowchart TD
    P[LpPR] --> T{cells}
    T -- "small: < ~30k" --> T1[single-level tree\ndense n2n\nexact W - numpy Cholesky]
    T -- "medium: ~30k - 500k" --> T2[multilevel FMM tree\nnear n2n ~ 27*leaf^3/node\n(33 GB @ 400k, leaf 5)]
    T -- "large: > ~500k" --> T3[circulant single-level\n(20 GB @ 400k) --\nBUILD SLOW: 87 min @ 400k,\nprofiling docketed]
    T2 --> W{n2nchol exists?}
    T1 --> WE[wsolve='exact' auto]
    W -- "yes (rare: compact,\nno dielectric layers)" --> WE2[wsolve='exact' auto]
    W -- "no (thin geometry,\ndielectric node layers)" --> WB[wsolve='band' auto\n+ true-residual postcheck]
    T3 --> WB
```

* Guard: an orientation with ZERO filaments (1-cell plates over an
  empty gap) is refused — thicken the plates.
* `LpPRSolver` handles the band-W rhs convention internally
  (`rhs = W*(P*injections)`); the true-residual postcheck runs on
  every solve (`info['true_residual']`) and warns loudly if the W is
  too weak. Never bypass it.
* Krylov memory: fgmres keeps ~`2*restrt*wholesize*16 B`. At >= 1M
  unknowns use `restrt<=100, maxiter=3..5` (restart 200 at 1.5M
  unknowns OOM'd a 62 GB box).
* Dielectric accuracy law: C within +-5% of converged truth at 1–4
  cells across the dielectric (BEM-validated); multilevel operator
  seam ~1e-3 (part E).

### 2b. Preconditioner / frequency (then choose this)

The crossover is geometric: f_c ~ where wL/R = 1 per cell,
f_c ∝ 1/(sigma d^2) — ~60 MHz at 50 um copper cells, ~1 GHz at
12.5 um.

```mermaid
flowchart TD
    F{dielectric cells?} -- yes --> DD["precond='diagschur' at ALL\nfrequencies (reluctance is\nINCOMPATIBLE with dielectric\nbranches -- guard raises;\nmaterial-split hybrid docketed)"]
    F -- no --> FC{frequency vs crossover}
    FC -- "f << f_c\n(resistive regime)" --> D[precond='diagschur']
    FC -- "f >~ f_c\n(inductive regime)" --> R[precond='reluctance']
    FC -- "near/above first\ncavity resonance" --> RES[reluctance +\nccap='band' or 'full']
    D --> DC{ccap}
    DD --> DC
    DC -- "small (< ~50k ext nodes)" --> DC1[ccap='diag' ok]
    DC -- "at scale" --> DC2[ccap='band' REQUIRED\n(diag needs dense eye-probe:\n684 GiB @ 300k ext)]
    DC -- "many frequencies, scale" --> DC3[sdsolve='amg'\n(guarded, falls back)]
    R --> RO["Schur ordering: MMD_ATA\n(in code since 2026-08-08)"]
```

**Measured 2026-08-08 (160^2 filled board):** reluctance +
dielectrics = true residual 0.52 at full budget (N_Z models every
branch as metal; dielectric branch admittance is ~10 orders away).
`LpPRSolver` now refuses the combination. For dielectric boards
above the crossover, diagschur remains correct with growing counts
(154 @ 160^2/1e8); the material-split hybrid (K~ on metal rows +
exact diagonal on dielectric rows) is the docketed path.

| decision | grouped configurations | setting / consequence |
|---|---|---|
| diagschur | f << f_c, any materials incl. superconductor + dielectric | complex-r probe fixed 2026-08-08 (subtracts r before reading Lp); `_Rdiag` refreshes per frequency (dielectric r is f-dependent) |
| reluctance | f >~ f_c | Schur factor ordering **MMD_ATA** — the old MMD_AT_PLUS_A is 58–90x slower on PERFORATED geometry (antipads/vias/slots derail greedy minimum degree: 2550 s vs 44 s at 99k nodes) while still best on unperforated compact volumes; MMD_ATA is robust everywhere measured. cholmod-METIS comparable (33 s) if a new path is ever needed. Dielectrics do NOT enter S~ values — only the graph size |
| ccap='full' | small near-resonance studies | dense P_ext^-1 block; O(n_ext^2) |
| per-frequency cost | reluctance & diagschur both refactor S per frequency | reluctance S~: ~44 s @ 100k nodes (MMD_ATA); diagschur S_d: 7-point, cheaper; sweep warm-starting is the open lever |
| flat geometry | boards, planes | pancake trees automatic; part-E seam applies |
| anisotropic pitch | capacitive path | **CAUTION: not explicitly validated** (aniso program validated the inductive path; panels carry per-axis pitch but no capacitive aniso gate exists) |
| GPU | LpPR matvec | only the traverseRL half has a GPU path; traverseP3 GPU port is docketed; diagschur/reluctance applies are CPU sparse |

### 2c. Known open items on this branch

* Multilevel capacitive memory (27*leaf^3 per node) is the scaling
  wall; smaller leaves shrink it; circulant build time (87 min @
  400k) needs profiling before circulant is the default at scale.
* `ccap='diag'` silent OOM deaths at 320^2 (two independent
  configurations) — suspected S_d fill spike; low priority since
  band is the at-scale route. Possibly the same perforation/ordering
  pathology as the reluctance Schur (S_d also uses MMD_AT_PLUS_A);
  untested.
* Warm-started frequency sweeps: designed, not implemented.
* diagschur-vs-reluctance matvec crossover on filigree boards:
  measurement in flight (160^2, this session).

---

## 3. Size classes, summarized across both branches

| size | LpR | LpPR |
|---|---|---|
| small (< 30–50k cells) | anything; defaults | single-level, exact W, ccap='diag' fine |
| medium (to ~500k) | defaults + CPU knobs; overcomplete if memory-tight | multilevel + band W + ccap='band'; watch near-field n2n RAM (leaf size) |
| large (0.5–3M+) | overcomplete/AMG; GPU on compact trees; setup dominates | multilevel with restart<=100 (33 GB @ 400k) or circulant (20 GB, slow build); band W + postcheck mandatory-in-practice |
| beyond (10M+) | matvec fine (6.4M: 9 s, 15.5 GB); solve setup is the frontier | uncharted; memory law says circulant or smaller leaves |

## 4. Quick invocation reference

```python
# LpR port solve
S = EquiTerminalSolver(model, M, port, basis='auto',
                       subdivide='auto')
z, i, info = S.solve(freq)

# LpPR port solve (dielectrics, PDN, capacitive)
M = model.build_tree(leaf, levels, capacitive=True)   # or circulant=True
model.prepare(M, freq)
S = LpPRSolver(model, M, precond='diagschur'|'reluctance',
               ccap='band', wsolve='auto')
z, x, info = S.solve(freq, restrt=100, maxiter=3)
# ALWAYS check info['true_residual']
```

Environment: `SPPEEC_SCHEME=cell` (always), `SPPEEC_GPU=1` (opt-in),
`OPENBLAS_NUM_THREADS` / `FFTW_THREADS_TOP` per the CPU-track notes.
