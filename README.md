# SuperPEEC

SuperPEEC (abbreviated *sppeec*) is a broadband electromagnetic field
solver for voxelized 3-D geometries, built on the partial element
equivalent circuit (PEEC) method. From a single TOML input file it
extracts frequency-dependent resistance, inductance, capacitance and
full port impedance matrices Z(f) for structures such as power-module
substrates, PDN plane pairs, interconnects and bond-wire assemblies —
covering skin and proximity effects, dielectric charge and loss, and
superconducting kinetic inductance within one discretisation. It
occupies the niche of the classic FastHenry/VoxHenry family of
extractors, unifying their capabilities in one maintained code with a
declarative input format.

![Top-surface current density of a DBC half-bridge power module at
1 MHz, with bond wires coloured by their solved chain
currents](docs/dbc_r4_plan.png)

*A 51-million-cell DBC half-bridge power module at 1 MHz: skin- and
proximity-driven current crowding on the top copper, with the bond
wires coloured by their solved chain currents. Solved on a 64 GB
desktop; figure produced by `studies/dbc_r4_plot.py` from the
solver's `.vti`/`.vtp` exports.*

## Features

* **Scalability is the defining feature.** SuperPEEC couples a
  block-Toeplitz FFT near field on the uniform voxel lattice with a
  fast-multipole (FMM) far field, so cost scales with the *occupied*
  cells rather than the padded bounding box that whole-domain FFT
  solvers pay for — on sparse structures (wires, coils, patterned
  planes) this wins by orders of magnitude, and on empty-heavy
  geometries the advantage grows without bound. Memory-flat
  preconditioning (geometric multigrid on the loop basis) and
  single-precision storage tiers hold the measured peak to
  **~0.6 kB per cell**: a 50-million-cell power module solves on a
  64 GB desktop, and billion-cell problems are projected to fit a
  single large-memory node.
* Two formulations, selected automatically from the declared
  materials: magnetoquasistatic **LpR** (R, L, skin/proximity) and
  full **LpPR** (R-L-P-C with charge, resonance and dielectrics).
* Materials: per-cell conductivity; dielectrics with loss tangent and
  the causal Djordjevic–Sarkar wideband-Debye dispersion model;
  two-fluid London superconductors (kinetic inductance, VoxHenry's
  material law).
* Subpixel geometry: round conductors (`[[cylinder]]`) voxelized
  with per-cell fill fractions — partial-cell resistance is exact,
  collapsing the staircase DC error (measured 11.6% -> ~1% on a
  3-cell-radius wire).
* Bond wires as polylines: validated round-wire cross-section model,
  wire–wire and wire–plane proximity, and a lattice-Green's-function
  calibrated foot (contact) resistance.
* Ports: prescribed current strips, solved-split equipotential
  terminals, and multi-port open-circuit Z matrices (reciprocity is a
  solved property, verified to ~1e-13).
* Opt-in GPU acceleration (CuPy): device-resident FMM top level and
  multigrid preconditioner apply, including a two-GPU VRAM-pooling
  split for hierarchies larger than one card.
* ParaView export: volume current density as `.vti` (with a
  slab-streamed writer whose memory stays O(one slab) at any grid
  size) and bond wires as `.vtp` polylines carrying solved currents.
* Adapters for VoxHenry `.vhr` and PyPEEC input files.
* A 36-script validation suite gated on analytic identities, closed
  forms and dense oracles — not just regression tolerances.

## Installation (Linux)

System packages (Debian/Ubuntu names):

    sudo apt install gcc gfortran make libfftw3-dev libopenblas-dev \
                     libsuitesparse-dev

Python 3.12+ with the scientific stack (a virtual environment is
recommended):

    pip install numpy scipy pyamg pyfftw cvxopt "scikit-sparse<0.5"

Optional, for GPU acceleration on CUDA 12 systems:

    pip install "cupy-cuda12x[ctk]"

Build the native extension modules (Fortran FMM kernels and the
H-matrix solver) in the repository root:

    make -f Makefile_multipole

The extension modules land in `src/` beside the code that imports
them. Then verify the build by running the validation suite (takes tens of
minutes; every script prints PASS/FAIL):

    bash validation/run_baseline.sh

Windows and macOS are untested.

## Command line

    python src/sppeec_cli.py input.toml [options]

The TOML file defines the problem; the command line holds run policy
only — nothing on the command line can change converged numbers.

| Option | Meaning |
|---|---|
| `--freq HZ [HZ ...]` | run only these points of the declared sweep (each must match a declared frequency — the CLI narrows the file, it never extends it) |
| `--export-vti` | write the volume current density per frequency (streaming `.vti` plus a binned quick-look companion) |
| `--export-wires` | write bond wires per frequency (`.vtp` polylines with solved chain currents) |
| `--export-dir DIR` | directory for exported files (default `results/`) |
| `--quicklook N` | bin factor for the quick-look `.vti` (default 4; 0 disables) |
| `-v`, `--verbose` | solver diagnostics: setup timings, preconditioner engagement, residuals |

Exports pair in ParaView: load the `.vti` under the `.vtp` wires and
apply a Tube filter (radius from the `radius` cell array).

## The TOML input file

One input file is the complete, reproducible statement of a problem:
geometry, materials, ports, frequencies and solver-relevant settings
all live in the file, so that running the same file always reproduces
the same output. Anything that would change the answer belongs in the
file, not on the command line; unknown keys are hard errors rather
than silently ignored defaults, so a typo cannot produce a subtly
wrong model. All quantities are SI (metres, S/m, Hz) in one world
frame with the origin at the corner of voxel (0,0,0).

The tables:

* `[grid]` — `dims` (cells per axis) and `pitch` (metres; scalar or
  per-axis for anisotropic cells).
* `[[block]]` — axis-aligned material regions as half-open cell
  ranges `from`/`to` (or `from_m`/`to_m` in metres). Materials:
  `sigma` (conductor, S/m), `epsilon` + optional `loss_tangent`
  (dielectric; add `dispersion = "djordjevic"` and `f_ref` for the
  causal wideband model), `lambda_l` (London depth — a
  superconductor, alone or two-fluid with `sigma`).
* `[[wire]]` — a bond wire as a polyline: `points` (list of 3-D
  coordinates; first and last are the bond contacts, whose pad feet
  are found automatically), `radius`, `sigma`, and optional
  discretisation overrides.
* `[port]` / `[[port]]` — cell-style ports (`p_cells`/`n_cells` or
  `p_box`/`n_box`) for the wire path; face-style ports
  (`p_faces`/`n_faces`, entries `[ix, iy, iz, "+z"]` on conductor
  faces) for the LpPR path, several of them for a multi-port Z
  matrix; `equipotential = true` solves the terminal current split
  instead of prescribing it.
* `[solve]` — `freq` as a literal list or a sweep expression
  `{ from, to, points, spacing = "log" }`, plus optional `rtol`,
  `formulation = "auto" | "LpR" | "LpPR"` (auto derives the
  formulation from the declared materials) and solver-policy
  overrides, including `skin = { ... }`: the sub-cell skin-effect
  engine on equipotential ports. Its default (`mode = "auto"`,
  `basis = "conduction"`) engages frequency-tracked conduction
  modes at k = 7 quadrature — the measured-best cross-section
  basis — but only when
  the cell size fails to resolve the skin depth at the sweep's
  highest frequency, so it costs nothing where it buys nothing;
  `mode = "off"` disables it, and `k`, `f_ref`, `rc_uu`/`rc_cross`
  and `boundary_only` tune it (see the doctrine's rule 13).

The full set of rules — including the conventions that make results
refinement-stable — is `docs/input_doctrine.md`, and `examples/`
contains runnable inputs from a three-wire power module to dispersive
plate capacitors and a superconducting bar.

## License

SuperPEEC is distributed under the MIT license — see
[LICENSE](LICENSE). Every source file carries an
`SPDX-License-Identifier` line stating its individual license, so a
file's terms remain unambiguous even when it is copied out of the
repository.

One exception: `examples/dbc_halfbridge.toml` and the reference files
under `examples/powersynth_2d_case_3/` are adapted from the
[PowerSynth2-pkg](https://github.com/e3da/PowerSynth2-pkg) sample
designs and are therefore GPL-3.0 (geometry description data only —
the solver itself is unaffected; see the SPDX headers in those
files).
