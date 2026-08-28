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
  with per-cell fill fractions — partial-cell resistance is exact
  (staircase DC error collapses 11.6% -> ~1% on a 3-cell-radius
  wire) and a sparse exact-integral correction does the same for
  the near-field inductance, leaving the FFT/FMM structure
  untouched.
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
* Blender export: the conductor *surface* as glTF 2.0 (`.glb`) —
  exposed voxel faces carrying tangential |J|, one object per named
  block, with bond wires as swept tubes coloured by chain current and
  their contacts flared into the model's own feet.
  Opens natively in Blender with no addon; `studies/blender_view.py`
  builds the shaded scene and renders it headlessly.
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
| `--export-glb` | write the conductor surface and bond wires per frequency as glTF 2.0 (`.glb`) for Blender |
| `--wire-scale F` | draw `.glb` wire tubes and feet at F x the physical radius (default 1.0; recorded in the legend and the render stamp) |
| `--export-dir DIR` | directory for exported files (default `results/`) |
| `--quicklook N` | bin factor for the quick-look `.vti` (default 4; 0 disables) |
| `-v`, `--verbose` | solver diagnostics: setup timings, preconditioner engagement, residuals |
| `--status` | live progress line on stderr (current task, percent, ETA) |
| `--status-file PATH` | machine-readable JSON progress, atomically updated — poll it from a notebook or GUI (`SPPEEC_STATUS=PATH` works from any entry point) |
| `--status-events PATH` | append-only JSONL timeline: task durations, per-point results (`SPPEEC_STATUS_EVENTS=PATH`) |

The status file's schema and guarantees are frozen in
`docs/status_api.md`; `examples/status_monitor.ipynb` is a worked
live monitor with a filling-in Z(f) plot.

For multi-hour solves, `SPPEEC_CHECKPOINT=/path/ck.npz` makes the
Krylov loop dump its iterate periodically (`SPPEEC_CHECKPOINT_S`
seconds, default 300, atomic) and a relaunch of the same solve
resumes from it instead of starting cold — a killed 4-hour run
restarts minutes from convergence. The file is removed on success,
and a checkpoint from a different system is recognised and ignored.

Exports pair in ParaView: load the `.vti` under the `.vtp` wires and
apply a Tube filter (radius from the `radius` cell array).

### Blender

`--export-glb` answers a different question from the ParaView export.
`.vti` is the volume field, for interrogating what the current does
*inside* the metal; `.glb` is the conductor **surface**, for seeing
what the module looks like and where the current crowds on the parts
you can see — lit, shaded and renderable to a figure or a turntable
without a VTK stack.

**Write the file:**

```
python src/sppeec_cli.py examples/dbc_halfbridge_r3.toml --export-glb
```

which writes, per solved frequency, into `--export-dir` (default
`results/`):

```
results/dbc_halfbridge_r3_1e06Hz.glb           geometry + field
results/dbc_halfbridge_r3_1e06Hz_legend.json   the colour scale
```

Open the `.glb` in Blender with **File ▸ Import ▸ glTF 2.0** — no
addon, default import settings, and the model arrives in the solver's
own axes at one Blender unit per millimetre. glTF 2.0 is a published
Khronos standard rather than a Blender format, so the same file is
readable by other tooling that accepts glTF, and it does not rot
against Blender releases the way a `.blend` would.

| Export option | Meaning |
|---|---|
| `--export-glb` | write the surface and bond wires as `.glb` per frequency |
| `--wire-scale F` | draw wire tubes and feet at F × the physical radius (default 1.0) |
| `--export-dir DIR` | where the files land (default `results/`) |
| `--freq HZ` | narrow the declared sweep, so you export one frequency instead of all |

A 0.13 mm bond wire on a 40 mm module is a quarter of a percent of
the frame — correct, and invisible beside lit copper. `--wire-scale`
exaggerates it the way ParaView's Tube filter does; **1.5 is a good
value** for this module, and much above that the feet of a bonded
pair start to collide (at 3.0 they are 1.56 mm across on a 1.5 mm
pair pitch). Any value other than 1.0 is recorded in the legend and
burned into the render stamp, because an unlabelled exaggeration is
just a wrong picture.

**Render it** (optional — this is the only step that needs Blender
installed, and it never touches the numbers):

```
blender -b --python studies/blender_view.py -- \
    results/dbc_halfbridge_r3_1e06Hz.glb \
    --view iso --hide bottom_metal --render
```

writing `<stem>_iso.blend` (open it and orbit) and, with `--render`,
`<stem>_iso.png`. Drop `-b` to land straight in the Blender GUI with
the scene already built.

| `blender_view.py` option | Meaning |
|---|---|
| `--view iso\|plan\|front` | camera placement (default `iso`) |
| `--hide A,B` | drop objects whose names contain these — almost always `--hide bottom_metal` |
| `--render` | also write a PNG |
| `--res N` | long edge in pixels (default 1600) |
| `--emit F` | how much of the image is the field rather than the lighting (default 0.92; 1.0 makes the rendered pixel exactly the colourmap value) |

**From Python**, if you are post-processing a solve you already have:

```python
import blendout

blendout.export_scene(
    m, M, info['i_f'], 'module.glb',
    parts=prob.block_cells(m),               # one object per [[block]]
    wires=sw.sol.wires, i_w=info['i_w'],
    seg0=sw.sol.wc.seg0, wire_of_seg=sw.sol.wire_of_seg,
    foot_cell=sw.sol.foot_cell, foot_r0=sw.sol.foot_r0,
    freq=freq)
```

Every argument after `path` is optional: without `parts` the skin is
one object, and without the wire arguments there are no wires.

#### What is in the file

Only the voxel faces where metal meets air are written, so a
10-million-cell module is ~10⁶ quads rather than 10⁷ boxes. Each
face carries the *tangential* current density of the cell behind it
(the normal component through an exposed face is zero by
construction). The skin is split into one object per declared
`[[block]]`, which is what makes the file usable: a power module's
ground plane is a 40 × 50 mm sheet that occludes the top copper from
above and out-glows it from every other angle, and split it is one
click — or `--hide bottom_metal` — to remove.

Every vertex carries the colour *and* the number. `COLOR_0` is a
baked logarithmic ramp so the file is right the moment it opens;
`_JMAG` is the raw scalar in A/m² (A on the wires), which Blender
imports as a mesh attribute so the ramp can be rebuilt in shader
nodes against true values without re-exporting. The colour range is
global across every object — per-object normalisation would give a
dead signal trace the same ramp as the commutation loop — and is
written alongside as `<stem>_legend.json` and burned into the render
as stamp metadata, because an image of a current density with no
scale on it is decoration rather than a measurement.

Bond-wire contacts are drawn as the **feet** the solver actually
models: each flares from the wire radius out to the contact radius
`foot_r0` (default twice the wire radius) and seats on a filled disc
lying on the outward face of the solver's own anchor cell. This is
not dressing — the foot constriction in the answer is
`R_disc(r0, rho) * h(r0/dx)` from `footcal`, computed *for that disc*,
so a wire drawn as a bare cylinder ending in mid-air would omit a term
that is in the extracted R. The disc also shows where each wire was
landed. Only the taper between the two radii is a reading rather than
a modelled surface: the solver has a wire of one radius and a contact
disc of another, and nothing in between.

Geometry is metres, scaled by 1000 on the way out so one Blender unit
is one millimetre (a 40 mm module at 1 unit = 1 m sits inside the
default near clip and is invisible), and written Y-up so Blender's
importer lands it in the solver's own axes with default settings.
`validation/validate_blendout.py` gates that orientation by
round-tripping an asymmetric marker through a real Blender import; it
skips cleanly where Blender is not installed, which is why Blender is
not a dependency of the solve.

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
