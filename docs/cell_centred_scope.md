# Scoping spike: edge-filament -> cell-centred discretisation

Status: scoping only, 2026-07-31. No code changed.

## Why

SuperPEEC places current filaments on cell EDGES and gives every one the
full cell cross-section `dx^2`. An `N x N` cell cross-section therefore
carries `(N+1)^2` filaments of area `dx^2` instead of `N^2`, so DC
resistance is low by `N^2/(N+1)^2`. Measured against VoxHenry: solid
20x20 bar 0.909 (predicted 0.9070), round wire 0.785 (predicted 0.7921);
inductance low by 2% and 5% respectively. The scheme is FIRST-ORDER
accurate (error ~ perimeter x dx / area), which is why the thin round
wire is far worse than the solid bar.

The fix is to put nodes at cell centres and filaments between adjacent
cell centres, which is what FastHenry's segment path and every
conventional voxel-PEEC code do. See `validate_port_impedance.py` for the measurements.

## What it buys

Cell-centred is not only exact, it is SMALLER, and the saving grows in
exactly the geometry class SuperPEEC targets:

| model | edge fil | edge node | centre fil | centre node | change |
|---|---|---|---|---|---|
| solid bar 60x20x20 | 77700 | 26901 | 69200 | 24000 | -11% |
| round wire 50x10x10 | 14230 | 5151 | 10920 | 4000 | -23% |
| circular coil 310x310x10 | 268155 | 97348 | 202592 | 74680 | -24% |

## The structural payoff: panels become congruent for free

Put nodes at cell centres. Then an x-directed filament sits at
`(i+1/2, j, k)` and an x-normal charge panel (a whole cell face) sits at
`(i+1/2, j, k)` -- the SAME lattice. Current and charge for a given
direction share one staggered lattice and the continuity coupling is
local by construction.

Today the panels are quartered into 2x2 sub-faces (`pxstruc` is built on
`(ntx+2, 2*nty, 2*ntz)`) purely so each quarter can belong to one CORNER
node. That machinery collapses to whole faces and gets simpler. The
cross-orientation stagger (x-normal vs y-normal panels offset by half a
cell) survives, so the common-lattice work in `leaf_poten.py` is not
wasted -- it runs at coarser resolution.

## Touch points

### Tier 1 -- the actual change (element definition)

| site | now | becomes |
|---|---|---|
| `tree.py:104` | `ntotal = (ntx+1, nty+1, ntz+1)` | `ntotal = (ntx, nty, ntz)` |
| `tree.py:344` nodestruc | OR over 8 corner cells | identity: one node per occupied cell |
| `tree.py:363,372,381` e/f/g struc | OR over the 4 cells round an edge | AND over the 2 cells sharing a face |
| `tree.py:371,380,389` e/f/g `n` | `(ntx+1,nty,ntz+1)` etc. | `(ntx,nty-1,ntz)` etc. |
| `tree.py:390,400,410` px/py/pz struc | abs-diff on the `2*nty x 2*ntz` quartered lattice | abs-diff on the `nty x ntz` face lattice |
| `tree.py:160,166,170` npx/npy/npz | `(ntx+1, 2*nty, 2*ntz)` etc. | `(ntx+1, nty, ntz)` etc. |
| `tree.py:136-142` nleafp{x,y,z} | `*= 2` on transverse axes | no doubling |
| `tree.py:143-157` lfp{x,y,z} | `/= 2` on transverse axes | no halving |
| `multipole_aux.f:286` `get_node_size` | 7 stencils over `s(1..8)` | same 7 rules, rewritten |
| `multipole_aux.f:382` `get_idx` | the SAME 7 stencils, duplicated | same edit, twice |

The two Fortran routines must stay in lockstep -- they encode identical
rules and are the only definition of element existence for the
multilevel path. Consider factoring the predicate into one helper so a
future edit cannot desynchronise them.

### Tier 2 -- geometry and kernel parameters (need re-derivation)

- `levels.py:170-201` -- element positions. Built from `x0` (half-integer)
  and `x1` (integer) offsets, assigned per orientation. **The filament
  stagger maps over unchanged**: `e` is `(half, INT, half)` today, which
  under a half-cell origin shift is "node position transversely,
  cell-centre along its own axis"; cell-centred filaments have exactly
  the same pattern relative to the cell lattice. Only `self.n` changes.
  **The panel stagger does need re-deriving** -- this is the fiddliest
  part of the whole job.
- `leaf_induct.py:110-119` -- `genL3D` calls are UNCHANGED. It still sees
  identical bars on a lattice whose pitch equals the bar size, which is
  what its superposition identity requires.
- `greens.py` `gen2_p_parz` / `gen_p_per` -- same functions, called with
  full-cell panel dimensions instead of half-cell.

### Tier 3 -- dimension-driven, adapt automatically (verify only)

Confirmed by reading: these take lattice dimensions as parameters and
contain no hard-coded `nt+1`.

- All incidence kernels: `node2file/filf/filg`, `file2node/filf2node/
  filg2node`, `node2filament`, `filament2node`. The arithmetic is e.g.
  `datae(i) = singlegroup(z,y,x) - singlegroup(z,y+1,x)` -- a filament
  still joins exactly two nodes at the same relative offset. Driven by
  `ne`/`nf`/`ng`/`nnode`.
- `tree.py:1879 adjmats` and `meshgraph_aux` -- driven by `.n`.
- `reluctance.py:224` -- decodes node positions from `M.ntotal`; follows
  once `ntotal` is redefined.
- `levels.py` m2m/m2l/l2l -- box-level, indifferent to element layout.
- `port_impedance.py`, `vhr.py` -- dimension-driven.

### Tier 4 -- consumers needing re-anchoring

- `main.py` `setup1/2/3`: geometry sizes and the hard-coded port node
  coordinates (`snx/sny/snz`) all shift by the node-grid redefinition.
- All three byte anchors move. Unavoidable for any real fix.
- Expected values in the `validate_*` suite.
- `vhr.py` `_face_nodes`: a VoxHenry port face currently maps to the 4
  corner nodes of that face. Under cell-centred nodes it maps to the ONE
  cell the face belongs to -- simpler, but it is a real change and
  `validate_vhr.py` PART B pins the current behaviour.
- The `N >= NT+1` invariant (recorded in project memory) is stated in
  terms of the `nt+1` node grid and must be re-derived.

## The terminal wrinkle

Cell-centred nodes give an L-cell bar L nodes but only L-1 filaments, so
the conduction length is `(L-1)dx`, not `L*dx`. Three resolutions:

1. **Terminal half-filaments** (length `dx/2`) at port cells. Exact, and
   confined to port cells. Needs mutual partial inductance between
   UNEQUAL parallel bars -- `genL3D`'s identical-bar superposition does
   not provide it. The Hoer-Love closed form does; FastHenry implements
   it in `src/fasthenry/mutual.c`, which is now compiled locally.
2. Define the port at the terminal cell centre and accept a model `dx/2`
   short at each end.
3. Pad the geometry by one cell at ports.

Recommend (1). It is the only exact option and the formula is standard.

FastHenry itself never hits this: its nodes are at the conductor's
physical endpoints and `assignFil` subdivides only ACROSS the
cross-section, never along the axis.

## FastHenry as an oracle (checked 2026-07-31)

- **Segment (`E...`) models are a clean oracle.** `assignFil`
  (`induct.c:840`) gives `Hdiv = Hinc`, `Wdiv = Winc` for uniform
  spacing, so `Winc x Hinc` filaments tile the specified `w x h`
  EXACTLY, and node-to-node length is exact. Build validation geometries
  from segments.
- **Ground-plane models are NOT.** `readGeom.c:1249` sets
  `nodes1 = seg1+1`, `nodes2 = seg2+1`, dimensions `segs1` as
  `[seg1][nodes2]`, and hands every column -- including both edges --
  the same `segwid1 = length2/seg2` (the full pitch). FastHenry's ground
  planes have the IDENTICAL edge over-count as SuperPEEC today. Two
  consequences: comparing current SuperPEEC against a FastHenry plane can
  show SPURIOUS AGREEMENT (both wrong the same way), and after this
  rewrite SuperPEEC will stop agreeing with FastHenry planes by roughly
  `(seg+1)/seg` -- that will look like a regression and will not be one.
  Note `main.py setup3` is a plane geometry.
- Partial workaround: `segwid1 = length2/(seg2+1)` makes the total plane
  cross-section exact and is legal (the guard only rejects
  `segwid1 > segfull1`). Fixes DC resistance only -- interior segments
  become narrower than their spacing, leaving unphysical gaps -- so use
  it for a resistance check, never for inductance.

That FastHenry is exact precisely where it uses cell-centred subdivision
and over-counted precisely where it uses a node-centred grid is
independent support for this rewrite.

## Loose end found while reading

`multipole_aux.f:465-474` (`get_idx`): the `e` filament's EXISTENCE test
is `s(1)|s(2)|s(5)|s(6)` (the 4 cells round a y-edge, correct) but its
stored value is `struce = max(s(1),s(2),s(3),s(4))` -- a different index
set, and the one `g` correctly uses. `f` and `g` are self-consistent.

Textually this allows an `e` filament to exist with `struc == 0`. I
could NOT trigger it: 30 random 25%-fill geometries and a z-half slab at
`numlevels=2` all gave zero occurrences. So it is either masked by the
guard/offset conventions or needs an unusual configuration. Not a
confirmed live bug -- resolve it while rewriting these stencils rather
than chasing it now.

## Recommended order

1. Rewrite `get_node_size` / `get_idx` stencils behind one shared
   predicate; rewrite the `tree.py` numlevels==1 branch to match.
2. Re-derive the panel position stagger in `levels.py` (highest risk).
3. Terminal half-filaments + Hoer-Love unequal-bar mutuals.
4. Re-anchor `main.py`, the byte anchors, and the validate suite.
5. **Refinement study**: halve `dx` and confirm the error falls as
   `O(dx^2)` rather than `O(dx)`. This is the claim a reviewer will
   test, and the only one that distinguishes a fix from a fudge.

The riskiest item is (2); the most tedious is (4). Nothing in the FMM
ladder, the circulant near field, the preconditioner family, meshgraph,
or the port extraction needs to change.
