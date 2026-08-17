# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Deep memory census of live solver state.

Answers "where are the bytes?" by walking the object graph from given
roots and attributing every reachable buffer to the attribute path it
was found at. Written for the fp32 campaign (which needs to know which
arrays are worth converting) and for hero-machine sizing; reusable for
any memory-law re-base.

Why the earlier ad-hoc walker under-counted by ~4x (2026-08-13, R3:
0.9 of 4.3 GB): it followed ``__dict__`` only, capped depth at 3, and
never entered LISTS OF OBJECTS -- ``M.lv`` (the level list), leaf
triples, pyamg level lists. This walker:

  * traverses ``__dict__``, ``__slots__``, dict/list/tuple/set members
    with an explicit stack (no depth cap, cycle-safe by id);
  * counts every numpy array ONCE, attributing views to their base so
    a slice never double-counts;
  * understands scipy sparse (data+indices+indptr / row+col) and
    cupy device arrays (reported as a SEPARATE VRAM total, not RSS);
  * reports opaque extension objects (cholmod factors, FFTW plans,
    SuperLU) BY TYPE AND COUNT rather than pretending they are free --
    the census prints what it cannot see so the RSS gap is explicit.

Use::

    rows, opaque, vram = memcensus.census({'sol': sol, 'M': M})
    memcensus.report(rows, opaque, vram, groups=memcensus.DBC_GROUPS)
"""
import re

import numpy as np
import scipy.sparse as sp

try:
    import cupy as _cp
except Exception:                                    # GPU-less boxes
    _cp = None

# Types never worth entering: they own no solver arrays, and some
# (modules especially) would drag the whole interpreter into the walk.
_STOP = (str, bytes, bytearray, type, type(np), type(np.sum),
         np.dtype, np.ufunc)


def _sparse_bytes(a):
    tot = 0
    for name in ('data', 'indices', 'indptr', 'row', 'col', 'offsets'):
        arr = getattr(a, name, None)
        if isinstance(arr, np.ndarray):
            tot += arr.nbytes
    return tot


def census(roots, max_items=2_000_000):
    """Walk from ``roots`` (a dict name -> object).

    Returns ``(rows, opaque, vram)``:
      rows   [(bytes, path, dtype, shape)] for host numpy/sparse
             buffers, each COUNTED ONCE (views attributed to base);
      opaque {type_name: count} for extension objects the walk cannot
             size (their bytes are in RSS but not in rows);
      vram   [(bytes, path, dtype, shape)] for cupy device arrays.
    """
    seen = set()          # object ids already visited
    owned = set()         # ids of array BASES already counted
    rows, vram = [], []
    opaque = {}
    stack = [(o, name) for name, o in roots.items()]
    n = 0
    while stack:
        obj, path = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        n += 1
        if n > max_items:
            raise RuntimeError("census walked past %d items -- a root "
                               "reaches far more than solver state; "
                               "narrow the roots" % max_items)
        if isinstance(obj, _STOP) or obj is None:
            continue
        if isinstance(obj, np.ndarray):
            base = obj.base if isinstance(obj.base, np.ndarray) else obj
            if id(base) not in owned:
                owned.add(id(base))
                rows.append((base.nbytes, path, str(base.dtype),
                             tuple(base.shape)))
            continue
        if _cp is not None and isinstance(obj, _cp.ndarray):
            base = obj.base if isinstance(getattr(obj, 'base', None),
                                          _cp.ndarray) else obj
            if id(base) not in owned:
                owned.add(id(base))
                vram.append((base.nbytes, path, str(base.dtype),
                             tuple(base.shape)))
            continue
        if sp.issparse(obj):
            if id(obj) not in owned:
                owned.add(id(obj))
                fresh = 0
                # claim the constituent arrays so the recursive walk
                # cannot count them again; arrays SHARED with an
                # already-counted matrix contribute 0 here (dedup)
                for name in ('data', 'indices', 'indptr', 'row',
                             'col', 'offsets'):
                    arr = getattr(obj, name, None)
                    if isinstance(arr, np.ndarray):
                        base = arr.base \
                            if isinstance(arr.base, np.ndarray) else arr
                        if id(base) not in owned:
                            owned.add(id(base))
                            fresh += base.nbytes
                rows.append((fresh, path,
                             'sparse/' + type(obj).__name__,
                             tuple(obj.shape)))
            # fall through: walk remaining attributes
        if isinstance(obj, dict):
            for k, v in obj.items():
                stack.append((v, "%s[%r]" % (path, k)))
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            # attribute all elements to one bracketed path; per-index
            # paths on a million-entry list would swamp the report
            for i, v in enumerate(obj):
                stack.append((v, "%s[%d]" % (path, i)))
            continue
        import types
        if isinstance(obj, types.FunctionType):
            # CLOSURES HOLD SOLVER STATE in this codebase: _GeoSplit's
            # __call__ captures the whole GeoMG factor + Schur cholesky
            # as closure cells, not attributes (found 2026-08-13 when
            # the gram hierarchy was invisible to the census).
            for cell in obj.__closure__ or ():
                try:
                    stack.append((cell.cell_contents,
                                  path + '.<closure>'))
                except ValueError:
                    pass
            for i, dflt in enumerate(obj.__defaults__ or ()):
                stack.append((dflt, path + '.<default%d>' % i))
            continue
        if isinstance(obj, types.MethodType):
            stack.append((obj.__func__, path))
            stack.append((obj.__self__, path + '.<self>'))
            continue
        walked = False
        d = getattr(obj, '__dict__', None)
        if isinstance(d, dict):
            walked = True
            for k, v in d.items():
                stack.append((v, path + '.' + k))
            # locally-defined helper classes (e.g. _GeoSplit) keep
            # state in their METHODS' closures; instance __dict__
            # never reaches those, so walk the class functions too
            if '<locals>' in getattr(type(obj), '__qualname__', ''):
                for v in type(obj).__dict__.values():
                    if isinstance(v, types.FunctionType):
                        stack.append((v, path + '.' + v.__name__))
        for slot in getattr(type(obj), '__slots__', ()) or ():
            try:
                stack.append((getattr(obj, slot), path + '.' + slot))
                walked = True
            except AttributeError:
                pass
        if not walked and type(obj).__module__ not in (
                'builtins', 'numpy'):
            # an extension object we cannot enter: count it honestly
            key = type(obj).__module__ + '.' + type(obj).__name__
            opaque[key] = opaque.get(key, 0) + 1
    rows.sort(reverse=True)
    vram.sort(reverse=True)
    return rows, opaque, vram


# Subsystem grouping for the DBC/wire stack. First match wins.
DBC_GROUPS = [
    ('m2l spectra (ftrans)',   r'\.ftrans'),
    ('tree: level data',       r'\bM\.lv\['),
    ('tree: leaf buffers',     r'\bM\.(e|f|g)\.'),
    ('tree: top/other',        r'\bM\.'),
    ('wire near kernels',      r'\.(_kern_wv|_kern_ww|Cn|_Cn)'),
    ('wire coupler',           r'\.wc\.|wirecoupler'),
    ('loop basis (B/Y/Bmat)',  r'\.(Bmat|B|Y|YT|Byt)\b'),
    ('gram/precond hierarchy', r'\.(ml|mg|chol|geo)\b|levels\['),
    ('model (struc/sigma)',    r'\.model\.|\.sigma|\.struc'),
    ('solver vectors',         r'\.(ihat|whole|fil_cell|i_f|i_w)'),
]


def group(rows, groups=DBC_GROUPS):
    """Sum rows into named subsystems; unmatched rows -> 'other'."""
    out = {name: 0 for name, _ in groups}
    out['other'] = 0
    other_top = []
    for b, path, dt, shape in rows:
        for name, pat in groups:
            if re.search(pat, path):
                out[name] += b
                break
        else:
            out['other'] += b
            other_top.append((b, path))
    other_top.sort(reverse=True)
    return out, other_top[:10]


def fp32_projection(rows):
    """Bytes now vs bytes if every float64/complex128 buffer halved --
    the ceiling for the fp32 campaign on the walked state."""
    now = sum(b for b, _, dt, _ in rows)
    then = sum(b//2 if ('float64' in dt or 'complex128' in dt) else b
               for b, _, dt, _ in rows)
    return now, then


def report(rows, opaque, vram, groups=DBC_GROUPS, top=25, rss=None):
    tot = sum(b for b, _, _, _ in rows)
    print("host buffers walked: %.2f GB in %d arrays" % (tot/1e9,
                                                         len(rows)))
    if rss is not None:
        print("process RSS %.2f GB -> unaccounted (interpreter, "
              "allocator, opaque, pools): %.2f GB" % (rss, rss - tot/1e9))
    g, other_top = group(rows, groups)
    print("-- by subsystem --")
    for name, b in sorted(g.items(), key=lambda kv: -kv[1]):
        if b:
            print("  %8.1f MB  %s" % (b/1e6, name))
    print("-- top arrays --")
    for b, path, dt, shape in rows[:top]:
        print("  %8.1f MB  %-14s %-18s %s" % (b/1e6, dt, shape, path))
    if other_top:
        print("-- largest 'other' (grouping gaps to fix) --")
        for b, path in other_top[:5]:
            print("  %8.1f MB  %s" % (b/1e6, path))
    if opaque:
        print("-- opaque (in RSS, NOT counted above) --")
        for k, c in sorted(opaque.items(), key=lambda kv: -kv[1]):
            print("  %6dx  %s" % (c, k))
    if vram:
        print("-- device (VRAM, not RSS) --")
        for b, path, dt, shape in vram[:10]:
            print("  %8.1f MB  %-14s %s" % (b/1e6, dt, path))
        print("  VRAM total %.2f GB" % (sum(b for b, *_ in vram)/1e9))
    now, then = fp32_projection(rows)
    print("fp32 ceiling on walked state: %.2f -> %.2f GB (-%.0f%%)"
          % (now/1e9, then/1e9, 100*(1 - then/max(now, 1))))
