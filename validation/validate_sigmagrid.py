# -*- coding: utf-8 -*-
"""The conductivity palette grid (:mod:`sigmagrid`) against a dense grid.

Its claim is exactness on every access the model makes: indexing by
scalar, slice, fancy index and mask; scalar comparison including with
values absent from the table; materialisation; writes that add values;
and refusal when a byte cannot index the values present.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(_op.path.abspath(__file__)), '..', 'src')]

import numpy as np
from sigmagrid import PaletteGrid

FAIL = []


def check(name, ok, note=''):
    print('    %s %s%s' % ('ok  ' if ok else 'FAIL', name, ('  ' + note) if note else ''))
    if not ok:
        FAIL.append(name)


def main():
    rng = np.random.default_rng(2)
    dims = (40, 50, 12)
    dense = np.zeros(dims, np.float32)
    dense[5:30, 10:40, 2:9] = np.float32(5.8e7)
    dense[32:38, 3:8, :] = np.float32(1.2e6)
    dense[0, 0, 0] = np.float32(3.5e7)
    g = PaletteGrid.compact(dense)
    check('compacts a four-value grid', g is not None and g.table.size == 4)
    if g is None:
        return 1
    check('shape, ndim, size, dtype match', g.shape == dims and g.ndim == 3
          and g.size == dense.size and g.dtype == np.float32)
    check('one byte per voxel', g.code.nbytes == dense.size)
    check('scalar read, same type', g[7, 12, 3] == dense[7, 12, 3]
          and type(g[7, 12, 3]) is type(dense[7, 12, 3]))
    check('slice read bit-identical',
          np.array_equal(g[5:12, 8:20, 1:5], dense[5:12, 8:20, 1:5]))
    pos = tuple(rng.integers(0, n, 300) for n in dims)
    check('fancy read bit-identical', np.array_equal(g[pos], dense[pos]))
    check('!= 0 mask identical', np.array_equal(g != 0.0, dense != 0.0))
    check('== 0 mask identical', np.array_equal(g == 0.0, dense == 0.0))
    check('== value in table', np.array_equal(g == 5.8e7, dense == np.float32(5.8e7)))
    check('== value absent from table', np.array_equal(g == 1.0, dense == 1.0))
    check('!= value absent from table', np.array_equal(g != 1.0, dense != 1.0))
    check('mask index then unique (sigma_values pattern)',
          np.array_equal(np.unique(g[g != 0.0]), np.unique(dense[dense != 0.0])))
    check('np.asarray materialises identically',
          np.array_equal(np.asarray(g, dtype=np.float64), dense.astype(np.float64)))
    d2, g2 = dense.copy(), PaletteGrid.compact(dense)
    d2[1, 2, 3] = np.float32(9.9e6); g2[1, 2, 3] = np.float32(9.9e6)
    d2[20:25, 20:25, 4:6] = 0.0;      g2[20:25, 20:25, 4:6] = 0.0
    w = np.where(d2[pos] > 0, d2[pos], np.float32(4.4e6))
    d2[pos] = w; g2[pos] = w
    check('writes (scalar, block, fancy; new values) identical',
          np.array_equal(g2.todense(), d2) and g2.table.size == 6)
    check('min/max from the table', g.min() == dense.min() and g.max() == dense.max())
    check('sum exact to 1e-12 relative',
          abs(g.sum() - dense.astype(np.float64).sum()) <= 1e-12*abs(dense.astype(np.float64).sum()))
    check('> and <= masks identical',
          np.array_equal(g > 2e6, dense > 2e6) and np.array_equal(g <= 0.0, dense <= 0.0))
    gc = g.copy(); gc[0, 0, 0] = 0.0
    check('copy is independent', g[0, 0, 0] == dense[0, 0, 0] and gc[0, 0, 0] == 0.0)
    check('nonzero matches', all(np.array_equal(a, b) for a, b in zip(g.nonzero(), dense.nonzero())))
    check('ravel materialises identically', np.array_equal(g.ravel(), dense.ravel()))
    check('refuses more values than a byte indexes',
          PaletteGrid.compact(rng.standard_normal(dims).astype(np.float32)) is None)
    if FAIL:
        print('FAIL: %d check(s): %s' % (len(FAIL), FAIL))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
