# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""multipole.py - backward-compatible facade.

The original ~4000-line module was split into multipole_common, special,
greens, levels, leaf_induct, leaf_poten, and tree.  `import multipole as mp`
still exposes everything (mp.Tree, etc.).
"""
# Thread defaults -- see sppeec_threads.py. The library defaults cost
# 6.3x on the FMM path (OpenBLAS spawning a dozen threads for each of
# many small gemv calls) and 1.8x even on the dense LpPR path, and the
# penalty is LARGEST on the coarse meshes the skin/corner studies use.
# The runtime call works whatever the import order; the environment
# block inside the module covers OMP and FFTW for anyone importing
# early.
try:
    import sppeec_threads as _spthreads
    _spthreads.enforce_blas()
except Exception:                    # tuning must never break a solve
    _spthreads = None

from multipole_common import *  # noqa: F401,F403
from special import *  # noqa: F401,F403
from greens import *  # noqa: F401,F403
from levels import *  # noqa: F401,F403
from leaf_induct import *  # noqa: F401,F403
from leaf_poten import *  # noqa: F401,F403
from tree import *  # noqa: F401,F403
