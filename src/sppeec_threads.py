# SPDX-License-Identifier: MIT
"""Thread-count defaults, applied BEFORE numpy/scipy load their BLAS.

WHY THIS FILE EXISTS. The library defaults are catastrophically wrong for
this solver. Measured on a 192000-cell straight conductor, 4 frequencies,
identical answers:

    nothing set (library defaults)                      1498.2 s
    OPENBLAS=1, OMP=4, FFTW_TOP=6                        237.4 s   6.3x
    OPENBLAS=1, OMP=8, FFTW_TOP=8                        268.3 s
    OPENBLAS=12, OMP=4, FFTW_TOP=6                      1295.5 s

The third row isolates the cause: holding OMP and FFTW at their good
values and changing ONLY OpenBLAS from 1 to 12 costs 5.5x. The FMM's
p2m/l2p stages are many SMALL gemv/gemm calls, and spawning and
synchronising twelve threads per call dwarfs the arithmetic. OMP=8 being
worse than OMP=4 is the same story in the p2p kernel, which is
bandwidth-bound past four threads.

THE PENALTY IS LARGEST ON SMALL MODELS, which is what makes this matter
for the skin/corner studies specifically: measured against VoxHenry on
the same corpus, pinning the threads was worth 10.0x at 2104 cells and
only 1.10x at 604800, because the pathology is many small BLAS calls and
a coarse mesh is nearly all small calls.

These values were known -- they were written down as tuned optima -- but
NOTHING IN THE CODE SET THEM, so every run that did not export them by
hand paid the penalty. That is a defect, not a missing optimisation,
which is why this is a module rather than a README line.

USAGE: import this BEFORE numpy/scipy. OpenBLAS reads its environment
when the shared library loads, so a later assignment is silently
ignored. Every knob honours a value the caller has already set.

    import sppeec_threads             # noqa: F401  (must precede numpy)
    import numpy as np
"""
import os

# One BLAS thread. This is the big one; see the table above.
_DEFAULTS = {
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '4',        # p2p is bandwidth-bound past ~4
    'FFTW_THREADS_TOP': '6',       # top-level FFT only
}

applied = {}
for _k, _v in _DEFAULTS.items():
    if os.environ.get(_k) is None:
        os.environ[_k] = _v
        applied[_k] = _v


def enforce_blas(nthreads=None):
    """Set the BLAS thread count AT RUNTIME, whatever the import order.

    The environment variables above only work if this module is imported
    before numpy: OpenBLAS reads its environment when the shared object
    loads. That is a real constraint to place on a library's users, and
    one they will forget -- so this also asks the already-loaded OpenBLAS
    directly, which works whenever it is called.

    Returns the count set, or None if the running BLAS is not an OpenBLAS
    that exposes the setter (MKL and Accelerate do not; the environment
    variables above remain the fallback for those).
    """
    if nthreads is None:
        # Read the ENVIRONMENT, not the default: the block above only
        # filled it in if the caller had not already chosen. Someone who
        # exports OPENBLAS_NUM_THREADS=8 means it, and a library has no
        # business overriding that.
        nthreads = int(os.environ.get('SPPEEC_BLAS_THREADS')
                       or os.environ.get('OPENBLAS_NUM_THREADS')
                       or _DEFAULTS['OPENBLAS_NUM_THREADS'])
    try:
        import ctypes
        seen = set()
        for line in open('/proc/self/maps'):
            for tok in line.split():
                if '.so' in tok and 'openblas' in tok.lower():
                    seen.add(tok)
        for path in sorted(seen):
            try:
                lib = ctypes.CDLL(path)
            except OSError:
                continue
            for fn in ('openblas_set_num_threads',
                       'openblas_set_num_threads64_'):
                if hasattr(lib, fn):
                    getattr(lib, fn)(int(nthreads))
                    return int(nthreads)
    except Exception:
        pass
    return None


def report():
    """What is actually in force (for a solver banner or a log line)."""
    return {k: os.environ.get(k) for k in _DEFAULTS}
