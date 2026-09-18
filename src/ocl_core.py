# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""OpenCL runtime for the device operator: programs, buffers, transforms.

The CUDA paths are written as CuPy array expressions. Transliterating
them would reproduce their traffic, and that traffic is the reason the
device only returns about twice the host's speed on a card with ten
times its bandwidth: a CuPy expression materialises every temporary and
re-reads it, so the top-level M2L moves roughly 14.7 GB per call on the
R3 flagship where the arithmetic needs 0.6.

So the OpenCL path is written as fused kernels instead, one per phase,
with the operands tiled into local memory. This module holds what they
all need: the program cache, the transform cache, and thin buffer
helpers. The kernels themselves live next to the phase they implement.

Precision: complex128 is ``double2`` and needs ``cl_khr_fp64``, which
every discrete GPU worth running this on reports. Parts whose double
precision is emulated (some Intel generations) want the complex64 path
instead; :func:`ctype` names the scalar so a program can be built
either way from one source.
"""
import numpy as np

import backend

_programs = {}
_ffts = {}


def queue(index=0):
    """The command queue for device ``index``."""
    return backend.ocl_queue(index)


def context():
    """The OpenCL context."""
    return backend.ocl_context()


def ctype(dtype):
    """``('double', 'double2')`` or ``('float', 'float2')``."""
    dt = np.dtype(dtype)
    if dt == np.complex128:
        return 'double', 'double2'
    if dt == np.complex64:
        return 'float', 'float2'
    raise TypeError("no OpenCL complex type for %s" % dt)


PRELUDE = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef REAL real_t;
typedef CPLX cplx_t;

inline cplx_t cmul(cplx_t a, cplx_t b) {
    return (cplx_t)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
inline cplx_t cconj(cplx_t a) { return (cplx_t)(a.x, -a.y); }
"""


def program(source, dtype=np.complex128, defines=None, key=None):
    """Build (once) and return a program from ``source``.

    ``source`` is prefixed with :data:`PRELUDE`, which supplies the
    complex type and helpers. ``defines`` become ``-D`` options, so the
    harmonic counts and tile sizes are compile-time constants and the
    inner loops unroll.
    """
    real, cplx = ctype(dtype)
    opts = ['-DREAL=%s' % real, '-DCPLX=%s' % cplx]
    for k, v in sorted((defines or {}).items()):
        opts.append('-D%s=%s' % (k, v))
    ck = (key or source, tuple(opts))
    prg = _programs.get(ck)
    if prg is None:
        import pyopencl as cl
        prg = cl.Program(context(), PRELUDE + source).build(options=opts)
        _programs[ck] = prg
    return prg


_kernels = {}


def kernel(prg, fname):
    """A cached kernel handle.

    ``prg.name(...)`` builds a fresh kernel object on every call, which
    pyopencl warns about and which shows up as real time when a kernel
    is enqueued once per slab per matvec.
    """
    key = (id(prg), fname)
    k = _kernels.get(key)
    if k is None:
        import pyopencl as cl
        k = cl.Kernel(prg, fname)
        _kernels[key] = k
    return k


def empty(shape, dtype=np.complex128, index=0):
    """An uninitialised device array."""
    import pyopencl.array as cla
    return cla.empty(queue(index), shape, dtype=dtype)


def zeros(shape, dtype=np.complex128, index=0):
    """A zeroed device array."""
    import pyopencl.array as cla
    return cla.zeros(queue(index), shape, dtype=dtype)


CHUNK = 64 << 20          # bytes per upload slice


def to_device(a, dtype=None, index=0):
    """Device copy of host array ``a``, allocated empty and filled.

    NOT ``pyopencl.array.to_device``. That creates the buffer with the
    host pointer copied in, and the runtime then keeps that host copy
    resident for the life of the buffer: measured 1.5 GB of resident
    memory for a 1.5 GB array, still there after the host array is
    freed. On R4 the top-level M2L's channel spectra alone are 1.8 GB,
    which is most of the 2.1 GB by which this backend's peak sat above
    the CUDA one.

    So the buffer is allocated empty and filled by copy, in slices, the
    same discipline :mod:`gpu_xfer` applies to the CUDA driver's
    staging arena for the same reason. Allocate-then-copy costs nothing
    resident.
    """
    import pyopencl as cl
    import pyopencl.array as cla
    a = np.ascontiguousarray(a if dtype is None
                             else a.astype(dtype, copy=False))
    q = queue(index)
    d = cla.empty(q, a.shape, dtype=a.dtype)
    if a.nbytes:
        flat = a.reshape(-1)
        step = max(1, CHUNK//a.dtype.itemsize)
        for i in range(0, flat.size, step):
            cl.enqueue_copy(q, d.data, flat[i:i + step],
                            device_offset=i*a.dtype.itemsize)
        q.finish()
    return d


def to_host(d, out=None):
    """Host copy of device array ``d``."""
    return d.get(ary=out)


def fft_app(shape, dtype=np.complex128, ndim=3, index=0):
    """A cached VkFFT plan for an in-place batched transform.

    ``shape`` is the whole buffer shape; the last ``ndim`` axes are
    transformed and the leading ones are the batch. The inverse carries
    the 1/N normalisation, matching numpy and CuPy.
    """
    key = (tuple(int(s) for s in shape), np.dtype(dtype).str, int(ndim),
           int(index))
    app = _ffts.get(key)
    if app is None:
        from pyvkfft.opencl import VkFFTApp
        app = VkFFTApp(tuple(int(s) for s in shape), np.dtype(dtype),
                       queue=queue(index), ndim=int(ndim), inplace=True)
        _ffts[key] = app
    return app


def clear():
    """Drop cached programs and transforms (frees their device memory)."""
    _programs.clear()
    _ffts.clear()
    _kernels.clear()
