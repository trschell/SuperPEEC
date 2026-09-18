# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Device backend selection: which library drives the GPU, if any.

The solver's device paths were written directly against CuPy/CUDA. This
module is the seam that lets a second backend (PyOpenCL, for AMD and
Intel Arc parts) supply the same facilities, and it is the only place
that decides which one is in play.

Selection
---------
``SPPEEC_BACKEND`` picks the backend explicitly:

``auto`` (default)
    CUDA when CuPy imports and reports a device, otherwise no device.
    This is exactly the behaviour that predates this module: OpenCL is
    never selected implicitly, so an existing install cannot change
    what it runs by accident.
``cuda``
    CuPy. Fails over to no device the same way ``auto`` does.
``opencl``
    PyOpenCL. Opt-in only.
``none``
    Host paths everywhere.

``SPPEEC_GPU`` keeps its established meaning as the master gate and is
checked first: ``'0'`` disables every device path regardless of
backend, ``'1'`` forces one on and makes failures loud, anything else
probes. The per-phase gates (``SPPEEC_GPU_P2P``, ``SPPEEC_GPU_LEAF``,
``SPPEEC_MODE_APPLY_GPU``, ...) are unchanged and still live at their
call sites; :func:`probe` takes one as an argument.

What a backend supplies
-----------------------
:func:`array_module` returns the array namespace (``cupy`` for CUDA)
and :func:`sparse_module` the sparse one. For CUDA these are the real
modules, so call sites behave exactly as they did before this seam
existed. The remaining functions cover the things that are not array
operations: device counting, placement context, free-memory queries
and cache release. Transfers live in :mod:`gpu_xfer`, which takes its
array module from here when the caller does not pass one.

Porting status: the CUDA backend is complete. The OpenCL backend
reports its devices and hands out a context and a queue; the array and
sparse namespaces are not built yet and raise a clear error naming the
phase that adds them.
"""
import os as _os
import warnings as _warnings
from contextlib import contextmanager as _contextmanager

__all__ = ['name', 'available', 'probe', 'device_count', 'array_module',
           'sparse_module', 'memory_info', 'device', 'free_pools',
           'forced', 'disabled', 'reset', 'describe', 'scatter_add',
           'device_memory_total',
           'ocl_context', 'ocl_queue', 'BackendUnavailable']

_CUDA, _OPENCL, _NONE = 'cuda', 'opencl', 'none'

# resolved lazily and cached; reset() clears (tests flip the env vars)
_state = {}


class BackendUnavailable(RuntimeError):
    """The selected backend cannot supply what a call site asked for."""


# --------------------------------------------------------------- gates


def disabled():
    """True when ``SPPEEC_GPU=0`` switches every device path off."""
    return _os.environ.get('SPPEEC_GPU', 'auto') == '0'


def forced():
    """True when ``SPPEEC_GPU=1`` demands a device (failures are loud)."""
    return _os.environ.get('SPPEEC_GPU') == '1'


def _requested():
    """The backend the environment asks for, before probing hardware."""
    if disabled():
        return _NONE
    want = _os.environ.get('SPPEEC_BACKEND', 'auto').strip().lower()
    if want in (_CUDA, _OPENCL, _NONE):
        return want
    if want not in ('auto', ''):
        _warnings.warn("SPPEEC_BACKEND=%r is not one of auto/cuda/opencl/"
                       "none -- treating it as auto" % want)
    return 'auto'


# ------------------------------------------------------------ selection


def _try_cuda():
    """The cupy module when a CUDA device is present, else None."""
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception:
        pass
    return None


def _try_opencl():
    """The pyopencl module when an OpenCL device is present, else None."""
    try:
        import pyopencl
        for plat in pyopencl.get_platforms():
            if plat.get_devices(pyopencl.device_type.GPU):
                return pyopencl
    except Exception:
        pass
    return None


def _resolve():
    """Pick the backend once and remember it.

    The hardware probe is cached, but ``SPPEEC_GPU=0`` is re-read every
    call and never cached: a process that flips the master gate (the
    device validators do) must see the change, and caching a 'none'
    that came from the gate rather than from the hardware would pin
    the process to the host paths for good.
    """
    if disabled():
        return _NONE
    if 'name' in _state:
        return _state['name']
    want = _requested()
    chosen, mod = _NONE, None
    if want == _NONE:
        pass
    elif want == _CUDA or want == 'auto':
        mod = _try_cuda()
        chosen = _CUDA if mod is not None else _NONE
        if mod is None and want == _CUDA and forced():
            _warnings.warn("SPPEEC_BACKEND=cuda and SPPEEC_GPU=1 but no "
                           "CUDA device is usable -- host paths")
    elif want == _OPENCL:
        mod = _try_opencl()
        chosen = _OPENCL if mod is not None else _NONE
        if mod is None:
            _warnings.warn("SPPEEC_BACKEND=opencl but no OpenCL GPU device "
                           "is usable -- host paths")
    _state['name'] = chosen
    _state['module'] = mod
    return chosen


def reset():
    """Forget the resolved backend (the environment may have changed)."""
    _state.clear()


def name():
    """``'cuda'``, ``'opencl'`` or ``'none'``."""
    return _resolve()


def available():
    """True when a device backend is in play."""
    return _resolve() != _NONE


def describe():
    """One line naming the backend and its device, for logs and status."""
    n = _resolve()
    if n == _NONE:
        return 'no device backend (host paths)'
    if n == _CUDA:
        try:
            cp = _state['module']
            props = cp.cuda.runtime.getDeviceProperties(
                cp.cuda.runtime.getDevice())
            return 'cuda: %s (%d device(s))' % (
                props['name'].decode(), device_count())
        except Exception:
            return 'cuda: %d device(s)' % device_count()
    try:
        return 'opencl: %s' % ocl_context().devices[0].name.strip()
    except Exception:
        return 'opencl: %d device(s)' % device_count()


def probe(gate=None, force_on_error=True):
    """True when this call site should take its device path.

    ``gate`` names an additional environment variable that opts the
    phase out when set to ``'0'`` (``SPPEEC_GPU_P2P`` and friends).
    ``force_on_error`` reproduces the established behaviour of the
    top-level M2L probe, which takes the device path on ``SPPEEC_GPU=1``
    even when the hardware query itself fails, so that the failure is
    reported by the path rather than swallowed by the probe.
    """
    if disabled():
        return False
    if gate is not None and _os.environ.get(gate, 'auto') == '0':
        return False
    try:
        return device_count() > 0
    except Exception:
        return bool(force_on_error) and forced()


# ------------------------------------------------------------ namespaces


def array_module():
    """The array namespace: ``cupy`` under CUDA.

    Raises :class:`BackendUnavailable` when there is no device backend,
    so that a call site which probed first never sees the exception and
    one which did not gets a clear message instead of an ImportError.
    """
    n = _resolve()
    if n == _CUDA:
        return _state['module']
    if n == _OPENCL:
        raise BackendUnavailable(
            "the OpenCL backend has no array namespace yet (port phase 1 "
            "supplies the fused operator kernels); run with "
            "SPPEEC_BACKEND=cuda or none")
    raise BackendUnavailable("no device backend is available")


def sparse_module():
    """The sparse namespace: ``cupyx.scipy.sparse`` under CUDA."""
    n = _resolve()
    if n == _CUDA:
        import cupyx.scipy.sparse as csp
        return csp
    if n == _OPENCL:
        raise BackendUnavailable(
            "the OpenCL backend has no sparse namespace yet (port phase 2 "
            "supplies a deterministic CSR product); run with "
            "SPPEEC_BACKEND=cuda or none")
    raise BackendUnavailable("no device backend is available")


# --------------------------------------------------------------- devices


def device_count():
    """Number of usable devices; 0 without a backend."""
    n = _resolve()
    if n == _CUDA:
        return int(_state['module'].cuda.runtime.getDeviceCount())
    if n == _OPENCL:
        return len(ocl_context().devices)
    return 0


@_contextmanager
def device(index):
    """Run a block with device ``index`` current.

    CUDA has a current-device notion and the multi-device GeoMG split
    relies on it. OpenCL addresses devices through queues instead, so
    this is a no-op there and placement will be explicit.
    """
    n = _resolve()
    if n == _CUDA and index is not None:
        with _state['module'].cuda.Device(int(index)):
            yield
    else:
        yield


def device_memory_total():
    """Total bytes of VRAM on the local device, or ``None``.

    A hardware fact, not a device path: the tree cost model asks for it
    to size its recommendation even when ``SPPEEC_GPU=0`` has switched
    the device paths off, so this deliberately ignores the master gate.
    """
    mod = _try_cuda()
    if mod is not None:
        try:
            return int(mod.cuda.runtime.memGetInfo()[1])
        except Exception:
            pass
    mod = _try_opencl()
    if mod is not None:
        try:
            for plat in mod.get_platforms():
                devs = plat.get_devices(mod.device_type.GPU)
                if devs:
                    return int(devs[0].global_mem_size)
        except Exception:
            pass
    return None


def memory_info(index=None):
    """``(free, total)`` device bytes, or ``None`` when unknown.

    OpenCL has no portable free-memory query, so it reports ``None``
    for the free half and placement decisions must fall back to a
    declared budget (``SPPEEC_GPU_BUDGET_GB``) with allocation failure
    as the signal.
    """
    n = _resolve()
    if n == _CUDA:
        try:
            with device(index):
                return tuple(int(v) for v in
                             _state['module'].cuda.runtime.memGetInfo())
        except Exception:
            return None
    if n == _OPENCL:
        try:
            dev = ocl_context().devices[0 if index is None else int(index)]
            return (None, int(dev.global_mem_size))
        except Exception:
            return None
    return None


def free_pools(device=True, pinned=True):
    """Release cached device (and host staging) blocks, where that exists.

    The CUDA allocator keeps freed blocks in a pool and the pinned host
    pool keeps transfer staging buffers; both show up in the resident
    set and both are released explicitly at the points where the solver
    has just dropped something large. Silent no-op on backends without
    a pool.
    """
    n = _resolve()
    if n != _CUDA:
        return
    cp = _state['module']
    if device:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    if pinned:
        try:
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


def scatter_add(target, indices, values):
    """``target[indices] += values``, accumulating repeated indices.

    CUDA runs this with atomics, so the summation order is not fixed
    and repeated calls are not bit-reproducible. That is tolerated only
    where the caller needs agreement at rounding level; it must never
    be used inside the preconditioner apply, whose map has to be the
    same every call or a long GMRES cycle stalls. Where the index set
    is collision-free, prefer a plain indexed write.
    """
    n = _resolve()
    if n == _CUDA:
        import cupyx
        cupyx.scatter_add(target, indices, values)
        return
    raise BackendUnavailable(
        "scatter-add is not implemented for the %s backend" % n)


# ---------------------------------------------------------------- OpenCL


def ocl_context():
    """The process-wide OpenCL context (GPU devices of one platform)."""
    ctx = _state.get('ocl_ctx')
    if ctx is not None:
        return ctx
    if _resolve() != _OPENCL:
        raise BackendUnavailable("the OpenCL backend is not selected")
    import pyopencl as cl
    devs = []
    for plat in cl.get_platforms():
        devs = plat.get_devices(cl.device_type.GPU)
        if devs:
            break
    if not devs:
        raise BackendUnavailable("no OpenCL GPU device")
    ctx = cl.Context(devices=devs)
    _state['ocl_ctx'] = ctx
    return ctx


def ocl_queue(index=0):
    """A command queue on device ``index`` of the OpenCL context."""
    key = 'ocl_q%d' % int(index)
    q = _state.get(key)
    if q is None:
        import pyopencl as cl
        ctx = ocl_context()
        q = cl.CommandQueue(ctx, device=ctx.devices[int(index)])
        _state[key] = q
    return q
