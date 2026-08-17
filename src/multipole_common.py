# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Shared imports and optional-dependency guards for the
multipole package (numpy, toeplitz, the f2py extensions, scipy pieces,
and guarded GPU/cholmod).  Split out of multipole.py.
"""
import time
import numpy as np
import toeplitz as tp
import FMMtop
import f_setup
import mp_fortran
import meshgraph_aux
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import lsmr
from scipy.linalg.blas import zgemv
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
# sksparse.cholmod (SuiteSparse) is optional; only the RDF/Cholesky
# preconditioner path uses it. Guarded so import succeeds without scikit-sparse.
try:
    import sksparse.cholmod as cholmod
except ImportError:
    cholmod = None
# (legacy plotly/matplotlib/bokeh plotting scaffolding removed
# 2026-08-16 together with levels.plotslice; export fields with
# vtkout and view them in ParaView instead)

try:
    device = cuda.Device(0)
    attribute = pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK
    MAX_THREADS_PER_BLOCK = cuda.Device.get_attribute(device, attribute)
except Exception:
    # No CUDA device/driver available: GPU paths are disabled. Provide a
    # fallback so import-time references to MAX_THREADS_PER_BLOCK don't crash.
    device = None
    MAX_THREADS_PER_BLOCK = 1024
