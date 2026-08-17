# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""
Created on Wed Jun 11 21:28:34 2014

@author: timmy
"""

import numpy as np
import pyfftw
import os as _os

# FFT worker threads for the Toeplitz workspaces. pyfftw defaults to 1;
# nothing in this module ever passed a thread count, so every FFT path
# here has been single-threaded. Env-gated so the default stays exactly
# as measured.
_FFTW_THREADS = int(_os.environ.get('FFTW_THREADS', '1'))
# The TOP-LEVEL whole-domain transform is the one place threading pays
# (measured 2026-08-04: top ifft 237 -> 109 ms at 6 threads while the
# same setting made the per-leaf near field 2x SLOWER -- hence the
# per-instance split, see ToeplitzM2L). Consumed by TopLevel.topinit.
_FFTW_THREADS_TOP = int(_os.environ.get(
    'FFTW_THREADS_TOP', str(min(6, _os.cpu_count() or 1))))

# NOTE: pyfftw.n_byte_align_empty(shape, n, dtype) was removed from pyfftw.
# The modern equivalent is pyfftw.empty_aligned(shape, dtype=..., n=...).
# import scipy


class Toeplitz:
    def __init__(self, A, N):
        self.ndimA = A.ndim
        self.N = N
        self.fC = self.T3mult3(A)
        self.Atype = A.dtype

    def __mul__(self, X):
        return self.T3mult4(X)

    def T3mult3(self, T):
        NW = self.N[0]
        NL = self.N[1]
        NT = self.N[2]
        C = np.zeros((2*NW, 2*NL, 2*NT), dtype=T.dtype)
        C[:NW, :NL, :NT] = T[:NW, :NL, :NT]
        C[:NW, :NL, NT+1:2*NT] = T[:NW, :NL, NT-1:0:-1]
        C[:NW, NL+1:2*NL, :NT] = T[:NW, NL-1:0:-1, :NT]
        C[:NW, NL+1:2*NL, NT+1:2*NT] = T[:NW, NL-1:0:-1, NT-1:0:-1]
        C[NW+1:2*NW, :NL, :NT] = T[NW-1:0:-1, :NL, :NT]
        C[NW+1:2*NW, :NL, NT+1:2*NT] = T[NW-1:0:-1, :NL, NT-1:0:-1]
        C[NW+1:2*NW, NL+1:2*NL, :NT] = T[NW-1:0:-1, NL-1:0:-1, :NT]
        C[NW+1:2*NW, NL+1:2*NL, NT+1:2*NT] = T[NW-1:0:-1, NL-1:0:-1, NT-1:0:-1]
        C = np.fft.fftn(C)
        return C

    def T3mult4(self, X):
        NW = self.N[0]
        NL = self.N[1]
        NT = self.N[2]
        XX = np.zeros((2*NW, 2*NL, 2*NT), dtype=np.complex128)
        XX[:NW, :NL, :NT] = X[:NW, :NL, :NT]
        XX[:, :NL, :NT] = np.fft.fft(XX[:, :NL, :NT], 2*NW, 0)
        XX[:, :, :NT] = np.fft.fft(XX[:, :, :NT], 2*NL, 1)
        XX = np.fft.fft(XX, 2*NT, 2)
        XX = self.fC*XX
        XX = np.fft.ifft(XX, 2*NT, 2)
        XX[:, :, :NT] = np.fft.ifft(XX[:, :, :NT], 2*NL, 1)
        XX[:, :NL, :NT] = np.fft.ifft(XX[:, :NL, :NT], 2*NW, 0)
        XX = XX[:NW, :NL, :NT]
        return XX

    def unpack(self):
        NX = self.N[0]
        NY = self.N[1]
        NZ = self.N[2]
        NN = np.prod(self.N)
        U = np.zeros((NN, NN), dtype=self.Atype)
        for yl in range(NY):
            for zl in range(NZ):
                x = NX - (yl-zl)
                for xl in range(NL):
                    for yl in range(NL):
                        pass


def t3multnonsym3(T, NW, NL, NT):
    # C = pyfftw.empty_aligned((2*NW, 2*NL, 2*NT), dtype='complex128', n=16)
    # fftC = pyfftw.FFTW(C, C, axes=(0,1,2), flags=('FFTW_MEASURE',))
    # C[...] = 0
    C = np.zeros((2*NW, 2*NL, 2*NT), dtype=np.complex128)
    C[:NW, :NL, :NT] = T[NW-1:2*NW-1, NL-1:2*NL-1, NT-1:2*NT-1]
    C[:NW, :NL, NT+1:2*NT] = T[NW-1:2*NW-1, NL-1:2*NL-1, :NT-1]
    C[:NW, NL+1:2*NL, :NT] = T[NW-1:2*NW-1, :NL-1, NT-1:2*NT-1]
    C[:NW, NL+1:2*NL, NT+1:2*NT] = T[NW-1:2*NW-1, :NL-1, :NT-1]
    C[NW+1:2*NW, :NL, :NT] = T[:NW-1, NL-1:2*NL-1, NT-1:2*NT-1]
    C[NW+1:2*NW, :NL, NT+1:2*NT] = T[:NW-1, NL-1:2*NL-1, :NT-1]
    C[NW+1:2*NW, NL+1:2*NL, :NT] = T[:NW-1, :NL-1, NT-1:2*NT-1]
    C[NW+1:2*NW, NL+1:2*NL, NT+1:2*NT] = T[:NW-1, :NL-1, :NT-1]
    return np.fft.fftn(C)


def t3multnonsym4(T, nx, ny, nz):
    # nx is the size of the negative x of the T array
    (npx, npy, npz) = np.shape(T)
    npx, npy, npz = npx+1, npy+1, npz+1
    # npx is nx+px, where px is the size of the positive side of the T array
    C = np.zeros((npx, npy, npz), dtype=T.dtype)
    midx, midy, midz = npx - nx, npy - ny, npz - nz
    C[:midx, :midy, :midz] = T[nx-1:npx-1, ny-1:npy-1, nz-1:npz-1]
    C[:midx, :midy, midz+1:npz] = T[nx-1:npx-1, ny-1:npy-1, :nz-1]
    C[:midx, midy+1:npy, :midz] = T[nx-1:npx-1, :ny-1, nz-1:npz-1]
    C[:midx, midy+1:npy, midz+1:npz] = T[nx-1:npx-1, :ny-1, :nz-1]
    C[midx+1:npx, :midy, :midz] = T[:nx-1, ny-1:npy-1, nz-1:npz-1]
    C[midx+1:npx, :midy, midz+1:npz] = T[:nx-1, ny-1:npy-1, :nz-1]
    C[midx+1:npx, midy+1:npy, :midz] = T[:nx-1, :ny-1, nz-1:npz-1]
    C[midx+1:npx, midy+1:npy, midz+1:npz] = T[:nx-1, :ny-1, :nz-1]
    return np.fft.fftn(C)


def T3mult5(fC, fX, NW, NL, NT):
    YY = np.fft.ifftn(fC*fX)
    return YY[:NW, :NL, :NT]


def t3mult6(X, N):
    NW = N.x
    NL = N.y
    NT = N.z
    XX = np.zeros((2*NW, 2*NL, 2*NT), dtype=np.complex128)
    XX[:NW, :NL, :NT] = X[:NW, :NL, :NT]
    XX[:, :NL, :NT] = np.fft.fft(XX[:, :NL, :NT], 2*NW, 0)
    XX[:, :, :NT] = np.fft.fft(XX[:, :, :NT], 2*NL, 1)
    XX = np.fft.fft(XX, 2*NT, 2)
    return XX


class ToeplitzP2P():
    def __init__(self, n_size):
        self.n_x = n_size[0]
        self.n_y = n_size[1]
        self.n_z = n_size[2]
        self.a = pyfftw.empty_aligned((2*self.n_x, self.n_y, self.n_z),
                                           dtype='complex128', n=16)
        self.b = pyfftw.empty_aligned((2*self.n_x, 2*self.n_y, self.n_z),
                                           dtype='complex128', n=16)
        self.c = pyfftw.empty_aligned((2*self.n_x, 2*self.n_y,
                                            2*self.n_z),
                                           dtype='complex128', n=16)
        self.fftab = pyfftw.FFTW(self.a, self.a, axes=(0,),
                                 flags=('FFTW_MEASURE', ), threads=_FFTW_THREADS)
        self.fftbc = pyfftw.FFTW(self.b, self.b, axes=(1,),
                                 flags=('FFTW_MEASURE', ), threads=_FFTW_THREADS)
        self.fftcc = pyfftw.FFTW(self.c, self.c, axes=(2,),
                                 flags=('FFTW_MEASURE', ), threads=_FFTW_THREADS)
        self.ifftcc = pyfftw.FFTW(self.c, self.c, axes=(2,),
                                  flags=('FFTW_MEASURE', ), threads=_FFTW_THREADS,
                                  direction='FFTW_BACKWARD')
        self.ifftcb = pyfftw.FFTW(self.b, self.b, axes=(1,),
                                  flags=('FFTW_MEASURE', ), threads=_FFTW_THREADS,
                                  direction='FFTW_BACKWARD')
        self.ifftba = pyfftw.FFTW(self.a, self.a, axes=(0,),
                                  flags=('FFTW_MEASURE', ), threads=_FFTW_THREADS,
                                  direction='FFTW_BACKWARD')

    def fft(self, x_values):
        self.a[:] = 0.0
        self.a[:self.n_x, :, :] = x_values
        self.a = self.fftab()
        self.b[:, :self.n_y, :] = self.a
        self.b = self.fftbc()
        self.c[:, :, :self.n_z] = self.b
        self.c = self.fftcc()
        return self.c

    def ifft(self, x_values):
        self.c[:, :, :] = x_values
        self.c = self.ifftcc()
        self.b[:, :, :] = self.c[:, :, :self.n_z]
        self.b = self.ifftcb()
        self.a[:, :, :] = self.b[:, :self.n_y, :]
        self.a = self.ifftba()
        return self.a[:self.n_x, :, :]


class ToeplitzCoeffPoten():
    def __init__(self, nin, nout, ngroups):
        self.n_x = nin[0]
        self.n_y = nin[1]
        self.n_z = nin[2]
        self.a = pyfftw.empty_aligned((ngroups, nout[0], nin[1], nin[2]),
                                           dtype='complex128', n=16)
        self.b = pyfftw.empty_aligned((ngroups, nout[0], nout[1], nin[2]),
                                           dtype='complex128', n=16)
        self.c = pyfftw.empty_aligned((ngroups, nout[0], nout[1],
                                            nout[2]), dtype='complex128', n=16)
        # Plans MUST be built here, on the still-empty buffers, exactly as
        # ToeplitzM2L does. FFTW_MEASURE planning EXECUTES candidate
        # transforms on the arrays it is handed, so constructing these
        # lazily inside fftab()/fftbc()/... (the original code) destroyed
        # the caller's data before it was ever transformed -- the source
        # slab arrived at the p2p kernel identically zero.
        self.fftab = pyfftw.FFTW(self.a, self.a, axes=(1,),
                                 flags=('FFTW_MEASURE', ), threads=_FFTW_THREADS)
        self.fftbc = pyfftw.FFTW(self.b, self.b, axes=(2,),
                                 flags=('FFTW_MEASURE', ), threads=_FFTW_THREADS)
        self.fftcc = pyfftw.FFTW(self.c, self.c, axes=(3,),
                                 flags=('FFTW_MEASURE', ), threads=_FFTW_THREADS)
        self.ifftcc = pyfftw.FFTW(self.c, self.c, axes=(3,),
                                  flags=('FFTW_MEASURE', ), threads=_FFTW_THREADS,
                                  direction='FFTW_BACKWARD')
        self.ifftcb = pyfftw.FFTW(self.b, self.b, axes=(2,),
                                  flags=('FFTW_MEASURE', ), threads=_FFTW_THREADS,
                                  direction='FFTW_BACKWARD')
        self.ifftba = pyfftw.FFTW(self.a, self.a, axes=(1,),
                                  flags=('FFTW_MEASURE', ), threads=_FFTW_THREADS,
                                  direction='FFTW_BACKWARD')


class ToeplitzM2L():
    def __init__(self, n_size, nnmax, threads=None):
        # threads: per-instance FFTW thread count. Default is the
        # module-level _FFTW_THREADS (1) -- right for the per-leaf-slab
        # instances, where threading is a measured NET LOSS. The
        # whole-domain TOP-LEVEL instance is the one place threading
        # pays (it gets _FFTW_THREADS_TOP from TopLevel.topinit); the
        # old module-global constant could not express that split.
        self.n_x = n_size[0]
        self.n_y = n_size[1]
        self.n_z = n_size[2]
        self.nnmax = nnmax
        _threads = _FFTW_THREADS if threads is None else int(threads)
        self.a = pyfftw.empty_aligned((self.nnmax, 2*self.n_x, self.n_y,
                                            self.n_z), dtype='complex128', n=16)
        self.b = pyfftw.empty_aligned((self.nnmax, 2*self.n_x, 2*self.n_y,
                                            self.n_z), dtype='complex128', n=16)
        self.c = pyfftw.empty_aligned((self.nnmax, 2*self.n_x, 2*self.n_y,
                                            2*self.n_z), dtype='complex128', n=16)
        self.fftab = pyfftw.FFTW(self.a, self.a, axes=(1,),
                                 flags=('FFTW_MEASURE', ), threads=_threads)
        self.fftbc = pyfftw.FFTW(self.b, self.b, axes=(2,),
                                 flags=('FFTW_MEASURE', ), threads=_threads)
        self.fftcc = pyfftw.FFTW(self.c, self.c, axes=(3,),
                                 flags=('FFTW_MEASURE', ), threads=_threads)
        self.ifftcc = pyfftw.FFTW(self.c, self.c, axes=(3,),
                                  flags=('FFTW_MEASURE', ), threads=_threads,
                                  direction='FFTW_BACKWARD')
        self.ifftcb = pyfftw.FFTW(self.b, self.b, axes=(2,),
                                  flags=('FFTW_MEASURE', ), threads=_threads,
                                  direction='FFTW_BACKWARD')
        self.ifftba = pyfftw.FFTW(self.a, self.a, axes=(1,),
                                  flags=('FFTW_MEASURE', ), threads=_threads,
                                  direction='FFTW_BACKWARD')

    def fft(self, x_values):
        # The PADDING of b and c must be zero for the padded transform
        # -- the p2p caller zeroes all three buffers itself before
        # driving these plans, but this method's other user (the
        # top-level m2lfortran forward) relies on it happening here.
        # Stale padding is not a subtle error: it NaNs the whole
        # transfer (found the hard way 2026-08-06).
        self.a[:] = 0.0
        self.b[:, :, self.n_y:, :] = 0.0
        self.c[:, :, :, self.n_z:] = 0.0
        self.a[:, :self.n_x, :, :] = x_values
        self.a = self.fftab()
        self.b[:, :, :self.n_y, :] = self.a
        self.b = self.fftbc()
        self.c[:, :, :, :self.n_z] = self.b
        self.c = self.fftcc()
        return self.c

    def ifft(self, x_values):
        self.c[:, :, :, :] = x_values
        self.c = self.ifftcc()
        self.b[:, :, :, :] = self.c[:, :, :, :self.n_z]
        self.b = self.ifftcb()
        self.a[:, :, :, :] = self.b[:, :, :self.n_y, :]
        self.a = self.ifftba()
        return self.a[:, :self.n_x, :, :]


def t3mult7(x_values, nnmax, n_size):
    """ Same as t3mult6, but includes 4th dimension for nnmax"""
    n_w = n_size[0]
    n_l = n_size[1]
    n_t = n_size[2]
    x_x = np.zeros((nnmax, 2*n_w, 2*n_l, 2*n_t), dtype=np.complex128)
    x_x[:, :n_w, :n_l, :n_t] = x_values[:, :n_w, :n_l, :n_t]
    x_x[:, :, :n_l, :n_t] = np.fft.fft(x_x[:, :, :n_l, :n_t], 2*n_w, 1)
    x_x[:, :, :, :n_t] = np.fft.fft(x_x[:, :, :, :n_t], 2*n_l, 2)
    x_x = np.fft.fft(x_x, 2*n_t, 3)
    return x_x


def t3mult8(Lnm, n_size):
    n_w = n_size.W
    n_l = n_size.L
    n_t = n_size.T
    Lnm = np.fft.ifft(Lnm, 2*n_t, 2)
    Lnm = np.fft.ifft(Lnm[:, :, :n_t, :], 2*n_l, 1)
    Lnm = np.fft.ifft(Lnm[:, :n_l, :n_t, :], 2*n_w, 0)
    return Lnm


def t3mult9(points, n_size):
    n_w = n_size.x
    n_l = n_size.y
    n_t = n_size.z
    points = np.fft.ifft(points, 2*n_t, 2)
    points[:, :, :n_t] = np.fft.ifft(points[:, :, :n_t], 2*n_l, 1)
    points[:, :n_l, :n_t] = np.fft.ifft(points[:, :n_l, :n_t], 2*n_w, 0)
    return points[:n_w, :n_l, :n_t]


class ToeplitzMS2LS():
    def __init__(self, n_size, nnmax):
        self.n_w = n_size.W//3
        self.n_l = n_size.L//3
        self.n_t = n_size.T//3
        self.nnmax = nnmax
        self.a = pyfftw.empty_aligned((self.nnmax, 6*self.n_w, 3*self.n_l,
                                            3*self.n_t), dtype='complex128', n=16)
        self.b = pyfftw.empty_aligned((self.nnmax, 6*self.n_w, 6*self.n_l,
                                            3*self.n_t), dtype='complex128', n=16)
        self.c = pyfftw.empty_aligned((self.nnmax, 6*self.n_w, 6*self.n_l,
                                            6*self.n_t), dtype='complex128', n=16)
        self.d = pyfftw.empty_aligned((self.nnmax, 6*self.n_w, 6*self.n_l,
                                            self.n_t), dtype='complex128', n=16)
        self.e = pyfftw.empty_aligned((self.nnmax, 6*self.n_w, self.n_l,
                                            self.n_t), dtype='complex128', n=16)
        self.fftab = pyfftw.FFTW(self.a, self.a, axes=(1, 2, 3),
                                 flags=('FFTW_PATIENT', ))
        self.fftbc = pyfftw.FFTW(self.b, self.b, axes=(1, 2, 3),
                                 flags=('FFTW_PATIENT', ))
        self.fftcc = pyfftw.FFTW(self.c, self.c, axes=(1, 2, 3),
                                 flags=('FFTW_PATIENT', ))
        self.ifftcc = pyfftw.FFTW(self.c, self.c, axes=(1, 2, 3),
                                  flags=('FFTW_PATIENT', ),
                                  direction='FFTW_BACKWARD')
        self.ifftcd = pyfftw.FFTW(self.d, self.d, axes=(1, 2, 3),
                                  flags=('FFTW_PATIENT', ),
                                  direction='FFTW_BACKWARD')
        self.ifftde = pyfftw.FFTW(self.e, self.e, axes=(1, 2, 3),
                                  flags=('FFTW_PATIENT', ),
                                  direction='FFTW_BACKWARD')

    def fft(self, x_values):
        self.a[:] = 0.0
        self.a[:, :3*self.n_w, :, :] = x_values
        self.a = self.fftab()
        self.b[:, :, :3*self.n_l, :] = self.a
        self.b = self.fftbc()
        self.c[:, :, :, :3*self.n_t] = self.b
        self.c = self.fftcc()
        return self.c

    def ifft(self, x_values):
        self.c[:] = 0.0
        self.c[:, :6*self.n_w, :6*self.n_l, :6*self.n_t] = x_values
        self.c = self.ifftcc()
        self.d = self.c[:, :, :, self.n_t:2*self.n_t]
        self.d = self.ifftcd()
        self.e = self.d[:, :, self.n_l:2*self.n_l, :]
        self.e = self.ifftde()
        return self.e[:, self.n_w:2*self.n_w, :, :]
