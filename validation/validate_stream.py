# -*- coding: utf-8 -*-
"""Validator: the streamed Krylov solver (krylov_stream.gmres_stream).

Dense complex system, spectrum over three decades plus a non-normal
part, and a lopsided left preconditioner (exact on the large
eigen-directions, crude on the small ones) so that |M r| and |r|
disagree as they do on the DBC. Checks:

  1. full GMRES (no restart) reaches the direct solve;
  2. short cycles (restart 8) WITH the LGMRES augmentation reach it
     too, and in no more matvecs than scipy's lgmres with the same
     inner_m and outer_k (the algorithm it reproduces; measured
     2026-09-30 at 279 vs 278 on this system), within a margin;
  3. the plain restart (augmentation off) is the slower solver, which
     is the reason the augmentation exists (R6: 20-step plain cycles
     did not converge at all).

Runs on the host in seconds; the basis files go to the stream
directory (~/.cache/sppeec/krylov) as in a real solve.
"""
import sys

import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres

import krylov_stream as ks

rng = np.random.default_rng(1)
n = 600
d = np.exp(rng.uniform(0, np.log(1e3), n))*np.exp(1j*rng.uniform(-0.3, 0.3, n))
Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n)))
A = (Q*d) @ Q.conj().T + 0.02*rng.standard_normal((n, n))
Minv = np.linalg.inv(A + 300.0*np.eye(n))
b = rng.standard_normal(n) + 1j*rng.standard_normal(n)
xref = np.linalg.solve(A, b)
bnorm = np.linalg.norm(b)
Mop = LinearOperator((n, n), matvec=lambda v: Minv @ v, dtype=np.complex128)
RTOL = 1e-8
ks._VERBOSE = False


def counted():
    cnt = [0]

    def mv(v):
        cnt[0] += 1
        return A @ v
    return LinearOperator((n, n), matvec=mv, dtype=np.complex128), cnt


def stream(K, restart):
    ks.AUG_K = K
    Aop, cnt = counted()
    x, flag, nmv = ks.gmres_stream(Aop, b, Mop, rtol=RTOL, budget=800,
                                   restart=restart,
                                   basis_dtype=np.complex128)
    assert cnt[0] == nmv, "matvec accounting %d vs %d" % (cnt[0], nmv)
    return (flag, nmv, np.linalg.norm(b - A @ x)/bnorm,
            np.linalg.norm(x - xref)/np.linalg.norm(xref))


def scipy_count(K, inner_m):
    Aop, cnt = counted()
    x, flag = lgmres(Aop, b, M=Mop, rtol=RTOL, atol=0.0, inner_m=inner_m,
                     outer_k=K, maxiter=200)
    return flag, cnt[0]


ok = True
full = stream(0, 800)
print("full GMRES        : flag %d  matvecs %3d  |r|/|b| %.1e  err %.1e"
      % full)
ok &= full[0] == 0 and full[2] <= RTOL and full[3] < 1e-6
plain = stream(0, 8)
print("restart 8, plain  : flag %d  matvecs %3d  |r|/|b| %.1e  err %.1e"
      % plain)
aug = stream(3, 8)
print("restart 8, 3 pairs: flag %d  matvecs %3d  |r|/|b| %.1e  err %.1e"
      % aug)
ok &= aug[0] == 0 and aug[2] <= RTOL and aug[3] < 1e-6
sflag, snmv = scipy_count(3, 8)
print("scipy lgmres 8+3  : flag %d  matvecs %3d" % (sflag, snmv))
ok &= aug[1] <= snmv*1.05 + 2
ok &= plain[0] != 0 or plain[1] > aug[1]
print("PASS" if ok else "FAIL: streamed LGMRES")
sys.exit(0 if ok else 1)
