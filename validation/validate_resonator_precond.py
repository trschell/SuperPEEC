# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Reluctance preconditioner on the RLC resonator, THROUGH its resonance.

All reluctance-preconditioner validation so far ran on setup1, which sits
~500 GHz below its self-resonance -- the capacitance there is a small
perturbation. This test drives the PRODUCTION preconditioner (SystemMat.
reluctanceprecinit / precondReluctance / rescaleLpPR, exec'd straight out of
main.py) on the folded parallel-plate resonator of validate_rlc_resonance.py
(f_res = 59.73 GHz, verified there), swept 0.1x-10x through resonance -- the
capacitance-DOMINATED regime where the S~ = -jw*C_cap - (1/jw)*A^T K~ A
Schur approximation actually has to balance its two terms.

The geometry is also structurally adversarial: thin plates mean ALL nodes are
external (no internal/bulk nodes) -- this exercises the all-external gauge
fallback in reluctanceprecinit/loopprecinit (originally both indexed the
first INTERNAL node and would crash here), and the windowed K~ extraction on
a boundary-dominated (non-cube) filament set.

This test CAUGHT A REAL BUG on first contact: the original preconditioner
pinned a gauge node in S~ and zeroed the matching residual entry in the
apply, making M singular along that node. On setup1 the pinned node was
internal (pure continuity, no capacitance) and the defect was invisible;
here every node carries capacitance, and GMRES converged its preconditioned
residual to 1e-8 while the currents were off by up to 140%. The fix (S~
factored as-is -- nonsingular, since -jw C_cap breaks the Laplacian null
space -- and the residual untouched) is what this file now validates.

Checks:
  1. the scalable kernel K~ extraction still matches the dense-extracted K~
     on this thin-plate geometry;
  2. production reluctance GMRES converges at every frequency INCLUDING at
     resonance, with currents matching a dense direct solve;
  3. the resonance peak |Z(f_res)| is reproduced through the iterative solve;
  4. through resonance the reluctance preconditioner DEGRADES LESS than the
     IDEAL inductive saddle (dense pinv of [[Z,A],[A^T,0]], exact Z -- the
     best any loop-reduction preconditioner could do), because it carries
     the capacitance the saddle ignores;
  5. the ccap='full' option (full P_ext^{-1} block in S~, not just the
     diagonal) sharply cuts iterations at resonance -- the off-diagonal
     capacitive coupling is what S~ must balance there.

Run inside the toolbox:  python3 validate_resonator_precond.py
Exits nonzero on failure.
"""
import os as _op
import sys as _sp
_sp.path[:0] = [_op.path.join(_op.path.dirname(
    _op.path.abspath(__file__)), _d) for _d in ('../src', '.')]

import sys
import numpy as np
import stencils as st

FAIL = []
def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAIL.append(name)

# ---- production SystemMat straight from main.py (exec the pre-driver head)
with open(_op.path.join(_op.path.dirname(_op.path.abspath(__file__)), 'main.py')) as fh:
    src = fh.read()
head = src[:src.index('\nconductivity = 5.81e7')]
ns = {}
exec(compile(head, '<main.py head>', 'exec'), ns)
mp = ns['mp']
LinearOperator = ns['LinearOperator']
lu_factor, lu_solve = ns['lu_factor'], ns['lu_solve']
gmres = ns['gmres']

# ---- resonator geometry (identical to validate_rlc_resonance.py)
conductivity = 5.81e7
NT = np.array([8, 4, 3])
cell = 5e-4
LT = NT*cell
fullstruc = np.zeros(NT, dtype=np.int8)
fullstruc[:, :, 0] = 1
fullstruc[:, :, -1] = 1
fullstruc[-1, :, :] = 1
M = mp.Tree(fullstruc, st.single_level_nleaf(NT), LT, 1, 1e0,
            None, capacitive=True)
M.e.r = M.e.l[1]/(M.e.l[0]*M.e.l[2]*conductivity)
M.f.r = M.f.l[0]/(M.f.l[1]*M.f.l[2]*conductivity)
M.g.r = M.g.l[2]/(M.g.l[0]*M.g.l[1]*conductivity)
M.lv[0].beta = 1.0
S = ns['SystemMat'](M, 1j)
ns['S'] = S                          # connectBT reads the global S
S.M.lv[0].neumann = False
S.M.alpha = 0
M.RDFinit()
efg, nn = S.efgsize, S.nodesize
ext = M.external
print("resonator: %d filaments, %d nodes (%d external -- %s internal)"
      % (efg, nn, ext.size, "NO" if ext.size == nn else "%d" % (nn-ext.size)))

# ---- dense reference operators (probed from the code, as in the rlc test)
Rd = np.concatenate([np.full(S.esize, M.e.r), np.full(S.fsize, M.f.r),
                     np.full(S.gsize, M.g.r)]).astype(np.complex128)
M.jomega = 1j
Lp = np.zeros((efg, efg), np.complex128)
for k in range(efg):
    S.wholedata[:efg] = 0
    S.wholedata[k] = 1.0
    M.traverseRL()
    Lp[:, k] = S.wholedata[:efg]
Lp = (Lp - np.diag(Rd))/1j
Lp = 0.5*(Lp + Lp.T)
Amat = np.zeros((efg, nn), np.complex128)
for k in range(nn):
    M.lv[0].data[:] = 0
    M.lv[0].data[k] = 1.0
    a = M.connectA()
    Amat[:, k] = np.concatenate(a)
S.wholedata[:] = 0
P = np.asarray(M.n2n, np.complex128)
luP = lu_factor(P[np.ix_(ext, ext)].real)

# ---- K~ extraction on the thin-plate geometry (kernel vs dense getter)
Kk = ns['rel'].extract_reluctance(M, verbose=True)
Kd = ns['rel'].extract_reluctance(M, Lp_dense=np.real(Lp))
kerr = (np.linalg.norm((Kk-Kd).toarray())/np.linalg.norm(Kd.toarray()))
check("kernel K~ matches dense-extracted K~ on thin plates", kerr < 3e-2,
      "|Kk-Kd|/|Kd| = %.2e" % kerr)

# ---- port (as in validate_rlc_resonance.py) and production preconditioner
def node_index(x, y, z):
    mark = M.parsesource(np.array([x]), np.array([y]), np.array([z]),
                         np.array([1.0], dtype=np.complex128), 'node')
    return int(np.nonzero(mark)[0][0])

iA = node_index(0, 0, NT[2])
iB = node_index(0, 0, 0)
f_res = 5.9729e10      # predicted + verified by validate_rlc_resonance.py

S.reluctanceprecinit()   # exercises the all-external fallback-gauge path
check("reluctanceprecinit survives an all-external structure",
      True, "fallback gauge node %d (external)" % S._gnd)


def counted(matvec):
    c = [0]
    def mv(v):
        c[0] += 1
        return matvec(v)
    return c, mv


print("\n%-8s | %-6s %-9s %-10s | %-7s %-9s | %-9s %-9s"
      % ("f/f_res", "rel_it", "rel_vsdir", "rel_flag", "sad_it", "sad_vsdir",
         "|Z|_iter", "|Z|_dense"))
rows = []
for frac in [0.1, 0.3, 0.6, 1.0, 1.7, 3.0, 10.0]:
    f = frac*f_res
    jw = 1j*2*np.pi*f
    S.jomega = jw
    M.jomega = jw
    S._factorReluctanceSchur()
    # dense mixed-row K (all nodes external) and its W-rescaled K'
    Z = np.diag(Rd) + jw*Lp
    K = np.zeros((efg+nn,)*2, np.complex128)
    K[:efg, :efg] = Z
    K[:efg, efg:] = Amat
    K[efg:, :efg] = P @ Amat.T
    K[efg:, efg:] = -jw*np.eye(nn)
    Kp = K.copy()
    rb = K[efg+ext]      # here ext = all nodes, but keep the general form
    Kp[efg+ext] = lu_solve(luP, rb.real) + 1j*lu_solve(luP, rb.imag)
    # port RHS, production mixed-row transform + rescaling
    sn = np.zeros(nn, np.complex128)
    sn[iA] = 1.0
    sn[iB] = -1.0
    Source = np.zeros(efg+nn, np.complex128)
    Source[efg:] = P @ sn                      # external rows get P s_n
    Sourcep = S.rescaleRHS(Source)
    # dense equilibrated reference on the same K'
    rhsp = Source.copy()
    rhsp[efg:][ext] = lu_solve(luP, Source[efg:][ext].real) \
        + 1j*lu_solve(luP, Source[efg:][ext].imag)
    rs = np.abs(Kp).max(1); rs[rs == 0] = 1
    K1 = Kp/rs[:, None]
    cs = np.abs(K1).max(0); cs[cs == 0] = 1
    xref = np.linalg.solve(K1/cs[None, :], rhsp/rs)/cs
    # production reluctance solve (standard GMRES, production settings)
    cnt, mv = counted(S.rescaleLpPR)
    Kop = LinearOperator((efg+nn,)*2, matvec=mv, dtype=np.complex128)
    PRel = LinearOperator((efg+nn,)*2, matvec=S.precondReluctance,
                          dtype=np.complex128)
    x, flag = gmres(Kop, Sourcep, M=PRel, tol=1e-8, maxiter=1, restrt=300)
    x = np.asarray(x).ravel()
    vsd = (np.linalg.norm(x[:efg]-xref[:efg])
           / np.linalg.norm(xref[:efg]))
    # ideal inductive saddle (dense pinv) -- loop reduction's best case
    Ksad = np.zeros_like(K)
    Ksad[:efg, :efg] = Z
    Ksad[:efg, efg:] = Amat
    Ksad[efg:, :efg] = Amat.T
    Msad = np.linalg.pinv(Ksad)
    cnt2, mv2 = counted(lambda v: Kp @ v)
    Kop2 = LinearOperator((efg+nn,)*2, matvec=mv2, dtype=np.complex128)
    Mop2 = LinearOperator((efg+nn,)*2, matvec=lambda v: Msad @ v,
                          dtype=np.complex128)
    x2, flag2 = gmres(Kop2, rhsp, M=Mop2, tol=1e-8, maxiter=1, restrt=300)
    x2 = np.asarray(x2).ravel()
    vsd2 = (np.linalg.norm(x2[:efg]-xref[:efg])
            / np.linalg.norm(xref[:efg]))
    zi = abs(-(x[efg:][iA]-x[efg:][iB]))
    zd = abs(-(xref[efg:][iA]-xref[efg:][iB]))
    rows.append(dict(frac=frac, it=cnt[0], vsd=vsd, flag=flag,
                     it2=cnt2[0], vsd2=vsd2, flag2=flag2, zi=zi, zd=zd))
    if frac == 1.0:
        res_ctx = dict(jw=jw, Sourcep=Sourcep, xref=xref)
    print("%-8.2f | %-6d %-9.2e %-10s | %-7d %-9.2e | %-9.2f %-9.2f"
          % (frac, cnt[0], vsd, ("ok" if flag == 0 else "CAP(%d)" % flag),
             cnt2[0], vsd2, zi, zd))

# ---- ccap='full' at resonance: full P_ext^{-1} block in S~
S.jomega = res_ctx['jw']
M.jomega = res_ctx['jw']
S.reluctanceprecinit(ccap='full')
cntf, mvf = counted(S.rescaleLpPR)
Kopf = LinearOperator((efg+nn,)*2, matvec=mvf, dtype=np.complex128)
PRelf = LinearOperator((efg+nn,)*2, matvec=S.precondReluctance,
                       dtype=np.complex128)
xf, flagf = gmres(Kopf, res_ctx['Sourcep'], M=PRelf, tol=1e-8, maxiter=1,
                  restrt=300)
xf = np.asarray(xf).ravel()
vsdf = (np.linalg.norm(xf[:efg]-res_ctx['xref'][:efg])
        / np.linalg.norm(res_ctx['xref'][:efg]))
print("\nccap='full' at f_res: %d matvecs, vs_direct %.2e" % (cntf[0], vsdf))

# ---- ccap='band' at resonance: sparse windowed P_ext^{-1}
# (reluctance.extract_ccap, window=30/retain=18 from profile_ccap_band.py) --
# must match the 'full' iteration class while keeping S~ sparse
S.reluctanceprecinit(ccap='band')
cntb, mvb = counted(S.rescaleLpPR)
Kopb = LinearOperator((efg+nn,)*2, matvec=mvb, dtype=np.complex128)
PRelb = LinearOperator((efg+nn,)*2, matvec=S.precondReluctance,
                       dtype=np.complex128)
xb, flagb = gmres(Kopb, res_ctx['Sourcep'], M=PRelb, tol=1e-8, maxiter=1,
                  restrt=300)
xb = np.asarray(xb).ravel()
vsdb = (np.linalg.norm(xb[:efg]-res_ctx['xref'][:efg])
        / np.linalg.norm(res_ctx['xref'][:efg]))
print("ccap='band' at f_res: %d matvecs, vs_direct %.2e (%.1f nnz/row)"
      % (cntb[0], vsdb, S._CcapBand.nnz/max(ext.size, 1)))

print()
worst_vsd = max(r['vsd'] for r in rows)
worst_it = max(r['it'] for r in rows)
res = next(r for r in rows if r['frac'] == 1.0)
check("production reluctance GMRES converges at ALL freqs incl. resonance",
      all(r['flag'] == 0 for r in rows) and worst_it <= 150,
      "max matvecs = %d" % worst_it)
check("currents match dense direct at all freqs (< 1e-5)",
      worst_vsd < 1e-5, "worst = %.2e" % worst_vsd)
check("resonance peak reproduced through the iterative solve",
      abs(res['zi']/res['zd'] - 1) < 0.02,
      "|Z(f_res)| iter %.1f vs dense %.1f ohm" % (res['zi'], res['zd']))
# The saddle here uses the EXACT dense Z inverse (unavailable scalably), so
# absolute counts favor it; the honest capacitance signal is the DEGRADATION
# through resonance relative to each preconditioner's own low-frequency self.
lo_r = next(r for r in rows if r['frac'] == 0.1)
hi_r = max(rows, key=lambda r: r['frac'])
grow_rel = hi_r['it']/lo_r['it']
grow_sad = hi_r['it2']/lo_r['it2']
check("through resonance, reluctance degrades LESS than the ideal saddle",
      grow_rel < grow_sad,
      "matvec growth 0.1x->10x: rel %.1fx vs saddle %.1fx"
      % (grow_rel, grow_sad))
check("ccap='full' sharply cuts iterations at resonance",
      flagf == 0 and vsdf < 1e-5 and cntf[0] <= res['it']//2,
      "%d matvecs vs %d with diag C_cap" % (cntf[0], res['it']))
check("ccap='band' matches the 'full' iteration class at SPARSE cost",
      flagb == 0 and vsdb < 1e-5 and cntb[0] <= res['it']//2
      and cntb[0] <= cntf[0] + 5 and S._CcapFull is None,
      "band %d vs full %d vs diag %d matvecs" % (cntb[0], cntf[0], res['it']))

print()
if FAIL:
    print("FAILURES:", ", ".join(FAIL))
    sys.exit(1)
print("all checks passed -- production reluctance preconditioner handles the "
      "capacitance-dominated (resonant) regime")
