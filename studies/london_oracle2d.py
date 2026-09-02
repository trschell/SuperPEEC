"""2-D cross-section London oracle: the TRUE kinetic term of the
# SPDX-License-Identifier: MIT
finite microstrip, from fine-grid energy minimisation.

Per unit length, x-directed currents j(y,z) on strip + ground:
  E = (1/2) sum_pq i_p G_pq i_q + (mu0 lam^2 / 2) sum_p j_p^2 A_p
  G_pq = -(mu0/2pi) ln(r_pq),  r_pp = 0.44705*a  (square-cell GMD)
minimised at fixed totals (+1 strip, -1 ground). The gauge constant
cancels in the lambda difference. Compare the true kinetic
L'(lam)-L'(tiny) against the per-square formula the film bench used."""
import numpy as np
MU0 = 4e-7*np.pi
T = 200e-9; H = 200e-9; LAM = 9e-8; TINY = 1e-9

def build(W, WG, a):
    cells = []
    tag = []
    nz = int(round(T/a))
    # ground: y in [0, WG], z in [0, T]
    for iy in range(int(round(WG/a))):
        for iz in range(nz):
            cells.append(((iy + .5)*a, (iz + .5)*a)); tag.append(0)
    # strip centred: y in [(WG-W)/2, (WG+W)/2], z in [T+H, 2T+H]
    y0 = (WG - W)/2
    for iy in range(int(round(W/a))):
        for iz in range(nz):
            cells.append((y0 + (iy + .5)*a, T + H + (iz + .5)*a)); tag.append(1)
    P = np.array(cells); tag = np.array(tag)
    d = P[:, None, :] - P[None, :, :]
    r = np.sqrt((d*d).sum(-1))
    np.fill_diagonal(r, 0.44705*a)
    G = -(MU0/(2*np.pi))*np.log(r)
    return P, tag, G, a*a

def L_of(lam, W=2e-6, WG=5e-6, a=12.5e-9):
    P, tag, G, A = build(W, WG, a)
    n = len(P)
    K = G + np.eye(n)*(MU0*lam*lam/A)
    C = np.zeros((2, n)); C[0, tag == 0] = 1; C[1, tag == 1] = 1
    M = np.block([[K, C.T], [C, np.zeros((2, 2))]])
    rhs = np.concatenate([np.zeros(n), [-1.0, 1.0]])
    sol = np.linalg.solve(M, rhs)
    i = sol[:n]
    return i @ K @ i

if __name__ == '__main__':
    import sys
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 12.5e-9
    coth = lambda t, l: l/np.tanh(t/l)
    for W, WG in ((2e-6, 5e-6), (4e-6, 8e-6)):
        kin = L_of(LAM, W, WG, a) - L_of(TINY, W, WG, a)
        persq = 2*MU0*(coth(T, LAM) - coth(T, TINY))/W
        print("W=%gum a=%gnm: TRUE kinetic %.5e H/m | per-square/W %.5e | ratio %.4f"
              % (W*1e6, a*1e9, kin, persq, kin/persq), flush=True)
