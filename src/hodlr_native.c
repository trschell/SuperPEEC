// SPDX-License-Identifier: MIT
/* Native (C + BLAS/LAPACK) solve for hodlr.py's recursive-Woodbury
 * factorization -- the hot path of the H-LU program: the S_c sampling
 * construction drives tens of thousands of columns through the HODLR
 * solve, and the Python tree walk costs ~10x the underlying flops
 * (measured: 2592 s for the 19^3 HODBF build, ~37 ms/column vs a
 * few-ms flop budget).
 *
 * The tree is passed as flat per-node arrays plus POINTER TABLES into
 * the numpy factor arrays owned by hodlr.HodlrZ -- nothing is copied.
 * Layout notes: leaf LU and Woodbury-core G are Fortran-ordered copies
 * made at factor time (with 1-based int32 pivots); U (m1 x r), V
 * (m2 x r), Y (m x 2r) stay numpy C-ordered, which equals the
 * Fortran-ordered TRANSPOSE -- exactly what the U^T/V^T products want,
 * and Y is applied through a transposed GEMM.
 *
 * Build (toolbox):
 *   gcc -O2 -shared -fPIC hodlr_native.c -o libhodlrnat.so -lopenblas
 */
#include <stdlib.h>
#include <complex.h>

typedef double _Complex zc;

extern void zgemm_(const char*, const char*, const int*, const int*,
                   const int*, const zc*, const zc*, const int*, const zc*,
                   const int*, const zc*, zc*, const int*);
extern void zgetrs_(const char*, const int*, const int*, const zc*,
                    const int*, const int*, zc*, const int*, int*);

typedef struct {
  const int *kind, *m, *r, *left, *right;
  const long long *p_lu, *p_piv, *p_U, *p_V, *p_Y, *p_G, *p_Gpiv;
} Tree;

static void solve_node(const Tree* T, int i, zc* B, int ldB, int nrhs,
                       zc* work) {
  int m = T->m[i];
  if (T->kind[i] == 0) {                       /* leaf: LU backsolve */
    int info;
    zgetrs_("N", &m, &nrhs, (const zc*)(size_t)T->p_lu[i], &m,
            (const int*)(size_t)T->p_piv[i], B, &ldB, &info);
    return;
  }
  int l = T->left[i], g = T->right[i];
  int m1 = T->m[l];
  solve_node(T, l, B, ldB, nrhs, work);
  solve_node(T, g, B + m1, ldB, nrhs, work);
  int r = T->r[i];
  if (r == 0) return;
  int two_r = 2*r, m2 = m - m1, info;
  zc one = 1.0, zero = 0.0, neg = -1.0;
  zc* z = work;
  /* z[:r]  = V^T B2 : V is numpy-C (m2 x r) == fortran (r x m2) = V^T */
  zgemm_("N", "N", &r, &nrhs, &m2, &one, (const zc*)(size_t)T->p_V[i], &r,
         B + m1, &ldB, &zero, z, &two_r);
  /* z[r:]  = U^T B1 */
  zgemm_("N", "N", &r, &nrhs, &m1, &one, (const zc*)(size_t)T->p_U[i], &r,
         B, &ldB, &zero, z + r, &two_r);
  zgetrs_("N", &two_r, &nrhs, (const zc*)(size_t)T->p_G[i], &two_r,
          (const int*)(size_t)T->p_Gpiv[i], z, &two_r, &info);
  /* B -= Y z : Y is numpy-C (m x 2r) == fortran (2r x m) = Y^T */
  zgemm_("T", "N", &m, &nrhs, &two_r, &neg, (const zc*)(size_t)T->p_Y[i],
         &two_r, z, &two_r, &one, B, &ldB);
}

void hodlr_solve(const int* kind, const int* m, const int* r,
                 const int* left, const int* right,
                 const long long* p_lu, const long long* p_piv,
                 const long long* p_U, const long long* p_V,
                 const long long* p_Y, const long long* p_G,
                 const long long* p_Gpiv,
                 int root, int n, zc* B, int nrhs, int rmax) {
  Tree T = {kind, m, r, left, right, p_lu, p_piv, p_U, p_V, p_Y, p_G,
            p_Gpiv};
  zc* work = (zc*)malloc(sizeof(zc)*2*(size_t)(rmax > 0 ? rmax : 1)*nrhs);
  solve_node(&T, root, B, n, nrhs, work);
  free(work);
}

/* Transpose solve Z^T x = b. Mirrors hodlr.py _solve_node_t: leaf and
 * core solves run with trans='T' (G is REUSED -- the transpose Woodbury
 * core is G^T), the correction uses Yt = D^{-T} W^T (numpy C (m x 2r),
 * applied through the same transposed-GEMM trick as Y), and jw scales
 * the U^T/V^T products via the GEMM alpha (packed U/V are unscaled;
 * on the forward path jw lives inside Y instead). */
typedef struct {
  const Tree* T;
  const long long* p_Yt;
  zc jw;
} TreeT;

static void solve_node_t(const TreeT* Tt, int i, zc* B, int ldB, int nrhs,
                         zc* work) {
  const Tree* T = Tt->T;
  int m = T->m[i];
  if (T->kind[i] == 0) {
    int info;
    zgetrs_("T", &m, &nrhs, (const zc*)(size_t)T->p_lu[i], &m,
            (const int*)(size_t)T->p_piv[i], B, &ldB, &info);
    return;
  }
  int l = T->left[i], g = T->right[i];
  int m1 = T->m[l];
  solve_node_t(Tt, l, B, ldB, nrhs, work);
  solve_node_t(Tt, g, B + m1, ldB, nrhs, work);
  int r = T->r[i];
  if (r == 0) return;
  int two_r = 2*r, m2 = m - m1, info;
  zc one = 1.0, zero = 0.0, neg = -1.0;
  zc* z = work;
  /* z[:r] = jw U^T B1, z[r:] = jw V^T B2 */
  zgemm_("N", "N", &r, &nrhs, &m1, &Tt->jw, (const zc*)(size_t)T->p_U[i],
         &r, B, &ldB, &zero, z, &two_r);
  zgemm_("N", "N", &r, &nrhs, &m2, &Tt->jw, (const zc*)(size_t)T->p_V[i],
         &r, B + m1, &ldB, &zero, z + r, &two_r);
  zgetrs_("T", &two_r, &nrhs, (const zc*)(size_t)T->p_G[i], &two_r,
          (const int*)(size_t)T->p_Gpiv[i], z, &two_r, &info);
  zgemm_("T", "N", &m, &nrhs, &two_r, &neg,
         (const zc*)(size_t)Tt->p_Yt[i], &two_r, z, &two_r, &one, B, &ldB);
}

void hodlr_solve_t(const int* kind, const int* m, const int* r,
                   const int* left, const int* right,
                   const long long* p_lu, const long long* p_piv,
                   const long long* p_U, const long long* p_V,
                   const long long* p_Yt, const long long* p_G,
                   const long long* p_Gpiv, double jw_re, double jw_im,
                   int root, int n, zc* B, int nrhs, int rmax) {
  Tree T = {kind, m, r, left, right, p_lu, p_piv, p_U, p_V, 0, p_G,
            p_Gpiv};
  TreeT Tt = {&T, p_Yt, jw_re + jw_im*I};
  zc* work = (zc*)malloc(sizeof(zc)*2*(size_t)(rmax > 0 ? rmax : 1)*nrhs);
  solve_node_t(&Tt, root, B, n, nrhs, work);
  free(work);
}
