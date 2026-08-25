C SPDX-License-Identifier: MIT
C
      SUBROUTINE MID_M2L(TRANSFER, MDATA, IDX, FARNEIGHBORS,
     +                   OUTMAT, SIZEIDX, NNMAX, NFAR, NPAR)
C
C     NFAR/NPAR were the literals 189/8 -- the 2x2x2 child geometry.
C     Generalized 2026-08-06 for per-axis nmidlev (an UNSPLIT axis has
C     nmidlev = 1: e.g. [2,2,1] gives NPAR = 4, NFAR = 81), which is
C     what a flat (pancake) geometry's mid levels need. The caller
C     (levels.MidLevel.midm2linit) builds the tables to match.
C
CF2PY INTENT(OUT) :: OUTMAT
CF2PY DOUBLE COMPLEX :: TRANSFER, MDATA
CF2PY INTEGER :: IDX, FARNEIGHBORS
CF2PY INTEGER, INTENT(HIDE), DEPEND(MDATA) :: NNMAX=SHAPE(MDATA, 0)
CF2PY INTEGER, INTENT(HIDE), DEPEND(IDX) :: SIZEIDX=SIZE(IDX)
CF2PY INTEGER, INTENT(HIDE), DEPEND(FARNEIGHBORS) :: NFAR=SHAPE(FARNEIGHBORS, 0)
CF2PY INTEGER, INTENT(HIDE), DEPEND(TRANSFER) :: NPAR=SHAPE(TRANSFER, 3)
      INTEGER NNMAX, SIZEIDX, NFAR, NPAR
      INTEGER IDX(SIZEIDX)
      INTEGER FARNEIGHBORS(NFAR, SIZEIDX)
      DOUBLE COMPLEX MDATA(NNMAX, SIZEIDX)
      DOUBLE COMPLEX OUTMAT(NNMAX, SIZEIDX)
      DOUBLE COMPLEX TRANSFER(NNMAX, NNMAX, NFAR, NPAR)
C
      INTEGER GIDX, POS, NEIGH, NGROUP, IDXNM, IDXJK
C
C     Each GIDX owns OUTMAT(:, GIDX) alone and the accumulation order
C     within a column is unchanged, so parallel results are
C     BIT-IDENTICAL to serial (the P2P recipe). Thread count follows
C     OMP_NUM_THREADS.
!$OMP PARALLEL DO DEFAULT(SHARED)
!$OMP& PRIVATE(GIDX,POS,NEIGH,NGROUP,IDXNM,IDXJK)
      DO GIDX = 1, SIZEIDX
          POS = IDX(GIDX) + 1
          DO IDXNM = 1, NNMAX
              OUTMAT(IDXNM, GIDX) = 0.0
          ENDDO
          DO NEIGH = 1, NFAR
              NGROUP = FARNEIGHBORS(NEIGH, GIDX) + 1
              IF (NGROUP.GE.1) THEN
                  DO IDXNM = 1, NNMAX
                      DO IDXJK = 1, NNMAX
                          OUTMAT(IDXNM, GIDX) = OUTMAT(IDXNM, GIDX) +
     +                        TRANSFER(IDXJK, IDXNM, NEIGH, POS) *
     +                        MDATA(IDXJK, NGROUP)
                      ENDDO
                  ENDDO
              ENDIF
          ENDDO
      ENDDO
!$OMP END PARALLEL DO
C
      END
C
C
      SUBROUTINE MID_M2L_C64(TRANSFER, MDATA, IDX, FARNEIGHBORS,
     +                       OUTMAT, SIZEIDX, NNMAX, NFAR, NPAR)
C
C     fp32-storage twin of MID_M2L (phase 3c, 2026-08-15): TRANSFER is
C     COMPLEX (single) -- the caller stores it NORMALISED (raw SI
C     r**-(j+n+1) entries overflow single's exponent range on fine
C     pitches; levels.midm2linit factors the magnitude into fp64
C     per-channel scales u(jk)*v(nm) and folds them into MDATA/OUTMAT).
C     MDATA, OUTMAT and the ACCUMULATION stay DOUBLE COMPLEX: storage
C     and read bandwidth are single, compute is double -- the same
C     split the top-level ftrans made.
C
CF2PY INTENT(OUT) :: OUTMAT
CF2PY COMPLEX :: TRANSFER
CF2PY DOUBLE COMPLEX :: MDATA
CF2PY INTEGER :: IDX, FARNEIGHBORS
CF2PY INTEGER, INTENT(HIDE), DEPEND(MDATA) :: NNMAX=SHAPE(MDATA, 0)
CF2PY INTEGER, INTENT(HIDE), DEPEND(IDX) :: SIZEIDX=SIZE(IDX)
CF2PY INTEGER, INTENT(HIDE), DEPEND(FARNEIGHBORS) :: NFAR=SHAPE(FARNEIGHBORS, 0)
CF2PY INTEGER, INTENT(HIDE), DEPEND(TRANSFER) :: NPAR=SHAPE(TRANSFER, 3)
      INTEGER NNMAX, SIZEIDX, NFAR, NPAR
      INTEGER IDX(SIZEIDX)
      INTEGER FARNEIGHBORS(NFAR, SIZEIDX)
      DOUBLE COMPLEX MDATA(NNMAX, SIZEIDX)
      DOUBLE COMPLEX OUTMAT(NNMAX, SIZEIDX)
      COMPLEX TRANSFER(NNMAX, NNMAX, NFAR, NPAR)
C
      INTEGER GIDX, POS, NEIGH, NGROUP, IDXNM, IDXJK
C
C     Each GIDX owns OUTMAT(:, GIDX) alone and the accumulation order
C     within a column is unchanged, so parallel results are
C     BIT-IDENTICAL to serial (the P2P recipe). Thread count follows
C     OMP_NUM_THREADS.
!$OMP PARALLEL DO DEFAULT(SHARED)
!$OMP& PRIVATE(GIDX,POS,NEIGH,NGROUP,IDXNM,IDXJK)
      DO GIDX = 1, SIZEIDX
          POS = IDX(GIDX) + 1
          DO IDXNM = 1, NNMAX
              OUTMAT(IDXNM, GIDX) = 0.0
          ENDDO
          DO NEIGH = 1, NFAR
              NGROUP = FARNEIGHBORS(NEIGH, GIDX) + 1
              IF (NGROUP.GE.1) THEN
                  DO IDXNM = 1, NNMAX
                      DO IDXJK = 1, NNMAX
                          OUTMAT(IDXNM, GIDX) = OUTMAT(IDXNM, GIDX) +
     +                        TRANSFER(IDXJK, IDXNM, NEIGH, POS) *
     +                        MDATA(IDXJK, NGROUP)
                      ENDDO
                  ENDDO
              ENDIF
          ENDDO
      ENDDO
!$OMP END PARALLEL DO
C
      END
C
C
      SUBROUTINE FFT1D(INMAT, OUTMAT, SIZEMAT)
C
CF2PY INTENT(OUT) :: OUTMAT
CF2PY DOUBLE COMPLEX :: INMAT
CF2PY INTEGER, INTENT(HIDE), DEPEND(INMAT) :: SIZEMAT=SIZE(INMAT)
      INTEGER SIZEMAT
      DOUBLE COMPLEX INMAT(SIZEMAT)
      DOUBLE COMPLEX OUTMAT(SIZEMAT)
C
#include "fftw3.f"
C
      INTEGER*8 PLAN
C
      CALL DFFTW_PLAN_DFT_1D(PLAN, SIZEMAT, INMAT, OUTMAT,
     +                       FFTW_FORWARD, FFTW_ESTIMATE)
      CALL DFFTW_EXECUTE_DFT(PLAN, INMAT, OUTMAT)
      CALL DFFTW_DESTROY_PLAN(PLAN)
C
      END
C
C
      SUBROUTINE P2P(BEHIND, CURRENT, AHEAD, ISLABIDX, OSLABIDX,
     +               COUNTX, NEIGHBORS, XIDX, P2P_TRANSFER, REVSLABIDX,
     +               OREVSLABIDX, OUTMAT, BSIZE, CSIZE, ASIZE, GSIZE,
     +               OSIZE, NX, NY, NZ)
C
CF2PY INTENT(OUT) :: OUTMAT
CF2PY DOUBLE COMPLEX :: BEHIND, CURRENT, AHEAD, P2P_TRANSFER
CF2PY INTEGER :: INSLABIDX, OUTSLABIDX, COUNTX, NEIGHBORS, XIDX
CF2PY INTEGER :: REVSLABIDX, OREVSLABIDX
CF2PY INTEGER, INTENT(HIDE), DEPEND(BEHIND) :: BSIZE=SHAPE(BEHIND, 3)
CF2PY INTEGER, INTENT(HIDE), DEPEND(CURRENT) :: CSIZE=SHAPE(CURRENT, 3)
CF2PY INTEGER, INTENT(HIDE), DEPEND(AHEAD) :: ASIZE=SHAPE(AHEAD, 3)
CF2PY INTEGER, INTENT(HIDE), DEPEND(XIDX) :: GSIZE=SIZE(XIDX)
CF2PY INTEGER, INTENT(HIDE), DEPEND(OSLABIDX) :: OSIZE=SIZE(OSLABIDX)
CF2PY INTEGER, INTENT(HIDE), DEPEND(CURRENT) :: NX=SHAPE(CURRENT, 2)
CF2PY INTEGER, INTENT(HIDE), DEPEND(CURRENT) :: NY=SHAPE(CURRENT, 1)
CF2PY INTEGER, INTENT(HIDE), DEPEND(CURRENT) :: NZ=SHAPE(CURRENT, 0)
      INTEGER BSIZE, CSIZE, ASIZE, GSIZE, OSIZE, NX, NY, NZ
      DOUBLE COMPLEX OUTMAT(NZ, NY, NX, OSIZE)
      DOUBLE COMPLEX BEHIND(NZ, NY, NX, BSIZE)
      DOUBLE COMPLEX CURRENT(NZ, NY, NX, CSIZE)
      DOUBLE COMPLEX AHEAD(NZ, NY, NX, ASIZE)
      DOUBLE COMPLEX P2P_TRANSFER(NZ, NY, NX, 27)
      INTEGER ISLABIDX(CSIZE), OSLABIDX(OSIZE), NEIGHBORS(GSIZE, 27)
      INTEGER XIDX(GSIZE), REVSLABIDX(GSIZE), OREVSLABIDX(GSIZE), COUNTX
C
      INTEGER COUNTG, COUNTN, GROUP, NEIGHGROUP, X, Y, Z, XOFF, YZIDX
      INTEGER OYZIDX
      DOUBLE COMPLEX TRANS, SDATA
C
      DO COUNTG = 1, OSIZE
          DO X = 1, NX
              DO Y = 1, NY
                  DO Z = 1, NZ
                      OUTMAT(Z, Y, X, COUNTG) = 0
                  ENDDO
              ENDDO
          ENDDO
      ENDDO
C     Each COUNTG accumulates into its own OUTMAT slot (OYZIDX is that
C     group's reverse-slab position, distinct per iteration), so the
C     iterations are independent and each slot summation order is
C     unchanged under OpenMP: parallel results are BIT-IDENTICAL to
C     serial. Thread count follows OMP_NUM_THREADS.
!$OMP PARALLEL DO DEFAULT(SHARED)
!$OMP& PRIVATE(COUNTG,GROUP,OYZIDX,COUNTN,NEIGHGROUP,YZIDX,XOFF)
!$OMP& PRIVATE(X,Y,Z,TRANS,SDATA)
      DO COUNTG = 1, CSIZE
          GROUP = ISLABIDX(COUNTG) + 1
          OYZIDX = OREVSLABIDX(GROUP) + 1
          IF (OYZIDX.GE.1) THEN
              DO COUNTN = 1, 27
                  NEIGHGROUP = NEIGHBORS(GROUP, COUNTN) + 1
                  IF (NEIGHGROUP.GE.1) THEN
                      YZIDX = REVSLABIDX(NEIGHGROUP) + 1
                  ENDIF
                  IF (NEIGHGROUP.GE.1 .AND. YZIDX.GE.1) THEN
                      XOFF = XIDX(NEIGHGROUP) - COUNTX
                      DO X = 1, NX
                          DO Y = 1, NY
                              DO Z = 1, NZ
                                  TRANS = P2P_TRANSFER(Z, Y, X, COUNTN)
                                  IF (XOFF.EQ.-1) THEN
                                      SDATA = BEHIND(Z, Y, X, YZIDX)
                                  ELSEIF (XOFF.EQ.0) THEN
                                      SDATA = CURRENT(Z, Y, X, YZIDX)
                                  ELSEIF (XOFF.EQ.1) THEN
                                      SDATA = AHEAD(Z, Y, X, YZIDX)
                                  ENDIF
                                  OUTMAT(Z, Y, X, OYZIDX) =
     +                                OUTMAT(Z, Y, X, OYZIDX) +
     +                                TRANS*SDATA
                              ENDDO
                          ENDDO
                      ENDDO
                  ENDIF
              ENDDO
          ENDIF
      ENDDO
!$OMP END PARALLEL DO
C
      END
C
C
      SUBROUTINE P2PINNERLOOP(BEHIND, CURRENT, AHEAD, ISLABIDX,
     +               OSLABIDX, COUNTX, NEIGHBORS, XIDX, P2P_TRANSFER,
     +               REVSLABIDX, OREVSLABIDX, OUTMAT,
     +               BSIZE, CSIZE, ASIZE, GSIZE, OSIZE, NX, NY, NZ)
C
CF2PY INTENT(OUT) :: OUTMAT
CF2PY DOUBLE COMPLEX :: BEHIND, CURRENT, AHEAD, P2P_TRANSFER
CF2PY INTEGER :: INSLABIDX, OUTSLABIDX, COUNTX, NEIGHBORS, XIDX
CF2PY INTEGER :: REVSLABIDX, OREVSLABIDX
CF2PY INTEGER, INTENT(HIDE), DEPEND(BEHIND) :: BSIZE=SHAPE(BEHIND, 3)
CF2PY INTEGER, INTENT(HIDE), DEPEND(CURRENT) :: CSIZE=SHAPE(CURRENT, 3)
CF2PY INTEGER, INTENT(HIDE), DEPEND(AHEAD) :: ASIZE=SHAPE(AHEAD, 3)
CF2PY INTEGER, INTENT(HIDE), DEPEND(XIDX) :: GSIZE=SIZE(XIDX)
CF2PY INTEGER, INTENT(HIDE), DEPEND(OSLABIDX) :: OSIZE=SIZE(OSLABIDX)
CF2PY INTEGER, INTENT(HIDE), DEPEND(CURRENT) :: NX=SHAPE(CURRENT, 2)
CF2PY INTEGER, INTENT(HIDE), DEPEND(CURRENT) :: NY=SHAPE(CURRENT, 1)
CF2PY INTEGER, INTENT(HIDE), DEPEND(CURRENT) :: NZ=SHAPE(CURRENT, 0)
      INTEGER BSIZE, CSIZE, ASIZE, GSIZE, OSIZE, NX, NY, NZ
      DOUBLE COMPLEX OUTMAT(NZ, NY, NX, OSIZE)
      DOUBLE COMPLEX BEHIND(NZ, NY, NX, BSIZE)
      DOUBLE COMPLEX CURRENT(NZ, NY, NX, CSIZE)
      DOUBLE COMPLEX AHEAD(NZ, NY, NX, ASIZE)
      DOUBLE COMPLEX P2P_TRANSFER(NZ, NY, NX, 27)
      INTEGER ISLABIDX(CSIZE), OSLABIDX(OSIZE), NEIGHBORS(GSIZE, 27)
      INTEGER XIDX(GSIZE), REVSLABIDX(GSIZE), OREVSLABIDX(GSIZE), COUNTX
C
      INTEGER COUNTG, COUNTN, GROUP, NEIGHGROUP, X, Y, Z, XOFF, YZIDX
      INTEGER OYZIDX
      DOUBLE COMPLEX TRANS, SDATA
C
      DO COUNTG = 1, OSIZE
          DO X = 1, NX
              DO Y = 1, NY
                  DO Z = 1, NZ
                      OUTMAT(Z, Y, X, COUNTG) = 0
                  ENDDO
              ENDDO
          ENDDO
      ENDDO
      DO COUNTG = 1, CSIZE
          GROUP = ISLABIDX(COUNTG) + 1
          OYZIDX = OREVSLABIDX(GROUP) + 1
          IF (OYZIDX.GE.1) THEN
              DO COUNTN = 1, 27
                  NEIGHGROUP = NEIGHBORS(GROUP, COUNTN) + 1
                  IF (NEIGHGROUP.GE.1) THEN
                      YZIDX = REVSLABIDX(NEIGHGROUP) + 1
                  ENDIF
                  IF (NEIGHGROUP.GE.1 .AND. YZIDX.GE.1) THEN
                      XOFF = XIDX(NEIGHGROUP) - COUNTX
                      DO X = 1, NX
                          DO Y = 1, NY
                              DO Z = 1, NZ
                                  TRANS = P2P_TRANSFER(Z, Y, X, COUNTN)
                                  IF (XOFF.EQ.-1) THEN
                                      SDATA = BEHIND(Z, Y, X, YZIDX)
                                  ELSEIF (XOFF.EQ.0) THEN
                                      SDATA = CURRENT(Z, Y, X, YZIDX)
                                  ELSEIF (XOFF.EQ.1) THEN
                                      SDATA = AHEAD(Z, Y, X, YZIDX)
                                  ENDIF
                                  OUTMAT(Z, Y, X, OYZIDX) =
     +                                OUTMAT(Z, Y, X, OYZIDX) +
     +                                TRANS*SDATA
                              ENDDO
                          ENDDO
                      ENDDO
                  ENDIF
              ENDDO
          ENDIF
      ENDDO
C
      END
C
C
      subroutine node2filament(nodedata, idx, idx0, idxe, idx0e, idxf,
     +                         idx0f, idxg, idx0g, neighbors, ne, nf,
     +                         ng, nnode, sizeidx, sizeidx0, sizee,
     +                         sizef, sizeg, datae, dataf, datag)
c
cf2py intent(out) :: datae, dataf, datag
cf2py double complex :: nodedata
cf2py integer :: idx, idx0
cf2py integer :: neighbors
cf2py integer :: idxe, idx0e, idxf, idx0f, idxg, idx0g
cf2py integer :: ne, nf, ng, nnode
cf2py integer, intent(hidden), depend(idx) :: sizeidx=size(idx)
cf2py integer, intent(hidden), depend(idx0) :: sizeidx0=size(idx0)
cf2py integer, intent(hidden), depend(idxe) :: sizee=size(idxe)
cf2py integer, intent(hidden), depend(idxf) :: sizef=size(idxf)
cf2py integer, intent(hidden), depend(idxg) :: sizeg=size(idxg)
c
      integer sizeidx, sizeidx0, sizee, sizef, sizeg
      double complex datae(sizee), dataf(sizef), datag(sizeg)
      integer idx(sizeidx), idx0(sizeidx0)
      integer idxe(sizee), idxf(sizef), idxg(sizeg)
      integer idx0e(sizeidx0), idx0f(sizeidx0), idx0g(sizeidx0)
      double complex nodedata(sizeidx)
      integer neighbors(sizeidx0 - 1, 27)
      integer ne(3), nf(3), ng(3), nnode(3)
c
      double complex singlegroup(nnode(3)+1, nnode(2)+1, nnode(1)+1)
      integer nextneigh(8), group, nn, i, ii, neigh, nexti
      integer nnx, nny, nnz, x, y, z, xx, yy, zz
c
      data nextneigh/14,15,17,18,23,24,26,27/
      do group = 1, sizeidx0-1
          do x = 1, nnode(1)+1
              do y = 1, nnode(2)+1
                  do z = 1, nnode(3)+1
                      singlegroup(z, y, x) = 0.0
                  enddo
              enddo
          enddo
          do nn = 1, 8
              neigh = nextneigh(nn)
              neighgroup = neighbors(group, neigh) + 1
              if (neighgroup.ge.1) then
                  nnx = (nn-1)/4
                  nny = mod((nn-1)/2, 2)
                  nnz = mod((nn-1), 2)
                  i = idx0(neighgroup) + 1
                  nexti = idx0(neighgroup+1)
                  do x = 1, (1-nnx)*nnode(1) + nnx
                      do y = 1, (1-nny)*nnode(2) + nny
                          do z = 1, (1-nnz)*nnode(3) + nnz
                              pos = nnode(2)*nnode(3)*(x-1) +
     +                              nnode(3)*(y-1) + z
                              ii = idx(i) + 1
                              do while (ii.lt.pos .and. i.lt.nexti)
                                  i = i + 1
                                  ii = idx(i) + 1
                              enddo
                              if (ii.eq.pos) then
                                  xx = nnx*nnode(1) + x
                                  yy = nny*nnode(2) + y
                                  zz = nnz*nnode(3) + z
                                  singlegroup(zz, yy, xx) = nodedata(i)
                              endif
                          enddo
                      enddo
                  enddo
              endif
          enddo
          do i = idx0e(group)+1, idx0e(group+1)
              ii = idxe(i)
              x = int(ii/(ne(2)*ne(3))) + 1
              y = mod(int(ii/ne(3)), ne(2)) + 1
              z = mod(ii, ne(3)) + 1
              datae(i) = singlegroup(z, y, x) - singlegroup(z, y+1, x)
          enddo
          do i = idx0f(group)+1, idx0f(group+1)
              ii = idxf(i)
              x = int(ii/(nf(2)*nf(3))) + 1
              y = mod(int(ii/nf(3)), nf(2)) + 1
              z = mod(ii, nf(3)) + 1
              dataf(i) = singlegroup(z, y, x) - singlegroup(z, y, x+1)
          enddo
          do i = idx0g(group)+1, idx0g(group+1)
              ii = idxg(i)
              x = int(ii/(ng(2)*ng(3))) + 1
              y = mod(int(ii/ng(3)), ng(2)) + 1
              z = mod(ii, ng(3)) + 1
              datag(i) = singlegroup(z, y, x) - singlegroup(z+1, y, x)
          enddo
      enddo
      end
c
c
      subroutine node2file(nodedata, idx, idx0, idxe, idx0e, neighbors,
     +                     ne, nnode, sizeidx, sizeidx0, sizee, datae)
c
cf2py intent(out) :: datae
cf2py double complex :: nodedata
cf2py integer :: idx, idx0
cf2py integer :: neighbors
cf2py integer :: idxe, idx0e
cf2py integer :: ne, nnode
cf2py integer, intent(hidden), depend(idx) :: sizeidx=size(idx)
cf2py integer, intent(hidden), depend(idx0) :: sizeidx0=size(idx0)
cf2py integer, intent(hidden), depend(idxe) :: sizee=size(idxe)
c
      integer sizeidx, sizeidx0, sizee
      double complex datae(sizee)
      integer idx(sizeidx), idx0(sizeidx0)
      integer idxe(sizee)
      integer idx0e(sizeidx0)
      double complex nodedata(sizeidx)
      integer neighbors(sizeidx0 - 1, 27)
      integer ne(3), nnode(3)
c
      double complex singlegroup(nnode(3), nnode(2)+1, nnode(1))
      integer nextneigh(2), group, nn, i, ii, neigh
      integer nnx, nny, nnz, x, y, z, xx, yy, zz
c
      data nextneigh/14,17/
      do group = 1, sizeidx0-1
          do x = 1, nnode(1)
              do y = 1, nnode(2)+1
                  do z = 1, nnode(3)
                      singlegroup(z, y, x) = 0.0
                  enddo
              enddo
          enddo
          do nn = 1, 2
              neigh = nextneigh(nn)
              neighgroup = neighbors(group, neigh) + 1
              if (neighgroup.ge.1) then
                  nnx = (nn-1)/4
                  nny = mod((nn-1)/2, 2)
                  nnz = mod((nn-1), 2)
                  ii = idx0(neighgroup) + 1
                  do x = 1, (1-nnx)*nnode(1) + nnx
                      do y = 1, (1-nny)*nnode(2) + nny
                          do z = 1, (1-nnz)*nnode(3) + nnz
                              pos = nnode(2)*nnode(3)*(x-1) +
     +                              nnode(3)*(y-1) + z
                              do while (idx(ii)+1.lt.pos .and.
     +                                  ii.lt.idx0(neighgroup+1))
                                  ii = ii + 1
                              enddo
                              if (idx(ii)+1.eq.pos) then
                                  xx = nnx*nnode(1) + x
                                  yy = nny*nnode(2) + y
                                  zz = nnz*nnode(3) + z
                                  singlegroup(zz, yy, xx) = nodedata(ii)
                              endif
                          enddo
                      enddo
                  enddo
              endif
          enddo
          do i = idx0e(group)+1, idx0e(group+1)
              ii = idxe(i)
              x = int(ii/(ne(2)*ne(3))) + 1
              y = mod(int(ii/ne(3)), ne(2)) + 1
              z = mod(ii, ne(3)) + 1
              datae(i) = singlegroup(z, y, x) - singlegroup(z, y+1, x)
          enddo
      enddo
      end
c
c
      subroutine node2filf(nodedata, idx, idx0, idxf, idx0f, neighbors,
     +                     nf, nnode, sizeidx, sizeidx0, sizef, dataf)
c
cf2py intent(out) :: dataf
cf2py double complex :: nodedata
cf2py integer :: idx, idx0
cf2py integer :: neighbors
cf2py integer :: idxf, idx0f
cf2py integer :: nf, nnode
cf2py integer, intent(hidden), depend(idx) :: sizeidx=size(idx)
cf2py integer, intent(hidden), depend(idx0) :: sizeidx0=size(idx0)
cf2py integer, intent(hidden), depend(idxf) :: sizef=size(idxf)
c
      integer sizeidx, sizeidx0, sizef
      double complex dataf(sizef)
      integer idx(sizeidx), idx0(sizeidx0)
      integer idxf(sizef)
      integer idx0f(sizeidx0)
      double complex nodedata(sizeidx)
      integer neighbors(sizeidx0 - 1, 27)
      integer nf(3), nnode(3)
c
      double complex singlegroup(nnode(3), nnode(2), nnode(1)+1)
      integer nextneigh(2), group, nn, i, ii, neigh
      integer nnx, nny, nnz, x, y, z, xx, yy, zz
c
      data nextneigh/14,23/
      do group = 1, sizeidx0-1
          do x = 1, nnode(1)+1
              do y = 1, nnode(2)
                  do z = 1, nnode(3)
                      singlegroup(z, y, x) = 0.0
                  enddo
              enddo
          enddo
          do nn = 1, 2
              neigh = nextneigh(nn)
              neighgroup = neighbors(group, neigh) + 1
              if (neighgroup.ge.1) then
                  nnx = (nn-1)/4
                  nny = mod((nn-1)/2, 2)
                  nnz = mod((nn-1), 2)
                  ii = idx0(neighgroup) + 1
                  do x = 1, (1-nnx)*nnode(1) + nnx
                      do y = 1, (1-nny)*nnode(2) + nny
                          do z = 1, (1-nnz)*nnode(3) + nnz
                              pos = nnode(2)*nnode(3)*(x-1) +
     +                              nnode(3)*(y-1) + z
                              do while (idx(ii)+1.lt.pos .and.
     +                                  ii.lt.idx0(neighgroup+1))
                                  ii = ii + 1
                              enddo
                              if (idx(ii)+1.eq.pos) then
                                  xx = nnx*nnode(1) + x
                                  yy = nny*nnode(2) + y
                                  zz = nnz*nnode(3) + z
                                  singlegroup(zz, yy, xx) = nodedata(ii)
                              endif
                          enddo
                      enddo
                  enddo
              endif
          enddo
          do i = idx0f(group)+1, idx0f(group+1)
              ii = idxf(i)
              x = int(ii/(nf(2)*nf(3))) + 1
              y = mod(int(ii/nf(3)), nf(2)) + 1
              z = mod(ii, nf(3)) + 1
              dataf(i) = singlegroup(z, y, x) - singlegroup(z, y, x+1)
          enddo
      enddo
      end
c
c
      subroutine node2filg(nodedata, idx, idx0, idxg, idx0g, neighbors,
     +                     ng, nnode, sizeidx, sizeidx0, sizeg, datag)
c
cf2py intent(out) :: datag
cf2py double complex :: nodedata
cf2py integer :: idx, idx0
cf2py integer :: neighbors
cf2py integer :: idxg, idx0g
cf2py integer :: ng, nnode
cf2py integer, intent(hidden), depend(idx) :: sizeidx=size(idx)
cf2py integer, intent(hidden), depend(idx0) :: sizeidx0=size(idx0)
cf2py integer, intent(hidden), depend(idxg) :: sizeg=size(idxg)
c
      integer sizeidx, sizeidx0, sizeg
      double complex datag(sizeg)
      integer idx(sizeidx), idx0(sizeidx0)
      integer idxg(sizeg)
      integer idx0g(sizeidx0)
      double complex nodedata(sizeidx)
      integer neighbors(sizeidx0 - 1, 27)
      integer ng(3), nnode(3)
c
      double complex singlegroup(nnode(3)+1, nnode(2), nnode(1))
      integer nextneigh(2), group, nn, i, ii, neigh
      integer nnx, nny, nnz, x, y, z, xx, yy, zz
c
      data nextneigh/14,15/
      do group = 1, sizeidx0-1
          do x = 1, nnode(1)
              do y = 1, nnode(2)
                  do z = 1, nnode(3)+1
                      singlegroup(z, y, x) = 0.0
                  enddo
              enddo
          enddo
          do nn = 1, 2
              neigh = nextneigh(nn)
              neighgroup = neighbors(group, neigh) + 1
              if (neighgroup.ge.1) then
                  nnx = (nn-1)/4
                  nny = mod((nn-1)/2, 2)
                  nnz = mod((nn-1), 2)
                  ii = idx0(neighgroup) + 1
                  do x = 1, (1-nnx)*nnode(1) + nnx
                      do y = 1, (1-nny)*nnode(2) + nny
                          do z = 1, (1-nnz)*nnode(3) + nnz
                              pos = nnode(2)*nnode(3)*(x-1) +
     +                              nnode(3)*(y-1) + z
                              do while (idx(ii)+1.lt.pos .and.
     +                                  ii.lt.idx0(neighgroup+1))
                                  ii = ii + 1
                              enddo
                              if (idx(ii)+1.eq.pos) then
                                  xx = nnx*nnode(1) + x
                                  yy = nny*nnode(2) + y
                                  zz = nnz*nnode(3) + z
                                  singlegroup(zz, yy, xx) = nodedata(ii)
                              endif
                          enddo
                      enddo
                  enddo
              endif
          enddo
          do i = idx0g(group)+1, idx0g(group+1)
              ii = idxg(i)
              x = int(ii/(ng(2)*ng(3))) + 1
              y = mod(int(ii/ng(3)), ng(2)) + 1
              z = mod(ii, ng(3)) + 1
              datag(i) = singlegroup(z, y, x) - singlegroup(z+1, y, x)
          enddo
      enddo
      end
c
c
      subroutine filament2node(datae, dataf, datag, idx, idx0, idxe,
     +                      idx0e, idxf, idx0f, idxg, idx0g, neighbors,
     +                      ne, nf, ng, nnode, sizeidx, sizeidx0, sizee,
     +                      sizef, sizeg, nodedata)
c
cf2py intent(out) :: nodedata
cf2py double complex :: datae, dataf, datag
cf2py integer :: idx, idx0
cf2py integer :: idxe, idx0e, idxf, idx0f, idxg, idx0g
cf2py integer :: neighbors
cf2py integer :: ne, nf, ng, nnode
cf2py integer, intent(hidden), depend(idx) :: sizeidx=size(idx)
cf2py integer, intent(hidden), depend(idx0) :: sizeidx0=size(idx0)
cf2py integer, intent(hidden), depend(idxe) :: sizee=size(idxe)
cf2py integer, intent(hidden), depend(idxf) :: sizef=size(idxf)
cf2py integer, intent(hidden), depend(idxg) :: sizeg=size(idxg)
c
      integer sizeidx, sizeidx0, sizee, sizef, sizeg
      integer ne(3), nf(3), ng(3), nnode(3)
      double complex datae(sizee), dataf(sizef), datag(sizeg)
      integer idx(sizeidx), idx0(sizeidx0)
      integer idxe(sizee), idxf(sizef), idxg(sizeg)
      integer idx0e(sizeidx0), idx0f(sizeidx0), idx0g(sizeidx0)
      double complex nodedata(sizeidx)
      integer neighbors(sizeidx0 - 1, 27)
c
      integer i, ii, x, y, z, nn, pos
      integer group, neighe, neighf, neighg
      integer nextneighe(2), nextneighf(2), nextneighg(2)
      double complex singlegroupe(nnode(3), nnode(2)+1, nnode(1))
      double complex singlegroupf(nnode(3), nnode(2), nnode(1)+1)
      double complex singlegroupg(nnode(3)+1, nnode(2), nnode(1))
      data nextneighe/11,14/
      data nextneighf/5,14/
      data nextneighg/13,14/
c
      do group = 1, sizeidx0-1
          do x = 1, nnode(1)
              do y = 1, nnode(2)+1
                  do z = 1, nnode(3)
                      singlegroupe(z, y, x) = 0.0
                  enddo
              enddo
          enddo
          do x = 1, nnode(1)+1
              do y = 1, nnode(2)
                  do z = 1, nnode(3)
                      singlegroupf(z, y, x) = 0.0
                  enddo
              enddo
          enddo
          do x = 1, nnode(1)
              do y = 1, nnode(2)
                  do z = 1, nnode(3)+1
                      singlegroupg(z, y, x) = 0.0
                  enddo
              enddo
          enddo
          do nn = 0, 1
              neighe = neighbors(group, nextneighe(nn+1)) + 1
              neighf = neighbors(group, nextneighf(nn+1)) + 1
              neighg = neighbors(group, nextneighg(nn+1)) + 1
              if (neighe.ge.1) then
                  ii = idx0e(neighe) + 1
                  do x = 1, ne(1)
                      do y = (1-nn)*ne(2) + nn, ne(2)
                          do z = 1, ne(3)
                              pos = ne(2)*ne(3)*(x-1) + ne(3)*(y-1) + z
                              do while (idxe(ii)+1.lt.pos .and.
     +                                  ii.lt.idx0e(neighe+1))
                                  ii = ii + 1
                              enddo
                              if ((idxe(ii)+1.eq.pos) .and.
     +                            (ii.le.idx0e(neighe+1))) then
                                  singlegroupe(z, nn*y+1, x) = datae(ii)
                              endif
                          enddo
                      enddo
                  enddo
              endif
              if (neighf.ge.1) then
                  ii = idx0f(neighf) + 1
                  do x = (1-nn)*nf(1) + nn, nf(1)
                      do y = 1, nf(2)
                          do z = 1, nf(3)
                              pos = nf(2)*nf(3)*(x-1) + nf(3)*(y-1) + z
                              do while (idxf(ii)+1.lt.pos .and.
     +                                  ii.lt.idx0f(neighf+1))
                                  ii = ii + 1
                              enddo
                              if ((idxf(ii)+1.eq.pos) .and.
     +                            (ii.le.idx0f(neighf+1))) then
                                  singlegroupf(z, y, nn*x+1) = dataf(ii)
                              endif
                          enddo
                      enddo
                  enddo
              endif
              if (neighg.ge.1) then
                  ii = idx0g(neighg) + 1
                  do x = 1, ng(1)
                      do y = 1, ng(2)
                          do z = (1-nn)*ng(3) + nn, ng(3)
                              pos = ng(2)*ng(3)*(x-1) + ng(3)*(y-1) + z
                              do while (idxg(ii)+1.lt.pos .and.
     +                                  ii.lt.idx0g(neighg+1))
                                  ii = ii + 1
                              enddo
                              if ((idxg(ii)+1.eq.pos) .and.
     +                            (ii.le.idx0g(neighg+1))) then
                                  singlegroupg(nn*z+1, y, x) = datag(ii)
                              endif
                          enddo
                      enddo
                  enddo
              endif
          enddo
          do ii = idx0(group)+1, idx0(group+1)
              i = idx(ii)
              x = int(i/(nnode(2)*nnode(3))) + 1
              y = mod(int(i/nnode(3)), nnode(2)) + 1
              z = mod(i, nnode(3)) + 1
              nodedata(ii) = nodedata(ii) +
     +            singlegroupe(z, y+1, x) - singlegroupe(z, y, x) +
     +            singlegroupf(z, y, x+1) - singlegroupf(z, y, x) +
     +            singlegroupg(z+1, y, x) - singlegroupg(z, y, x)
          enddo
      enddo
      end
c
c
      subroutine file2node(datae, idx, idx0, idxe, idx0e, neighbors, ne,
     +                     nnode, sizeidx, sizeidx0, sizee, nodedata)
c
cf2py intent(out) :: nodedata
cf2py double complex :: datae
cf2py integer :: idx, idx0, idxe, idx0e, neighbors, ne, nnode
cf2py integer, intent(hidden), depend(idx) :: sizeidx=size(idx)
cf2py integer, intent(hidden), depend(idx0) :: sizeidx0=size(idx0)
cf2py integer, intent(hidden), depend(idxe) :: sizee=size(idxe)
c
      integer sizeidx, sizeidx0, sizee, ne(3), nnode(3)
      double complex datae(sizee)
      integer idx(sizeidx), idx0(sizeidx0), idxe(sizee), idx0e(sizeidx0)
      double complex nodedata(sizeidx)
      integer neighbors(sizeidx0 - 1, 27)
c
      integer i, ii, x, y, z, nn, pos, group, neighe, nextneighe(2)
      double complex singlegroupe(nnode(3), nnode(2)+1, nnode(1))
      data nextneighe/11,14/
c
      do group = 1, sizeidx0-1
          do x = 1, nnode(1)
              do y = 1, nnode(2)+1
                  do z = 1, nnode(3)
                      singlegroupe(z, y, x) = 0.0
                  enddo
              enddo
          enddo
          do nn = 0, 1
              neighe = neighbors(group, nextneighe(nn+1)) + 1
              if (neighe.ge.1) then
                  ii = idx0e(neighe) + 1
                  do x = 1, ne(1)
                      do y = (1-nn)*ne(2) + nn, ne(2)
                          do z = 1, ne(3)
                              pos = ne(2)*ne(3)*(x-1) + ne(3)*(y-1) + z
                              do while (idxe(ii)+1.lt.pos .and.
     +                                  ii.lt.idx0e(neighe+1))
                                  ii = ii + 1
                              enddo
                              if ((idxe(ii)+1.eq.pos) .and.
     +                            (ii.le.idx0e(neighe+1))) then
                                  singlegroupe(z, nn*y+1, x) = datae(ii)
                              endif
                          enddo
                      enddo
                  enddo
              endif
          enddo
          do ii = idx0(group)+1, idx0(group+1)
              i = idx(ii)
              x = int(i/(nnode(2)*nnode(3))) + 1
              y = mod(int(i/nnode(3)), nnode(2)) + 1
              z = mod(i, nnode(3)) + 1
              nodedata(ii) = nodedata(ii) +
     +            singlegroupe(z, y+1, x) - singlegroupe(z, y, x)
          enddo
      enddo
      end
c
c
      subroutine filf2node(dataf, idx, idx0, idxf, idx0f, neighbors, nf,
     +                     nnode, sizeidx, sizeidx0, sizef, nodedata)
c
cf2py intent(out) :: nodedata
cf2py double complex :: dataf
cf2py integer :: idx, idx0, idxf, idx0f, neighbors, nf, nnode
cf2py integer, intent(hidden), depend(idx) :: sizeidx=size(idx)
cf2py integer, intent(hidden), depend(idx0) :: sizeidx0=size(idx0)
cf2py integer, intent(hidden), depend(idxf) :: sizef=size(idxf)
c
      integer sizeidx, sizeidx0, sizef, nf(3), nnode(3)
      double complex dataf(sizef)
      integer idx(sizeidx), idx0(sizeidx0), idxf(sizef), idx0f(sizeidx0)
      double complex nodedata(sizeidx)
      integer neighbors(sizeidx0 - 1, 27)
c
      integer i, ii, x, y, z, nn, pos, group, neighf, nextneighf(2)
      double complex singlegroupf(nnode(3), nnode(2), nnode(1)+1)
      data nextneighf/5,14/
c
      do group = 1, sizeidx0-1
          do x = 1, nnode(1)+1
              do y = 1, nnode(2)
                  do z = 1, nnode(3)
                      singlegroupf(z, y, x) = 0.0
                  enddo
              enddo
          enddo
          do nn = 0, 1
              neighf = neighbors(group, nextneighf(nn+1)) + 1
              if (neighf.ge.1) then
                  ii = idx0f(neighf) + 1
                  do x = 1, (1-nn)*nf(1) + nn, nf(1)
                      do y = 1, nf(2)
                          do z = 1, nf(3)
                              pos = nf(2)*nf(3)*(x-1) + nf(3)*(y-1) + z
                              do while (idxf(ii)+1.lt.pos .and.
     +                                  ii.lt.idx0f(neighf+1))
                                  ii = ii + 1
                              enddo
                              if ((idxf(ii)+1.eq.pos) .and.
     +                            (ii.le.idx0f(neighf+1))) then
                                  singlegroupf(z, y, nn*x+1) = dataf(ii)
                              endif
                          enddo
                      enddo
                  enddo
              endif
          enddo
          do ii = idx0(group)+1, idx0(group+1)
              i = idx(ii)
              x = int(i/(nnode(2)*nnode(3))) + 1
              y = mod(int(i/nnode(3)), nnode(2)) + 1
              z = mod(i, nnode(3)) + 1
              nodedata(ii) = nodedata(ii) +
     +            singlegroupf(z, y, x+1) - singlegroupf(z, y, x)
          enddo
      enddo
      end
c
c
      subroutine filg2node(datag, idx, idx0, idxg, idx0g, neighbors, ng,
     +                     nnode, sizeidx, sizeidx0, sizeg, nodedata)
c
cf2py intent(out) :: nodedata
cf2py double complex :: datag
cf2py integer :: idx, idx0, idxg, idx0g, neighbors, ng, nnode
cf2py integer, intent(hidden), depend(idx) :: sizeidx=size(idx)
cf2py integer, intent(hidden), depend(idx0) :: sizeidx0=size(idx0)
cf2py integer, intent(hidden), depend(idxg) :: sizeg=size(idxg)
c
      integer sizeidx, sizeidx0, sizeg, ng(3), nnode(3)
      double complex datag(sizeg)
      integer idx(sizeidx), idx0(sizeidx0), idxg(sizeg), idx0g(sizeidx0)
      double complex nodedata(sizeidx)
      integer neighbors(sizeidx0 - 1, 27)
c
      integer i, ii, x, y, z, nn, pos, group, neighg, nextneighg(2)
      double complex singlegroupg(nnode(3)+1, nnode(2), nnode(1))
      data nextneighg/13,14/
c
      do group = 1, sizeidx0-1
          do x = 1, nnode(1)
              do y = 1, nnode(2)
                  do z = 1, nnode(3)+1
                      singlegroupg(z, y, x) = 0.0
                  enddo
              enddo
          enddo
          do nn = 0, 1
              neighg = neighbors(group, nextneighg(nn+1)) + 1
              if (neighg.ge.1) then
                  ii = idx0g(neighg) + 1
                  do x = 1, ng(1)
                      do y = 1, ng(2)
                          do z = (1-nn)*ng(3) + nn, ng(3)
                              pos = ng(2)*ng(3)*(x-1) + ng(3)*(y-1) + z
                              do while (idxg(ii)+1.lt.pos .and.
     +                                  ii.lt.idx0g(neighg+1))
                                  ii = ii + 1
                              enddo
                              if ((idxg(ii)+1.eq.pos) .and.
     +                            (ii.le.idx0g(neighg+1))) then
                                  singlegroupg(nn*z+1, y, x) = datag(ii)
                              endif
                          enddo
                      enddo
                  enddo
              endif
          enddo
          do ii = idx0(group)+1, idx0(group+1)
              i = idx(ii)
              x = int(i/(nnode(2)*nnode(3))) + 1
              y = mod(int(i/nnode(3)), nnode(2)) + 1
              z = mod(i, nnode(3)) + 1
              nodedata(ii) = nodedata(ii) +
     +            singlegroupg(z+1, y, x) - singlegroupg(z, y, x)
          enddo
      enddo
      end
c
c
      subroutine get_fft_plans(m, mat_a, mat_b, mat_c, plans,
     +                         mx, my, mz)
cf2py intent(out) :: plans
cf2py integer :: m
cf2py double complex :: mat_a, mat_b, mat_c
cf2py integer, intent(hide), depend(m) :: mx = m[0]
cf2py integer, intent(hide), depend(m) :: my = m[1]
cf2py integer, intent(hide), depend(m) :: mz = m[2]
c
      integer*8 plans(6)
      integer m(3), mx, my, mz, howmany_n(2), howmany_is(2)
      double complex mat_a(mz, my, 2*mx)
      double complex mat_b(mz, 2*my, 2*mx)
      double complex mat_c(2*mz, 2*my, 2*mx)
#include "fftw3.f"
c
      call dfftw_plan_many_dft(plans(1), 1, 2*m(1), m(2)*m(3), mat_a,
     +                         2*m(1), m(2)*m(3), 1, mat_a, 2*m(1),
     +                         m(2)*m(3), 1, FFTW_FORWARD, FFTW_MEASURE)
      howmany_n(1) = 2*m(1); howmany_n(2) = m(3)
      howmany_is(1) = 2*m(2)*m(3); howmany_is(2) = 1
      call dfftw_plan_guru_dft(plans(2), 1, 2*m(2), m(3), m(3), 2,
     +                         howmany_n, howmany_is, howmany_is, mat_b,
     +                         mat_b, FFTW_FORWARD, FFTW_MEASURE)
      call dfftw_plan_many_dft(plans(3), 1, 2*m(3), 4*m(1)*m(2), mat_c,
     +                         2*m(3), 1, 2*m(3), mat_c, 2*m(3),
     +                         1, 2*m(3), FFTW_FORWARD, FFTW_MEASURE)
      call dfftw_plan_many_dft(plans(4), 1, 2*m(3), 4*m(1)*m(2), mat_c,
     +                         2*m(3), 1, 2*m(3), mat_c, 2*m(3),
     +                         1, 2*m(3), FFTW_BACKWARD, FFTW_MEASURE)
      call dfftw_plan_guru_dft(plans(5), 1, 2*m(2), m(3), m(3), 2,
     +                         howmany_n, howmany_is, howmany_is, mat_b,
     +                         mat_b, FFTW_BACKWARD, FFTW_MEASURE)
      call dfftw_plan_many_dft(plans(6), 1, 2*m(1), m(2)*m(3), mat_a,
     +                         2*m(1), m(2)*m(3), 1, mat_a, 2*m(1),
     +                         m(2)*m(3), 1, FFTW_BACKWARD,
     +                         FFTW_MEASURE)
      end
c
c
      subroutine destroy_fft_plans(plans)
cf2py intent(inout) :: plans
c
      integer*8 plans(6)
#include "fftw3.f"
c
      call dfftw_destroy_plan(plans(1))
      call dfftw_destroy_plan(plans(2))
      call dfftw_destroy_plan(plans(3))
      call dfftw_destroy_plan(plans(4))
      call dfftw_destroy_plan(plans(5))
      call dfftw_destroy_plan(plans(6))
      end
c
c
      subroutine p2p_full(slabidx, slabidx0, neighbors, xidx, idx,
     +                    idx0, p2p_transfer, revslabidx, orevslabidx,
     +                    m, n, ng, singlegroup_a, singlegroup_b,
     +                    singlegroup_c, plans, selfdata, gsize, dsize,
     +                    mx, my, mz, ngx)
cf2py double complex, intent(inout) :: selfdata
cf2py integer :: slabidx, slabidx0, neighbors, xidx, idx, idx0
cf2py integer :: revslabidx, orevslabidx, m, n, ng
cf2py integer*8 :: plans
cf2py double complex, intent(inout) :: singlegroup_a, singlegroup_b
cf2py double complex, intent(inout) :: singlegroup_c
cf2py integer, intent(hide), depend(m) :: mx = m[0]
cf2py integer, intent(hide), depend(m) :: my = m[1]
cf2py integer, intent(hide), depend(m) :: mz = m[2]
cf2py integer, intent(hide), depend(ng) :: ngx = ng[0]
cf2py double complex :: p2p_transfer
cf2py integer, intent(hide), depend(xidx) :: gsize=size(xidx)
cf2py integer, intent(hide), depend(selfdata) :: dsize=size(selfdata)
      integer mx, my, mz, ngx, gsize, dsize, idx(dsize), idx0(gsize+1)
      integer*8 plans(6)
      integer neighbors(gsize, 27)
      integer slabidx(gsize), slabidx0(ngx+1), ng(3), n(3), m(3)
      double complex p2p_transfer(2*mz, 2*my, 2*mx, 27)
      double complex singlegroup_a(mz, my, 2*mx)
      double complex singlegroup_b(mz, 2*my, 2*mx)
      double complex singlegroup_c(2*mz, 2*my, 2*mx)
      double complex selfdata(dsize)
      integer xidx(gsize), revslabidx(gsize), orevslabidx(gsize)
c
      integer ahead, current, behind, islabidx(ng(2)*ng(3)), sizeslab(3)
      double complex inslab(2*m(3), 2*m(2), 2*m(1), ng(2)*ng(3), 3)
      double complex targetslab(2*m(3), 2*m(2), 2*m(1), ng(2)*ng(3))
      double complex singlegroupflat(n(1)*n(2)*n(3))
      integer i, ii, countx, slabng, oyzidx, sizeslabtarget
      integer flatix, flatiy, flatiz
      integer countg, countn, group, neighgroup, x, y, z, xoff, yzidx
      integer m11, m12, m13, m21, m22, m23, md1, md2, md3, n1, n2, n3
      double complex trans, sdata
      data sizeslab/ 3 * 0/
#include "fftw3.f"
c
      m11 = m(1); m12 = m(2); m13 = m(3)
      m21 = 2*m(1); m22 = 2*m(2); m23 = 2*m(3)
      md1 = dble(m21); md2 = dble(m22); md3 = dble(m23)
      n1 = n(1); n2 = n(2); n3 = n(3)
      slabng = ng(1) * ng(2)
      do i = 1, 3
          do ii = 1, slabng
              do x = 1, m21
                  do y = 1, m22
                      do z = 1, m23
                          inslab(z, y, x, ii, i) = 0.0
                          targetslab(z, y, x, ii) = 0.0
                      enddo
                  enddo
              enddo
          enddo
      enddo
      do countx = 0, ng(1)
          ahead = mod(countx+2, 3) + 1
          current = mod(countx+1, 3) + 1
          behind = mod(countx, 3) + 1
          if (countx.lt.ng(1)) then
              sizeslab(ahead) = slabidx0(countx+2) - slabidx0(countx+1)
              do x = 1, m21
                  do y = 1, m12
                      do z = 1, m13
                          singlegroup_a(z, y, x) = 0.0
                      enddo
                  enddo
              enddo
              do x = 1, m21
                  do y = 1, m22
                      do z = 1, m13
                          singlegroup_b(z, y, x) = 0.0
                      enddo
                  enddo
              enddo
              do x = 1, m21
                  do y = 1, m22
                      do z = 1, m23
                          singlegroup_c(z, y, x) = 0.0
                      enddo
                  enddo
              enddo
              do countg = 1, sizeslab(ahead)
                  group = slabidx(slabidx0(countx+1) + countg) + 1
                  do i = 1, n1*n2*n3
                      singlegroupflat(i) = 0.0
                  enddo
                  do i = idx0(group)+1, idx0(group+1)
                      singlegroupflat(idx(i)+1) = selfdata(i)
                  enddo
                  do x = 1, n1
                      flatix = (x-1)*n2*n3
                      do y = 1, n2
                          flatiy = flatix + (y-1)*n3
                          do z = 1, n3
                              flatiz = flatiy + z
                              singlegroup_a(z, y, x) =
     +                            singlegroupflat(flatiz)
                          enddo
                      enddo
                  enddo
                  call dfftw_execute_dft(plans(1), singlegroup_a,
     +                                   singlegroup_a)
                  do x = 1, m21
                      do y = 1, m12
                          do z = 1, m13
                              singlegroup_b(z, y, x) =
     +                            singlegroup_a(z, y, x)
                          enddo
                      enddo
                  enddo
                  call dfftw_execute_dft(plans(2), singlegroup_b,
     +                                   singlegroup_b)
                  do x = 1, m21
                      do y = 1, m22
                          do z = 1, m13
                              singlegroup_c(z, y, x) =
     +                            singlegroup_b(z, y, x)
                          enddo
                      enddo
                  enddo
                  call dfftw_execute_dft(plans(3), singlegroup_c,
     +                                   singlegroup_c)
                  do x = 1, m21
                      do y = 1, m22
                          do z = 1, m23
                              inslab(z, y, x, countg, ahead) =
     +                            singlegroup_c(z, y, x)
                          enddo
                      enddo
                  enddo
              enddo
          endif
          if ((countx.ge.1) .and. (sizeslab(current).gt.0)) then
              sizeslabtarget = slabidx0(countx+1) - slabidx0(countx)
              do i = slabidx0(countx)+1, slabidx0(countx+1)
                  islabidx(i) = slabidx(i)
              enddo
              do countg = 1, sizeslabtarget
                  group = islabidx(countg) + 1
                  oyzidx = orevslabidx(group) + 1
                  if (oyzidx.ge.1) then
                      do countn = 1, 27
                          neighgroup = neighbors(group, countn) + 1
                          if (neighgroup.ge.1) then
                              yzidx = revslabidx(neighgroup) + 1
                          endif
                          if (neighgroup.ge.1 .and. yzidx.ge.1) then
                              xoff = xidx(neighgroup)+1 - countx
                              do x = 1, m21
                                  do y = 1, m22
                                      do z = 1, m23
                                          trans = p2p_transfer(z, y, x,
     +                                                         countn)
                                          if (xoff.eq.-1) then
                                              sdata = inslab(z, y, x,
     +                                            yzidx, behind)
                                          elseif (xoff.eq.0) then
                                              sdata = inslab(z, y, x,
     +                                            yzidx, current)
                                          elseif (xoff.eq.1) then
                                              sdata = inslab(z, y, x,
     +                                            yzidx, ahead)
                                          else sdata = 0
                                          endif
                                          targetslab(z, y, x, oyzidx) =
     +                                      targetslab(z, y, x, oyzidx)
     +                                      + trans*sdata
                                      enddo
                                  enddo
                              enddo
                          endif
                      enddo
                  endif
              enddo
              do countg = 1, sizeslabtarget
                  do x = 1, m21
                      do y = 1, m22
                          do z = 1, m23
                              singlegroup_c(z, y, x) =
     +                            targetslab(z, y, x, countg)
                          enddo
                      enddo
                  enddo
                  call dfftw_execute_dft(plans(4), singlegroup_c,
     +                                   singlegroup_c)
                  do x = 1, m21
                      do y = 1, m22
                          do z = 1, m13
                              singlegroup_b(z, y, x) =
     +                            singlegroup_c(z, y, x) / md3
                          enddo
                      enddo
                  enddo
                  call dfftw_execute_dft(plans(5), singlegroup_b,
     +                                   singlegroup_b)
                  do x = 1, m21
                      do y = 1, m12
                          do z = 1, m13
                              singlegroup_a(z, y, x) =
     +                            singlegroup_b(z, y, x) / md2
                          enddo
                      enddo
                  enddo
                  call dfftw_execute_dft(plans(6), singlegroup_a,
     +                                   singlegroup_a)
                  group = slabidx(countg + slabidx0(countx)) + 1
                  do x = 1, n1
                      flatix = (x-1)*n2*n3
                      do y = 1, n2
                          flatiy = flatix + (y-1)*n3
                          do z = 1, n3
                              flatiz = flatiy + z
                              singlegroupflat(flatiz)
     +                            = singlegroup_a(z, y, x) / md1
                          enddo
                      enddo
                  enddo
                  do i = idx0(group)+1, idx0(group+1)
                      selfdata(i) = singlegroupflat(idx(i)+1)
                  enddo
              enddo
          endif
      enddo
      end
c
c
C
C
      SUBROUTINE CSRMV_S(INDPTR, INDICES, DATA, X, Y,
     +                    NROW, NNZ, NCOL)
C
C     y = A @ x for a CSR matrix -- REAL*4 data, INTEGER*4 indices.
C     Row-parallel: each Y(I) is one thread's ORDERED dot product
C     (ascending column index, same as scipy's csr_matvec), so
C     threaded results are BIT-IDENTICAL to serial at any OMP count.
C     Added 2026-08-25 for the GeoMG v-cycle, whose scipy SpMV was
C     the largest single-threaded line left in a CPU solve cycle
C     (7.9 s of ~26 at R4 after the MID_M2L/P2P threading).
C
CF2PY INTENT(OUT) :: Y
CF2PY REAL :: DATA, X, Y
CF2PY INTEGER :: INDPTR, INDICES
CF2PY INTEGER, INTENT(HIDE), DEPEND(INDPTR) :: NROW=SIZE(INDPTR)-1
CF2PY INTEGER, INTENT(HIDE), DEPEND(DATA) :: NNZ=SIZE(DATA)
CF2PY INTEGER, INTENT(HIDE), DEPEND(X) :: NCOL=SIZE(X)
      INTEGER NROW, NNZ, NCOL
      INTEGER INDPTR(NROW+1), INDICES(NNZ)
      REAL DATA(NNZ), X(NCOL)
      REAL Y(NROW)
      INTEGER I
      INTEGER J
      REAL ACC
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(I, J, ACC)
      DO I = 1, NROW
          ACC = 0.0
          DO J = INDPTR(I) + 1, INDPTR(I + 1)
              ACC = ACC + DATA(J)*X(INDICES(J) + 1)
          ENDDO
          Y(I) = ACC
      ENDDO
!$OMP END PARALLEL DO
      END
C
C
      SUBROUTINE CSRMV_D(INDPTR, INDICES, DATA, X, Y,
     +                    NROW, NNZ, NCOL)
C
C     y = A @ x for a CSR matrix -- REAL*8 data, INTEGER*4 indices.
C     Row-parallel: each Y(I) is one thread's ORDERED dot product
C     (ascending column index, same as scipy's csr_matvec), so
C     threaded results are BIT-IDENTICAL to serial at any OMP count.
C     Added 2026-08-25 for the GeoMG v-cycle, whose scipy SpMV was
C     the largest single-threaded line left in a CPU solve cycle
C     (7.9 s of ~26 at R4 after the MID_M2L/P2P threading).
C
CF2PY INTENT(OUT) :: Y
CF2PY REAL*8 :: DATA, X, Y
CF2PY INTEGER :: INDPTR, INDICES
CF2PY INTEGER, INTENT(HIDE), DEPEND(INDPTR) :: NROW=SIZE(INDPTR)-1
CF2PY INTEGER, INTENT(HIDE), DEPEND(DATA) :: NNZ=SIZE(DATA)
CF2PY INTEGER, INTENT(HIDE), DEPEND(X) :: NCOL=SIZE(X)
      INTEGER NROW, NNZ, NCOL
      INTEGER INDPTR(NROW+1), INDICES(NNZ)
      REAL*8 DATA(NNZ), X(NCOL)
      REAL*8 Y(NROW)
      INTEGER I
      INTEGER J
      REAL*8 ACC
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(I, J, ACC)
      DO I = 1, NROW
          ACC = 0.0
          DO J = INDPTR(I) + 1, INDPTR(I + 1)
              ACC = ACC + DATA(J)*X(INDICES(J) + 1)
          ENDDO
          Y(I) = ACC
      ENDDO
!$OMP END PARALLEL DO
      END
C
C
      SUBROUTINE CSRMV_SL(INDPTR, INDICES, DATA, X, Y,
     +                    NROW, NNZ, NCOL)
C
C     y = A @ x for a CSR matrix -- REAL*4 data, INTEGER*8 indices.
C     Row-parallel: each Y(I) is one thread's ORDERED dot product
C     (ascending column index, same as scipy's csr_matvec), so
C     threaded results are BIT-IDENTICAL to serial at any OMP count.
C     Added 2026-08-25 for the GeoMG v-cycle, whose scipy SpMV was
C     the largest single-threaded line left in a CPU solve cycle
C     (7.9 s of ~26 at R4 after the MID_M2L/P2P threading).
C
CF2PY INTENT(OUT) :: Y
CF2PY REAL :: DATA, X, Y
CF2PY INTEGER*8 :: INDPTR, INDICES
CF2PY INTEGER, INTENT(HIDE), DEPEND(INDPTR) :: NROW=SIZE(INDPTR)-1
CF2PY INTEGER, INTENT(HIDE), DEPEND(DATA) :: NNZ=SIZE(DATA)
CF2PY INTEGER, INTENT(HIDE), DEPEND(X) :: NCOL=SIZE(X)
      INTEGER NROW, NNZ, NCOL
      INTEGER*8 INDPTR(NROW+1), INDICES(NNZ)
      REAL DATA(NNZ), X(NCOL)
      REAL Y(NROW)
      INTEGER I
      INTEGER*8 J
      REAL ACC
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(I, J, ACC)
      DO I = 1, NROW
          ACC = 0.0
          DO J = INDPTR(I) + 1, INDPTR(I + 1)
              ACC = ACC + DATA(J)*X(INDICES(J) + 1)
          ENDDO
          Y(I) = ACC
      ENDDO
!$OMP END PARALLEL DO
      END
C
C
      SUBROUTINE CSRMV_DL(INDPTR, INDICES, DATA, X, Y,
     +                    NROW, NNZ, NCOL)
C
C     y = A @ x for a CSR matrix -- REAL*8 data, INTEGER*8 indices.
C     Row-parallel: each Y(I) is one thread's ORDERED dot product
C     (ascending column index, same as scipy's csr_matvec), so
C     threaded results are BIT-IDENTICAL to serial at any OMP count.
C     Added 2026-08-25 for the GeoMG v-cycle, whose scipy SpMV was
C     the largest single-threaded line left in a CPU solve cycle
C     (7.9 s of ~26 at R4 after the MID_M2L/P2P threading).
C
CF2PY INTENT(OUT) :: Y
CF2PY REAL*8 :: DATA, X, Y
CF2PY INTEGER*8 :: INDPTR, INDICES
CF2PY INTEGER, INTENT(HIDE), DEPEND(INDPTR) :: NROW=SIZE(INDPTR)-1
CF2PY INTEGER, INTENT(HIDE), DEPEND(DATA) :: NNZ=SIZE(DATA)
CF2PY INTEGER, INTENT(HIDE), DEPEND(X) :: NCOL=SIZE(X)
      INTEGER NROW, NNZ, NCOL
      INTEGER*8 INDPTR(NROW+1), INDICES(NNZ)
      REAL*8 DATA(NNZ), X(NCOL)
      REAL*8 Y(NROW)
      INTEGER I
      INTEGER*8 J
      REAL*8 ACC
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(I, J, ACC)
      DO I = 1, NROW
          ACC = 0.0
          DO J = INDPTR(I) + 1, INDPTR(I + 1)
              ACC = ACC + DATA(J)*X(INDICES(J) + 1)
          ENDDO
          Y(I) = ACC
      ENDDO
!$OMP END PARALLEL DO
      END
