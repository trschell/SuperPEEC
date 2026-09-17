C SPDX-License-Identifier: MIT
C
      SUBROUTINE M2L(FMG,FTRANS,C,NMAX,NW,NL,NT,LNM,NNMAX,NNMAX2)
C
CF2PY INTENT(OUT) :: LNM
CF2PY DOUBLE COMPLEX :: FMG
CF2PY DOUBLE COMPLEX :: FTRANS
CF2PY DOUBLE COMPLEX :: C
CF2PY INTEGER :: NMAX
CF2PY INTEGER, INTENT(HIDE), DEPEND(FMG) :: NNMAX=SHAPE(FMG,3)
CF2PY INTEGER, INTENT(HIDE), DEPEND(FTRANS) :: NNMAX2=SHAPE(FTRANS,3)
      INTEGER NMAX,NNMAX,NW,NL,NT,X,Y,Z
      DOUBLE COMPLEX C(*), FMG(2*NT,2*NL,2*NW,NNMAX)
      DOUBLE COMPLEX FTRANS(2*NT,2*NL,2*NW,NNMAX2)
      DOUBLE COMPLEX LNM(2*NT,2*NL,2*NW,NNMAX)
C
      INTEGER N,M,J,K,IDXNM,NNMAXIDXNM,IDXJK,IDXNMJK
C
C     Threaded over the (N,M) OUTPUT channels (2026-09-17): each
C     channel's accumulation keeps its serial order, so threaded ==
C     serial bit for bit; the kernel was the second-largest phase of a
C     CPU-only solve once the preconditioner was fixed (0.77 s per
C     matvec on R3, serial). Needs -fopenmp in Makefile_multipole.
!$OMP PARALLEL DO DEFAULT(SHARED) SCHEDULE(DYNAMIC)
!$OMP& PRIVATE(N, M, J, K, IDXNM, NNMAXIDXNM, IDXJK, IDXNMJK, X, Y, Z)
      DO N = 0, NMAX
          DO M = -N, N
              IDXNM = N**2 + N + M
              NNMAXIDXNM = NNMAX * IDXNM
              DO J = 0, NMAX
                  DO K = -J, J
                      IDXJK = J**2 + J + K
                      IDXNMJK = NNMAXIDXNM + IDXJK
                      DO X = 1, 2*NW
                          DO Y = 1, 2*NL
                              DO Z = 1, 2*NT
                                  LNM(Z,Y,X,IDXNM+1) =
     +                                LNM(Z,Y,X,IDXNM+1) + C(IDXNMJK+1)*
     +                                FTRANS(Z,Y,X,(J+N)**2+J+N+K-M+1) *
     +                                FMG(Z,Y,X,J**2+J+K+1)
                              ENDDO
                          ENDDO
                      ENDDO
                  ENDDO
              ENDDO
          ENDDO
      ENDDO
!$OMP END PARALLEL DO
      END
