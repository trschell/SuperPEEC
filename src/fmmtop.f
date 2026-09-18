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
      INTEGER N,M,J,K,IDXNM,NNMAXIDXNM,IDXJK,IDXNMJK,Y0,Y1,NSEG,YT
      INTEGER I,NPL
C
C     Plane-tiled (2026-09-17): the transfer table (NNMAX2 padded
C     grids) and the spectra do not fit in cache, and the old loop
C     order -- every (n,m,j,k) pair streaming its whole grid -- moved
C     NNMAX**2 x 32 B per grid point from memory (5.9 GB per call on
C     R3, 0.25 s at 23 GB/s). Now each thread takes one X plane and,
C     inside it, a run of Y columns short enough that the 131 table
C     and spectrum segments it needs stay in L2; all (n,m,j,k) pairs
C     are applied to that segment before moving on. Per output
C     element the (j,k) accumulation order is unchanged; the result
C     matches the old kernel to fp64 rounding (the compiler's FMA
C     contraction differs between the two loop forms: 4e-16 relative
C     on R3), R3 kernel 0.25 -> 0.07 s.
      NPL = 2*NT*2*NL*2*NW
      DO IDXNM = 1, NNMAX
          CALL M2LZERO(NPL, LNM(1,1,1,IDXNM))
      ENDDO
      YT = MAX(1, 256/(2*NT))
!$OMP PARALLEL DO DEFAULT(SHARED) SCHEDULE(DYNAMIC)
!$OMP& PRIVATE(X,Y0,Y1,NSEG,N,M,J,K,IDXNM,NNMAXIDXNM,IDXJK,IDXNMJK)
      DO X = 1, 2*NW
        DO Y0 = 1, 2*NL, YT
          Y1 = MIN(2*NL, Y0+YT-1)
          NSEG = 2*NT*(Y1-Y0+1)
          DO N = 0, NMAX
            DO M = -N, N
              IDXNM = N**2 + N + M
              NNMAXIDXNM = NNMAX * IDXNM
              DO J = 0, NMAX
                DO K = -J, J
                  IDXJK = J**2 + J + K
                  IDXNMJK = NNMAXIDXNM + IDXJK
                  CALL M2LSEG(NSEG, C(IDXNMJK+1),
     +                 FTRANS(1,Y0,X,(J+N)**2+J+N+K-M+1),
     +                 FMG(1,Y0,X,IDXJK+1), LNM(1,Y0,X,IDXNM+1))
                ENDDO
              ENDDO
            ENDDO
          ENDDO
        ENDDO
      ENDDO
!$OMP END PARALLEL DO
      END
C
C
      SUBROUTINE M2LSEG(N, CC, T, S, O)
C     O(I) = O(I) + CC*T(I)*S(I) over one contiguous segment.
      INTEGER N, I
      DOUBLE COMPLEX CC, T(N), S(N), O(N)
      DO I = 1, N
          O(I) = O(I) + CC*T(I)*S(I)
      ENDDO
      END
C
C
      SUBROUTINE M2LZERO(N, O)
      INTEGER N, I
      DOUBLE COMPLEX O(N)
      DO I = 1, N
          O(I) = 0
      ENDDO
      END
