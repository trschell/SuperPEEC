C SPDX-License-Identifier: MIT
c
      subroutine f_getneighbors(xidx, yidx, zidx, outmat, sizeidx)
c
c     Uses the xyz-coordinates (xidx, yidx, zidx) to construct an
c     array that contains the index of each potential neighbor.
c     Non-existent neighbors receive the index -1. The stored indices
c     are zero-based.
c
cf2py intent(out) :: outmat
cf2py integer :: xidx, yidx, zidx
cf2py integer, intent(hide), depend(xidx) :: sizeidx=size(xidx)
      integer sizeidx
      integer xidx(sizeidx), yidx(sizeidx), zidx(sizeidx)
      integer outmat(sizeidx, 27)
c
      integer neighbor, neighx, neighy, neighz, stopflag, i, j
c
      do neighbor = 1, 27
          neighx = (neighbor - 1)/9 - 1
          neighy = mod((neighbor - 1)/3, 3) - 1
          neighz = mod((neighbor - 1), 3) - 1
          stopflag = 0
          i = 0
10        continue
              i = i + 1
              j = 0
20            continue
                  j = j + 1
                  if (xidx(j) .eq. xidx(i)+neighx) then
                      if (yidx(j) .eq. yidx(i)+neighy) then
                          if (zidx(j) .eq. zidx(i)+neighz) then
                              stopflag = 1
                          endif
                      endif
                  endif
              if (j.lt.sizeidx .and. stopflag.eq.0) goto 20
              if (stopflag.eq.1) then
                  outmat(i, neighbor) = j - 1
                  stopflag = 0
              else
                  outmat(i, neighbor) = -1
              endif
          if (i.lt.sizeidx) goto 10
      enddo
      end
