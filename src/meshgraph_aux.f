C SPDX-License-Identifier: MIT
      subroutine adjmat(adjnode, node2fil, adjnnz, nn, adjdat, adjind,
     +                  adjindptr)
cf2py integer, intent(out) :: adjdat, adjind, adjindptr
cf2py integer :: adjnode, node2fil, adjnnz
cf2py integer, intent(hide), depend(adjnode) :: nn=shape(adjnode, 1)
c
      integer adjnnz, nn
      integer adjdat(adjnnz), adjind(adjnnz), adjindptr(nn+1)
      integer adjnode(3, nn), node2fil(3, nn)
c
      integer ii, adjptr, tmpind, tmpdat, startptr, stopptr
c
      adjptr = 1
      adjindptr(1) = 0
      do ii = 1, nn
          if (adjnode(1, ii) > 0) then
              adjdat(adjptr) = node2fil(1, ii)
              adjind(adjptr) = adjnode(1, ii) - 1
              adjptr = adjptr + 1
          endif
          if (adjnode(2, ii) > 0) then
              adjdat(adjptr) = node2fil(2, ii)
              adjind(adjptr) = adjnode(2, ii) - 1
              adjptr = adjptr + 1
          endif
            if (adjnode(3, ii) > 0) then
              adjdat(adjptr) = node2fil(3, ii)
              adjind(adjptr) = adjnode(3, ii) - 1
              adjptr = adjptr + 1
          endif
          adjindptr(ii+1) = adjptr - 1
          startptr = adjindptr(ii) + 1
          stopptr = adjindptr(ii+1)
          if (stopptr - startptr .eq. 1) then
              if (adjind(stopptr) .lt. adjind(startptr)) then
                  tmpind = adjind(startptr)
                  adjind(startptr) = adjind(stopptr)
                  adjind(stopptr) = tmpind
                  tmpdat = adjdat(startptr)
                  adjdat(startptr) = adjdat(stopptr)
                  adjdat(stopptr) = tmpdat
              endif
          elseif (stopptr - startptr .eq. 2) then
              if (adjind(startptr+1) .lt. adjind(startptr)) then
                  tmpind = adjind(startptr)
                  adjind(startptr) = adjind(startptr+1)
                  adjind(startptr+1) = tmpind
                  tmpdat = adjdat(startptr)
                  adjdat(startptr) = adjdat(startptr+1)
                  adjdat(startptr+1) = tmpdat
              endif
              if (adjind(stopptr) .lt. adjind(startptr+1)) then
                  tmpind = adjind(startptr+1)
                  adjind(startptr+1) = adjind(stopptr)
                  adjind(stopptr) = tmpind
                  tmpdat = adjdat(startptr+1)
                  adjdat(startptr+1) = adjdat(stopptr)
                  adjdat(stopptr) = tmpdat
              endif
              if (adjind(startptr+1) .lt. adjind(startptr)) then
                  tmpind = adjind(startptr)
                  adjind(startptr) = adjind(startptr+1)
                  adjind(startptr+1) = tmpind
                  tmpdat = adjdat(startptr)
                  adjdat(startptr) = adjdat(startptr+1)
                  adjdat(startptr+1) = tmpdat
              endif
          endif
      enddo
      end
c
c
      subroutine adjnode(idx, idx0, neighbors, nnode, sizeidx, sizeidx0,
     +                   adjnodes)
c
cf2py intent(out) :: adjnodes
cf2py integer :: idx, idx0
cf2py integer :: neighbors
cf2py integer :: nnode
cf2py integer, intent(hide), depend(idx) :: sizeidx=size(idx)
cf2py integer, intent(hide), depend(idx0) :: sizeidx0=size(idx0)
c
      integer sizeidx, sizeidx0
      integer adjnodes(3, sizeidx)
      integer idx(sizeidx), idx0(sizeidx0)
      integer neighbors(sizeidx0 - 1, 27)
      integer nnode(3)
c
      integer singlegroup(nnode(3)+1, nnode(2)+1, nnode(1)+1)
      integer nextneigh(8), group, nn, i, ii, neigh, nexti
      integer nnx, nny, nnz, x, y, z, xx, yy, zz
c
      data nextneigh/14,15,17,18,23,24,26,27/
      do group = 1, sizeidx0-1
          do x = 1, nnode(1)+1
              do y = 1, nnode(2)+1
                  do z = 1, nnode(3)+1
                      singlegroup(z, y, x) = 0
                  enddo
              enddo
          enddo
          do nn = 1, 8
              neigh = nextneigh(nn)
              neighgroup = neighbors(group, neigh) + 1
c                 An empty neighbour group would make the index below
c                 run one past the end of idx: with idx0(g) == idx0(g+1)
c                 the gather starts at idx0(g)+1 == size+1 and reads
c                 out of bounds, which corrupts the heap rather than
c                 raising. Empty groups became possible once boxes were
c                 stored for holding ANY element (a box can carry
c                 surface panels without carrying nodes).
              if (neighgroup.ge.1 .and.
     +            idx0(neighgroup+1).gt.idx0(neighgroup)) then
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
                                  singlegroup(zz, yy, xx) = i
                              endif
                          enddo
                      enddo
                  enddo
              endif
          enddo
          do i = idx0(group)+1, idx0(group+1)
              ii = idx(i)
              x = int(ii/(nnode(2)*nnode(3))) + 1
              y = mod(int(ii/nnode(3)), nnode(2)) + 1
              z = mod(ii, nnode(3)) + 1
              adjnodes(1, i) = singlegroup(z+1, y, x)
              adjnodes(2, i) = singlegroup(z, y+1, x)
              adjnodes(3, i) = singlegroup(z, y, x+1)
          enddo
      enddo
      end
c
c
      subroutine fil2nodesparse(idx, idx0, idxe, idx0e, idxf, idx0f,
     +                          idxg, idx0g, neighbors, ne, nf, ng,
     +                          nnode, sizeidx, sizeidx0,
     +                          sizee, sizef, sizeg, nodedata)
c
cf2py intent(out) :: nodedata
cf2py integer :: idx, idx0
cf2py integer :: idxe, idx0e, idxf, idx0f, idxg, idx0g
cf2py integer :: neighbors
cf2py integer :: ne, nf, ng, nnode
cf2py integer, intent(hide), depend(idx) :: sizeidx=size(idx)
cf2py integer, intent(hide), depend(idx0) :: sizeidx0=size(idx0)
cf2py integer, intent(hide), depend(idxe) :: sizee=size(idxe)
cf2py integer, intent(hide), depend(idxf) :: sizef=size(idxf)
cf2py integer, intent(hide), depend(idxg) :: sizeg=size(idxg)
c
      integer sizeidx, sizeidx0, sizee, sizef, sizeg
      integer ne(3), nf(3), ng(3), nnode(3)
      integer idx(sizeidx), idx0(sizeidx0)
      integer idxe(sizee), idxf(sizef), idxg(sizeg)
      integer idx0e(sizeidx0), idx0f(sizeidx0), idx0g(sizeidx0)
      integer nodedata(3, sizeidx)
      integer neighbors(sizeidx0 - 1, 27)
c
      integer i, ii, x, y, z, nn, pos
      integer group, neighe, neighf, neighg
      integer nextneighe(2), nextneighf(2), nextneighg(2)
      integer singlegroupe(nnode(3), nnode(2)+1, nnode(1))
      integer singlegroupf(nnode(3), nnode(2), nnode(1)+1)
      integer singlegroupg(nnode(3)+1, nnode(2), nnode(1))
      data nextneighe/11,14/
      data nextneighf/5,14/
      data nextneighg/13,14/
c
      do group = 1, sizeidx0-1
          do x = 1, nnode(1)
              do y = 1, nnode(2)+1
                  do z = 1, nnode(3)
                      singlegroupe(z, y, x) = 0
                  enddo
              enddo
          enddo
          do x = 1, nnode(1)+1
              do y = 1, nnode(2)
                  do z = 1, nnode(3)
                      singlegroupf(z, y, x) = 0
                  enddo
              enddo
          enddo
          do x = 1, nnode(1)
              do y = 1, nnode(2)
                  do z = 1, nnode(3)+1
                      singlegroupg(z, y, x) = 0
                  enddo
              enddo
          enddo
          do nn = 0, 1
              neighe = neighbors(group, nextneighe(nn+1)) + 1
              neighf = neighbors(group, nextneighf(nn+1)) + 1
              neighg = neighbors(group, nextneighg(nn+1)) + 1
c                 An empty neighbour group would make the index below
c                 run one past the end of idxe: with idx0(g) == idx0(g+1)
c                 the gather starts at idx0(g)+1 == size+1 and reads
c                 out of bounds, which corrupts the heap rather than
c                 raising. Empty groups became possible once boxes were
c                 stored for holding ANY element (a box can carry
c                 surface panels without carrying nodes).
              if (neighe.ge.1 .and.
     +            idx0e(neighe+1).gt.idx0e(neighe)) then
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
                                  singlegroupe(z, nn*y+1, x) = ii
                              endif
                          enddo
                      enddo
                  enddo
              endif
c                 An empty neighbour group would make the index below
c                 run one past the end of idxf: with idx0(g) == idx0(g+1)
c                 the gather starts at idx0(g)+1 == size+1 and reads
c                 out of bounds, which corrupts the heap rather than
c                 raising. Empty groups became possible once boxes were
c                 stored for holding ANY element (a box can carry
c                 surface panels without carrying nodes).
              if (neighf.ge.1 .and.
     +            idx0f(neighf+1).gt.idx0f(neighf)) then
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
                                  singlegroupf(z, y, nn*x+1) = sizee+ii
                              endif
                          enddo
                      enddo
                  enddo
              endif
c                 An empty neighbour group would make the index below
c                 run one past the end of idxg: with idx0(g) == idx0(g+1)
c                 the gather starts at idx0(g)+1 == size+1 and reads
c                 out of bounds, which corrupts the heap rather than
c                 raising. Empty groups became possible once boxes were
c                 stored for holding ANY element (a box can carry
c                 surface panels without carrying nodes).
              if (neighg.ge.1 .and.
     +            idx0g(neighg+1).gt.idx0g(neighg)) then
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
                                  singlegroupg(nn*z+1, y, x) =
     +                                sizee+sizef+ii
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
              nodedata(2, ii) = singlegroupe(z, y+1, x)
              nodedata(3, ii) = singlegroupf(z, y, x+1)
              nodedata(1, ii) = singlegroupg(z+1, y, x)
          enddo
      enddo
      end
c
c
      subroutine getmesh(adjind, adjindptr, adjdat, esize, efsize,
     +                   efgsize, ntree, nsize, nnzsize, Zdat, Zind)
c
cf2py intent(out) :: Zdat, Zind
cf2py integer :: adjind, adjindptr, adjdat, esize, efsize, efgsize
cf2py integer :: ntree
cf2py integer, intent(hide), depend(adjind) :: nnzsize=size(adjind)
cf2py integer, intent(hide), depend(adjindptr) :: nsize=size(adjindptr)
c
c   ntree = number of connected components (counttrees). The cycle
c   rank is E - V + ntree, and the BFS emits roughly one quad per
c   non-tree edge PER COMPONENT -- the old sizing
c   4*(efgsize-nsize+3) assumed ONE component and overflowed the
c   heap by 4*(ntree-1) entries on multi-component models (found
c   2026-08-11 on the DBC flagship: 8 components -- floating signal
c   traces and the bottom metal -- corrupted the allocator and
c   crashed the next scipy call). maxmesh guards the store so any
c   future shortfall surfaces as a deficient basis (the wrapper's
c   validated-count fallback), never as corruption.
c
      integer nnzsize, nsize, esize, efsize, efgsize, ntree, maxmesh
      integer adjind(nnzsize), adjindptr(nsize), adjdat(nnzsize)
      integer Zind(4*(efgsize-nsize+2+ntree))
      integer*1 Zdat(4*(efgsize-nsize+2+ntree))
c
      integer Q(nsize), u, v, vp, neighbor_vp
      integer i, ii, iii, n1, n2, n3, n4, fil(4), j, x
      integer startq, stopq, parent(nsize-1)
      integer*1 signs(4), signf1, sign13, sign24, y
      integer*1 color(nsize-1), white, grey, black
      data white,grey,black/0,1,2/
c
      do i = 1, nsize-1
          Q(i+1) = -1
          color(i) = white
      enddo
      Q(1) = 1
      mesh = 0
      maxmesh = efgsize - nsize + 2 + ntree
   10 continue
      startq = 1
      stopq = 2
      do while (startq.ne.stopq)
          u = Q(startq)
          startq = startq + 1
          do i = adjindptr(u)+1, adjindptr(u+1)
              v = adjind(i) + 1
              if (color(v).eq.white) then
                  color(v) = grey
                  parent(v) = u
                  Q(stopq) = v
                  stopq = stopq + 1
              elseif (color(v).eq.grey) then
                  do ii = adjindptr(parent(v))+1, adjindptr(parent(v)+1)
                      if (parent(u).eq.adjind(ii)+1) then
                          n1 = u; n2 = v; n3 = parent(v); n4 = parent(u)
                          go to 20
                      endif
                  enddo
                  do ii = adjindptr(v)+1, adjindptr(v+1)
                      vp = adjind(ii) + 1
                      do iii = adjindptr(vp)+1, adjindptr(vp+1)
                          neighbor_vp = adjind(iii) + 1
                          if ((parent(u).eq.neighbor_vp) .and.
     +                            (vp.ne.u)) then
                              n1 = u; n2 = v; n3 = vp; n4 = parent(u)
                              go to 20
                          endif
                      enddo
                  enddo
                  go to 70
   20             continue
                  do ii = adjindptr(n1)+1, adjindptr(n1+1)
                      if (adjind(ii)+1.eq.n2) then
                          fil(1) = adjdat(ii) - 1
                          go to 30
                      endif
                  enddo
   30             do ii = adjindptr(n2)+1, adjindptr(n2+1)
                      if (adjind(ii)+1.eq.n3) then
                          fil(2) = adjdat(ii) - 1
                          go to 40
                      endif
                  enddo
   40             do ii = adjindptr(n3)+1, adjindptr(n3+1)
                      if (adjind(ii)+1.eq.n4) then
                          fil(3) = adjdat(ii) - 1
                          go to 50
                      endif
                  enddo
   50             do ii = adjindptr(n4)+1, adjindptr(n4+1)
                      if (adjind(ii)+1.eq.n1) then
                          fil(4) = adjdat(ii) - 1
                          go to 60
                      endif
                  enddo
   60             if ((fil(3) - fil(1)) .ge. 0) then
                      sign13 = 1
                  else
                      sign13 = -1
                  endif
                  if ((fil(4) - fil(2)) .ge. 0) then
                      sign24 = 1
                  else
                      sign24 = -1
                  endif
                  signf1 = 0
                  if ((fil(1) < esize) .and. (esize <= fil(2))
     +                    .and. (fil(2) < efsize)) then
                      signf1 = -1
                  elseif ((esize <= fil(1)) .and. (fil(1) < efsize)
     +                    .and. (efsize <= fil(2))
     +                    .and. (fil(2) < efgsize)) then
                      signf1 = -1
                  elseif ((efsize <= fil(1))
     +                    .and. (fil(1) < efgsize)
     +                    .and. (fil(2) < esize)) then
                      signf1 = -1
                  endif
                  if (signf1 .ne. -1) signf1 = 1
                  signs(1) = sign13*signf1
                  signs(2) = -sign24*signf1
                  signs(3) = -sign13*signf1
                  signs(4) = sign24*signf1
c   fil and signs are reordered using an insertion sort of fil
c   This is necessary because scipy's csr_matrix requires ordered
c   indices.
                  do ii = 2, 4
                      x = fil(ii)
                      y = signs(ii)
                      j = ii - 1
c   Fortran does not short-circuit .and., so guard fil(j) with a
c   separate test: (j.ge.1).and.(fil(j).gt.x) would read fil(0) at j=0.
                      do while (j.ge.1)
                          if (fil(j).le.x) exit
                          fil(j+1) = fil(j)
                          signs(j+1) = signs(j)
                          j = j - 1
                      enddo
                      fil(j+1) = x
                      signs(j+1) = y
                  enddo
                  if (mesh .lt. maxmesh) then
                      do ii = 1, 4
                          iii = 4*mesh + ii
                          Zind(iii) = fil(ii)
                          Zdat(iii) = signs(ii)
                      enddo
                      mesh = mesh + 1
                  endif
   70             continue
              endif
          enddo
          color(u) = black
      enddo
c   Nodes are indexed 1..nsize-1 (nsize = size of the CSR pointer array);
c   scan only those, not color(nsize), which is one past the array. Matches
c   the equivalent loop in counttrees.
      do i = 1, nsize-1
          if (color(i).eq.white) then
              do ii = 2, nsize
                  Q(ii) = -1
              enddo
              Q(1) = i
              go to 10
          endif
      enddo
      end
c
c
      subroutine counttrees(adjind, adjindptr, nsize, nnzsize, numtrees)
c
cf2py intent(out) :: numtrees
cf2py integer :: adjind, adjindptr
cf2py integer, intent(hide), depend(adjind) :: nnzsize=size(adjind)
cf2py integer, intent(hide), depend(adjindptr) :: nsize=size(adjindptr)
c
      integer nnzsize, nsize, numtrees
      integer adjind(nnzsize), adjindptr(nsize)
c
      integer Q(nsize), u, v, i, ii, startq, stopq, lasti
      integer*1 color(nsize-1), white, black
      data white,black/0,1/
c
      numtrees = 1
      do i = 1, nsize-1
          Q(i+1) = -1
          color(i) = white
      enddo
      lasti = 1
      Q(1) = 1
   10 continue
      startq = 1
      stopq = 2
      do while (startq.ne.stopq)
          u = Q(startq)
          startq = startq + 1
          do i = adjindptr(u)+1, adjindptr(u+1)
              v = adjind(i) + 1
              if (color(v).eq.white) then
                  color(v) = black
                  Q(stopq) = v
                  stopq = stopq + 1
              endif
          enddo
          color(u) = black
      enddo
      do i = lasti+1, nsize-1
          if (color(i).eq.white) then
              do ii = 2, nsize
                  Q(ii) = -1
              enddo
              lasti = i
              Q(1) = i
              numtrees = numtrees + 1
              go to 10
          endif
      enddo
      end
      subroutine getmeshfull(adjind, adjindptr, adjdat, esize, efsize,
     +                   efgsize, maxq, nsize, nnzsize, nq, Zdat, Zind)
c
c   ALL plaquettes (every 4-cycle), not a spanning set.
c
c   getmesh above is a BFS FUNDAMENTAL-cycle finder: it emits one quad
c   per non-tree edge, hence exactly the cycle rank, hence an INDEPENDENT
c   basis. That selection is what ill-conditions Y^T Y (measured: cond
c   2.9e4, a 361-mode near-null cluster, chol fill 27.7x). Keeping ALL
c   plaquettes gives a singular-but-consistent Gram whose kernel is the
c   cube boundaries -- physically meaningless, since currents differing
c   by a cube boundary are the same current -- and AMG then solves it in
c   ~1.2x memory instead of 27.7x fill.
c
c   Enumeration: u is the MINIMUM node of the quad; v,w are u's two
c   neighbours in it (unordered pair, taken once via ii>i); x is the
c   node opposite u. Each plaquette is therefore emitted exactly once.
c   Signs and the index insertion sort are identical to getmesh.
c
c   nq returns the number of quads written, or -1 on overflow of maxq.
c
cf2py intent(out) :: nq, Zdat, Zind
cf2py integer :: adjind, adjindptr, adjdat, esize, efsize, efgsize, maxq
cf2py integer, intent(hide), depend(adjind) :: nnzsize=size(adjind)
cf2py integer, intent(hide), depend(adjindptr) :: nsize=size(adjindptr)
c
      integer nnzsize, nsize, esize, efsize, efgsize, maxq, nq
      integer adjind(nnzsize), adjindptr(nsize), adjdat(nnzsize)
      integer Zind(4*maxq)
      integer*1 Zdat(4*maxq)
c
      integer u, v, w, x, i, ii, iii, iv, j, fil(4), xx
      integer*1 signs(4), signf1, sign13, sign24, y
c
      nq = 0
      do u = 1, nsize-1
        do i = adjindptr(u)+1, adjindptr(u+1)
          v = adjind(i) + 1
          if (v .le. u) go to 110
          do ii = i+1, adjindptr(u+1)
            w = adjind(ii) + 1
            if (w .le. u) go to 120
            do iii = adjindptr(v)+1, adjindptr(v+1)
              x = adjind(iii) + 1
              if (x .le. u) go to 130
              if (x .eq. w) go to 130
              do iv = adjindptr(x)+1, adjindptr(x+1)
                if (adjind(iv)+1 .ne. w) go to 140
                fil(1) = adjdat(i) - 1
                fil(2) = adjdat(iii) - 1
                fil(3) = adjdat(iv) - 1
                fil(4) = adjdat(ii) - 1
                if ((fil(3) - fil(1)) .ge. 0) then
                    sign13 = 1
                else
                    sign13 = -1
                endif
                if ((fil(4) - fil(2)) .ge. 0) then
                    sign24 = 1
                else
                    sign24 = -1
                endif
                signf1 = 0
                if ((fil(1) < esize) .and. (esize <= fil(2))
     +                  .and. (fil(2) < efsize)) then
                    signf1 = -1
                elseif ((esize <= fil(1)) .and. (fil(1) < efsize)
     +                  .and. (efsize <= fil(2))
     +                  .and. (fil(2) < efgsize)) then
                    signf1 = -1
                elseif ((efsize <= fil(1))
     +                  .and. (fil(1) < efgsize)
     +                  .and. (fil(2) < esize)) then
                    signf1 = -1
                endif
                if (signf1 .ne. -1) signf1 = 1
                signs(1) = sign13*signf1
                signs(2) = -sign24*signf1
                signs(3) = -sign13*signf1
                signs(4) = sign24*signf1
                do j = 2, 4
                    xx = fil(j)
                    y = signs(j)
                    jj = j - 1
                    do while (jj.ge.1)
                        if (fil(jj).le.xx) exit
                        fil(jj+1) = fil(jj)
                        signs(jj+1) = signs(jj)
                        jj = jj - 1
                    enddo
                    fil(jj+1) = xx
                    signs(jj+1) = y
                enddo
                if (nq .ge. maxq) then
                    nq = -1
                    return
                endif
                do j = 1, 4
                    Zind(4*nq + j) = fil(j)
                    Zdat(4*nq + j) = signs(j)
                enddo
                nq = nq + 1
  140           continue
              enddo
  130         continue
            enddo
  120       continue
          enddo
  110     continue
        enddo
      enddo
      end
c
