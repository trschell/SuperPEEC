! Streamed-basis Krylov kernels (2026-09-15): the projections and the
! block update of krylov_stream.gmres_stream, OpenMP-threaded, reading
! the complex64 basis block once per pass and accumulating in double.
!
! numpy did the same work at 0.09-0.12 s per R4-sized vector
! (12 M loops), moving every vector through memory three times in
! complex128 on one core; these read the block once, in the stored
! precision, on all threads, with a row-chunked loop so the work vector
! stays in cache while the block streams past.
!
! The block is v(n, nb) in Fortran order: a C-contiguous numpy array of
! shape (nb, n) passed as its transpose, no copy.

subroutine block_dots(n, nb, v, w, h)
  ! h(i) = conjg(v(:, i)) . w
  implicit none
  integer, intent(in) :: n, nb
  complex(4), intent(in) :: v(n, nb)
  complex(8), intent(in) :: w(n)
  complex(8), intent(out) :: h(nb)
  integer, parameter :: chunk = 8192
  integer :: c, i, j, jend
  complex(8) :: s
  complex(8) :: hp(nb)
  h = (0d0, 0d0)
  !$omp parallel private(c, i, j, jend, s, hp)
  hp = (0d0, 0d0)
  !$omp do schedule(static)
  do c = 1, n, chunk
     jend = min(c + chunk - 1, n)
     do i = 1, nb
        s = (0d0, 0d0)
        do j = c, jend
           s = s + conjg(cmplx(v(j, i), kind=8))*w(j)
        end do
        hp(i) = hp(i) + s
     end do
  end do
  !$omp end do
  !$omp critical
  h = h + hp
  !$omp end critical
  !$omp end parallel
end subroutine block_dots

subroutine block_update(n, nb, v, h, w)
  ! w = w - v h   (pass -y for an accumulation u = u + V y)
  implicit none
  integer, intent(in) :: n, nb
  complex(4), intent(in) :: v(n, nb)
  complex(8), intent(in) :: h(nb)
  complex(8), intent(inout) :: w(n)
  integer, parameter :: chunk = 8192
  integer :: c, i, j, jend
  complex(8) :: s
  !$omp parallel do private(c, i, j, jend, s) schedule(static)
  do c = 1, n, chunk
     jend = min(c + chunk - 1, n)
     do j = c, jend
        s = w(j)
        do i = 1, nb
           s = s - h(i)*cmplx(v(j, i), kind=8)
        end do
        w(j) = s
     end do
  end do
  !$omp end parallel do
end subroutine block_update


! Spanning forest of the lattice node graph (2026-09-27): the DFS of
! wireassembly._forest_walk, step for step -- roots in the given
! order, a root skipped when already reached or isolated, each node's
! edges in adjacency order, one LIFO stack -- so the forest, and every
! gauge-dependent byte downstream, is the one the Python loop made.
! The Python loop ran over .tolist() copies of the adjacency (boxed
! ints: ~9 GiB at R5, the build's high-water instant once the solve's
! own copies were gone); this walks the arrays as they are. All
! indices are 0-based on both sides.

subroutine forest_walk(nn, m, nroots, ptr, nbr, eid, sgn, roots, &
                       parent, pedge, psign, comp, ncomp)
  implicit none
  integer, intent(in) :: nn, m, nroots
  integer(8), intent(in) :: ptr(nn + 1), nbr(m), eid(m), roots(nroots)
  integer(1), intent(in) :: sgn(m)
  integer(8), intent(out) :: parent(nn), pedge(nn), comp(nn)
  real(8), intent(out) :: psign(nn)
  integer, intent(out) :: ncomp
  integer(8), allocatable :: stack(:)
  integer(8) :: top, u, v, i, r
  integer :: k
  parent = -1
  pedge = -1
  comp = -1
  psign = 0d0
  allocate(stack(nn))
  ncomp = 0
  do k = 1, nroots
    r = roots(k)
    if (comp(r + 1) >= 0 .or. ptr(r + 1) == ptr(r + 2)) cycle
    comp(r + 1) = ncomp
    top = 1
    stack(1) = r
    do while (top > 0)
      u = stack(top)
      top = top - 1
      do i = ptr(u + 1), ptr(u + 2) - 1
        v = nbr(i + 1)
        if (comp(v + 1) < 0) then
          comp(v + 1) = ncomp
          parent(v + 1) = u
          pedge(v + 1) = eid(i + 1)
          psign(v + 1) = dble(sgn(i + 1))
          top = top + 1
          stack(top) = v
        end if
      end do
    end do
    ncomp = ncomp + 1
  end do
  deallocate(stack)
end subroutine forest_walk


! ---- the plaquette basis from the lattice (2026-09-28) ---------------
! The plaquette block of the loop basis needs no stored indices. A
! plaquette is (normal, base cell); its four filaments are the face's
! edges in ONE fixed entry pattern per normal (checked on the built
! basis: scratch/basis_structure.py); and both numberings are lattice
! formulas -- plaquettes in tile order (tile, normal, z, y, x) over
! the stencil's occupancy mask, filaments per orientation block by
! leaf group (x, y, z order) then local lattice index (x, y, z) with a
! per-group occupancy mask. Both products are gathers over their
! output index, threaded, adding in the order the stored CSC/CSR form
! adds (columns ascending; entries in ascending filament index), so
! the bits are the same.
!
! filament of (axis a, cell c), 1-based:
!   g = ggrid(c/n + 1, a + 1); l = ((cx mod n0)*n1 + cy mod n1)*n2 + cz mod n2
!   f = gbase(g + 1) + popcount(gmask bits below l) + 1
! plaquette of (normal on, base b), 1-based:
!   t = tgrid(b/TL + 1); l = ((on*TL + bz mod TL)*TL + by mod TL)*TL + bx mod TL
!   j = tpre(t + 1) + popcount(tmask bits below l) + 1

subroutine lattice_bt(nt, nwt, tmask, tpre, tcoord, tl, n0, n1, n2, &
                      gx, gy, gz, ggrid, ng, nwg, gbase, gmask, pat, &
                      nf, xr, xi, npl, yr, yi)
  ! y(j) = sum_k s_k x(f_k(j)) for every plaquette j (two vectors)
  implicit none
  integer, intent(in) :: nt, nwt, tl, n0, n1, n2, gx, gy, gz, ng, nwg, nf, npl
  integer(8), intent(in) :: tmask(nwt, nt), gmask(nwg, ng)
  integer(8), intent(in) :: tpre(nt), gbase(ng)
  integer, intent(in) :: tcoord(3, nt), ggrid(gx, gy, gz, 3), pat(5, 4, 3)
  real(8), intent(in) :: xr(nf), xi(nf)
  real(8), intent(out) :: yr(npl), yi(npl)
  integer :: t, l, w, b, on, rem, lx, ly, lz, k, a, cx, cy, cz
  integer :: gi, gj, gk, q, l2, w2, b2, ww, cube
  integer(8) :: j, f
  real(8) :: sr, si, sg
  cube = tl*tl*tl
  !$omp parallel do schedule(dynamic, 64) default(shared) &
  !$omp private(t, l, w, b, on, rem, lx, ly, lz, k, a, cx, cy, cz, &
  !$omp         gi, gj, gk, q, l2, w2, b2, ww, j, f, sr, si, sg)
  do t = 1, nt
    j = tpre(t)
    do l = 0, 3*cube - 1
      w = l/64 + 1
      b = l - 64*(w - 1)
      if (.not. btest(tmask(w, t), b)) cycle
      j = j + 1
      on = l/cube
      rem = l - on*cube
      lz = rem/(tl*tl)
      ly = (rem - lz*tl*tl)/tl
      lx = rem - lz*tl*tl - ly*tl
      sr = 0d0
      si = 0d0
      do k = 1, 4
        a  = pat(1, k, on + 1)
        cx = tcoord(1, t)*tl + lx + pat(2, k, on + 1)
        cy = tcoord(2, t)*tl + ly + pat(3, k, on + 1)
        cz = tcoord(3, t)*tl + lz + pat(4, k, on + 1)
        sg = dble(pat(5, k, on + 1))
        gi = cx/n0
        gj = cy/n1
        gk = cz/n2
        q = ggrid(gi + 1, gj + 1, gk + 1, a + 1)
        l2 = ((cx - gi*n0)*n1 + (cy - gj*n1))*n2 + (cz - gk*n2)
        w2 = l2/64 + 1
        b2 = l2 - 64*(w2 - 1)
        f = gbase(q + 1) + 1
        do ww = 1, w2 - 1
          f = f + popcnt(gmask(ww, q + 1))
        end do
        f = f + popcnt(iand(gmask(w2, q + 1), not(ishft(-1_8, b2))))
        sr = sr + sg*xr(f)
        si = si + sg*xi(f)
      end do
      yr(j) = sr
      yi(j) = si
    end do
  end do
end subroutine lattice_bt


subroutine lattice_b(nt, nwt, tmask, tpre, tx, ty, tz, tgrid, tl, sgn, &
                     nf, fa, fc, npl, xr, xi, yr, yi)
  ! y(f) = sum over the plaquettes containing filament f, in ascending
  ! plaquette index, of s x(j) (two vectors); zero where none does
  implicit none
  integer, intent(in) :: nt, nwt, tx, ty, tz, tl, nf, npl
  integer(8), intent(in) :: tmask(nwt, nt), tpre(nt)
  integer, intent(in) :: tgrid(tx, ty, tz), sgn(2, 3, 3)
  integer(1), intent(in) :: fa(nf)
  integer(2), intent(in) :: fc(3, nf)
  real(8), intent(in) :: xr(npl), xi(npl)
  real(8), intent(out) :: yr(nf), yi(nf)
  integer :: f, a, on, tt, d, gi, gj, gk, t, l, w, b, ww, m, k, kk
  integer :: c(3), bb(3)
  integer(8) :: j, jj(4), jt
  real(8) :: ss(4), st, sr, si
  !$omp parallel do schedule(static) default(shared) &
  !$omp private(f, a, on, tt, d, gi, gj, gk, t, l, w, b, ww, m, k, kk, &
  !$omp         c, bb, j, jj, jt, ss, st, sr, si)
  do f = 1, nf
    a = fa(f)
    c(1) = fc(1, f)
    c(2) = fc(2, f)
    c(3) = fc(3, f)
    m = 0
    do on = 0, 2
      if (on == a) cycle
      tt = 3 - a - on
      do d = 0, 1
        bb = c
        bb(tt + 1) = bb(tt + 1) - d
        if (bb(1) < 0 .or. bb(2) < 0 .or. bb(3) < 0) cycle
        gi = bb(1)/tl
        gj = bb(2)/tl
        gk = bb(3)/tl
        if (gi >= tx .or. gj >= ty .or. gk >= tz) cycle
        t = tgrid(gi + 1, gj + 1, gk + 1)
        if (t < 0) cycle
        l = ((on*tl + (bb(3) - gk*tl))*tl + (bb(2) - gj*tl))*tl + (bb(1) - gi*tl)
        w = l/64 + 1
        b = l - 64*(w - 1)
        if (.not. btest(tmask(w, t + 1), b)) cycle
        j = tpre(t + 1) + 1
        do ww = 1, w - 1
          j = j + popcnt(tmask(ww, t + 1))
        end do
        j = j + popcnt(iand(tmask(w, t + 1), not(ishft(-1_8, b))))
        m = m + 1
        jj(m) = j
        ss(m) = dble(sgn(d + 1, a + 1, on + 1))
      end do
    end do
    ! ascending plaquette index: the CSC scatter's column order
    do k = 2, m
      jt = jj(k)
      st = ss(k)
      kk = k - 1
      do while (kk >= 1)
        if (jj(kk) <= jt) exit
        jj(kk + 1) = jj(kk)
        ss(kk + 1) = ss(kk)
        kk = kk - 1
      end do
      jj(kk + 1) = jt
      ss(kk + 1) = st
    end do
    sr = 0d0
    si = 0d0
    do k = 1, m
      sr = sr + ss(k)*xr(jj(k))
      si = si + ss(k)*xi(jj(k))
    end do
    yr(f) = sr
    yi(f) = si
  end do
end subroutine lattice_b
