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
