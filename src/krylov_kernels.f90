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
