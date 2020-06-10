subroutine foo(a)
    implicit none
    integer,intent(in) :: a
    write(*,*) "Hello from Fortran!"
    write(*,*) "a=",a
    return
end subroutine foo