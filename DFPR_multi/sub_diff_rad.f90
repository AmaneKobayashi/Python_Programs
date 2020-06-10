subroutine diff_rad(diff,nd,beam_stop_value,rad)
    implicit none

    integer,intent(in) :: nd
    real*4,dimension(nd,nd),intent(in) :: diff
    real*4,intent(out) :: rad(nd,nd)
    real*4,intent(in) :: beam_stop_value
    
    integer :: i,ii,x,y,n
    integer,allocatable :: temp_rad_x(:),temp_rad_y(:)
    real*4,allocatable :: distance(:)
    real*4 :: pi
    integer :: radius
    integer :: min_index(1)

    pi=acos(-1.0e0)
    allocate(temp_rad_x(int(2*pi*nd)),&
    &temp_rad_y(int(2*pi*nd)),&
    &distance(int(2*pi*nd)))   

    write(*,*)"nd = ",nd
    

    do ii=1,nd
        do i=1,nd
            if(diff(i,ii)==0.0e0)then
                radius=int(sqrt(real(nd/2-i)**2+real(nd/2-ii)**2))
                n=1
                distance(:)=1.0e5
                do y=1,nd
                    do x=1,nd
                        if(int(sqrt(real(nd/2-x)**2+real(nd/2-y)**2)) == radius)then
                            if(diff(x,y)/=0.0e0)then
                                temp_rad_x(n)=x
                                temp_rad_y(n)=y
                                distance(n)=sqrt(real(i-x)**2+real(ii-y)**2)
                                n=n+1
                            endif
                        endif
                    enddo
                enddo
                if(n==1)then
                    rad(i,ii)=beam_stop_value
                else
                    min_index=minloc(distance)
                    rad(i,ii)=diff(temp_rad_x(min_index(1)),temp_rad_y(min_index(1)))
                endif
            else
                rad(i,ii)=diff(i,ii)                
            endif
        enddo
    enddo
    return
end subroutine diff_rad
