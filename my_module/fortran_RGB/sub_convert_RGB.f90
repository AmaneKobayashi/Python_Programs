subroutine convert_RGB_chainer(np_diff,ixpix,iypix,min_range,max_range,np_diff_log_rgb)
    implicit none

    integer,intent(in) :: ixpix,iypix
    real*4,intent(in) :: np_diff(ixpix,iypix)
    real*4,intent(in) :: min_range,max_range
    real*4,intent(out) :: np_diff_log_rgb(ixpix,iypix,3)

    integer :: i,ii
    real*4 :: cdict_Red_X(8),cdict_Red_Y(8)
    real*4 :: cdict_Green_X(8),cdict_Green_Y(8)
    real*4 :: cdict_Blue_X(8),cdict_Blue_Y(8)
    integer,parameter :: precision=10000
    real*4 :: Red(precision)
    real*4 :: Green(precision)
    real*4 :: Blue(precision)
    real*4 :: a_Red,b_Red,a_Green,b_Green,a_Blue,b_Blue
    integer :: Red_index,Green_index,Blue_index
    real*4 :: step

    real*4 :: np_diff_log(ixpix,iypix)
    integer :: np_diff_log_X(ixpix,iypix)

    !write(*,*) "ixpix = ",ixpix
    !write(*,*) "iypix = ",iypix
    !write(*,*) "min_range = ",min_range
    !write(*,*) "max_range = ",max_range

    cdict_Red_X(1)=0.0 
    cdict_Red_X(2)=0.375
    cdict_Red_X(3)=0.492
    cdict_Red_X(4)=0.75 
    cdict_Red_X(5)=0.80 
    cdict_Red_X(6)=0.875
    cdict_Red_X(7)=0.895
    cdict_Red_X(8)=1.0

    cdict_Red_Y(1)=0.0  
    cdict_Red_Y(2)=0.0  
    cdict_Red_Y(3)=1.0  
    cdict_Red_Y(4)=1.0  
    cdict_Red_Y(5)=0.867
    cdict_Red_Y(6)=0.867
    cdict_Red_Y(7)=1.0  
    cdict_Red_Y(8)=1.0 

    cdict_Green_X(1)=0.0  
    cdict_Green_X(2)=0.129 
    cdict_Green_X(3)=0.3125
    cdict_Green_X(4)=0.4375
    cdict_Green_X(5)=0.75  
    cdict_Green_X(6)=0.8125
    cdict_Green_X(7)=0.9375
    cdict_Green_X(8)=1.0   

    cdict_Green_Y(1)=0.0  
    cdict_Green_Y(2)=0.0  
    cdict_Green_Y(3)=1.0  
    cdict_Green_Y(4)=1.0  
    cdict_Green_Y(5)=0.0  
    cdict_Green_Y(6)=0.734
    cdict_Green_Y(7)=1.0  
    cdict_Green_Y(8)=1.0  

    cdict_Blue_X(1)=0.0   
    cdict_Blue_X(2)=0.1875
    cdict_Blue_X(3)=0.375 
    cdict_Blue_X(4)=0.4375
    cdict_Blue_X(5)=0.754 
    cdict_Blue_X(6)=0.8125
    cdict_Blue_X(7)=0.9375
    cdict_Blue_X(8)=1.0   

    cdict_Blue_Y(1)=0.03 
    cdict_Blue_Y(2)=1.0  
    cdict_Blue_Y(3)=1.0  
    cdict_Blue_Y(4)=0.0  
    cdict_Blue_Y(5)=0.0  
    cdict_Blue_Y(6)=0.715
    cdict_Blue_Y(7)=1.0  
    cdict_Blue_Y(8)=1.0  

    cdict_Red_X(:)=cdict_Red_X(:)*real(precision)
    cdict_Green_X(:)=cdict_Green_X(:)*real(precision)
    cdict_Blue_X(:)=cdict_Blue_X(:)*real(precision)

    Red_index=1
    Green_index=1
    Blue_index=1

    a_Red=(cdict_Red_Y(Red_index+1) - cdict_Red_Y(Red_index)) / (cdict_Red_X(Red_index+1) - cdict_Red_X(Red_index))
    b_Red=(cdict_Red_X(Red_index+1) * cdict_Red_Y(Red_index) - cdict_Red_X(Red_index) * cdict_Red_Y(Red_index+1)) /&
        & (cdict_Red_X(Red_index+1) - cdict_Red_X(Red_index))  
    a_Green=(cdict_Green_Y(Green_index+1) - cdict_Green_Y(Green_index)) /&
        & (cdict_Green_X(Green_index+1) - cdict_Green_X(Green_index))
    b_Green=(cdict_Green_X(Green_index+1) * cdict_Green_Y(Green_index) - cdict_Green_X(Green_index) &
        &* cdict_Green_Y(Green_index+1)) / (cdict_Green_X(Green_index+1) - cdict_Green_X(Green_index))
    a_Blue=(cdict_Blue_Y(Blue_index+1) - cdict_Blue_Y(Blue_index)) /&
        &(cdict_Blue_X(Blue_index+1) - cdict_Blue_X(Blue_index))
    b_Blue=(cdict_Blue_X(Blue_index+1) * cdict_Blue_Y(Blue_index) - cdict_Blue_X(Blue_index) &
        &* cdict_Blue_Y(Blue_index+1)) / (cdict_Blue_X(Blue_index+1) - cdict_Blue_X(Blue_index))      

    do i=1,precision
        if(i > int(cdict_Red_X(Red_index+1)))then
            Red_index=Red_index+1
            a_Red=(cdict_Red_Y(Red_index+1) - cdict_Red_Y(Red_index)) / (cdict_Red_X(Red_index+1) - cdict_Red_X(Red_index))
            b_Red=(cdict_Red_X(Red_index+1) * cdict_Red_Y(Red_index) - cdict_Red_X(Red_index) * cdict_Red_Y(Red_index+1)) /&
                & (cdict_Red_X(Red_index+1) - cdict_Red_X(Red_index))        
        endif

        if(i > int(cdict_Green_X(Green_index+1)))then
            Green_index=Green_index+1
            a_Green=(cdict_Green_Y(Green_index+1) - cdict_Green_Y(Green_index)) /&
                & (cdict_Green_X(Green_index+1) - cdict_Green_X(Green_index))
            b_Green=(cdict_Green_X(Green_index+1) * cdict_Green_Y(Green_index) - cdict_Green_X(Green_index) &
                &* cdict_Green_Y(Green_index+1)) / (cdict_Green_X(Green_index+1) - cdict_Green_X(Green_index))                
        endif

        if(i > int(cdict_Blue_X(Blue_index+1)))then
            Blue_index=Blue_index+1
            a_Blue=(cdict_Blue_Y(Blue_index+1) - cdict_Blue_Y(Blue_index)) /&
                & (cdict_Blue_X(Blue_index+1) - cdict_Blue_X(Blue_index))
            b_Blue=(cdict_Blue_X(Blue_index+1) * cdict_Blue_Y(Blue_index) - cdict_Blue_X(Blue_index) &
                &* cdict_Blue_Y(Blue_index+1)) / (cdict_Blue_X(Blue_index+1) - cdict_Blue_X(Blue_index))
        endif

        Red(i)=a_Red * real(i) + b_Red
        Green(i)=a_Green * real(i) + b_Green
        Blue(i)=a_Blue * real(i) + b_Blue
    enddo

    np_diff_log(:,:)=0.0
    do ii=1,iypix
        do i=1,ixpix
            if(np_diff(i,ii)>0.0)then
                np_diff_log(i,ii)=log10(np_diff(i,ii))
                !write(*,*)i,ii,np_diff(i,ii),np_diff_log(i,ii)
            endif
        enddo
    enddo

    step=(max_range - min_range) / real(precision)
    !write(*,*)"step = ",step
    do ii=1,iypix
        do i=1,ixpix
            np_diff_log_X(i,ii)=int( (np_diff_log(i,ii)-min_range) / step ) +1 
            if(np_diff_log(i,ii) <= min_range)then
                np_diff_log_X(i,ii)=1
            elseif(np_diff_log(i,ii) >= max_range)then
                np_diff_log_X(i,ii)=precision
            endif
        enddo
    enddo

    do ii=1,iypix
        do i=1,ixpix
            np_diff_log_rgb(i,ii,1)=Red(np_diff_log_X(i,ii))
            np_diff_log_rgb(i,ii,2)=Green(np_diff_log_X(i,ii))
            np_diff_log_rgb(i,ii,3)=Blue(np_diff_log_X(i,ii))
            !write(*,*) i,ii,np_diff_log_X(i,ii)
        enddo
    enddo

    return
end subroutine convert_RGB_chainer

subroutine convert_RGB(np_diff,ixpix,iypix,min_range,max_range,np_diff_log_rgb)
    implicit none

    integer,intent(in) :: ixpix,iypix
    real*4,intent(in) :: np_diff(ixpix,iypix)
    real*4,intent(in) :: min_range,max_range
    real*4,intent(out) :: np_diff_log_rgb(ixpix,iypix,3)

    integer :: i,ii
    real*4 :: cdict_Red_X(8),cdict_Red_Y(8)
    real*4 :: cdict_Green_X(8),cdict_Green_Y(8)
    real*4 :: cdict_Blue_X(8),cdict_Blue_Y(8)
    integer,parameter :: precision=10000
    real*4 :: Red(precision)
    real*4 :: Green(precision)
    real*4 :: Blue(precision)
    real*4 :: a_Red,b_Red,a_Green,b_Green,a_Blue,b_Blue
    integer :: Red_index,Green_index,Blue_index
    real*4 :: step

    real*4 :: np_diff_log(ixpix,iypix)
    integer :: np_diff_log_X(ixpix,iypix)

    !write(*,*) "ixpix = ",ixpix
    !write(*,*) "iypix = ",iypix
    !write(*,*) "min_range = ",min_range
    !write(*,*) "max_range = ",max_range

    cdict_Red_X(1)=0.0 
    cdict_Red_X(2)=0.375
    cdict_Red_X(3)=0.492
    cdict_Red_X(4)=0.75 
    cdict_Red_X(5)=0.80 
    cdict_Red_X(6)=0.875
    cdict_Red_X(7)=0.895
    cdict_Red_X(8)=1.0

    cdict_Red_Y(1)=0.0  
    cdict_Red_Y(2)=0.0  
    cdict_Red_Y(3)=1.0  
    cdict_Red_Y(4)=1.0  
    cdict_Red_Y(5)=0.867
    cdict_Red_Y(6)=0.867
    cdict_Red_Y(7)=1.0  
    cdict_Red_Y(8)=1.0 

    cdict_Green_X(1)=0.0  
    cdict_Green_X(2)=0.129 
    cdict_Green_X(3)=0.3125
    cdict_Green_X(4)=0.4375
    cdict_Green_X(5)=0.75  
    cdict_Green_X(6)=0.8125
    cdict_Green_X(7)=0.9375
    cdict_Green_X(8)=1.0   

    cdict_Green_Y(1)=0.0  
    cdict_Green_Y(2)=0.0  
    cdict_Green_Y(3)=1.0  
    cdict_Green_Y(4)=1.0  
    cdict_Green_Y(5)=0.0  
    cdict_Green_Y(6)=0.734
    cdict_Green_Y(7)=1.0  
    cdict_Green_Y(8)=1.0  

    cdict_Blue_X(1)=0.0   
    cdict_Blue_X(2)=0.1875
    cdict_Blue_X(3)=0.375 
    cdict_Blue_X(4)=0.4375
    cdict_Blue_X(5)=0.754 
    cdict_Blue_X(6)=0.8125
    cdict_Blue_X(7)=0.9375
    cdict_Blue_X(8)=1.0   

    cdict_Blue_Y(1)=0.03 
    cdict_Blue_Y(2)=1.0  
    cdict_Blue_Y(3)=1.0  
    cdict_Blue_Y(4)=0.0  
    cdict_Blue_Y(5)=0.0  
    cdict_Blue_Y(6)=0.715
    cdict_Blue_Y(7)=1.0  
    cdict_Blue_Y(8)=1.0  

    cdict_Red_X(:)=cdict_Red_X(:)*real(precision)
    cdict_Green_X(:)=cdict_Green_X(:)*real(precision)
    cdict_Blue_X(:)=cdict_Blue_X(:)*real(precision)

    cdict_Red_Y(:)=cdict_Red_Y(:)*real(255)
    cdict_Green_Y(:)=cdict_Green_Y(:)*real(255)
    cdict_Blue_Y(:)=cdict_Blue_Y(:)*real(255)

    Red_index=1
    Green_index=1
    Blue_index=1

    a_Red=(cdict_Red_Y(Red_index+1) - cdict_Red_Y(Red_index)) / (cdict_Red_X(Red_index+1) - cdict_Red_X(Red_index))
    b_Red=(cdict_Red_X(Red_index+1) * cdict_Red_Y(Red_index) - cdict_Red_X(Red_index) * cdict_Red_Y(Red_index+1)) /&
        & (cdict_Red_X(Red_index+1) - cdict_Red_X(Red_index))  
    a_Green=(cdict_Green_Y(Green_index+1) - cdict_Green_Y(Green_index)) /&
        & (cdict_Green_X(Green_index+1) - cdict_Green_X(Green_index))
    b_Green=(cdict_Green_X(Green_index+1) * cdict_Green_Y(Green_index) - cdict_Green_X(Green_index) &
        &* cdict_Green_Y(Green_index+1)) / (cdict_Green_X(Green_index+1) - cdict_Green_X(Green_index))
    a_Blue=(cdict_Blue_Y(Blue_index+1) - cdict_Blue_Y(Blue_index)) /&
        &(cdict_Blue_X(Blue_index+1) - cdict_Blue_X(Blue_index))
    b_Blue=(cdict_Blue_X(Blue_index+1) * cdict_Blue_Y(Blue_index) - cdict_Blue_X(Blue_index) &
        &* cdict_Blue_Y(Blue_index+1)) / (cdict_Blue_X(Blue_index+1) - cdict_Blue_X(Blue_index))      

    do i=1,precision
        if(i > int(cdict_Red_X(Red_index+1)))then
            Red_index=Red_index+1
            a_Red=(cdict_Red_Y(Red_index+1) - cdict_Red_Y(Red_index)) / (cdict_Red_X(Red_index+1) - cdict_Red_X(Red_index))
            b_Red=(cdict_Red_X(Red_index+1) * cdict_Red_Y(Red_index) - cdict_Red_X(Red_index) * cdict_Red_Y(Red_index+1)) /&
                & (cdict_Red_X(Red_index+1) - cdict_Red_X(Red_index))        
        endif

        if(i > int(cdict_Green_X(Green_index+1)))then
            Green_index=Green_index+1
            a_Green=(cdict_Green_Y(Green_index+1) - cdict_Green_Y(Green_index)) /&
                & (cdict_Green_X(Green_index+1) - cdict_Green_X(Green_index))
            b_Green=(cdict_Green_X(Green_index+1) * cdict_Green_Y(Green_index) - cdict_Green_X(Green_index) &
                &* cdict_Green_Y(Green_index+1)) / (cdict_Green_X(Green_index+1) - cdict_Green_X(Green_index))                
        endif

        if(i > int(cdict_Blue_X(Blue_index+1)))then
            Blue_index=Blue_index+1
            a_Blue=(cdict_Blue_Y(Blue_index+1) - cdict_Blue_Y(Blue_index)) /&
                & (cdict_Blue_X(Blue_index+1) - cdict_Blue_X(Blue_index))
            b_Blue=(cdict_Blue_X(Blue_index+1) * cdict_Blue_Y(Blue_index) - cdict_Blue_X(Blue_index) &
                &* cdict_Blue_Y(Blue_index+1)) / (cdict_Blue_X(Blue_index+1) - cdict_Blue_X(Blue_index))
        endif

        Red(i)=a_Red * real(i) + b_Red
        Green(i)=a_Green * real(i) + b_Green
        Blue(i)=a_Blue * real(i) + b_Blue
    enddo

    np_diff_log(:,:)=0.0
    do ii=1,iypix
        do i=1,ixpix
            if(np_diff(i,ii)>0.0)then
                np_diff_log(i,ii)=log10(np_diff(i,ii))
                !write(*,*)i,ii,np_diff(i,ii),np_diff_log(i,ii)
            endif
        enddo
    enddo

    step=(max_range - min_range) / real(precision)
    !write(*,*)"step = ",step
    do ii=1,iypix
        do i=1,ixpix
            np_diff_log_X(i,ii)=int( (np_diff_log(i,ii)-min_range) / step ) +1 
            if(np_diff_log(i,ii) <= min_range)then
                np_diff_log_X(i,ii)=1
            elseif(np_diff_log(i,ii) >= max_range)then
                np_diff_log_X(i,ii)=precision
            endif
        enddo
    enddo

    do ii=1,iypix
        do i=1,ixpix
            np_diff_log_rgb(i,ii,1)=Red(np_diff_log_X(i,ii))
            np_diff_log_rgb(i,ii,2)=Green(np_diff_log_X(i,ii))
            np_diff_log_rgb(i,ii,3)=Blue(np_diff_log_X(i,ii))
            !write(*,*) i,ii,np_diff_log_X(i,ii)
        enddo
    enddo

    return
end subroutine convert_RGB