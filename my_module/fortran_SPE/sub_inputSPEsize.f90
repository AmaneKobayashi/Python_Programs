subroutine inputSPEsize(finame,ixpix,iypix)
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !										!
    !		Subroutine for reading size of image.SPE			!
    !		Coded by AMANE KOBAYASHI in 2014/01/27				!
    !										!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    implicit real*8 (a-h,o-z)
    !
    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
    !
    !  Parameters of binary image file
    !
    !          rdata : two-dimensional real data array
    !          ixpix : maximum number of pixels (x)
    !          iypix : maximum number of pixels (y)
    !          dabyte: number of bytes per one pixel value
    !          hebyte: number of bytes for header
    !          itype : data type
    !
    !++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++!
    !
          integer dabyte, hebyte, mxndat2
          integer,intent(out) :: ixpix,iypix
          integer*2 :: ixtmp,iytmp
          parameter(dabyte=4,hebyte=4100,mxndat2=2000*200000)
          character*1000,intent(in) :: finame
          parameter(ndin=11,nddat=12,ndout=13,ndlog=14)
          
          open(ndin,file=finame(1:len_trim(finame)),access='direct',form='unformatted',recl=2,status='old')
             read(ndin,rec=22) ixtmp
             read(ndin,rec=329) iytmp
             ixpix=int(ixtmp)
             iypix=int(iytmp)
    !         write(*,*) "ixpix = ",ixpix
    !         write(*,*) "iypix = ",iypix,iytmp
          close(ndin)
        
          return
end subroutine inputSPEsize
    