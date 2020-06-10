  subroutine READSPE(rdata,ixpix,iypix,finame)

  implicit real*8 (a-h,o-z)

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
  integer :: dabyte, hebyte, mxndat2
  parameter(dabyte=4,hebyte=4100,mxndat2=3599*2399)
  integer,intent(in) :: ixpix,iypix 
  real*4,intent(out) :: rdata(ixpix,iypix)
  real*4 :: rtemp(1:mxndat2+hebyte/dabyte)
!      ^dabyte
  integer*4 :: itemp(1:mxndat2+hebyte/dabyte)
!         ^dabyte
  character :: ctemp*10000000
  integer*2 itype
  character*1000,intent(in) :: finame
  parameter(ndin=11,nddat=12,ndout=13,ndlog=14)
!
  equivalence(ctemp,rtemp(1))
  equivalence(ctemp,itemp(1))
!
  lenfin=len_trim(finame)
!
!
! <<<<<<<<<<         Reading all data as 1 record         <<<<<<<<<<
!
  open(ndin,file=finame(1:lenfin),access='direct',&
  form='unformatted',recl=2,status='old')
     read(ndin,rec=55) itype             
  close(ndin)

  lrec=int(dabyte)*ixpix
  nrec=iypix
  lenrec=lrec*nrec+int(hebyte)

  open(ndin,file=finame(1:lenfin),access='direct',&
  form='unformatted',recl=lenrec,status='old')
     read(ndin,rec=1) ctemp(1:lrec*nrec+hebyte)
  close(ndin)
!
! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
!
!

  if(itype.eq.0) then
     do j=1,iypix
        do i=1,ixpix
           rdata(i,j)=real(rtemp(hebyte/dabyte+i+(j-1)*ixpix))
        end do
     end do
  else
     do j=1,iypix
        do i=1,ixpix
           rdata(i,j)=real(itemp(hebyte/dabyte+i+(j-1)*iypix))
        end do
     end do
  end if
!
!
!
  return
end subroutine READSPE