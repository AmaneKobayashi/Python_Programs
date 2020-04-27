#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import cupy as cp
import time
import tifffile
import os
import mrcfile

from PIL import Image
from skimage import io

if ((len(sys.argv)==1)):
	print("command:python3 FFT3_test.py [-finame]")
	exit()

n_parameter=1
parameter_name_list=[""]*n_parameter
flag_list=[0]*n_parameter

parameter_name_list[0]="-finame"

for i in range(len(sys.argv)):
	if(sys.argv[i]=="-finame"):
		finame=sys.argv[i+1]
		flag_list[0]=1
	if(sys.argv[i]=="--help"):
		print("command:python3 FFT3_test.py [-finame]")
		exit()

input_parameter=0

for i in range(n_parameter):
	if(flag_list[i]==0):
		print("please input parameter : [" + parameter_name_list[i] + "]")
		input_parameter=1
	if(input_parameter==1):
		exit()

print("finame = " + finame)

t1=time.time()

with mrcfile.open(finame, permissive=True) as mrc:
#	mrc.header.map = mrcfile.constants.MAP_ID
	np_finame=np.asarray(mrc.data,dtype="float32")
mrc.close

t2=time.time()

print("np_finame size = " + str(np_finame.size))
print("np_finame shape = " + str(np_finame.shape))
print("np_finame size = " + str(type(np_finame.size)))
print("np_finame dtype = " + str(np_finame.dtype))

cp_finame=cp.asarray(np_finame,dtype="float32")

cp_finame = cp.fft.fftn(cp_finame, axes=(0,1,2), norm="ortho")
#cp_finame = cp.fft.fftshift(cp_finame)
cp_amp = cp.absolute(cp_finame)

t3=time.time()

np_finame = cp.asnumpy(cp_amp)
foname=finame[finame.rfind("/")+1:len(finame)-4] + "_FFT.mrc" 
with mrcfile.new(foname, overwrite=True) as mrc2:
	mrc2.set_data(np_finame)
mrc2.close

t4=time.time()

print("open time : " + str(t2-t1))
print("fft time  : " + str(t3-t2))
print("output    : " + str(t4-t3))
print("total time: " + str(t4-t1))









