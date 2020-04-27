#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import cupy as cp
import time
import os
import mrcfile
import tifffile

from PIL import Image
from skimage import io

if ((len(sys.argv)==1)):
	print("command:python3 fft2d_multi_test.py [-finame]")
	exit()

n_parameter=1
parameter_name_list=[""]*n_parameter
flag_list=[0]*n_parameter

parameter_name_list[0]="-finame"

input_parameter=0

for i in range(len(sys.argv)):
	if(sys.argv[i]=="-finame"):
		finame=sys.argv[i+1]
		flag_list[0]=1
	if(sys.argv[i]=="--help"):
		print("command:python3 fft2d_multi_test.py [-finame]")
		exit()

for i in range(n_parameter):
	if(flag_list[i]==0):
		print("please input parameter : [" + parameter_name_list[i] + "]")
		input_parameter=1
if(input_parameter==1):
	exit()

with mrcfile.open(finame, permissive=True) as mrc:
	cp_temp=cp.asarray(mrc.data,dtype="float32")
mrc.close

print("cp_temp dtype = " + str(cp_temp.dtype))
print("cp_temp shape = " + str(cp_temp.shape))

cp_temp = cp.fft.fftn(cp_temp, axes=(1,2), norm="ortho")#【フーリエ変換】
cp_temp = cp.fft.fftshift(cp_temp, axes=(1,2))#fftshiftを使ってシフト

cp_temp = cp.asnumpy(cp_temp)#cupy配列 ⇒ numpy配列に変換
with mrcfile.new('test_temp.mrc', overwrite=True) as mrc:
	mrc.set_data(np.absolute(cp_temp))
mrc.close






