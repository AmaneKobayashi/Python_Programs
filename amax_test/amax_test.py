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
	print("command:python3 amax_test.py [-finame]")
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
		print("command:python3 amax_test.py [-finame]")
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

amax_axis0=cp.amax(cp_temp,axis=(0,1))
amax_axis1=cp.amax(cp_temp,axis=(1,2))
amax_axis2=cp.amax(cp_temp,axis=(0,2))
amax=cp.amax(cp_temp)

print("axis 01 = " + str(amax_axis0))
print("axis 12 = " + str(amax_axis1))
print("axis 02 = " + str(amax_axis2))
print("axis 012 = " + str(amax))
print()
print("axis 01 shape = " + str(amax_axis0.shape))
print("axis 12 shape = " + str(amax_axis1.shape))
print("axis 02 shape = " + str(amax_axis2.shape))
print()
print("cp_temp shape = " + str(cp_temp.shape))

for i in range(cp_temp.shape[0]):
	cp_temp[i,:,:]=cp.where(cp_temp[i,:,:]>0.8*amax_axis1[i],1.0,0.0)

cp_temp=cp_temp.astype(cp.float32)

cp_temp = cp.asnumpy(cp_temp)#cupy配列 ⇒ numpy配列に変換
with mrcfile.new('amax_test.mrc', overwrite=True) as mrc:
	mrc.set_data(cp_temp)
mrc.close



