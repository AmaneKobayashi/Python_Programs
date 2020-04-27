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
	print("command:python3 make_support_from_autocorrelation.py [-diff] [-threshold]")
	exit()

n_parameter=2
parameter_name_list=[""]*n_parameter
flag_list=[0]*n_parameter

parameter_name_list[0]="-diff"
parameter_name_list[1]="-threshold"

input_parameter=0

for i in range(len(sys.argv)):
	if(sys.argv[i]=="-diff"):
		finame=sys.argv[i+1]
		flag_list[0]=1
	if(sys.argv[i]=="-threshold"):
		threshold=sys.argv[i+1]
		flag_list[1]=1
	if(sys.argv[i]=="--help"):
		print("command:python3 make_support_from_autocorrelation.py [-diff] [-threshold]")
		exit()

for i in range(n_parameter):
	if(flag_list[i]==0):
		print("please input parameter : [" + parameter_name_list[i] + "]")
		input_parameter=1
if(input_parameter==1):
	exit()

print("diff = " + finame)
print("threshold = " + threshold)

if(finame.find("tif")!=-1):
	diff=Image.open(finame)
	np_diff=np.asarray(diff,dtype="float32")
if(finame.find("mrc")!=-1):
	with mrcfile.open(finame, permissive=True) as mrc:
		np_diff=np.asarray(mrc.data,dtype="float32")
	mrc.close

print("np_diff dtype = " + str(np_diff.dtype))
print("np_diff shape = " + str(np_diff.shape))
print("np_diff ndim = " + str(np_diff.ndim))


row=np_diff.shape[0]
col=np_diff.shape[1]
print("row of diff = " + str(row))
print("col of diff = " + str(col))

if(np_diff.ndim==3):
	sta=np_diff.shape[2]
	print("sta of diff = " + str(sta))
print("")

np_temp=np.zeros(np_diff.shape,dtype="float32")
np_temp=np.where(np_diff>0.0,np_diff,0.0)
np_diff=np_temp

#cupy array

cp_diff = cp.asarray(np_diff,dtype="float32")

#FFT

if(np_diff.ndim==2):
	cp_autocorrelation = cp.fft.fft2(cp_diff, norm="ortho")
if(np_diff.ndim==3):
	cp_autocorrelation = cp.fft.fftn(cp_diff, axes=(0,1,2), norm="ortho")

#threshold

cp_autocorrelation = cp.fft.fftshift(cp_autocorrelation)
cp_autocorrelation_abs=cp.absolute(cp_autocorrelation)
th = float(threshold) * cp.amax(cp_autocorrelation_abs)
cp_sup=cp.where(cp_autocorrelation_abs>=th,float(1),float(0))
cp_sup=cp_sup.astype(cp.float32)

#OS_ratio

if(np_diff.ndim==2):
	OS_ratio = float(row) * float(col) / cp.sum(cp_sup)
if(np_diff.ndim==3):
	OS_ratio = float(row) * float(col) * float(sta) / cp.sum(cp_sup)
print("OS_ratio = " + str(OS_ratio))

#output

np_sup = cp.asnumpy(cp_sup)
np_autocorrelation_abs=cp.asnumpy(cp_autocorrelation_abs)
if(np_diff.ndim==2):
	foname=finame[finame.rfind("/")+1:len(finame)-4] + "_threshold_" + threshold + ".tif"
	tifffile.imsave(foname ,np_sup)
#	foname=finame[finame.rfind("/")+1:len(finame)-4] + "_autoc.tif"
#	tifffile.imsave(foname ,np_autocorrelation_abs)
if(np_diff.ndim==3):
	foname=finame[finame.rfind("/")+1:len(finame)-4] + "_threshold_" + threshold + ".mrc"
	with mrcfile.new(foname, overwrite=True) as mrc:
		mrc.set_data(np_sup)
	mrc.close







