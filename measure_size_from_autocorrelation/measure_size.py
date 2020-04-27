#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import cupy as cp
import time
import os
import mrcfile
import tifffile
import cv2

from PIL import Image
from skimage import io

if ((len(sys.argv)==1)):
	print("command:python3 make_support_from_autocorrelation.py [-diff] [-FWHM]")
	exit()

n_parameter=2
parameter_name_list=[""]*n_parameter
flag_list=[0]*n_parameter

parameter_name_list[0]="-diff"
parameter_name_list[1]="-FWHM"

input_parameter=0

for i in range(len(sys.argv)):
	if(sys.argv[i]=="-diff"):
		finame=sys.argv[i+1]
		flag_list[0]=1
	if(sys.argv[i]=="-FWHM"):
		FWHM=sys.argv[i+1]
		flag_list[1]=1
	if(sys.argv[i]=="--help"):
		print("command:python3 make_support_from_autocorrelation.py [-diff] [-FWHM]")
		exit()

for i in range(n_parameter):
	if(flag_list[i]==0):
		print("please input parameter : [" + parameter_name_list[i] + "]")
		input_parameter=1
if(input_parameter==1):
	exit()

print("diff = " + finame)
print("FWHM = " + FWHM)

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
ave_autocorrelation=cp.average(cp_autocorrelation_abs)
std_autocorrelation=cp.std(cp_autocorrelation_abs)
max_autocorrelation=cp.max(cp_autocorrelation_abs)
print("average = " + str(ave_autocorrelation))
print("std = " + str(std_autocorrelation))
print("max = " + str(max_autocorrelation))

th = ave_autocorrelation + std_autocorrelation
print("threshold = " + str(th))
cp_th=cp.where(cp_autocorrelation_abs>=th,float(1),float(0))
cp_th=cp_th.astype(cp.float32)

np_th=cp.asnumpy(cp_th)
foname=finame[finame.rfind("/")+1:len(finame)-4] + "_th.tif"
tifffile.imsave(foname ,np_th)

Wgauss= cp.log(max_autocorrelation/th) * cp.log(2) 
Wgauss=cp.sqrt(Wgauss)
Wgauss=Wgauss/cp.pi
Wgauss=float(4) * Wgauss / float(FWHM)
Wgauss=Wgauss * float(col)
print("Wgauss = " + str(Wgauss))

kernel=np.ones((int(Wgauss),int(Wgauss)), np.float32)
dilate=cv2.dilate(np_th,kernel)

kernel=np.ones((int(2*Wgauss),int(2*Wgauss)), np.float32)
erode=cv2.erode(dilate,kernel)

#cp_th2=cp.zeros(cp_diff.shape)
#for x in range(int(col)):
#	for y in range(int(row)):
#		print(x,y)
#		if(cp_th[x,y] > 0):
#			for i in range(int(Wgauss)):
#				for ii in range(int(Wgauss)):
#					if(cp.sqrt(cp.square(x-(x-i))+cp.square(y-(y-ii))) <= Wgauss):
#						cp_th2[x-i,y-ii]=float(1)
#						cp_th2[x+i,y-ii]=float(1)						
#						cp_th2[x-i,y+ii]=float(1)
#						cp_th2[x+i,y+ii]=float(1)
			

#OS_ratio

if(np_diff.ndim==2):
	OS_ratio = float(row) * float(col) / np.sum(erode)
if(np_diff.ndim==3):
	OS_ratio = float(row) * float(col) * float(sta) / np.sum(erode)
print("OS_ratio = " + str(OS_ratio))

#size
size=np.sum(erode)
print("size = " + str(size))

#output

np_autocorrelation_abs=cp.asnumpy(cp_autocorrelation_abs)
foname=finame[finame.rfind("/")+1:len(finame)-4] + "_autocorrelation.tif"
tifffile.imsave(foname ,np_autocorrelation_abs)


foname=finame[finame.rfind("/")+1:len(finame)-4] + "_dilate.tif"
tifffile.imsave(foname ,dilate)
foname=finame[finame.rfind("/")+1:len(finame)-4] + "_erode.tif"
tifffile.imsave(foname ,erode)








