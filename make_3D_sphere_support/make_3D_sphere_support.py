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
	print("command:python3 make_3D_sphere_support.py [-size] [-diameter]")
	exit()

n_parameter=2
parameter_name_list=[""]*n_parameter
flag_list=[0]*n_parameter

parameter_name_list[0]="-size"
parameter_name_list[1]="-diameter"

input_parameter=0

for i in range(len(sys.argv)):
	if(sys.argv[i]=="-size"):
		size=sys.argv[i+1]
		flag_list[0]=1
	if(sys.argv[i]=="-diameter"):
		diameter=sys.argv[i+1]
		flag_list[1]=1
	if(sys.argv[i]=="--help"):
		print("command:python3 make_3D_sphere_support.py [-size] [-diameter]")
		exit()

for i in range(n_parameter):
	if(flag_list[i]==0):
		print("please input parameter : [" + parameter_name_list[i] + "]")
		input_parameter=1
if(input_parameter==1):
	exit()

print("size = " + size)
print("diameter = " + diameter)

def D(x,y,z):
	distance_from_center=np.sqrt((x-float(size)/2.0)**2 + (y-float(size)/2.0)**2 + (z-float(size)/2.0)**2)
	return distance_from_center
x=y=z=np.arange(0,int(size),1)
X,Y,Z=np.meshgrid(x,y,z)
D_kernel=D(X,Y,Z)
np_D_kernel=np.asarray(D_kernel,dtype="float32")

np_sup=np.where(np_D_kernel*2.0<=float(diameter),1,0)
np_sup=np_sup.astype(np.float32)

foname="sphere_support_size_" + size + "_diameter_" + diameter + ".mrc"
with mrcfile.new(foname, overwrite=True) as mrc:
	mrc.set_data(np_sup)












