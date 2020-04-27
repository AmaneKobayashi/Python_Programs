#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import time
import tifffile
import os
import mrcfile

from PIL import Image
from skimage import io

if ((len(sys.argv)==1)):
	print("command:python3 make_random_density.py [-size] [-dimension 2 or 3 or multi] [-n_density_image]")
	exit()

n_parameter=3
parameter_name_list=[""]*n_parameter
flag_list=[0]*n_parameter

parameter_name_list[0]="-size"
parameter_name_list[1]="-dimension"
parameter_name_list[2]="-n_density_image"

for i in range(len(sys.argv)):
	if(sys.argv[i]=="-size"):
		size=sys.argv[i+1]
		flag_list[0]=1
	if(sys.argv[i]=="-dimension"):
		dimension=sys.argv[i+1]
		flag_list[1]=1
	if(sys.argv[i]=="-n_density_image"):
		n_density_image=sys.argv[i+1]
		flag_list[2]=1
	if(sys.argv[i]=="--help"):
		print("command:python3 make_random_density.py [-size] [-dimension 2 or 3 or multi] [-n_density_image]")
		exit()

input_parameter=0

for i in range(n_parameter):
	if(flag_list[i]==0):
		print("please input parameter : [" + parameter_name_list[i] + "]")
		input_parameter=1
	if(input_parameter==1):
		exit()

print("size = " + size)
print("dimension = " + dimension)
print("n_density_image = " + n_density_image)

t1=time.time()

if(dimension=="multi"):

	np.random.seed(1)	

	np_random_density=np.random.rand(int(n_density_image), int(size), int(size))
	np_random_density=np_random_density.astype(np.float32)

	foname="random_" + size + "_x_" + size + "_x_" + n_density_image + ".mrc"
	with mrcfile.new(foname, overwrite=True) as mrc:
		mrc.set_data(np_random_density)

else:
	for i in range(int(n_density_image)):
		
		np.random.seed(i)
	
		if(dimension=="2"):
			np_random_density=np.random.rand(int(size), int(size))
			np_random_density=np_random_density.astype(np.float32)
			foname="random_" + size + "_" + str(i+1).zfill(6) + ".tif"
			tifffile.imsave(foname ,np_random_density)

		if(dimension=="3"):
			np_random_density=np.random.rand(int(size), int(size), int(size))
			np_random_density=np_random_density.astype(np.float32)
	
			foname="random_" + size + "_" + str(i+1).zfill(6) + ".mrc"
			with mrcfile.new(foname, overwrite=True) as mrc:
				mrc.set_data(np_random_density)

t2=time.time()
print("total time : " + str(t2-t1))






