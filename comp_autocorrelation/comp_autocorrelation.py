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

print("ver. 20191024")

if(len(sys.argv)!=3):
	print("command: comp_autocorrelation.py autocorrelation_stack temp_autocorrelation")
	exit()

autocorrelation_stack=sys.argv[1]
temp_autocorrelation=sys.argv[2]
print("autocorrelation_stack = " + autocorrelation_stack)
print("temp_autocorrelation = " + temp_autocorrelation)

#stack
with mrcfile.open(autocorrelation_stack, permissive=True) as mrc:
	cp_autocorrelation_stack=cp.asarray(mrc.data,dtype="float32")
mrc.close

average_autocorrelation=cp.average(cp_autocorrelation_stack, axis=(1,2))

average_sub_correlation=cp.zeros(cp_autocorrelation_stack.shape)

n_autocorrelation=cp_autocorrelation_stack.shape[0]
for i in range(n_autocorrelation):
	average_sub_correlation[i,:,:]=cp_autocorrelation_stack[i,:,:]-average_autocorrelation[i]

#temp_autocorrelation
cp_temp_autocorrelation=Image.open(temp_autocorrelation)
cp_temp_autocorrelation=cp.asarray(cp_temp_autocorrelation, dtype="float32")

cp_temp_autocorrelation=cp.flipud(cp_temp_autocorrelation)

average_temp_autocorrelation=cp.average(cp_temp_autocorrelation)
average_sub_temp_autocorrelation=cp.zeros(cp_temp_autocorrelation.shape)
average_sub_temp_autocorrelation=cp_temp_autocorrelation[:,:]-average_temp_autocorrelation

#
sub_abs=cp.zeros(cp_autocorrelation_stack.shape[0])
for i in range(n_autocorrelation):
	sub_abs[i]=cp.sum(cp.absolute(average_sub_correlation[i,:,:]-average_sub_temp_autocorrelation[:,:]))
	
print(cp_temp_autocorrelation)
print()
print(cp_autocorrelation_stack[0,:,:])

print(sub_abs)
print("min correlation = " + str(cp.amin(sub_abs)))
print("index = " + str(cp.argmin(sub_abs)))