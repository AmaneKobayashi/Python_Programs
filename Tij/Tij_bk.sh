#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import cupy as cp
import time
import os
import mrcfile
import cv2
import tifffile
import gc
import subprocess
import chainer

from PIL import Image
from skimage import io

if ((len(sys.argv)!=3)):
	print("command:python.exe Tij.py [stack.mrc] [shift]")
	exit()

stack=sys.argv[1]
print("stack = " + stack)
shift=sys.argv[2]
print("shift = " + shift)

with mrcfile.open(stack, permissive=True) as mrc:
	cp_dens=cp.asarray(mrc.data,dtype="float32")
mrc.close

sta_dens=cp_dens.shape[0]
print("number of density images = " + str(sta_dens))

cp_dens_rot=cp.rot90(cp_dens,axes=(1,2))
cp_dens_rot=cp.rot90(cp_dens_rot,axes=(1,2))

t1=time.time()

i_shift=int(shift)
Tij=cp.zeros((sta_dens,sta_dens))
temp_dens=cp.zeros(cp_dens.shape)
for i in range(sta_dens):
	Tij_shift_rot=cp.zeros((2,sta_dens,2*i_shift,2*i_shift))
	temp_dens[:,:,:]=cp_dens[i,:,:]
	for x in range(2*i_shift):
		for y in range(2*i_shift):
			if((y-i_shift == 0) & (x-i_shift == 0)):
				cp_dens_shift[:,:,:]=cp_dens[:,:,:]
				cp_dens_rot_shift[:,:,:]=cp_dens_rot[:,:,:]
			elif((y-i_shift == 0) & (x-i_shift != 0)):
				cp_dens_shift=cp.roll(cp_dens, x-i_shift, axis=1)
				cp_dens_rot_shift=cp.roll(cp_dens_rot, x-i_shift, axis=1)
			elif((y-i_shift != 0) & (x-i_shift == 0)):
				cp_dens_shift=cp.roll(cp_dens, y-i_shift, axis=2)
				cp_dens_rot_shift=cp.roll(cp_dens_rot, y-i_shift, axis=2)
			else:
				cp_dens_shift=cp.roll(cp_dens, x-i_shift, axis=1)
				cp_dens_rot_shift=cp.roll(cp_dens_rot, x-i_shift, axis=1)
				cp_dens_shift=cp.roll(cp_dens_shift, y-i_shift, axis=2)
				cp_dens_rot_shift=cp.roll(cp_dens_rot_shift, y-i_shift, axis=2)
		
			nume=cp.absolute(temp_dens[:,:,:]-cp_dens_shift[:,:,:])
			nume=cp.sum(nume,axis=(1,2))
			deno=cp.absolute(temp_dens[:,:,:]+cp_dens_shift[:,:,:])
			deno=cp.sum(deno,axis=(1,2))
			Tij_shift_rot[0,:,x,y]=nume/deno

			nume=cp.absolute(temp_dens[:,:,:]-cp_dens_rot_shift[:,:,:])
			nume=cp.sum(nume,axis=(1,2))
			deno=cp.absolute(temp_dens[:,:,:]+cp_dens_rot_shift[:,:,:])
			deno=cp.sum(deno,axis=(1,2))
			Tij_shift_rot[1,:,x,y]=nume/deno
			
#	print(Tij_shift_rot[0,0,i_shift,i_shift])
	Tij_shift_rot[:,i,:,:]=100.0
#	Tij_shift_rot=cp.where(Tij_shift_rot==0,100,Tij_shift_rot)
#	print(Tij_shift_rot[0,0,i_shift,i_shift])

	Tij[i,:]=cp.amin(Tij_shift_rot,axis=(0,2,3))

#index=cp.unravel_index(cp.argmin(Tij), Tij.shape)
index=cp.argmin(Tij)
index_0=index // sta_dens
index_1=index % sta_dens

#print(cp.amin(Tij))
#print(index)
#print(index_0)
#print(index_1)
#print(Tij[index_0,index_1])
#print(Tij[index_1,index_0])

#csv fileçÏê¨
log_path=stack[0:stack.rfind(".")] + "_Tij.csv"
with open(log_path, mode='w') as log:
	log.write("i_min," + str(index_0) + ",j_min," + str(index_1) + ",Tij_min," + str(Tij[index_0,index_1]) + "\n")
with open(log_path, mode='a') as log:
	log.write("i,j,Tij" + "\n")
for i in range(sta_dens):
	for ii in range(sta_dens-i-1):
		j=ii+i+1
		with open(log_path, mode='a') as log:
			log.write(str(i) + "," + str(j) + "," + str(Tij[i,j]) + "\n")

#np_dens = cp.asnumpy(cp_dens)#cupyîzóÒ ÅÀ numpyîzóÒÇ…ïœä∑
#foname=stack[0:stack.rfind(".")] + "_i.tif"
#tifffile.imsave(foname ,cp_dens[index_0,:,:])
#foname=stack[0:stack.rfind(".")] + "_j.tif"
#tifffile.imsave(foname ,cp_dens[index_1,:,:])

t2=time.time()
print("calculation time = " + str(t2-t1))