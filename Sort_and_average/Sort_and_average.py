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

temp_dens_flag=0
if len(sys.argv)==3:
	density_stack=sys.argv[1]
	shift=sys.argv[2]
	print('density_stack = ' + density_stack)
	print("shift = " + shift)
	print("")
elif len(sys.argv)==4:
	density_stack=sys.argv[1]
	shift=sys.argv[2]
	temp_dens_name=sys.argv[3]
	temp_dens_flag=1
	print('density_stack = ' + density_stack)
	print("shift = " + shift)
	print("temp_dens_name = " + temp_dens_name)
	print("")
else:
	print("command:python3 Sort_and_average.py density_stack shift (temp_dens_name)")
	exit()

t1=time.time()

# open mrc file

with mrcfile.open(density_stack, permissive=True) as mrc:
	cp_density_stack=cp.asarray(mrc.data,dtype="float32")
mrc.close

sta_dens=cp_density_stack.shape[0]
row=cp_density_stack.shape[1]
col=cp_density_stack.shape[2]
print("sta_dens = " + str(sta_dens))
print("row = " + str(row))
print("col = " + str(col))
print("")

average_dens=cp.average(cp_density_stack, axis=(1,2))
#print(average_dens)
#print(average_dens.shape)

temp_dens=cp.zeros(cp_density_stack.shape)
if(temp_dens_flag==1):
	temp_dens_2D=Image.open(temp_dens_name)
	temp_dens_2D=cp.asarray(temp_dens_2D,dtype="float32")
	temp_dens_2D=cp.flipud(temp_dens_2D)
	temp_dens_2D_ave=cp.average(temp_dens_2D)
	temp_dens_2D[:,:]=temp_dens_2D[:,:]-temp_dens_2D_ave
	temp_dens[:,:,:]=temp_dens_2D[:,:]
else:
	temp_dens[:,:,:]=cp_density_stack[0,:,:]

#average_temp_dens=cp.average(temp_dens,axis=(1,2))

average_dens_expa=cp.zeros(cp_density_stack.shape)
for i in range(sta_dens):
	average_dens_expa[i,:,:]=average_dens[i]

average_temp_dens_expa=cp.zeros(cp_density_stack.shape)
average_temp_dens_expa[:,:,:]=average_dens[0]

ave_sub_density_stack=cp_density_stack[:,:,:]-average_dens_expa[:,:,:]
if(temp_dens_flag==1):
	ave_sub_temp_density_stack=temp_dens[:,:,:]
else:
	ave_sub_temp_density_stack=temp_dens[:,:,:]-average_temp_dens_expa[:,:,:]

A_square=cp.square(ave_sub_density_stack)
B_square=cp.square(ave_sub_temp_density_stack)

A=cp.sum(A_square,axis=(1,2))
B=cp.sum(B_square,axis=(1,2))
C=cp.sqrt(A*B)

ave_sub_density_stack_rot=cp.rot90(ave_sub_density_stack,axes=(1,2))
ave_sub_density_stack_rot=cp.rot90(ave_sub_density_stack_rot,axes=(1,2))
cp_density_stack_rot=cp.rot90(cp_density_stack,axes=(1,2))
cp_density_stack_rot=cp.rot90(cp_density_stack_rot,axes=(1,2))
#print(ave_sub_density_stack_rot.shape)

i_shift=int(shift)
correlation=cp.zeros((2,sta_dens,2*i_shift,2*i_shift))
for x in range(2*i_shift):
	for y in range(2*i_shift):
		if((y-i_shift == 0) & (x-i_shift == 0)):
			ave_sub_density_stack_shift[:,:,:]=ave_sub_density_stack[:,:,:]
			ave_sub_density_stack_rot_shift[:,:,:]=ave_sub_density_stack_rot[:,:,:]
		elif((y-i_shift == 0) & (x-i_shift != 0)):
			ave_sub_density_stack_shift=cp.roll(ave_sub_density_stack, x-i_shift, axis=1)
			ave_sub_density_stack_rot_shift=cp.roll(ave_sub_density_stack_rot, x-i_shift, axis=1)
		elif((y-i_shift != 0) & (x-i_shift == 0)):
			ave_sub_density_stack_shift=cp.roll(ave_sub_density_stack_shift, y-i_shift, axis=2)
			ave_sub_density_stack_rot_shift=cp.roll(ave_sub_density_stack_rot_shift, y-i_shift, axis=2)
		else:
			ave_sub_density_stack_shift=cp.roll(ave_sub_density_stack, x-i_shift, axis=1)
			ave_sub_density_stack_rot_shift=cp.roll(ave_sub_density_stack_rot, x-i_shift, axis=1)
			ave_sub_density_stack_shift=cp.roll(ave_sub_density_stack_shift, y-i_shift, axis=2)
			ave_sub_density_stack_rot_shift=cp.roll(ave_sub_density_stack_rot_shift, y-i_shift, axis=2)

		AB_presum=ave_sub_density_stack_shift[:,:,:]*ave_sub_temp_density_stack[:,:,:]
		AB=cp.sum(AB_presum,axis=(1,2))
		correlation[0,:,x,y]=AB/C

		AB=cp.sum(ave_sub_density_stack_rot_shift*ave_sub_temp_density_stack,axis=(1,2))
		correlation[1,:,x,y]=AB/C

print(correlation[0,:,i_shift,i_shift])
print("max correlation = " + str(cp.amax(correlation[0,:,i_shift,i_shift])))
print("index = " + str(cp.argmax(correlation[0,:,i_shift,i_shift])))

index=cp.argmax(correlation,axis=(0,2,3))

index_rot=index[:] / (2*i_shift*2*i_shift)
index_rot=index_rot.astype(cp.int)

index_x=(index[:]-(2*i_shift*2*i_shift)*index_rot[:])/(2*i_shift)
index_x=index_x.astype(cp.int)

index_y=(index[:]-(2*i_shift*2*i_shift)*index_rot[:]) % (2*i_shift)
index_y=index_y.astype(cp.int)

#n=0
#print(correlation[:,n,:,:])
#print("index = " + str(index[n]))
#print("index_rot = " + str(index_rot[n]))
#print("index_x = " + str(index_x[n]))
#print("index_y = " + str(index_y[n]))
#print(correlation[index_rot[n],n,index_x[n],index_y[n]])
#print(correlation[:,n,:,:].shape)

#cupyîzóÒ ÅÀ numpyîzóÒÇ…ïœä∑
#cp_density_stack = cp.asnumpy(cp_density_stack)
#average_dens = cp.asnumpy(average_dens)

for i in range(sta_dens):
	if(index_rot[i]==0):
		cp_density_stack_mod=cp_density_stack[i,:,:]
	else:
		cp_density_stack_mod=cp_density_stack_rot[i,:,:]
	
	if(index_x[i]-i_shift != 0):
		cp_density_stack_mod=cp.roll(cp_density_stack_mod, int(index_x[i]-i_shift), axis=1)
	if(index_y[i]-i_shift != 0):
		cp_density_stack_mod=cp.roll(cp_density_stack_mod, int(index_y[i]-i_shift), axis=0)
	
	cp_density_stack[i,:,:]=cp_density_stack_mod*(average_dens[0]/average_dens[i])

cp_density_stack = cp.asnumpy(cp_density_stack)
with mrcfile.new(density_stack[0:len(density_stack)-4] + '_sort.mrc', overwrite=True) as mrc:
	mrc.set_data(cp_density_stack)

mrc.close

t2=time.time()

print("time = " + str(t2-t1))

#print(cp.argmax(correlation,axis=(0,2,3)).shape)

















