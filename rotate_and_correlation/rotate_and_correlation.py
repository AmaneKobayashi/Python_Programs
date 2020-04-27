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

print("ver. 20200306")

temp_dens_flag=0

if len(sys.argv)==5:
	obs_projection=sys.argv[1]
	calc_projection=sys.argv[2]
	shift=sys.argv[3]
	log_file_name=sys.argv[4]
	print('obs_projection = ' + obs_projection)
	print('calc_projection = ' + calc_projection)
	print("shift = " + shift)
	print("log_file_name = " + log_file_name)	
	print("")
else:
	print("command:python3 rotate_and_correlation.py obs_projection calc_projection shift log_file_name(csv)")
	exit()

t1=time.time()

i_shift=int(shift)

# open mrc file

with mrcfile.open(obs_projection, permissive=True) as mrc:
	cp_obs_projection=cp.asarray(mrc.data,dtype="float32")
mrc.close

with mrcfile.open(calc_projection, permissive=True) as mrc:
	cp_calc_projection=cp.asarray(mrc.data,dtype="float32")
mrc.close

sta_dens_obs=cp_obs_projection.shape[0]
sta_dens_calc=cp_calc_projection.shape[0]
row=cp_calc_projection.shape[1]
col=cp_calc_projection.shape[2]
print("sta_dens_obs = " + str(sta_dens_obs))
print("sta_dens_calc = " + str(sta_dens_calc))
print("row = " + str(row))
print("col = " + str(col))
print("")

average_dens_obs=cp.average(cp_obs_projection, axis=(1,2))
average_dens_calc=cp.average(cp_calc_projection, axis=(1,2))

average_dens_calc_expa=cp.zeros(cp_calc_projection.shape)
ave_sub_calc_dens=cp.zeros(cp_calc_projection.shape)
for i in range(sta_dens_calc):
	average_dens_calc_expa[i,:,:]=average_dens_calc[i]
	ave_sub_calc_dens[i,:,:]=cp_calc_projection[i,:,:]-average_dens_calc_expa[i,:,:]

B_square=cp.square(ave_sub_calc_dens)
B=cp.sum(B_square,axis=(1,2))

obs_dens=cp.zeros(cp_calc_projection.shape)
average_dens_obs_expa=cp.zeros(cp_calc_projection.shape)

#
log_path=log_file_name
with open(log_path, mode='w') as log:
	log.write('obs_projection,' + obs_projection +"\n")
	log.write('calc_projection,' + calc_projection +"\n")
	log.write("shift," + shift +"\n")	
	log.write("projection_number,index_orientation,correlation,index_rot,index_shift_x,index_shift_y" +"\n")


for i_dens_obs in range(sta_dens_obs):
	obs_dens[:,:,:]=cp_obs_projection[i_dens_obs,:,:]
	average_dens_obs_expa[:,:,:]=average_dens_obs[i_dens_obs]
	
	ave_sub_obs_dens=obs_dens[:,:,:]-average_dens_obs_expa[:,:,:]

	A_square=cp.square(ave_sub_obs_dens)
	A=cp.sum(A_square,axis=(1,2))
	C=cp.sqrt(A*B)

	ave_sub_obs_dens_rot=cp.rot90(ave_sub_obs_dens,axes=(1,2))
	ave_sub_obs_dens_rot=cp.rot90(ave_sub_obs_dens_rot,axes=(1,2))
#	cp_density_stack_rot=cp.rot90(cp_density_stack,axes=(1,2))
#	cp_density_stack_rot=cp.rot90(cp_density_stack_rot,axes=(1,2))
#print(ave_sub_density_stack_rot.shape)

	correlation=cp.zeros((4,sta_dens_calc,2*i_shift,2*i_shift))
	for x in range(2*i_shift):
		for y in range(2*i_shift):
			if((y-i_shift == 0) & (x-i_shift == 0)):
				ave_sub_density_stack_shift[:,:,:]=ave_sub_obs_dens[:,:,:]
				ave_sub_density_stack_rot_shift[:,:,:]=ave_sub_obs_dens_rot[:,:,:]
			elif((y-i_shift == 0) & (x-i_shift != 0)):
				ave_sub_density_stack_shift=cp.roll(ave_sub_obs_dens, x-i_shift, axis=1)
				ave_sub_density_stack_rot_shift=cp.roll(ave_sub_obs_dens_rot, x-i_shift, axis=1)
			elif((y-i_shift != 0) & (x-i_shift == 0)):
				ave_sub_density_stack_shift=cp.roll(ave_sub_obs_dens, y-i_shift, axis=2)
				ave_sub_density_stack_rot_shift=cp.roll(ave_sub_obs_dens_rot, y-i_shift, axis=2)
			else:
				ave_sub_density_stack_shift=cp.roll(ave_sub_obs_dens, x-i_shift, axis=1)
				ave_sub_density_stack_rot_shift=cp.roll(ave_sub_obs_dens_rot, x-i_shift, axis=1)
				ave_sub_density_stack_shift=cp.roll(ave_sub_density_stack_shift, y-i_shift, axis=2)
				ave_sub_density_stack_rot_shift=cp.roll(ave_sub_density_stack_rot_shift, y-i_shift, axis=2)

			AB_presum=ave_sub_density_stack_shift[:,:,:]*ave_sub_calc_dens[:,:,:]
			AB=cp.sum(AB_presum,axis=(1,2))
			correlation[0,:,x,y]=AB/C

			AB_presum=ave_sub_density_stack_rot_shift[:,:,:]*ave_sub_calc_dens[:,:,:]
			AB=cp.sum(AB_presum,axis=(1,2))
			correlation[1,:,x,y]=AB/C

			ave_sub_density_stack_shift=cp.flip(ave_sub_density_stack_shift,axis=1)
			ave_sub_density_stack_rot_shift=cp.flip(ave_sub_density_stack_rot_shift,axis=1)

			AB_presum=ave_sub_density_stack_shift[:,:,:]*ave_sub_calc_dens[:,:,:]
			AB=cp.sum(AB_presum,axis=(1,2))
			correlation[2,:,x,y]=AB/C

			AB_presum=ave_sub_density_stack_rot_shift[:,:,:]*ave_sub_calc_dens[:,:,:]
			AB=cp.sum(AB_presum,axis=(1,2))
			correlation[3,:,x,y]=AB/C




	index=cp.argmax(correlation,axis=(0,2,3))

	index_rot=index[:] / (2*i_shift*2*i_shift)
	index_rot=index_rot.astype(cp.int)

	index_x=(index[:]-(2*i_shift*2*i_shift)*index_rot[:])/(2*i_shift)
	index_x=index_x.astype(cp.int)

	index_y=(index[:]-(2*i_shift*2*i_shift)*index_rot[:]) % (2*i_shift)
	index_y=index_y.astype(cp.int)
	
	correlation_orientation=cp.zeros(index.shape)
	for i in range(sta_dens_calc):
		correlation_orientation[i]=correlation[index_rot[i],i,index_x[i],index_y[i]]
	index_orientation=cp.argmax(correlation_orientation)

#	print(correlation[0,:,i_shift,i_shift])
	print("i_dens_obs = " + str(i_dens_obs))
#	print("max correlation = " + str(cp.amax(correlation)))
	print("max correlation = " + str(correlation_orientation[index_orientation]))
	print("index_orientation = " + str(index_orientation))
	print("index_rot = " + str(index_rot[index_orientation]))
	print("index_x = " + str(index_x[index_orientation]))
	print("index_y = " + str(index_y[index_orientation]))
	print("")
	with open(log_path, mode='a') as log:
		log.write(str(i_dens_obs) + "," + str(index_orientation) + "," + str(cp.amax(correlation)) + "," + str(index_rot[index_orientation]) + "," + str(index_x[index_orientation]) + "," + str(index_y[index_orientation]) +"\n")

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

#for i in range(sta_dens):
#	if(index_rot[i]==0):
#		cp_density_stack_mod=cp_density_stack[i,:,:]
#	else:
#		cp_density_stack_mod=cp_density_stack_rot[i,:,:]
	
#	if(index_x[i]-i_shift != 0):
#		cp_density_stack_mod=cp.roll(cp_density_stack_mod, int(index_x[i]-i_shift), axis=1)
#	if(index_y[i]-i_shift != 0):
#		cp_density_stack_mod=cp.roll(cp_density_stack_mod, int(index_y[i]-i_shift), axis=0)
	
#	cp_density_stack[i,:,:]=cp_density_stack_mod*(average_dens[0]/average_dens[i])

#cp_density_stack = cp.asnumpy(cp_density_stack)
#with mrcfile.new(density_stack[0:len(density_stack)-4] + '_sort.mrc', overwrite=True) as mrc:
#	mrc.set_data(cp_density_stack)

#mrc.close

t2=time.time()

print("time = " + str(t2-t1))

#print(cp.argmax(correlation,axis=(0,2,3)).shape)

















