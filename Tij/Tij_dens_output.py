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
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from skimage import io
from my_module import Rf

t1=time.time()

if ((len(sys.argv)!=3)):
	print("command:python.exe Tij.py [stack.mrc] [shift]")
	exit()

stack=sys.argv[1]
print("stack = " + stack)
shift=sys.argv[2]
print("shift = " + shift)
print()

i_shift=int(shift)

csv_path=stack[0:stack.rfind(".")] + "_Tij.csv"
if(os.path.isfile(csv_path)):
	print("csv_file already exists")
	print()
else:
	with mrcfile.open(stack, permissive=True) as mrc:
		cp_dens=cp.asarray(mrc.data,dtype="float32")
	mrc.close

	sta_dens=cp_dens.shape[0]
	print("number of density images = " + str(sta_dens))
	print()

	cp_dens_rot=cp.rot90(cp_dens,axes=(1,2))
	cp_dens_rot=cp.rot90(cp_dens_rot,axes=(1,2))

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
	with open(csv_path, mode='w') as log:
		log.write("i_min," + str(index_0) + ",j_min," + str(index_1) + ",Tij_min," + str(Tij[index_0,index_1]) + "\n")
	with open(csv_path, mode='a') as log:
		log.write("i,j,Tij" + "\n")
	for i in range(sta_dens):
		for ii in range(sta_dens-i-1):
			j=ii+i+1
			with open(csv_path, mode='a') as log:
				log.write(str(i) + "," + str(j) + "," + str(Tij[i,j]) + "\n")

Tij_ave_dens_path=stack[0:stack.rfind(".")] + "_Tij_ave.tif"
if(os.path.isfile(Tij_ave_dens_path)):
	print("Tij ave dens file already exists")
	print()
else:
	dcsv = pd.read_csv(csv_path,header=None)
	i_min=int(dcsv.iat[0,1])
	j_min=int(dcsv.iat[0,3])
	Tij_min=float(dcsv.iat[0,5])
	print("i_min = " + str(i_min))
	print("j_min = " + str(j_min))	
	print("Tij_min = " + str(Tij_min))
	print()

	with mrcfile.open(stack, permissive=True) as mrc:
		np_dens=np.asarray(mrc.data,dtype="float32")
	mrc.close
	
	dens_i=np.zeros((np_dens.shape[1],np_dens.shape[2]),dtype=np.float32)
	dens_j=np.zeros((np_dens.shape[1],np_dens.shape[2]),dtype=np.float32)	
	ave_dens_i=np.zeros((np_dens.shape[1],np_dens.shape[2]),dtype=np.float32)
	ave_dens_j=np.zeros((np_dens.shape[1],np_dens.shape[2]),dtype=np.float32)

	dens_i[:,:]=np_dens[i_min,:,:]
	dens_j[:,:]=np_dens[j_min,:,:]

	ave_i_dens=np.average(dens_i)
	ave_j_dens=np.average(dens_j)	
	ave_dens_i[:,:]=ave_i_dens
	ave_dens_j[:,:]=ave_j_dens

	dens_i_rot=cp.rot90(dens_i)
	dens_i_rot=cp.rot90(dens_i_rot)

	B=dens_j-ave_dens_j
	B_square=np.square(B)
	B_square_sum=np.sum(B_square)

	correlation=np.zeros((2,2*i_shift,2*i_shift))
	for x in range(2*i_shift):
		for y in range(2*i_shift):
			if((y-i_shift == 0) & (x-i_shift == 0)):
				dens_i_shift[:,:]=dens_i[:,:]
				dens_i_rot_shift[:,:]=dens_i_rot[:,:]
			elif((y-i_shift == 0) & (x-i_shift != 0)):
				dens_i_shift=np.roll(dens_i, x-i_shift, axis=0)
				dens_i_rot_shift=np.roll(dens_i_rot, x-i_shift, axis=0)
			elif((y-i_shift != 0) & (x-i_shift == 0)):
				dens_i_shift=np.roll(dens_i, y-i_shift, axis=1)
				dens_i_rot_shift=np.roll(dens_i_rot, y-i_shift, axis=1)
			else:
				dens_i_shift=np.roll(dens_i, x-i_shift, axis=0)
				dens_i_rot_shift=np.roll(dens_i_rot, x-i_shift, axis=0)
				dens_i_shift=np.roll(dens_i_shift, y-i_shift, axis=1)
				dens_i_rot_shift=np.roll(dens_i_rot_shift, y-i_shift, axis=1)
				
			A=dens_i_shift-ave_dens_i
			A_square=np.square(A)
						
			nume=np.sum(A * B)
			deno=np.sum(A_square)*B_square_sum
			deno=np.sqrt(deno)
			
			ZNCC=nume/deno
			
			correlation[0,x,y]=ZNCC
			
			A=dens_i_rot_shift-ave_dens_i
			A_square=np.square(A)
						
			nume=np.sum(A * B)
			deno=np.sum(A_square)*B_square_sum
			deno=np.sqrt(deno)
			
			ZNCC=nume/deno
			
			correlation[1,x,y]=ZNCC			
			
	
	max_corr=np.amax(correlation)
	print("max_corr = " + str(max_corr))
	
	index=np.argmax(correlation)
	index_rot=index // (2*i_shift*2*i_shift)
	index_x=(index % (2*i_shift*2*i_shift)) // (2*i_shift)
	index_y=(index % (2*i_shift*2*i_shift)) % (2*i_shift)
	print("index = " + str(index))
	print("index_rot = " + str(index_rot))
	print("index_x = " + str(index_x))
	print("index_y = " + str(index_y))
	print("corr[index_rot,index_x,indexz_y] = " + str(correlation[index_rot,index_x,index_y]))
	print()
	
	if(index_rot == 1):
		dens_i=dens_i_rot

	if((index_y-i_shift == 0) & (index_x-i_shift == 0)):
		dens_i_shift[:,:]=dens_i[:,:]
	elif((index_y-i_shift == 0) & (index_x-i_shift != 0)):
		dens_i_shift=np.roll(dens_i, index_x-i_shift, axis=0)
	elif((index_y-i_shift != 0) & (index_x-i_shift == 0)):
		dens_i_shift=np.roll(dens_i, index_y-i_shift, axis=1)
	else:
		dens_i_shift=np.roll(dens_i, index_x-i_shift, axis=0)
		dens_i_shift=np.roll(dens_i_shift, index_y-i_shift, axis=1)
	
	foname=stack[0:stack.rfind(".")] + "_i_min.tif"
	tifffile.imsave(foname ,dens_i_shift)
	foname=stack[0:stack.rfind(".")] + "_j_min.tif"
	tifffile.imsave(foname ,dens_j)
	
	ave_dens=np.zeros(dens_i.shape,dtype=np.float32)
	ave_dens[:,:]=(dens_i_shift[:,:]+dens_j[:,:])/2.0
	
	tifffile.imsave(Tij_ave_dens_path ,ave_dens)

log_path=stack[0:stack.rfind(".")] + "_Tij.log"
if(os.path.isfile(log_path)):
	print("log file already exists")
	print()
else:

	MPRlog=stack[0:stack.rfind("_final")] + "_MPR.log"
	df=pd.read_csv(MPRlog, sep=" ",names=[1,2,3,4,5,6,7])
	diff=os.path.join("G:\DFPR", df.iat[1,2])

	print("diff = " + diff)
	print()
	
	np_diff=np.asarray(Image.open(diff),dtype="float32")
	ave_dens=np.asarray(Image.open(Tij_ave_dens_path),dtype="float32")
	dens_i=np.asarray(Image.open(stack[0:stack.rfind(".")] + "_i_min.tif"),dtype="float32")
	dens_j=np.asarray(Image.open(stack[0:stack.rfind(".")] + "_j_min.tif"),dtype="float32")
	if __name__ == '__main__':
		Rf_ave_dens,scale_ave_dens=Rf.Rf(ave_dens,np_diff)
		Rf_dens_i,scale_dens_i=Rf.Rf(dens_i,np_diff)
		Rf_dens_j,scale_dens_j=Rf.Rf(dens_j,np_diff)

	print("Rf_ave_dens = " + str(Rf_ave_dens))
	print("Rf_dens_i = " + str(Rf_dens_i))
	print("Rf_dens_j = " + str(Rf_dens_j))
	print("scale_ave_dens = " + str(scale_ave_dens))
	print("scale_dens_i = " + str(scale_dens_i))
	print("scale_dens_j = " + str(scale_dens_j))
	print()
	
	dcsv = pd.read_csv(csv_path,header=None)
	i_min=int(dcsv.iat[0,1])
	j_min=int(dcsv.iat[0,3])
	Tij_min=float(dcsv.iat[0,5])
	print("i_min = " + str(i_min))
	print("j_min = " + str(j_min))	
	print("Tij_min = " + str(Tij_min))
	print()

	with open(log_path, mode='w') as log2:
		log2.write("diff        : " + diff + "\n"
		           "image_i     : " + str(i_min) + "\n" +
		           "image_j     : " + str(j_min) + "\n" +
		           "rfactor_i   : " + str(Rf_dens_i) + "\n" +
		           "rfactor_j   : " + str(Rf_dens_j) + "\n" +
		           "Tij_minimum : " + str(Tij_min) + "\n" +
		           "rfactor     : " + str(Rf_ave_dens) + "\n" +
		           "scalefactor : " + str(scale_ave_dens) + "\n")

histogram_path=stack[0:stack.rfind(".")] + "_Tij_hisotgram.png"
if(os.path.isfile(histogram_path)):
	print("histogram already exists")
	print()
else:
	dcsv = pd.read_csv(csv_path,header=None)
	len_csv=len(dcsv)-2
	hist_value=np.zeros(len_csv,dtype=np.float32)
	for i in range(len_csv):
		hist_value[i]=float(dcsv.iat[2+i,2])	
	
	basename=os.path.basename(stack)
	
	fig=plt.figure()
	plt.hist(hist_value, bins=200, range=(0.0,1.0), color="red", rwidth=1.0, label=basename)
	plt.legend()
	plt.xlabel("Tij")
	plt.ylabel("frequency")
	plt.xlim(0.0,1.0)
	
	
	fig.savefig(histogram_path)


t2=time.time()
print("calculation time = " + str(t2-t1))