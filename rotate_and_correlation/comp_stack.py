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
import pandas as pd
import codecs

print("ver. 20200311")

temp_dens_flag=0

if len(sys.argv)==2:
	csv_analyze=sys.argv[1]	
	print("csv_analyze = " + csv_analyze)
	print("")
else:
	print("command:python3 comp_stack.py csv_analyze")
	exit()

t1=time.time()

danalyze = pd.read_csv(csv_analyze,header=0)
print("number of obs_projection = " + str(len(danalyze)))

projection_number=np.zeros([len(danalyze)])
correlation=np.zeros([len(danalyze)])
index_orientation=np.zeros([len(danalyze)])
index_rot=np.zeros([len(danalyze)])
index_x=np.zeros([len(danalyze)])
index_y=np.zeros([len(danalyze)])
filename=[""]*len(danalyze)

for n in range(len(danalyze)):
	projection_number[n]=int(danalyze.iat[n,0])
	index_orientation[n]=int(danalyze.iat[n,1])
	correlation[n]=float(danalyze.iat[n,2])
	index_rot[n]=int(danalyze.iat[n,3])
	index_x[n]=int(danalyze.iat[n,4])
	index_y[n]=int(danalyze.iat[n,5])
	filename[n]=danalyze.iat[n,6]

dfile = pd.read_csv(filename[0],header=None,names=[0,1,2,3,4,5])
obs_projection=dfile.iat[0,1]
calc_projection=dfile.iat[1,1]
#print("obs_projection = " + obs_projection)
#print("calc_projection = " + calc_projection)

with mrcfile.open(obs_projection, permissive=True) as mrc:
	cp_obs_projection=cp.asarray(mrc.data,dtype="float32")
mrc.close

cp_obs_projection_sort=cp.zeros(cp_obs_projection.shape,dtype="float32")

with mrcfile.open(calc_projection, permissive=True) as mrc:
	cp_calc_projection=cp.asarray(mrc.data,dtype="float32")
mrc.close

cp_calc_projection_sort=cp.zeros(cp_calc_projection.shape,dtype="float32")

calc_projection_150="E:/XFEL CXDI MED4 data/201607-201802_512_Tij_for_EMAN/projection_from_3D/MED4_projection_5deg.mrc"
with mrcfile.open(calc_projection_150, permissive=True) as mrc:
	cp_calc_projection_150=cp.asarray(mrc.data,dtype="float32")
mrc.close

obs_projection_150="E:/XFEL CXDI MED4 data/201607-201802_512_Tij_for_EMAN/dens_pre_scale/obs_projection-150.mrc"
with mrcfile.open(obs_projection_150, permissive=True) as mrc:
	cp_obs_projection_150=cp.asarray(mrc.data,dtype="float32")
mrc.close

cp_calc_projection_150_sort=cp.zeros(cp_obs_projection_150.shape,dtype="float32")
cp_obs_projection_150_sort=cp.zeros(cp_obs_projection_150.shape,dtype="float32")


for n in range(len(danalyze)):
	dfile = pd.read_csv(filename[n],header=None,names=[0,1,2,3,4,5])
	obs_projection=dfile.iat[0,1]
	deg=filename[n][filename[n].rfind("_"):filename[n].rfind(".")-3]
	obs_projection_150_temp=obs_projection_150[0:obs_projection_150.rfind(".")] + deg + ".mrc"
	print(obs_projection_150_temp)

	with mrcfile.open(obs_projection, permissive=True) as mrc:
		cp_obs_projection=cp.asarray(mrc.data,dtype="float32")
	mrc.close

	with mrcfile.open(obs_projection_150_temp, permissive=True) as mrc:
		cp_obs_projection_150_temp=cp.asarray(mrc.data,dtype="float32")
	mrc.close

	if(index_rot[n]==0):
		cp_obs_projection=cp_obs_projection
	elif(index_rot[n]==1):
		cp_obs_projection=cp.rot90(cp_obs_projection,axes=(1,2))
		cp_obs_projection=cp.rot90(cp_obs_projection,axes=(1,2))
	elif(index_rot[n]==2):
		cp_obs_projection=cp.flip(cp_obs_projection,axis=1)
	elif(index_rot[n]==3):
		cp_obs_projection=cp.rot90(cp_obs_projection,axes=(1,2))
		cp_obs_projection=cp.rot90(cp_obs_projection,axes=(1,2))
		cp_obs_projection=cp.flip(cp_obs_projection,axis=1)

	if(index_rot[n]==0):
		cp_obs_projection_150_temp=cp_obs_projection_150_temp
	elif(index_rot[n]==1):
		cp_obs_projection_150_temp=cp.rot90(cp_obs_projection_150_temp,axes=(1,2))
		cp_obs_projection_150_temp=cp.rot90(cp_obs_projection_150_temp,axes=(1,2))
	elif(index_rot[n]==2):
		cp_obs_projection_150_temp=cp.flip(cp_obs_projection_150_temp,axis=1)
	elif(index_rot[n]==3):
		cp_obs_projection_150_temp=cp.rot90(cp_obs_projection_150_temp,axes=(1,2))
		cp_obs_projection_150_temp=cp.rot90(cp_obs_projection_150_temp,axes=(1,2))
		cp_obs_projection_150_temp=cp.flip(cp_obs_projection_150_temp,axis=1)
	
		
	cp_obs_projection_sort[n,:,:]=cp_obs_projection[n,:,:]
	cp_calc_projection_sort[n,:,:]=cp_calc_projection[index_orientation[n],:,:]
	cp_calc_projection_150_sort[n,:,:]=cp_calc_projection_150[index_orientation[n],:,:]
	cp_obs_projection_150_sort[n,:,:]=cp_obs_projection_150_temp[n,:,:]


cp_obs_projection_sort = cp.asnumpy(cp_obs_projection_sort)#cupy配列 ⇒ numpy配列に変換
with mrcfile.new(csv_analyze[0:csv_analyze.rfind(".")] + '_obs_sort.mrc', overwrite=True) as mrc:
	mrc.set_data(cp_obs_projection_sort)
mrc.close

cp_calc_projection_sort = cp.asnumpy(cp_calc_projection_sort)#cupy配列 ⇒ numpy配列に変換
with mrcfile.new(csv_analyze[0:csv_analyze.rfind(".")] + '_calc_sort.mrc', overwrite=True) as mrc:
	mrc.set_data(cp_calc_projection_sort)
mrc.close

cp_calc_projection_150_sort = cp.asnumpy(cp_calc_projection_150_sort)#cupy配列 ⇒ numpy配列に変換
with mrcfile.new(csv_analyze[0:csv_analyze.rfind(".")] + '_calc_150_sort.mrc', overwrite=True) as mrc:
	mrc.set_data(cp_calc_projection_150_sort)
mrc.close

cp_obs_projection_150_sort = cp.asnumpy(cp_obs_projection_150_sort)#cupy配列 ⇒ numpy配列に変換
with mrcfile.new(csv_analyze[0:csv_analyze.rfind(".")] + '_obs_150_sort.mrc', overwrite=True) as mrc:
	mrc.set_data(cp_obs_projection_150_sort)
mrc.close

t2=time.time()

print("time = " + str(t2-t1))