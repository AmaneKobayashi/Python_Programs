#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import tifffile
import pandas as pd
import sys
import os
from PIL import Image
import mrcfile
import shutil
import glob

#runの指定
if len(sys.argv)!=6:
#	print("command:python3 Extract.py file_name_master BG_file_name_master flat_field mask trim_size saturation_value")
	print("command:python3 Extract.py file_name_master BG_file_name_master flat_field mask trim_size")
	exit()

file_name_master=sys.argv[1]
BG_file_name_master=sys.argv[2]
flat_field=sys.argv[3]
mask=sys.argv[4]
trim_size=sys.argv[5]
#saturation_value=sys.argv[6]
print('file_name_master = ' + file_name_master)
print("BG_file_name_master = " + BG_file_name_master)
print('flat_field = ' + flat_field)
print('mask = ' + mask)
print('trim_size = ' + trim_size)
#print("saturation_value  = " + saturation_value)
print("")


#
dir=file_name_master[0:len(file_name_master)-10] + "/Preprocessing"
os.makedirs(dir, exist_ok=True)

#file名

u1=file_name_master.rfind('Sample_')
u2=file_name_master.rfind('_ang_')
u3=file_name_master.rfind('_T_')
u4=file_name_master.rfind('_n_')
u5=file_name_master.rfind('_Y_')
u6=file_name_master.rfind('_Z_')
u7=file_name_master.rfind('_dY_')
u8=file_name_master.rfind('_dZ_')
u9=file_name_master[u8+len('_dZ_'):len(file_name_master)].find('_')+u8+len('_dZ_')
u10=file_name_master[u9+len('_'):len(file_name_master)].find('_')+u9+len('_')

sample_name=file_name_master[u1+len("Sample_"):u2]
ang=file_name_master[u2+len('_ang_'):u3]
T=file_name_master[u3+len('_T_'):u4]
n=file_name_master[u4+len('_n_'):u5]
Y=file_name_master[u5+len('_Y_'):u6]
Z=file_name_master[u6+len('_Z_'):u7]
dY=file_name_master[u7+len('_dY_'):u8]
dZ=file_name_master[u8+len('_dZ_'):u9]
n_EIGER=file_name_master[u9+len("_"):u10]

print("sample_name = " + sample_name)
print("ang = " + ang)
print("T = " + T)
print("n = " + n)
print("Y = " + Y)
print("Z = " + Z)
print("dY = " + dY)
print("dZ = " + dZ)
print("n_EIGER = " + n_EIGER)
print("")

file_name_data=file_name_master[0:len(file_name_master)-9] + "data_000001.h5"
print("file_name_data = " + file_name_data)

BG_file_name_data = BG_file_name_master[0:len(BG_file_name_master)-9] + "data_000001.h5"

#list_dir=str(os.listdir())
#list_dir=str(glob.glob("BG_" + sample_name + "_ang_" + ang + "*.h5"))
#u1=list_dir[0:len(list_dir)].find("BG_" + sample_name + "_ang_" + ang)
#u2=list_dir[u1:len(list_dir)].find(",")+u1
#BG_file_name=list_dir[u1:u2-1]

#if(BG_file_name.find("master")==-1):
#	BG_file_name_master = BG_file_name[0:len(BG_file_name)-14] + "master.h5"
#	BG_file_name_data = BG_file_name
#else:
#	BG_file_name_master = BG_file_name
#	BG_file_name_data = BG_file_name[0:len(BG_file_name)-9] + "data_000001.h5"

#print("BG_file_name_master = " + BG_file_name_master)
print("BG_file_name_data = " + BG_file_name_data)
print("")

u3=BG_file_name_master.rfind('_T_')
u4=BG_file_name_master.rfind('_n_')
u5=BG_file_name_master.rfind('_Y_')
u6=BG_file_name_master.rfind('_Z_')
u9=BG_file_name_master[u6+len('_Z_'):len(BG_file_name_master)].find('_')+u6+len('_Z_')
u10=BG_file_name_master[u9+len('_'):len(BG_file_name_master)].find('_')+u9+len('_')

BG_T=BG_file_name_master[u3+len('_T_'):u4]
BG_n=BG_file_name_master[u4+len('_n_'):u5]
BG_Y=BG_file_name_master[u5+len('_Y_'):u6]
BG_Z=BG_file_name_master[u6+len('_Z_'):u9]
BG_n_EIGER=BG_file_name_master[u9+len('_'):u10]

print("BG_T = " + BG_T)
print("BG_n = " + BG_n)
print("BG_Y = " + BG_Y)
print("BG_Z = " + BG_Z)
print("BG_n_EIGER = " + BG_n_EIGER)
print("")

#read & write Sample
#list(f.keys())
f=h5py.File(file_name_data,"r")
fimg=np.array(f['entry/data/data'])
fimg_90=np.rot90(fimg,axes=(1,2))
fimg_9090=np.rot90(fimg_90,axes=(1,2))
fimg_flip=np.flip(fimg_9090,axis=2)

fimg_float=fimg_flip.astype(np.float32)

basename=os.path.basename(file_name_master)
with mrcfile.new(dir + "/" + basename[0:len(basename)-10] + '.mrc', overwrite=True) as mrc:
	mrc.set_data(fimg_float)

mrc.close

fimg_sum=np.sum(fimg,axis=0)
fimg_sum_float=fimg_sum.astype(np.float32)
tifffile.imsave(dir + "/" + basename[0:len(basename)-10] +"_sum.tif" ,fimg_sum_float)

f.close()

#read & write BG

f=h5py.File(BG_file_name_data,"r")
fimg=np.array(f['entry/data/data'])
fimg_90=np.rot90(fimg,axes=(1,2))
fimg_9090=np.rot90(fimg_90,axes=(1,2))
fimg_flip=np.flip(fimg_9090,axis=2)

fimg_float=fimg_flip.astype(np.float32)

basename=os.path.basename(BG_file_name_master)
with mrcfile.new(dir + "/" + basename[0:len(basename)-10]  + '.mrc', overwrite=True) as mrc:
	mrc.set_data(fimg_float)

mrc.close

fimg_sum=np.sum(fimg,axis=0)
BG_fimg_sum_float=fimg_sum.astype(np.float32)
tifffile.imsave(dir + "/" + basename[0:len(basename)-10] +"_sum.tif" ,BG_fimg_sum_float)

f.close()

#BG subtraction

scale_factor=(float(T)*float(n))/(float(BG_T)*float(BG_n))
print("BG scale factor = " + str(scale_factor))
print("")

#saturated pixel

#fimg_sum_float=np.where(fimg_sum_float > 700000.0*float(n)*float(T),0,fimg_sum_float)
#BG_fimg_sum_float=np.where(BG_fimg_sum_float > 700000.0*float(BG_n)*float(BG_T),0,BG_fimg_sum_float)

#

BG_sub=fimg_sum_float-scale_factor*BG_fimg_sum_float
#BG_sub=fimg_sum_float
basename=os.path.basename(file_name_master)
tifffile.imsave(dir + "/" + basename[0:len(basename)-10]+"_sum_BGsub.tif" ,BG_sub)

# flat field & mask

FF=Image.open(flat_field)
np_FF=np.asarray(FF,dtype="float32")

M=Image.open(mask)
np_M=np.asarray(M,dtype="float32")

row=np_M.shape[0]
col=np_M.shape[1]

FF_BG_sub=np_FF*BG_sub
#tifffile.imsave(dir + "/" +  basename[0:len(basename)-10]+"_sum_BGsub_flat_field.tif" ,FF_BG_sub)

M_FF_BG_sub=np.where(np_M==-1.0,np_M,FF_BG_sub*np_M)
#tifffile.imsave(dir + "/" +  basename[0:len(basename)-10]+"_sum_BGsub_flat_field_mask.tif" ,M_FF_BG_sub)

# friedel comp

M_FF_BG_sub_rot=np.rot90(M_FF_BG_sub)
M_FF_BG_sub_rot=np.rot90(M_FF_BG_sub_rot)

base_m1=np.full((3*row,3*col),-1)
base_m1=base_m1.astype(np.float32)
base_m1[row+2*int(dZ)-int(row)+1 : 2*row+2*int(dZ)-int(row)+1 , col+2*int(dY)-int(col)+1 : 2*col+2*int(dY)-int(col)+1]=M_FF_BG_sub_rot
#tifffile.imsave(dir + "/" +  basename[0:len(basename)-10]+"_sum_BGsub_flat_field_mask_base_rot.tif" ,base_m1)

base_m2=np.full((3*row,3*col),-1)
base_m2=base_m2.astype(np.float32)
base_m2[row:2*row,col:2*col]=M_FF_BG_sub
#tifffile.imsave(dir + "/" +  basename[0:len(basename)-10]+"_sum_BGsub_flat_field_mask_base.tif" ,base_m2)

#base_m1=np.where(base_m1 > 1000000000*float(n)*float(T),0,base_m1)
base_comp=np.where(base_m2<0.0,base_m1,base_m2)
#tifffile.imsave(dir + "/" +  basename[0:len(basename)-10]+"_sum_base_comp.tif" ,base_comp)

comp=np.full((row,col),-1)
comp=comp.astype(np.float32)
comp=base_comp[row:2*row,col:2*col]
#comp=np.where(comp>float(saturation_value)*float(T)*float(n),-1,comp)
tifffile.imsave(dir + "/" +  basename[0:len(basename)-10]+"_sum_comp.tif" ,comp)

comp_trim=np.full((int(trim_size),int(trim_size)),-1)
comp_trim=comp_trim.astype(np.float32)
comp_trim=base_comp[row+int(dZ)-int(int(trim_size)/2):row+int(dZ)+int(int(trim_size)/2),col+int(dY)-int(int(trim_size)/2):col+int(dY)+int(int(trim_size)/2)]
tifffile.imsave(dir + "/" +  basename[0:len(basename)-10]+"_sum_comp_trim.tif" ,comp_trim)


#shutil.move(file_name_master,dir)
#shutil.move(file_name_data,dir)
#shutil.move(BG_file_name_master,dir)
#shutil.move(BG_file_name_data,dir)





