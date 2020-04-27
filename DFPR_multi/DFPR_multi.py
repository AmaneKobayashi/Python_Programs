#!~/anaconda3/python.exe
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

#pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
#cp.cuda.set_allocator(pool.malloc)

if((len(sys.argv)==1)):
	print("command:python3 PR_multi_EIGER.py [-diff] [-sup] [-initial_dens] [-iteration] [-header] [-output_interval] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-initial_SW_delta] [-last_SW_ips] [-n_SW_delta] [-additional_iteration] [-OSS_interval] [-DFPR] [-target_size] [-donut_mask]")
	print("HIO only			: [-diff] [-sup] [-initial_dens] '''[-iteration 0]''' [-header] [-additional_iteration]")
	print("HIO-SW			: [-diff] [-sup] [-initial_dens ] [-iteration] [-header] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-initial_SW_delta] [-last_SW_ips] [-n_SW_delta]")
	print("HIO-SW target_size	: [-diff] [-sup] [-initial_dens ] [-iteration] [-header] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-last_SW_ips] [-target_size]")
	print("OSS			: [-diff] [-sup] [-iteration] [-header] [-OSS_interval]")
	print("option			: [-output_interval] [-SW_sup_output] [-complex_constraint] [-DFPR] [-donut_mask] [-target_size] [-memory_transfer]")
	exit()

n_parameter=21
parameter_name_list=[""]*n_parameter
flag_list=[0]*n_parameter

parameter_name_list[0]="-diff"
parameter_name_list[1]="-sup"
parameter_name_list[2]="-initial_dens"
parameter_name_list[3]="-iteration"
parameter_name_list[4]="-header"
parameter_name_list[5]="-output_interval"
parameter_name_list[6]="-SW_interval"
parameter_name_list[7]="-initial_SW_ips"
parameter_name_list[8]="-SW_ips_step"
parameter_name_list[9]="-last_SW_ips"
parameter_name_list[10]="-initial_SW_delta"
parameter_name_list[11]="-additional_iteration"
parameter_name_list[12]="-OSS_interval"
parameter_name_list[13]="-SW_sup_output"
parameter_name_list[14]="-last_SW_delta"
parameter_name_list[15]="-n_SW_delta"
parameter_name_list[16]="-complex_constraint"
parameter_name_list[17]="-DFPR"
parameter_name_list[18]="-target_size"
parameter_name_list[19]="-donut_mask"
parameter_name_list[20]=="-memory_transfer"


diff_flag=0
sup_flag=0
intial_dens_flag=0
iteration_flag=0
header_flag=0
output_interval_flag=0
initial_SW_ips_flag=0
SW_ips_step_flag=0
last_SW_ips_flag=0
initial_SW_delta_flag=0
last_SW_delta_flag=0
n_SW_delta_flag=0
additional_iteration_flag=0
OSS_flag=0
SW_sup_output_flag=0
complex_constraint_flag=0
DFPR_flag=0
target_size_flag=0
donut_mask_flag=0
memory_transfer_flag=0

input_parameter=0

for i in range(len(sys.argv)):
	if(sys.argv[i]=="-diff"):
		finame=sys.argv[i+1]
		diff_flag=1
		flag_list[0]=1
	if(sys.argv[i]=="-sup"):
		support=sys.argv[i+1]
		sup_flag=1
		flag_list[1]=1
	if(sys.argv[i]=="-initial_dens"):
		initial_dens=sys.argv[i+1]
		intial_dens_flag=1
		flag_list[2]=1
	if(sys.argv[i]=="-iteration"):
		iteration=sys.argv[i+1]
		iteration_flag=1
		flag_list[3]=1
	if(sys.argv[i]=="-header"):
		header=sys.argv[i+1]
		header_flag=1
		flag_list[4]=1
	if(sys.argv[i]=="-output_interval"):
		output_interval=sys.argv[i+1]
		output_interval_flag=1
		flag_list[5]=1
	if(sys.argv[i]=="-SW_interval"):
		SW_interval=sys.argv[i+1]
		SW_interval_flag=1
		flag_list[6]=1
	if(sys.argv[i]=="-initial_SW_ips"):
		initial_SW_ips=sys.argv[i+1]
		initial_SW_ips_flag=1
		flag_list[7]=1
	if(sys.argv[i]=="-SW_ips_step"):
		SW_ips_step=sys.argv[i+1]
		SW_ips_step_flag=1
		flag_list[8]=1
	if(sys.argv[i]=="-last_SW_ips"):
		last_SW_ips=sys.argv[i+1]
		last_SW_ips_flag=1
		flag_list[9]=1
	if(sys.argv[i]=="-initial_SW_delta"):
		initial_SW_delta=sys.argv[i+1]
		initial_SW_delta_flag=1
		flag_list[10]=1
	if(sys.argv[i]=="-additional_iteration"):
		additional_iteration=sys.argv[i+1]
		additional_iteration_flag=1
		flag_list[11]=1
	if(sys.argv[i]=="-OSS_interval"):
		OSS_interval=sys.argv[i+1]
		OSS_flag=1
		flag_list[12]=1
	if(sys.argv[i]=="-SW_sup_output"):
		SW_sup_output_flag=1
		flag_list[13]=1
	if(sys.argv[i]=="-last_SW_delta"):
		last_SW_delta=sys.argv[i+1]
		last_SW_delta_flag=1
		flag_list[14]=1
	if(sys.argv[i]=="-n_SW_delta"):
		n_SW_delta=sys.argv[i+1]
		n_SW_delta_flag=1
		flag_list[15]=1
	if(sys.argv[i]=="-complex_constraint"):
		complex_constraint_flag=1
		flag_list[16]=1
	if(sys.argv[i]=="-DFPR"):
		DFPR_flag=1
		flag_list[17]=1
	if(sys.argv[i]=="-target_size"):
		target_size=sys.argv[i+1]
		target_size_flag=1
		flag_list[18]=1
	if(sys.argv[i]=="-donut_mask"):
		donut_mask=sys.argv[i+1]
		donut_mask_flag=1
		flag_list[19]=1
	if(sys.argv[i]=="-memory_transfer"):
		memory_transfer_flag=1
		flag_list[20]=1
	if(sys.argv[i]=="--help"):
		print("command:python3 PR_multi_EIGER.py [-diff] [-sup] [-initial_dens] [-iteration] [-header] [-output_interval] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-initial_SW_delta] [-last_SW_ips] [-n_SW_delta] [-additional_iteration] [-OSS_interval] [-DFPR] [-target_size] [-donut_mask]")
		print("HIO only			: [-diff] [-sup] [-initial_dens] '''[-iteration 0]''' [-header] [-additional_iteration]")
		print("HIO-SW			: [-diff] [-sup] [-initial_dens ] [-iteration] [-header] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-initial_SW_delta] [-last_SW_ips] [-n_SW_delta]")
		print("HIO-SW target_size	: [-diff] [-sup] [-initial_dens ] [-iteration] [-header] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-last_SW_ips] [-target_size]")
		print("OSS			: [-diff] [-sup] [-iteration] [-header] [-OSS_interval]")
		print("option			: [-output_interval] [-SW_sup_output] [-complex_constraint] [-DFPR] [-donut_mask] [-target_size] [-memory_transfer]")
		exit()

if(OSS_flag!=1):
	if(iteration=="0"):
		#HIO only
		for i in range(n_parameter):
			if((flag_list[i]==0) & (i!=12) & (i!=5) & (i!=6) & (i!=7) & (i!=8) & (i!=9) & (i!=10) & (i!=13) & (i!=14) & (i!=15) & (i!=16) & (i!=17) & (i!=18) & (i!=19) & (i!=20)):
				if(input_parameter==0):
					print("HIO only mode")
				print("please input parameter : [" + parameter_name_list[i] + "]")
				input_parameter=1
		if(input_parameter==1):
			exit()
	else:
		#HIO-SW
		if(target_size_flag == 1):
			for i in range(n_parameter):
				if((flag_list[i]==0) & (i!=1) & (i!=12) & (i!=5) & (i!=10) & (i!=13) & (i!=14) & (i!=15) & (i!=16) & (i!=17) & (i!=19) & (i!=20)):
					if(input_parameter==0):
						print("HIO-SW target_size mode")
					print("please input parameter : [" + parameter_name_list[i] + "]")
					input_parameter=1

		elif(donut_mask_flag == 1):
			for i in range(n_parameter):
				if((flag_list[i]==0) & (i!=1) &  (i!=12) & (i!=5) & (i!=10) & (i!=13) & (i!=14) & (i!=15) & (i!=16) & (i!=17) & (i!=18) & (i!=20)):
					if(input_parameter==0):
						print("HIO-SW target_size mode")
					print("please input parameter : [" + parameter_name_list[i] + "]")
					input_parameter=1
		else:
			for i in range(n_parameter):
				if((flag_list[i]==0) & (i!=1) &  (i!=12) & (i!=5) & (i!=13) & (i!=16) & (i!=17) & (i!=18) & (i!=19) & (i!=20)):
					if(input_parameter==0):
						print("HIO-SW mode")
					print("please input parameter : [" + parameter_name_list[i] + "]")
					input_parameter=1
		if(input_parameter==1):
			exit()
else:
	#OSS
	for i in range(n_parameter):
		if((flag_list[i]==0) & (i!=5) & (i!=6) & (i!=7) & (i!=8) & (i!=9) &(i!=10) & (i!=13) & (i!=14) & (i!=15) & (i!=16) & (i!=17) & (i!=18) & (i!=19) & (i!=20)):
			if(input_parameter==0):
				print("OSS mode")
			print("please input parameter : [" + parameter_name_list[i] + "]")	
			input_parameter=1
	if(input_parameter==1):
		exit()

#fixed parameters
target_area=200			#評価する暗視野自己相関関数の領域サイズ(pix)
small_angle_area=80		#極小角領域の定義(pix)
average_distance_limit=50	#暗視野自己相関関数の中心間平均距離の閾値。この値より大きい場合は位相回復をしない。(pix)
saturation_limit=800		#飽和ピクセルの上限値。これを超えたら位相回復をしない
std_limit=13.0			#標準偏差/平均の上限値。ズビシ判定

#

print("diff = " + finame)
if(donut_mask_flag == 1):
	print("donut_mask = " + donut_mask)
if(sup_flag==1):
	print("support = " + support)
print("initial_dens = " + initial_dens)
print("iteration = " + iteration)
print("header = " + header)
if(flag_list[5]==0):
	output_interval=10000000000
	print("final output only")
else:
	print("output_interval = " +output_interval)
if(OSS_flag!=1):
	if(iteration=="0"):
		print("additional_iteration = " + additional_iteration)
		print("HIO only mode")
	else:
		print("SW_interval = " + SW_interval)
		print("initial_SW_ips = " + initial_SW_ips)
		print("SW_ips_step = " + SW_ips_step)
		print("last_SW_ips = " + last_SW_ips)
		print("additional_iteration = " + additional_iteration)
		if((target_size_flag != 1) & (donut_mask_flag != 1)):
			print("initial_SW_delta = " + initial_SW_delta)
			print("last_SW_delta = " + last_SW_delta)
			print("n_SW_delta = " + n_SW_delta)
			print("HIO-SW mode")
		else:
			if(target_size_flag == 1):
				print("target_size = " + target_size)
				print("HIO-SW target_size mode")
			if(donut_mask_flag == 1):
				print("HIO-SW donut_mask mode")
else:	
	print("OSS mode")
print("")

#ディレクトリ作成
if(header.rfind("/")==-1):
	os.makedirs(header, exist_ok=True)
else:
	os.makedirs(header[0:header.rfind("/")], exist_ok=True)
if(header.rfind("/") == -1):
	header=header + "/" +header	
	log_path=header + "_MPR.log"
else:
	header=header + "/" +header[header.rfind("/")+1:len(header)]

#log file作成
log_path=header + "_MPR.log"
with open(log_path, mode='w') as log:
	log.write("program Phase retrieval multi ver.20191121")

# open mrc file

with mrcfile.open(initial_dens, permissive=True) as mrc:
	cp_dens=cp.asarray(mrc.data,dtype="float32")
mrc.close

if(sup_flag==1):
	if(support.find("tif") != 0):
		sup=Image.open(support)
		np_sup=np.asarray(sup,dtype="float32")
		cp_sup=[np_sup]*int(cp_dens.shape[0])
		cp_sup=cp.asarray(cp_sup,dtype="float32")
		cp_sup=cp.flip(cp_sup,axis=1)
		print("np_sup dtype = " + str(np_sup.dtype))
		print("np_sup shape = " + str(np_sup.shape))
		print("cp_sup shape = " + str(cp_sup.shape))
		
	else:
		with mrcfile.open(support, permissive=True) as mrc:
			cp_sup=cp.asarray(mrc.data,dtype="float32")
		mrc.close
		print("cp_sup dtype = " + str(cp_sup.dtype))
		print("cp_sup shape = " + str(cp_sup.shape))

	print("cp_initial_dens dtype = " + str(cp_dens.dtype))
	print("cp_initial_dens shape = " + str(cp_dens.shape))
	print("")

	if(cp_sup.shape != cp_dens.shape):
		print("shape of support and initial_density file is different !")
		exit()

sta_dens=cp_dens.shape[0]
print("sta_dens of diff (n_trial) = " + str(sta_dens))
print("")

# open diffraction pattern

diff=Image.open(finame)
np_diff=np.asarray(diff,dtype="float32")
row=np_diff.shape[0]
col=np_diff.shape[1]
print("np_diff dtype = " + str(np_diff.dtype))
print("np_diff shape = " + str(np_diff.shape))
print("row of diff = " + str(row))
print("col of diff = " + str(col))

if(target_size_flag==1):
	#飽和ピクセル数	
	np_zero_pixel_array=np.zeros((row,col),dtype="float32")
	np_zero_pixel_array=np.where(np_diff==0.0,1.0,0.0)
	num_zero=np.sum(np_zero_pixel_array[int((row-small_angle_area)/2):int((row+small_angle_area)/2),int((col-small_angle_area)/2):int((col+small_angle_area)/2)])
	print("zero pixel : " + str(num_zero))
	
	if(num_zero>saturation_limit):
		with open(log_path, mode='a') as log:
			log.write("\n" + "saturation limit exceed : " + str(num_zero))
		exit()
	
	#標準偏差/平均
	num_zero=np.sum(np_zero_pixel_array[:,:])
	average_intensity=np.sum(np_diff[:,:])/(row*col-num_zero)
	np_std_array=np.zeros((row,col),dtype="float32")
	np_std_array=np.where(np_diff!=0.0,np.square(np_diff-average_intensity),0.0)
	std=np.sqrt(np.sum(np_std_array)/(row*col-num_zero))
	std_ave=std/average_intensity
	print("std_ave : " + str(std_ave))

	if(std_ave>std_limit):
		with open(log_path, mode='a') as log:
			log.write("\n" + "std limit exceed : " + str(std_ave))
		exit()

	#donut_target dicision

	temp_rad=np.zeros((row,col))		
	temp_rad=temp_rad.astype(int)
	rad_x=np.zeros((int((row+col)/2), 2*int(np.pi*(row+col)/2)))
	rad_y=np.zeros((int((row+col)/2), 2*int(np.pi*(row+col)/2)))
	rad_num=np.zeros(int((row+col)/2))
	rad_x=rad_x.astype(int)
	rad_y=rad_y.astype(int)
	rad_num=rad_num.astype(int)
	for x in range(row):
		for y in range(col):
			temp_rad[x,y]=int(np.sqrt(np.square(x-row/2)+np.square(y-col/2)))
			rad_num[int(temp_rad[x,y])]=rad_num[int(temp_rad[x,y])]+1
			rad_x[int(temp_rad[x,y]),rad_num[int(temp_rad[x,y])]]=x
			rad_y[int(temp_rad[x,y]),rad_num[int(temp_rad[x,y])]]=y
		
	max_index=int(np.max(temp_rad))
	rad_num=np.zeros(max_index+1)
	rad_num=rad_num.astype(np.float32)
	rad_value=np.zeros(max_index+1)
	rad_value=rad_value.astype(np.float32)
	for x in range(row):
		for y in range(col):
			rad_num[int(temp_rad[x,y])]=rad_num[int(temp_rad[x,y])]+1.0
			rad_value[int(temp_rad[x,y])]=rad_value[int(temp_rad[x,y])]+np_diff[x,y]
	rad_diff=rad_value[:]/rad_num[:]

	donut_position=50
	SN_threshold=25.0
	i=0
	while(rad_diff[max_index-i]<SN_threshold):
		i=i+1
	donut_position=max_index-i
	print("donut_position = " + str(donut_position))
	if(donut_position>250):
		donut_position=150
	print("donut_position = " + str(donut_position))

	donut_bank_name="C:\Python_Programs\donut_FWHM050.mrc"
	with mrcfile.open(donut_bank_name, permissive=True) as mrc:
		donut_bank=np.asarray(mrc.data,dtype="float32")
	mrc.close

	np_donut_target=donut_bank[donut_position,:,:]
	#print(np_donut_target.shape)
	
	#donut_target=Image.open('/home/amanekobayashi/work/donut_mask/donut_R200_FWHM050.tif')
	#np_donut_target=np.asarray(donut_target,dtype="float32")
	cp_donut_target=cp.asarray(np_donut_target,dtype="float32")
	
	#欠損領域を円環平均の値で埋める　←　自己相関関数でギブズフリンジを低減するため。
	#np_diff_rad=np.zeros(np_diff.shape,dtype="float32")
	#for x in range(row):
	#	for y in range(col):
	#		if(np_diff[x,y]<=0.0):
	#			np_diff_rad[x,y]=rad_diff[int(temp_rad[x,y])]
	#		else:
	#			np_diff_rad[x,y]=np_diff[x,y]
	#foname=header + "_diff_rad.tif"
	#tifffile.imsave(foname ,np_diff_rad)
	
	#欠損領域を一番近いピクセルの値で埋める
	beam_stop_value=np.max(np_diff)
	print("beam_stop_value = " + str(beam_stop_value))

	np_diff_rad=np.zeros(np_diff.shape,dtype="float32")
	for x in range(row):
		for y in range(col):
			if(np_diff[x,y]==0.0):
				rad_distance=np.full(2*int(np.pi*(row+col)/2),100000.0)
				if(rad_diff[int(temp_rad[x,y])] != 0.0):
					for i in range(2*int(np.pi*(row+col)/2)):
						if((rad_x[int(temp_rad[x,y]),i] != 0) & (rad_y[int(temp_rad[x,y]),i] != 0) & (np_diff[rad_x[int(temp_rad[x,y]),i],rad_y[int(temp_rad[x,y]),i]] != 0)):
							rad_distance[i]=np.sqrt(np.square(x-rad_x[int(temp_rad[x,y]),i])+np.square(y-rad_y[int(temp_rad[x,y]),i]))
					i_min=np.argmin(rad_distance)
		#			print(i_min,rad_x[int(temp_rad[x,y]),i_min],rad_y[int(temp_rad[x,y]),i_min],np_diff[rad_x[int(temp_rad[x,y]),i_min],rad_y[int(temp_rad[x,y]),i_min]])
					np_diff_rad[x,y]=np_diff[rad_x[int(temp_rad[x,y]),i_min],rad_y[int(temp_rad[x,y]),i_min]]
				elif((rad_diff[int(temp_rad[x,y])] == 0.0) & (x > (row-small_angle_area)/2) & (x < (row+small_angle_area)/2) & (y > (col-small_angle_area)/2) & (y < (col+small_angle_area)/2)):
					np_diff_rad[x,y]=beam_stop_value
			else:
				np_diff_rad[x,y]=np_diff[x,y]
	
	foname=header + "_diff_rad.tif"
	tifffile.imsave(foname ,np_diff_rad)

	#
	np_diff_donut=np_diff_rad * np.square(np_donut_target)
	foname=header + "_donut_target_diff.tif"
	tifffile.imsave(foname ,np_diff_donut)
	
	if(donut_mask_flag == 1):
		donut=Image.open(donut_mask)
		np_donut=np.asarray(donut,dtype="float32")
		row_donut=np_donut.shape[0]
		col_donut=np_donut.shape[1]
	
		print("np_donut dtype = " + str(np_donut.dtype))
		print("np_donut shape = " + str(np_donut.shape))
		print("row of donut = " + str(row_donut))
		print("col of donut = " + str(col_donut))
	
		if((row != row_donut) | (col != col_donut)):
			print("check dim of donut_mask")
			exit()
		
		if(DFPR_flag == 1):
			np_diff=np_diff * np.square(np_donut)
	
np_diff=np.where(np_diff>0.0,np_diff,0.0)
np_diff=np.sqrt(np_diff)

if(target_size_flag == 1):
	if(donut_mask_flag == 1):
		small_angle_mask=Image.open('C:\Python_Programs\small_angle_mask.tif')
		np_small_angle_mask=np.asarray(small_angle_mask,dtype="float32")
		np_diff=np.where((np_small_angle_mask==0.0) & (np_diff==0.0),-1.0,np_diff)

cp_diff_amp=[np_diff]*int(sta_dens)
cp_diff_amp=cp.asarray(cp_diff_amp,dtype="float32")
cp_diff_amp=cp.flip(cp_diff_amp,axis=1)
cp_diff_amp=cp.fft.ifftshift(cp_diff_amp, axes=(1,2))

if(target_size_flag == 1):
	if(target_size.find(".tif") != -1):
		erode_pre=Image.open(target_size)
		erode=np.asarray(erode_pre,dtype="float32")
		target_size=str(np.sum(erode))
		print("target_size = " + target_size)
		d_kernel=np.ones((10,10), np.float32)
	else:
		erode=np.zeros(np_diff.shape)
		erode=np.asarray(erode,dtype="float32")
		d_kernel=np.ones((10,10), np.float32)

if(donut_mask_flag == 1):
	foname=header + "_donut_diff.tif"
	tifffile.imsave(foname ,np_diff)

if(target_size_flag == 1):
	#暗視自己相関関数の計算
	cp_diff=cp.asarray(np_diff_donut,dtype="float32")
	
	cp_autocorrelation = cp.fft.fft2(cp_diff, norm="ortho")
	cp_autocorrelation = cp.fft.fftshift(cp_autocorrelation)
	
	cp_autocorrelation_abs=cp.absolute(cp_autocorrelation)
	del cp_autocorrelation
	np_autocorrelation_abs=cp.asnumpy(cp_autocorrelation_abs)

	foname=header + "_autocorrelation.tif"
	tifffile.imsave(foname ,np_autocorrelation_abs)

	#TH_mode="OTSU"
	TH_mode="keiken"

	if(TH_mode=="keiken"):
		ave_autocorrelation=cp.average(cp_autocorrelation_abs)
		std_autocorrelation=cp.std(cp_autocorrelation_abs)
	
		th = ave_autocorrelation + std_autocorrelation
		print("threshold = " + str(th))
	elif(TH_mode=="OTSU"):
		max_autocorrelation=np.max(np_autocorrelation_abs*10)
		np_autocorrelation_abs_uint8=np.asarray(np_autocorrelation_abs*10, dtype=np.uint8)
		ret2, np_autocorrelation_abs_otsu = cv2.threshold(np_autocorrelation_abs_uint8, 0, int(max_autocorrelation), cv2.THRESH_OTSU)
		th=float(format(ret2))/10.0
		print("threshold = " + str(th))

	cp_th=cp.where(cp_autocorrelation_abs>=th,float(1),float(0))
	cp_th=cp_th.astype(cp.float32)
	np_th=cp.asnumpy(cp_th)

	foname=header + "_threshold.tif"
	tifffile.imsave(foname ,np_th)

	#FWHM=donut_mask[donut_mask.rfind("FWHM")+4:donut_mask.rfind(".tif")]
	FWHM="050"
	print("FWHM : " + FWHM)

	max_autocorrelation=cp.max(cp_autocorrelation_abs)
	Wgauss= cp.log(max_autocorrelation/th) * cp.log(2) 
	Wgauss=cp.sqrt(Wgauss)
	Wgauss=Wgauss/cp.pi
	Wgauss=float(4) * Wgauss / float(FWHM)
	Wgauss=Wgauss * float(col)
	print("Wgauss = " + str(Wgauss))
	Wgauss=10.0
	
	d_kernel=np.ones((int(Wgauss),int(Wgauss)), np.uint8)
	for x in range(int(Wgauss)):
		for y in range(int(Wgauss)):
			d=np.sqrt(np.square(x-Wgauss/2) + np.square(y-Wgauss/2))
			if(d>=Wgauss/2):
				d_kernel[x,y]=0
	d_kernel=np.asarray(d_kernel, dtype=np.uint8)
	d_kernel2=np.ones((2,2), np.uint8)
	d_kernel2=np.asarray(d_kernel2, dtype=np.uint8)

	erode=cv2.erode(np_th,d_kernel2)
	dilate=cv2.dilate(erode,d_kernel)
	erode=cv2.erode(dilate,d_kernel)

	#暗視野自己相関関数の平均距離の算出
	a=1
	distance=0.0
	for n in range(row):
		for nn in range(col):
			if(erode[n,nn]>0.0):
				distance=distance+np.sqrt(np.square(n-col/2)+np.square(nn-row/2))
				a=a+1
	average_distance=distance/np.float(a)
	print("average_distance of autocorrelation(1) : " + str(average_distance))
	
	#平均距離が指定値より大きい場合は二値化の閾値を見直してやり直す
	average_distance_retry_flag=0
	if(average_distance>average_distance_limit):
		th = ave_autocorrelation + 2*std_autocorrelation
		cp_th=cp.where(cp_autocorrelation_abs>=th,float(1),float(0))
		cp_th=cp_th.astype(cp.float32)
		np_th=cp.asnumpy(cp_th)
	
		erode=cv2.erode(np_th,d_kernel2)
		dilate=cv2.dilate(erode,d_kernel)
		erode=cv2.erode(dilate,d_kernel)
	
		a=1
		distance=0.0
		for n in range(row):
			for nn in range(col):
				if(erode[n,nn]>0.0):
					distance=distance+np.sqrt(np.square(n-col/2)+np.square(nn-row/2))
					a=a+1
		average_distance=distance/np.float(a)
		print("average_distance of autocorrelation(2) : " + str(average_distance))
		average_distance_retry_flag=1

	#foname=header + "_dilate.tif"
	#tifffile.imsave(foname ,dilate)	
	foname=header + "_erode.tif"
	tifffile.imsave(foname ,erode)

	#
if(sup_flag!=1):
	np_sup=erode
	cp_sup=[np_sup]*int(cp_dens.shape[0])
	cp_sup=cp.asarray(cp_sup,dtype="float32")	

if((target_size_flag != 1) & (initial_SW_delta_flag != 1) & (initial_SW_ips_flag == 1)):	
	i_target_size = np.sum(erode[int((col-target_area)/2):int((col+target_area)/2),int((row-target_area)/2):int((row+target_area)/2)])
	target_size = str(i_target_size)
	print("target_size = " + target_size)
if((target_size_flag==1)):
	if((target_size=="")):
		i_target_size = np.sum(erode[int((col-target_area)/2):int((col+target_area)/2),int((row-target_area)/2):int((row+target_area)/2)])
		target_size = str(i_target_size)
		print("target_size = " + target_size)
if((target_size_flag==1)):
	if((target_size[0]=="-")):
		i_target_size = np.sum(erode[int((col-target_area)/2):int((col+target_area)/2),int((row-target_area)/2):int((row+target_area)/2)])
		target_size = str(i_target_size)
		print("target_size = " + target_size)

log_text="\n\n" + "diff = " + finame + "\n"
if(sup_flag==1):
	log_text=log_text + "support = " + support + "\n"
else:
	log_text=log_text + "support = " + header + "_erode.tif (target)" + "\n"

log_text=log_text + "initial_dens = " + initial_dens + "\n" + "iteration = " + iteration + "\n" + "header = " + header

if(flag_list[5]==0):
	log_text=log_text + "\n" + "output_interval = final output only"
else:
	log_text=log_text + "\n" + "output_interval = " + output_interval
if(OSS_flag!=1):
	if(iteration=="0"):
		log_text=log_text + "\n" + "additional_iteration = " + additional_iteration + "\n"
		log_text=log_text + "\n" + "HIO only mode\n"
	else:
		log_text=log_text + "\n" + "SW_interval = " + SW_interval + "\n" + "initial_SW_ips = " + initial_SW_ips + "\n" + "SW_ips_step = " + SW_ips_step + "\n" + "last_SW_ips = " + last_SW_ips
		log_text=log_text + "\n" + "additional_iteration = " + additional_iteration + "\n"
		if((target_size_flag == 1) | (donut_mask_flag == 1)):
			log_text=log_text + "\n" + "HIO-SW target_size mode : " + target_size + "\n"
			if(average_distance_retry_flag==0):
				log_text=log_text + "average_distance_autocorrelation(1) : " + str(average_distance) +"\n"
			else:
				log_text=log_text + "average_distance_autocorrelation(2) : " + str(average_distance) +"\n"				
			log_text=log_text + "target_area : " + str(target_area) +"\n"
			log_text=log_text + "number of zero pixel in small angle area : " + str(num_zero) +"\n"
			log_text=log_text + "standard deviation / average intensity : " + str(std_ave) +"\n"
		else:
			log_text=log_text + "\n" + "initial_SW_delta = " + initial_SW_delta + "\n" + "last_SW_delta = " + last_SW_delta + "\n" + "n_SW_delta = " + n_SW_delta
			log_text=log_text + "\n" + "HIO-SW mode\n"
else:
	log_text=log_text + "\nOSS mode\n"

log_text=log_text + "diff shape = " + str(np_diff.shape) +"\n" 

if(donut_mask_flag == 1):
	log_text=log_text + "donut_mask = " + donut_mask + "\n"
	if(target_size_flag != 1):
		log_text=log_text + "target_size = " + target_size + "\n"

if(complex_constraint_flag == 1):
	log_text=log_text + "complex_constraint_mode" +"\n"
	print("complex_constraint_mode")
if(DFPR_flag == 1):
	log_text=log_text + "DFPR_mode (non positive constraint)" +"\n"	
	print("DFPR mode")

if((target_size_flag==1)):
	log_text=log_text + "donut_target position = " + str(donut_position) + "\n"
	log_text=log_text + "SN_threshold = " + str(SN_threshold) + "\n"
	log_text=log_text + "sta_dens of diff (n_trial) = " + str(sta_dens) +"\n\n"

if(memory_transfer_flag==1):
	log_text=log_text + "memory_transfer mode" + "\n"

with open(log_path, mode='a') as log:
	log.write(log_text)

if((target_size_flag==1)):
	if(average_distance>50.0):
		log_text=log_text + "average distance of autocorrelation is larger than 50 pix"
		with open(log_path, mode='a') as log:
			log.write(log_text)
		exit()
#
if(iteration!="0"):
	if((initial_SW_ips_flag ==1 ) & (SW_ips_step_flag ==1 ) & (last_SW_ips_flag ==1 ) & (initial_SW_delta_flag ==1 ) & (last_SW_delta_flag ==1 ) & (n_SW_delta_flag ==1 )):
		SW_delta=[0.0]*int(sta_dens)
		SW_delta=cp.asarray(SW_delta,dtype="float32")
		SW_delta_step=(float(last_SW_delta)-float(initial_SW_delta))/(float(n_SW_delta)-1.0)
		#print("SW_delta_step = " + str(SW_delta_step))
		sta_dens_step=int(int(sta_dens)/int(n_SW_delta))
		#print("sta_dens_step = " + str(sta_dens_step))
	
		for i in range(int(n_SW_delta)):
			SW_delta[i*sta_dens_step:(i+1)*sta_dens_step]=float(initial_SW_delta)+float(i)*SW_delta_step

#

t1=time.time()

#電子密度の複素数化

cp_dens=cp.array(cp_dens, dtype=cp.complex64)

#SW用ガウシアンの定義
if((iteration!="0") & (OSS_flag==0)):
	def G(x,y,ips):
		Z=cp.exp(-((x-row/2)**2 + (y-col/2)**2) / (2 * ips**2)) / (2 * cp.pi * ips**2)
		return Z
	x=cp.arange(0,int(row),1)
	y=cp.arange(0,int(col),1)
	X,Y=cp.meshgrid(x,y)
	SW_ips=float(initial_SW_ips)

#OSS用マスクの定義
if(OSS_flag==1):
	def W(x,y,alpha):
		M=cp.exp(-0.5*((x-row/2)**2 + (y-col/2)**2 )/alpha**2)
		return M
	x=cp.arange(0,int(row),1)
	y=cp.arange(0,int(col),1)
	X,Y=cp.meshgrid(x,y)
	OSS_alpha=float(row)
	OSS_alpha_step=(float(row)-1.0/float(row)) / (float(iteration)/float(OSS_interval)-1.0)

#

R_factor=[0.0]*sta_dens
gamma=[0.0]*sta_dens
OS_ratio=[0.0]*sta_dens
scale_factor=[0.0]*sta_dens

#
if(memory_transfer_flag==1):
	print("memory_transfer mode")

#start
if((target_size_flag == 1) | (donut_mask_flag == 1)):
	print("trial iteration scale_factor Rfactor OS_ratio gamma NOR_ex")
	with open(log_path, mode='a') as log:
		log.write("trial iteration scale_factor Rfactor OS_ratio gamma NOR_ex")
else:
	print("trial iteration scale_factor Rfactor OS_ratio gamma")
	with open(log_path, mode='a') as log:
		log.write("trial iteration scale_factor Rfactor OS_ratio gamma")

for i in range(int(iteration)+int(additional_iteration)):
	gc.collect()
#	print(i+1)

#	subprocess.run(["nvidia-smi"])
	
	cp_structure_factor = cp.fft.fftn(cp_dens, axes=(1,2), norm="ortho")#【フーリエ変換】
	cp_amp = cp.absolute(cp_structure_factor)#絶対値をとる
	
	#逆空間拘束

	if(donut_mask_flag==1):
		cp_structure_factor.real=cp.where((cp_diff_amp!=-1.0) & (cp_amp!=0.0),cp_structure_factor.real*cp_diff_amp/cp_amp,cp_structure_factor.real)
		cp_structure_factor.imag=cp.where((cp_diff_amp!=-1.0) & (cp_amp!=0.0),cp_structure_factor.imag*cp_diff_amp/cp_amp,cp_structure_factor.imag)
	else:
		cp_structure_factor.real=cp.where((cp_diff_amp>0.0) & (cp_amp!=0.0),cp_structure_factor.real*cp_diff_amp/cp_amp,cp_structure_factor.real)
		cp_structure_factor.imag=cp.where((cp_diff_amp>0.0) & (cp_amp!=0.0),cp_structure_factor.imag*cp_diff_amp/cp_amp,cp_structure_factor.imag)
#	subprocess.run(["nvidia-smi"])
	cp_dens_pre = cp.fft.ifftn(cp_structure_factor, axes=(1,2),norm="ortho")
	if(memory_transfer_flag==1):
		cp_structure_factor = cp.asnumpy(cp_structure_factor)#cupy配列 ⇒ numpy配列に変換
		cp_dens_pre = cp.asnumpy(cp_dens_pre)#cupy配列 ⇒ numpy配列に変換

	#Shrink Wrap
	
	if((OSS_flag!=1) & (iteration != 0)):
		if((i+1) <= int(iteration)):
			if((i+1) % int(SW_interval) == 0):

				if(float(SW_ips_step)>=1.0):
					if(SW_ips>=float(last_SW_ips)):
						SW_ips=float(last_SW_ips)
				if(float(SW_ips_step)<1.0):
					if(SW_ips<=float(last_SW_ips)):
						SW_ips=float(last_SW_ips)
				
#				print("SW_ips = " + str(SW_ips))

				G_kernel=G(X,Y,SW_ips)
				G_kernel=cp.asarray(G_kernel,dtype="float32")
				G_kernel=G_kernel/cp.amax(G_kernel)
	
				G_kernel=cp.fft.ifftshift(G_kernel)

				if(DFPR_flag == 1):
					if(memory_transfer_flag==1):
						cp_dens_pre = cp.asarray(cp_dens_pre, dtype=cp.complex64)#numpy配列に変換 ⇒ cupy配列 
					cp_dens_pre_abs=cp.absolute(cp_dens_pre)
					if(memory_transfer_flag==1):
						cp_dens_pre = cp.asnumpy(cp_dens_pre)#cupy配列 ⇒ numpy配列に変換
					cp_structure_factor_abs = cp.fft.fftn(cp_dens_pre_abs, axes=(1,2),norm="ortho")
					cp_structure_factor_abs.real=G_kernel*cp_structure_factor_abs.real
					cp_structure_factor_abs.imag=G_kernel*cp_structure_factor_abs.imag
					G_dens = cp.fft.ifftn(cp_structure_factor_abs, axes=(1,2),norm="ortho")		
				else:
					if(memory_transfer_flag==1):
						cp_structure_factor = cp.asarray(cp_structure_factor, dtype=cp.complex64)#numpy配列に変換 ⇒ cupy配列 	
					cp_structure_factor.real=G_kernel*cp_structure_factor.real
					cp_structure_factor.imag=G_kernel*cp_structure_factor.imag
				
					G_dens = cp.fft.ifftn(cp_structure_factor, axes=(1,2),norm="ortho")
					if(memory_transfer_flag==1):
						cp_structure_factor = cp.asnumpy(cp_structure_factor)#cupy配列 ⇒ numpy配列に変換

				if(initial_SW_delta_flag==1):
					if(DFPR_flag == 1):
						threshold = SW_delta*cp.amax(cp.abs(G_dens), axis=(1,2))
					else:
						threshold = SW_delta*cp.amax(cp.real(G_dens), axis=(1,2))
					threshold_3D=cp.repeat(threshold,int(col)*int(row))
					threshold_3D=threshold_3D.reshape(sta_dens,int(row),int(col))
#					print("normal SW")
	
					if(DFPR_flag == 1):
						cp_sup=cp.where(cp.absolute(G_dens)>=threshold_3D,float(1),float(0))
					else:
						cp_sup=cp.where(cp.real(G_dens)>=threshold_3D,float(1),float(0))
				if((initial_SW_delta_flag!=1) & ((target_size_flag == 1) | (donut_mask_flag == 1))):
#					print("delta free SW")
					if(DFPR_flag == 1):
						e_G_dens=cp.absolute(G_dens)
					else:
						e_G_dens=cp.real(G_dens)
					max_array=cp.amax(e_G_dens, axis=(1,2))

					SW_mode="S4"
#					SW_mode="AC"

					if(SW_mode=="S4"):
						e_target_size=float(target_size)/float(4)

						for n in range(sta_dens):
							div=max_array[n]/float(100)
							threshold=div*float(100-1)
							cp_sup_2D=cp.where(e_G_dens[n,:,:]>=threshold,float(1),float(0))
							size_sup_2D=cp.sum(cp_sup_2D)
							while (size_sup_2D <= e_target_size):
								threshold=threshold-div
								cp_sup_2D=cp.where(e_G_dens[n,:,:]>=threshold,float(1),float(0))
								size_sup_2D=cp.sum(cp_sup_2D)
								
							div=div/10
							while (size_sup_2D >= e_target_size):
								threshold=threshold+div
								cp_sup_2D=cp.where(e_G_dens[n,:,:]>=threshold,float(1),float(0))
								size_sup_2D=cp.sum(cp_sup_2D)							
							cp_sup[n,:,:]=cp_sup_2D[:,:]
					elif(SW_mode=="AC"):							
						cp_autocorrelation = cp.fft.fft2(e_G_dens, axes=(1,2), norm="ortho")
						cp_autocorrelation = cp.fft.fftshift(cp_autocorrelation)	
						cp_autocorrelation_abs=cp.absolute(cp_autocorrelation)
						max_array_AC=cp.amax(cp_autocorrelation_abs, axis=(1,2))

						for n in range(sta_dens):
							div=max_array_AC[n]/float(100000000)
							threshold=div
							cp_sup_AC=cp.where(cp_autocorrelation_abs[n,:,:]>=threshold,float(1),float(0))						
							size_sup_2D=cp.sum(cp_sup_AC)
							while (size_sup_2D >= float(target_size)):
								threshold=threshold+div
								cp_sup_AC=cp.where(cp_autocorrelation_abs[n,:,:]>=threshold,float(1),float(0))
								size_sup_2D=cp.sum(cp_sup_AC)
							cp_sup_2D=cp.where(e_G_dens[n,:,:]>=max_array[n]*cp.sqrt(threshold/max_array_AC[n]),float(1),float(0))						
							cp_sup[n,:,:]=cp_sup_2D[:,:]
						print(threshold,size_sup_2D,cp.sum(cp_sup_2D))

				cp_sup=cp_sup.astype(cp.float32)				
		
				SW_ips=SW_ips*float(SW_ips_step)

				if(SW_sup_output_flag==1):
					cp_sup = cp.asnumpy(cp_sup)
					with mrcfile.new(header + "_" + str(i+1).zfill(6) + '_sup.mrc', overwrite=True) as mrc:
						mrc.set_data(cp_sup)
					mrc.close
					cp_sup=cp.asarray(cp_sup,dtype="float32")

	#電子密度の出力

	if((i+1) % int(output_interval) == 0):
		cp_dens_pre = cp.asnumpy(cp_dens_pre)#cupy配列 ⇒ numpy配列に変換
		with mrcfile.new(header + "_" + str(i+1).zfill(6) + '_rdens.mrc', overwrite=True) as mrc:
			mrc.set_data(cp_dens_pre.real)
		mrc.close
		cp_dens_pre = cp.asarray(cp_dens_pre, dtype=cp.complex64)

	#OS比の計算

	if(i+1==int(iteration)+int(additional_iteration)):
		n_pixel=float(row) * float(col)
		OS_ratio = n_pixel / cp.sum(cp_sup, axis=(1,2))
		
	#gammaの計算

	if(i+1==int(iteration)+int(additional_iteration)):
		if(memory_transfer_flag==1):
			cp_dens_pre = cp.asarray(cp_dens_pre, dtype=cp.complex64)#numpy配列 ⇒ cupy配列に変換
		rdens_in=cp_sup*cp.absolute(cp_dens_pre)
		rdens_out=-(cp_sup-1.0)*cp.absolute(cp_dens_pre)
		if(memory_transfer_flag==1):
			cp_dens_pre = cp.asnumpy(cp_dens_pre)#cupy配列 ⇒ numpy配列に変換

		rdens_in_sum=cp.sum(rdens_in, axis=(1,2))
		rdens_out_sum=cp.sum(rdens_out, axis=(1,2))

		gamma = rdens_out_sum / ((OS_ratio - 1.0 ) * rdens_in_sum)

		n_min_gamma=cp.argmin(gamma)
		min_gamma=cp.min(gamma)

	#重心の計算

	if(i+1==int(iteration)+int(additional_iteration)):
		if(memory_transfer_flag==1):
			cp_dens_pre = cp.asarray(cp_dens_pre, dtype=cp.complex64)#numpy配列 ⇒ cupy配列に変換	
		R_dens=cp_dens_pre
		if(memory_transfer_flag==1):
			cp_dens_pre = cp.asnumpy(cp_dens_pre)#cupy配列 ⇒ numpy配列に変換
		R_dens = cp.asnumpy(R_dens)
		cp_sup = cp.asnumpy(cp_sup)
		R_dens.real=R_dens.real*cp_sup
		
		if(DFPR_flag == 1):
			R_dens_abs=np.absolute(R_dens)*cp_sup

#		if(complex_constraint_flag != 1):
#			R_dens.imag=R_dens.imag*cp_sup

#		print(R_dens.imag)
#		R_dens.flags.writeable = True
#		print(R_dens.flags)
#		R_dens.imag[:,:,:]=0.0
	
	if((i+1==int(iteration)+int(additional_iteration)) & (OSS_flag!=1) & (iteration!="0")):
	
		nx=np.arange(0,int(row),1)
		ny=np.arange(0,int(col),1)

		for n in range(sta_dens):
			if(DFPR_flag == 1):
				R_dens_real=R_dens_abs[n,:,:]
			else:
				R_dens_real=R_dens.real[n,:,:]			

			cp_mode='np'
			if(cp_mode=='np'):
				max_index=np.unravel_index(np.argmax(R_dens_real), R_dens_real.shape)

				R_dens[n,:,:]=np.roll(R_dens[n,:,:], int(row/2)-max_index[0], axis=0)
				R_dens[n,:,:]=np.roll(R_dens[n,:,:], int(col/2)-max_index[1], axis=1)

				weight_sum=np.sum(R_dens_real)
				x_axis_sum=np.sum(R_dens_real, axis=1)
				y_axis_sum=np.sum(R_dens_real, axis=0)
			
				x_sum=np.sum(x_axis_sum * nx)
				y_sum=np.sum(y_axis_sum * ny)
			elif(cp_mode=='cp'):
				max_index=cp.unravel_index(cp.argmax(R_dens_real), R_dens_real.shape)

				R_dens[n,:,:]=cp.roll(R_dens[n,:,:], int(row/2)-max_index[0], axis=0)
				R_dens[n,:,:]=cp.roll(R_dens[n,:,:], int(col/2)-max_index[1], axis=1)

				weight_sum=cp.sum(R_dens_real)
				x_axis_sum=cp.sum(R_dens_real, axis=1)
				y_axis_sum=cp.sum(R_dens_real, axis=0)
			
				x_sum=cp.sum(x_axis_sum * x)
				y_sum=cp.sum(y_axis_sum * y)
	
			if(weight_sum != 0.0):
		
				if(int((x_sum / weight_sum)*10.0) % 10 >= 5):
					x_G = int(x_sum / weight_sum)+1
				else:
					x_G = int(x_sum / weight_sum)
				if(int((y_sum / weight_sum)*10.0) % 10 >= 5):
					y_G = int(y_sum / weight_sum)+1
				else:
					y_G = int(y_sum / weight_sum)

				#print(str(n) + ", " + str(x_G) + ", " + str(y_G)) # + ", " + str(x_sum) + ", " + str(y_sum))

				if(cp_mode=='np'):
					R_dens[n,:,:]=np.roll(R_dens[n,:,:], int(row/2)-x_G, axis=0)
					R_dens[n,:,:]=np.roll(R_dens[n,:,:], int(col/2)-y_G, axis=1)
			
					cp_sup[n,:,:]=np.roll(cp_sup[n,:,:], int(row/2)-x_G, axis=0)	
					cp_sup[n,:,:]=np.roll(cp_sup[n,:,:], int(col/2)-y_G, axis=1)
				elif(cp_mode=='cp'):		
					R_dens[n,:,:]=cp.roll(R_dens[n,:,:], int(row/2)-x_G, axis=0)
					R_dens[n,:,:]=cp.roll(R_dens[n,:,:], int(col/2)-y_G, axis=1)
			
					cp_sup[n,:,:]=cp.roll(cp_sup[n,:,:], int(row/2)-x_G, axis=0)	
					cp_sup[n,:,:]=cp.roll(cp_sup[n,:,:], int(col/2)-y_G, axis=1)

		R_dens = cp.asarray(R_dens, dtype=cp.complex64)
		cp_sup = cp.asarray(cp_sup,dtype="float32")

	
	#最後の電子密度とパターンの出力

	if(i+1==int(iteration)+int(additional_iteration)):
		R_dens = cp.asnumpy(R_dens)#cupy配列 ⇒ numpy配列に変換
		with mrcfile.new(header + '_final_rdens.mrc', overwrite=True) as mrc:
			mrc.set_data(R_dens.real)
		mrc.close

		if(DFPR_flag==1):
			with mrcfile.new(header + '_final_adens.mrc', overwrite=True) as mrc:
				mrc.set_data(np.absolute(R_dens))
			mrc.close
		R_dens = cp.asarray(R_dens, dtype=cp.complex64)

		if(complex_constraint_flag == 1):
			R_dens = cp.asnumpy(R_dens)#cupy配列 ⇒ numpy配列に変換
			with mrcfile.new(header + '_final_idens.mrc', overwrite=True) as mrc:
				mrc.set_data(R_dens.imag)
			mrc.close
			R_dens = cp.asarray(R_dens, dtype=cp.complex64)		


		R_structure_factor = cp.fft.fftn(R_dens, axes=(1,2),norm="ortho")
		R_amp = cp.absolute(R_structure_factor)
		R_amp = cp.fft.fftshift(R_amp)
		R_amp = cp.asnumpy(R_amp)#cupy配列 ⇒ numpy配列に変換
		with mrcfile.new(header + '_final_diff.mrc', overwrite=True) as mrc:
			mrc.set_data(np.square(R_amp))
		mrc.close
		R_amp=cp.asarray(R_amp, dtype="float32")

		if((target_size_flag == 1) | (donut_mask_flag == 1)):
			if(DFPR_flag==1):
				R_structure_factor_adens = cp.fft.fftn(cp.absolute(R_dens), axes=(1,2),norm="ortho")
				R_amp_adens = cp.absolute(R_structure_factor_adens)
				R_amp_adens = cp.fft.fftshift(R_amp_adens)
				R_amp_donut=R_amp_adens
			else:
				R_amp_donut=cp.square(R_amp*cp_donut_target)

			cp_autocorrelation = cp.fft.fft2(R_amp_donut, axes=(1,2), norm="ortho")
			cp_autocorrelation = cp.fft.fftshift(cp_autocorrelation)	
			cp_autocorrelation_abs=cp.absolute(cp_autocorrelation)

			np_autocorrelation_abs=cp.asnumpy(cp_autocorrelation_abs)
			with mrcfile.new(header + '_final_autocorrelation.mrc', overwrite=True) as mrc:
				mrc.set_data(np_autocorrelation_abs)
			mrc.close

			NOR=np.zeros(sta_dens)
			NOR=np.asarray(NOR,dtype="float32")
			NOR_array=np.zeros(np_diff.shape,dtype="float32")
			NOR_array1=np.zeros(np_diff.shape,dtype="float32")
			NOR_array2=np.zeros(np_diff.shape,dtype="float32")

			erode=np.flipud(erode)

#			AC_mode="ave_std"
			AC_mode="S"

			if(AC_mode=="ave_std"):
				ave_autocorrelation=np.average(np_autocorrelation_abs, axis=(1,2))
				std_autocorrelation=np.std(np_autocorrelation_abs, axis=(1,2))
				threshold=ave_autocorrelation + std_autocorrelation
			elif(AC_mode=="S"):				
				max_array=np.amax(np_autocorrelation_abs, axis=(1,2))

			for n in range(sta_dens):
				if(AC_mode=="ave_std"):
					np_sup_2D=np.where(np_autocorrelation_abs[n,:,:]>=threshold[n],float(1),float(0))
					dilate=cv2.dilate(np_sup_2D,d_kernel)
					erode_ex=cv2.erode(dilate,d_kernel)
				elif(AC_mode=="S"):
					div=max_array[n]/float(10)
					threshold=div
					np_sup_2D=np.where(np_autocorrelation_abs[n,:,:]>=threshold,float(1),float(0))	

					dilate=cv2.dilate(np_sup_2D,d_kernel)
					erode_ex=cv2.erode(dilate,d_kernel)
					
					size_sup_2D=np.sum(erode_ex)
					np_sup_2D_pre=erode_ex
					size_sup_2D_pre=size_sup_2D
					while (size_sup_2D >= float(target_size)):
						np_sup_2D_pre=erode_ex
						size_sup_2D_pre=size_sup_2D
						threshold=threshold+div
						np_sup_2D=np.where(np_autocorrelation_abs[n,:,:]>=threshold,float(1),float(0))
	#
						dilate=cv2.dilate(np_sup_2D,d_kernel)
						erode_ex=cv2.erode(dilate,d_kernel)
	#
						size_sup_2D=np.sum(erode_ex)				
					
					div=div/10
					while (size_sup_2D <= float(target_size)):
						np_sup_2D_pre=erode_ex
						size_sup_2D_pre=size_sup_2D
						threshold=threshold-div
						np_sup_2D=np.where(np_autocorrelation_abs[n,:,:]>=threshold,float(1),float(0))
	#
						dilate=cv2.dilate(np_sup_2D,d_kernel)
						erode_ex=cv2.erode(dilate,d_kernel)
	#
						size_sup_2D=np.sum(erode_ex)
					if(cp.absolute(size_sup_2D-float(target_size)) >= cp.absolute(size_sup_2D_pre-float(target_size))):
						erode_ex=np_sup_2D_pre	


				NOR_array1=np.where((erode_ex==1.0) & (erode==0.0),1,0)
				NOR_array2=np.where((erode_ex==0.0) & (erode==1.0),1,0)
				NOR_array=NOR_array1+NOR_array2
				NOR[n]=np.sum(NOR_array[int((col-target_area)/2):int((col+target_area)/2),int((row-target_area)/2):int((row+target_area)/2)])/float(target_size)
				np_autocorrelation_abs[n,:,:]=NOR_array[:,:]
			
			n_min_NOR=np.argmin(NOR)
			min_NOR=np.min(NOR)

			with mrcfile.new(header + '_final_binarized_autocorrelation.mrc', overwrite=True) as mrc:
				mrc.set_data(np_autocorrelation_abs)
			mrc.close

			R_dens_sort=R_dens[np.argsort(NOR),:,:]
			R_dens_sort = cp.asnumpy(R_dens_sort)#cupy配列 ⇒ numpy配列に変換
			with mrcfile.new(header + '_final_rdens_sort.mrc', overwrite=True) as mrc:
				mrc.set_data(R_dens_sort.real)
			mrc.close

			cp_sup_sort=cp_sup[np.argsort(NOR),:,:]
			cp_sup_sort = cp.asnumpy(cp_sup_sort)#cupy配列 ⇒ numpy配列に変換
			with mrcfile.new(header + '_final_sup_sort.mrc', overwrite=True) as mrc:
				mrc.set_data(cp_sup_sort.real)
			mrc.close

			R_dens_min_NOR=R_dens[n_min_NOR,:,:]
			R_dens_min_NOR=cp.flipud(R_dens_min_NOR)
			R_dens_min_NOR = cp.asnumpy(R_dens_min_NOR)#cupy配列 ⇒ numpy配列に変換
			foname=header + "_final_rdens_min_NOR.tif"
			tifffile.imsave(foname ,R_dens_min_NOR.real)
			foname=header + "_final_adens_min_NOR.tif"
			tifffile.imsave(foname ,np.absolute(R_dens_min_NOR))

			cp_sup_min_NOR=cp_sup[n_min_NOR,:,:]
			cp_sup_min_NOR=cp.flipud(cp_sup_min_NOR)
			cp_sup_min_NOR = cp.asnumpy(cp_sup_min_NOR)#cupy配列 ⇒ numpy配列に変換
			foname=header + "_final_sup_min_NOR.tif"
			tifffile.imsave(foname ,cp_sup_min_NOR)

			

		cp_sup = cp.asnumpy(cp_sup)
		with mrcfile.new(header + '_final_sup.mrc', overwrite=True) as mrc:
			mrc.set_data(cp_sup)
		mrc.close
		cp_sup=cp.asarray(cp_sup,dtype="float32")

	#R-factorの計算

	if(i+1==int(iteration)+int(additional_iteration)):
		R_amp = cp.absolute(R_structure_factor)
		R_amp_square=cp.square(R_amp)
		R_amp_abs=cp.absolute(R_amp)
		cp_diff_amp_abs=cp.absolute(cp_diff_amp)
	
		R_amp_abs_cp_diff_amp_abs=R_amp_abs*cp_diff_amp_abs
	
		R_amp_square_pos=cp.zeros(cp_dens.shape,dtype="float32")
		R_amp_abs_pos=cp.zeros(cp_dens.shape,dtype="float32")
		R_amp_abs_cp_diff_amp_abs_pos=cp.zeros(cp_dens.shape,dtype="float32")
		cp_diff_amp_pos=cp.zeros(cp_dens.shape,dtype="float32")
		cp_diff_amp_abs_pos=cp.zeros(cp_dens.shape,dtype="float32")
		
		R_amp_square_pos[cp_diff_amp%2>0.0]=R_amp_square[cp_diff_amp%2>0.0]
		R_amp_abs_pos[cp_diff_amp%2>0.0]=R_amp_abs[cp_diff_amp%2>0.0]
		R_amp_abs_cp_diff_amp_abs_pos[cp_diff_amp%2>0.0]=R_amp_abs_cp_diff_amp_abs[cp_diff_amp%2>0.0]
		cp_diff_amp_pos[cp_diff_amp%2>0.0]=cp_diff_amp[cp_diff_amp%2>0.0]
		cp_diff_amp_abs_pos[cp_diff_amp%2>0.0]=cp_diff_amp_abs[cp_diff_amp%2>0.0]

		amp2_sum=cp.sum(R_amp_square_pos, axis=(1,2))
		amp_x_diff_amp_sum=cp.sum(R_amp_abs_cp_diff_amp_abs_pos, axis=(1,2))
		diff_amp_sum=cp.sum(cp_diff_amp_abs_pos, axis=(1,2))
		scale_factor=amp_x_diff_amp_sum/amp2_sum

		scale_factor_3D=cp.repeat(scale_factor,int(row)*int(col))
		scale_factor_3D=scale_factor_3D.reshape(sta_dens,int(row),int(col))
	
		diff_amp_scale=cp.sum(cp.absolute(cp_diff_amp_abs_pos-scale_factor_3D*R_amp_abs_pos), axis=(1,2))
		R_factor=diff_amp_scale/diff_amp_sum

		n_min_R=cp.argmin(R_factor)
		min_R=cp.min(R_factor)

	#ログファイルに最終のパラメーター書き出し
		for n in range(sta_dens):
			if((target_size_flag == 1) | (donut_mask_flag == 1)):
				print(str(n) + " " + str(i+1).zfill(6) + " " + str(scale_factor[n]) + " " + str(R_factor[n]) + " " + str(OS_ratio[n]) + " " + str(gamma[n]) + " " + str(NOR[n]))
				with open(log_path, mode='a') as log:
					log.write("\n" + str(n) + " " + str(i+1).zfill(6) + " " + str(scale_factor[n]) + " " + str(R_factor[n]) + " " + str(OS_ratio[n]) + " " + str(gamma[n]) + " " + str(NOR[n]))

			else:
				print(str(n) + " " + str(i+1).zfill(6) + " " + str(scale_factor[n]) + " " + str(R_factor[n]) + " " + str(OS_ratio[n]) + " " + str(gamma[n]))
				with open(log_path, mode='a') as log:
					log.write("\n" + str(n) + " " + str(i+1).zfill(6) + " " + str(scale_factor[n]) + " " + str(R_factor[n]) + " " + str(OS_ratio[n]) + " " + str(gamma[n]))
		if((target_size_flag == 1) | (donut_mask_flag == 1)):
			print("min_NOR(" + str(n_min_NOR) + ") = " + str(min_NOR))
			with open(log_path, mode='a') as log:
				log.write("\n" + "min_NOR(" + str(n_min_NOR) + ") = " + str(min_NOR))
		print("min_R_factor(" + str(n_min_R) + ") = " + str(min_R))
		print("min_gamma(" + str(n_min_gamma) + ") = " + str(min_gamma))
		with open(log_path, mode='a') as log:
			log.write("\n" + "min_R_factor(" + str(n_min_R) + ") = " + str(min_R))
			log.write("\n" + "min_gamma(" + str(n_min_gamma) + ") = " + str(min_gamma))
	
	#実空間拘束
	
	cp_dens_bk=cp.real(cp_dens)
	if(memory_transfer_flag==1):
		cp_dens_pre = cp.asarray(cp_dens_pre, dtype=cp.complex64)#numpy配列 ⇒ cupy配列に変換

	if(DFPR_flag == 1):
		cp_dens.real=cp.where(cp_sup==1, cp_dens_pre.real, cp_dens_bk-0.9*cp_dens_pre.real)
	else:
		cp_dens.real=cp.where((cp_dens_pre.real>=0.0) & (cp_sup==1), cp_dens_pre.real, cp_dens_bk-0.9*cp_dens_pre.real)

	if(complex_constraint_flag == 1):
		cp_dens_imag_bk=cp.imag(cp_dens)
		if(DFPR_flag == 1):
			cp_dens.imag=cp.where(cp_sup==1, cp_dens_pre.imag, cp_dens_imag_bk-0.9*cp_dens_pre.imag)
		else:
			cp_dens.imag=cp.where((cp_dens_pre.imag>=0.0) & (cp_sup==1), cp_dens_pre.imag, cp_dens_imag_bk-0.9*cp_dens_pre.imag)			
	else:
		cp_dens.imag[:,:,:]=0.0

	del cp_dens_pre
	del cp_dens_bk

	#OSS mask convolution
	
	if((OSS_flag==1) & ((i+1) <= int(iteration))):
		if(memory_transfer_flag==1):
			cp_structure_factor = cp.asarray(cp_structure_factor, dtype=cp.complex64)#numpy配列に変換 ⇒ cupy配列 
		cp_structure_factor = cp.fft.fftn(cp_dens, axes=(1,2),norm="ortho")#【フーリエ変換】
		
		if(i==0):
			W_kernel=W(X,Y,OSS_alpha)
			W_kernel=cp.asarray(W_kernel,dtype="float32")
			W_kernel=W_kernel / cp.amax(W_kernel)
			W_kernel = cp.fft.ifftshift(W_kernel)


		cp_structure_factor.real=W_kernel*cp_structure_factor.real
		cp_structure_factor.imag=W_kernel*cp_structure_factor.imag

		W_dens = cp.fft.ifftn(cp_structure_factor, axes=(1,2),norm="ortho")
		if(memory_transfer_flag==1):
			cp_structure_factor = cp.asnumpy(cp_structure_factor)#cupy配列 ⇒ numpy配列に変換

#		print(str(i+1).zfill(6) + " alpha = " + str(OSS_alpha))

		#OSS 実空間拘束

		cp_dens.real=cp.where(cp_sup==0, W_dens.real, cp_dens.real)
		cp_dens.imag[:,:,:]=0.0

	#OSS parameter update

	if(OSS_flag==1):
		if(((i+1) % int(OSS_interval) == 0) & ((i+1) < int(iteration))):
			OSS_alpha=OSS_alpha-OSS_alpha_step

			W_kernel=W(X,Y,OSS_alpha)
			W_kernel=cp.asarray(W_kernel,dtype="float32")
			W_kernel=W_kernel / cp.amax(W_kernel)
			W_kernel = cp.fft.ifftshift(W_kernel)

t5=time.time()

print("total time : " + str(t5-t1))

with open(log_path, mode='a') as log:
	log.write("\n" + "total time : " + str(t5-t1))

#del cp_dens
#del cp_sup
#del cp_donut_target
#del cp_diff_amp
#del cp_autocorrelation
#del cp_autocorrelation_abs
#del ave_autocorrelation
#del std_autocorrelation
#del th
#del cp_th
#del max_autocorrelation
#del Wgauss
#del cp_structure_factor
#del cp_amp
#del cp_dens_pre
#del G_dens
#del cp_sup_2D
#del rdens_in_sum
#del rdens_out_sum
#del R_dens
#del R_amp_square
#del R_amp_abs
#del cp_diff_amp_abs
#del cp_diff_amp_abs_pos
#del R_amp_abs_cp_diff_amp_abs
#del R_amp_square_pos
#del R_amp_abs_pos
#del R_amp_abs_cp_diff_amp_abs_pos
#del amp2_sum
#del amp_x_diff_amp_sum
#del diff_amp_sum
#del scale_factor
#del scale_factor_3D
#del diff_amp_scale
#del R_factor
#del cp_dens_bk

gc.collect()

print("complete del")
