#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import cupy as cp
import time
import os
import mrcfile

from PIL import Image
from skimage import io

pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)

if ((len(sys.argv)==1)):
	print("command:python3 PR_multi_EIGER.py [-diff] [-sup] [-initial_dens] [-iteration] [-header] [-output_interval] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-initial_SW_delta] [-last_SW_ips] [-n_SW_delta] [-additional_iteration] [-OSS_interval]")
	print("HIO only	: [-diff] [-sup] [-initial_dens] '''[-iteration 0]''' [-header] [-additional_iteration]")
	print("HIO-SW	: [-diff] [-sup] [-initial_dens ] [-iteration] [-header] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-initial_SW_delta] [-last_SW_ips] [-n_SW_delta]")
	print("OSS	: [-diff] [-sup] [-iteration] [-header] [-OSS_interval]")
	print("option	: [-output_interval] [-SW_sup_output] [-complex_constraint]")
	exit()

n_parameter=17
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
	if(sys.argv[i]=="--help"):
		print("command:python3 PR.py [-diff] [-sup] [-initial_dens] [-iteration] [-header] [-output_interval] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-initial_SW_delta] [-last_SW_ips] [-n_SW_delta] [-additional_iteration] [-OSS_interval]")
		print("HIO only : [-diff] [-sup] [-initial_dens] '''[-iteration 0]''' [-header] [-additional_iteration]")
		print("HIO-SW	: [-diff] [-sup] [-initial_dens] [-iteration] [-header] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-initial_SW_delta] [-last_SW_ips] [-n_SW_delta]")
		print("OSS	: [-diff] [-sup] [-initial_dens] [-iteration] [-header] [-OSS_interval] [-additional_iteration]")
		print("option	: [-output_interval] [-SW_sup_output] [-complex_constraint]")
		exit()

if(OSS_flag!=1):
	if(iteration=="0"):
		#HIO only
		for i in range(n_parameter):
			if((flag_list[i]==0) & (i!=12) & (i!=5) & (i!=6) & (i!=7) & (i!=8) & (i!=9) & (i!=10) & (i!=13) & (i!=16)):
				if(input_parameter==0):
					print("HIO only mode")
				print("please input parameter : [" + parameter_name_list[i] + "]")
				input_parameter=1
		if(input_parameter==1):
			exit()
	else:
		#HIO-SW
		for i in range(n_parameter):
			if((flag_list[i]==0) & (i!=12) & (i!=5) & (i!=13) & (i!=16)):
				if(input_parameter==0):
					print("HIO-SW mode")
				print("please input parameter : [" + parameter_name_list[i] + "]")
				input_parameter=1
		if(input_parameter==1):
			exit()
else:
	#OSS
	for i in range(n_parameter):
		if((flag_list[i]==0) & (i!=5) & (i!=6) & (i!=7) & (i!=8) & (i!=9) &(i!=10) & (i!=13) & (i!=16)):
			if(input_parameter==0):
				print("OSS mode")
			print("please input parameter : [" + parameter_name_list[i] + "]")	
			input_parameter=1
	if(input_parameter==1):
		exit()

print("diff = " + finame)
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
		print("initial_SW_delta = " + initial_SW_delta)
		print("last_SW_delta = " + last_SW_delta)
		print("n_SW_delta = " + n_SW_delta)
		print("additional_iteration = " + additional_iteration)
		print("HIO-SW mode")
else:
	print("OSS mode")
print("")

# open mrc file

with mrcfile.open(initial_dens, permissive=True) as mrc:
	cp_dens=cp.asarray(mrc.data,dtype="float32")
mrc.close

if(support.find("tif") != 0):
	sup=Image.open(support)
	np_sup=np.asarray(sup,dtype="float32")
	cp_sup=[np_sup]*int(cp_dens.shape[0])
	cp_sup=cp.asarray(cp_sup,dtype="float32")
	cp_sup=cp.flip(cp_sup,axis=1)
	print("np_sup dtype = " + str(np_sup.dtype))
	print("np_sup shape = " + str(np_sup.shape))
	
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

np_diff=np.where(np_diff>0.0,np_diff,0.0)
np_diff=np.sqrt(np_diff)

cp_diff_amp=[np_diff]*int(sta_dens)
cp_diff_amp=cp.asarray(cp_diff_amp,dtype="float32")
cp_diff_amp=cp.flip(cp_diff_amp,axis=1)
cp_diff_amp = cp.fft.ifftshift(cp_diff_amp, axes=(1,2))

#ディレクトリ作成

os.makedirs(header[0:header.rfind("/")], exist_ok=True)
#if(header.rfind("/") == -1):
#	header=header + "/" +header	
#	log_path=header + "_MPR.log"
#else:
#	header=header + "/" +header[header.rfind("/")+1:len(header)]
	
log_path=header + "_MPR.log"
with open(log_path, mode='w') as log:
	log.write("program Phase retrieval multi ver.20190701")

log_text="\n\n" + "diff = " + finame + "\n" + "support = " + support + "\n" + "initial_dens = " + initial_dens + "\n" + "iteration = " + iteration + "\n" + "header = " + header

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
		log_text=log_text + "\n" + "initial_SW_delta = " + initial_SW_delta + "\n" + "last_SW_delta = " + last_SW_delta + "\n" + "n_SW_delta = " + n_SW_delta
		log_text=log_text + "\n" + "additional_iteration = " + additional_iteration + "\n"
		log_text=log_text + "\n" + "HIO-SW mode\n"
else:
	log_text=log_text + "\nOSS mode\n"

log_text=log_text + "diff shape = " + str(np_diff.shape) +"\n" 


if(complex_constraint_flag == 1):
	log_text=log_text + "complex_constraint_mode" +"\n"
	log_text=log_text + "sta_dens of diff (n_trial) = " + str(sta_dens) +"\n\n"
else:
	log_text=log_text + "sta_dens of diff (n_trial) = " + str(sta_dens) +"\n\n"


with open(log_path, mode='a') as log:
	log.write(log_text)
#

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

#start

print("trial scale_factor Rfactor OS_ratio gamma")
with open(log_path, mode='a') as log:
	log.write("trial scale_factor Rfactor OS_ratio gamma")

for i in range(int(iteration)+int(additional_iteration)):

#	print(i+1)
	
	cp_structure_factor = cp.fft.fftn(cp_dens, axes=(1,2), norm="ortho")#【フーリエ変換】
	cp_amp = cp.absolute(cp_structure_factor)#絶対値をとる
	
	#逆空間拘束

	cp_structure_factor[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)]=cp_structure_factor[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)] * cp_diff_amp[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)] / cp_amp[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)]
	cp_dens_pre = cp.fft.ifftn(cp_structure_factor, axes=(1,2),norm="ortho")

	#Shrink Wrap
	
	if(OSS_flag!=1):
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

				cp_structure_factor.real=G_kernel*cp_structure_factor.real
				cp_structure_factor.imag=G_kernel*cp_structure_factor.imag
				
				G_dens = cp.fft.ifftn(cp_structure_factor, axes=(1,2),norm="ortho")

				threshold = SW_delta*cp.amax(cp.real(G_dens), axis=(1,2))
				threshold_3D=cp.repeat(threshold,int(col)*int(row))
				threshold_3D=threshold_3D.reshape(sta_dens,int(row),int(col))

				cp_sup=cp.where(cp.real(G_dens)>=threshold_3D,float(1),float(0))
	
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
		cp_dens_pre = cp.asarray(cp_dens_pre,dtype="float32")

	#OS比の計算

	if(i+1==int(iteration)+int(additional_iteration)):
		n_pixel=float(row) * float(col)
		OS_ratio = n_pixel / cp.sum(cp_sup, axis=(1,2))
		
	#gammaの計算

	if(i+1==int(iteration)+int(additional_iteration)):
		rdens_in=cp_sup*cp.absolute(cp_dens_pre)
		rdens_out=-(cp_sup-1.0)*cp.absolute(cp_dens_pre)

		rdens_in_sum=cp.sum(rdens_in, axis=(1,2))
		rdens_out_sum=cp.sum(rdens_out, axis=(1,2))

		gamma = rdens_out_sum / ((OS_ratio - 1.0 ) * rdens_in_sum)

	#重心の計算

	if(i+1==int(iteration)+int(additional_iteration)):	
		R_dens=cp_dens_pre
		R_dens.real=R_dens.real*cp_sup

#		if(complex_constraint_flag != 1):
#			R_dens.imag=R_dens.imag*cp_sup
		R_dens.imag[:,:,:]=0
	
	if((i+1==int(iteration)+int(additional_iteration)) & (OSS_flag!=1) & (iteration!="0")):
	
		for n in range(sta_dens):
			R_dens_real=R_dens.real[n,:,:]

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
		
				R_dens[n,:,:]=cp.roll(R_dens[n,:,:], int(row/2)-x_G, axis=0)
				R_dens[n,:,:]=cp.roll(R_dens[n,:,:], int(col/2)-y_G, axis=1)
		
				cp_sup[n,:,:]=cp.roll(cp_sup[n,:,:], int(row/2)-x_G, axis=0)	
				cp_sup[n,:,:]=cp.roll(cp_sup[n,:,:], int(col/2)-y_G, axis=1)


	
	#最後の電子密度とパターンの出力

	if(i+1==int(iteration)+int(additional_iteration)):
		R_dens = cp.asnumpy(R_dens)#cupy配列 ⇒ numpy配列に変換
		with mrcfile.new(header + '_final_rdens.mrc', overwrite=True) as mrc:
			mrc.set_data(R_dens.real)
		mrc.close
		R_dens = cp.asarray(R_dens,dtype="float32")

		if(complex_constraint_flag == 1):
			R_dens = cp.asnumpy(R_dens)#cupy配列 ⇒ numpy配列に変換
			with mrcfile.new(header + '_final_idens.mrc', overwrite=True) as mrc:
				mrc.set_data(R_dens.imag)
			mrc.close
			R_dens = cp.asarray(R_dens,dtype="float32")		


		R_structure_factor = cp.fft.fftn(R_dens, axes=(1,2),norm="ortho")
		R_amp = cp.absolute(R_structure_factor)
		R_amp = cp.fft.fftshift(R_amp)
		R_amp = cp.asnumpy(R_amp)#cupy配列 ⇒ numpy配列に変換
		with mrcfile.new(header + '_final_diff.mrc', overwrite=True) as mrc:
			mrc.set_data(np.square(R_amp))
		mrc.close
		R_amp=cp.asarray(R_amp,dtype="float32")

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

	#ログファイルに最終のパラメーター書き出し
		for n in range(sta_dens):
			print(str(n) + " " + str(i+1).zfill(6) + " " + str(scale_factor[n]) + " " + str(R_factor[n]) + " " + str(OS_ratio[n]) + " " + str(gamma[n]))
			with open(log_path, mode='a') as log:
				log.write("\n" + str(n) + " " + str(i+1).zfill(6) + " " + str(scale_factor[n]) + " " + str(R_factor[n]) + " " + str(OS_ratio[n]) + " " + str(gamma[n]))
	
	#実空間拘束
	
	cp_dens_bk=cp.real(cp_dens)

	cp_dens.real[cp_sup%2==1]=cp_dens_pre.real[cp_sup%2==1]
	cp_dens.real[(cp_dens_pre.real%2<0.0) | (cp_sup%2==0)]=cp_dens_bk[(cp_dens_pre.real%2<0.0) | (cp_sup%2==0)]-0.9*cp_dens_pre.real[(cp_dens_pre.real%2<0.0) | (cp_sup%2==0)]

	if(complex_constraint_flag == 1):
		cp_dens_imag_bk=cp.imag(cp_dens)
		cp_dens.imag[cp_sup%2==1]=cp_dens_pre.imag[cp_sup%2==1]
		cp_dens.imag[(cp_dens_pre.imag%2<0.0) | (cp_sup%2==0)]=cp_dens_imag_bk[(cp_dens_pre.imag%2<0.0) | (cp_sup%2==0)]-0.9*cp_dens_pre.imag[(cp_dens_pre.imag%2<0.0) | (cp_sup%2==0)]	
	else:
		cp_dens.imag[:,:,:]=0.0
	
	#OSS mask convolution
	
	if((OSS_flag==1) & ((i+1) <= int(iteration))):
		cp_structure_factor = cp.fft.fftn(cp_dens, axes=(1,2),norm="ortho")#【フーリエ変換】
		
		if(i==0):
			W_kernel=W(X,Y,OSS_alpha)
			W_kernel=cp.asarray(W_kernel,dtype="float32")
			W_kernel=W_kernel / cp.amax(W_kernel)
			W_kernel = cp.fft.ifftshift(W_kernel)


		cp_structure_factor.real=W_kernel*cp_structure_factor.real
		cp_structure_factor.imag=W_kernel*cp_structure_factor.imag

		W_dens = cp.fft.ifftshift(cp_structure_factor)

#		print(str(i+1).zfill(6) + " alpha = " + str(OSS_alpha))

		#OSS 実空間拘束

		cp_dens.real[cp_sup%2==0]=W_dens.real[cp_sup%2==0]
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



