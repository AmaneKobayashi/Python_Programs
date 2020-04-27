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
	print("command:python3 PR.py [-diff] [-sup] [-initial_dens] [-iteration] [-header] [-output_interval] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-SW_delta] [-additional_iteration] [-OSS_interval]")
	print("HIO only	: [-diff] [-sup] [-initial_dens] '''[-iteration 0]''' [-header] [-additional_iteration]")
	print("HIO-SW	: [-diff] [-sup] [-initial_dens ] [-iteration] [-header] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-SW_delta]")
	print("OSS	: [-diff] [-sup] [-iteration] [-header] [-OSS_interval]")
	print("option	: [-output_interval] [-SW_sup_output]")
	exit()

n_parameter=14
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
parameter_name_list[10]="-SW_delta"
parameter_name_list[11]="-additional_iteration"
parameter_name_list[12]="-OSS_interval"
parameter_name_list[13]="-SW_sup_output"

diff_flag=0
sup_flag=0
intial_dens_flag=0
iteration_flag=0
header_flag=0
output_interval_flag=0
initial_SW_ips_flag=0
SW_ips_step_flag=0
last_SW_ips_flag=0
SW_delta_flag=0
additional_iteration_flag=0
OSS_flag=0
SW_sup_output_flag=0

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
	if(sys.argv[i]=="-SW_delta"):
		SW_delta=sys.argv[i+1]
		SW_delta_flag=1
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
	if(sys.argv[i]=="--help"):
		print("command:python3 PR.py [-diff] [-sup] [-initial_dens] [-iteration] [-header] [-output_interval] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-SW_delta] [-additional_iteration] [-OSS_interval]")
		print("HIO only : [-diff] [-sup] [-initial_dens] '''[-iteration 0]''' [-header] [-additional_iteration]")
		print("HIO-SW	: [-diff] [-sup] [-initial_dens] [-iteration] [-header] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-SW_delta]")
		print("OSS	: [-diff] [-sup] [-initial_dens] [-iteration] [-header] [-OSS_interval] [-additional_iteration]")
		print("option	: [-output_interval] [-SW_sup_output]")
		exit()

if(OSS_flag!=1):
	if(iteration=="0"):
		#HIO only
		for i in range(n_parameter):
			if((flag_list[i]==0) & (i!=12) & (i!=5) & (i!=6) & (i!=7) & (i!=8) & (i!=9) & (i!=10) & (i!=13)):
				if(input_parameter==0):
					print("HIO only mode")
				print("please input parameter : [" + parameter_name_list[i] + "]")
				input_parameter=1
		if(input_parameter==1):
			exit()
	else:
		#HIO-SW
		for i in range(n_parameter):
			if((flag_list[i]==0) & (i!=12) & (i!=5) & (i!=13)):
				if(input_parameter==0):
					print("HIO-SW mode")
				print("please input parameter : [" + parameter_name_list[i] + "]")
				input_parameter=1
		if(input_parameter==1):
			exit()
else:
	#OSS
	for i in range(n_parameter):
		if((flag_list[i]==0) & (i!=5) & (i!=6) & (i!=7) & (i!=8) & (i!=9) &(i!=10) & (i!=13)):
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
		print("SW_delta = " + SW_delta)
		print("additional_iteration = " + additional_iteration)
		print("HIO-SW mode")
else:
	print("OSS mode")
print("")

# open mrc file

with mrcfile.open(finame, permissive=True) as mrc:
	cp_diff_amp=cp.asarray(mrc.data,dtype="float32")
mrc.close
cp_diff_amp = cp.fft.fftshift(cp_diff_amp)

with mrcfile.open(support, permissive=True) as mrc:
	cp_sup=cp.asarray(mrc.data,dtype="float32")
mrc.close

with mrcfile.open(initial_dens, permissive=True) as mrc:
	cp_initial_dens=cp.asarray(mrc.data,dtype="float32")
mrc.close

print("cp_diff dtype = " + str(cp_diff_amp.dtype))
print("cp_diff shape = " + str(cp_diff_amp.shape))
print("cp_sup dtype = " + str(cp_sup.dtype))
print("cp_sup shape = " + str(cp_sup.shape))
print("cp_initial_dens dtype = " + str(cp_initial_dens.dtype))
print("cp_initial_dens shape = " + str(cp_initial_dens.shape))
print("")

row=cp_diff_amp.shape[0]
col=cp_diff_amp.shape[1]
sta=cp_diff_amp.shape[2]
print("row of diff = " + str(row))
print("col of diff = " + str(col))
print("sta of diff = " + str(sta))
print("")

#ディレクトリ作成

os.makedirs(header, exist_ok=True)
if(header.rfind("/") == -1):
	header=header + "/" +header
	log_path=header + "_MPR.log"
else:
	header=header + "/" +header[header.rfind("/")+1:len(header)]
	log_path=header + "_MPR.log"

with open(log_path, mode='w') as log:
	log.write("program Phase retrieval ver.20190305")

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
		log_text=log_text + "\n" + "SW_delta = " + SW_delta + "\n" + "additional_iteration = " + additional_iteration + "\n"
		log_text=log_text + "\n" + "HIO-SW mode\n"
else:
	log_text=log_text + "\nOSS mode\n"

log_text=log_text + "diff shape = " + str(cp_diff_amp.shape) +"\n\n"

with open(log_path, mode='a') as log:
	log.write(log_text)

#

t1=time.time()

#実験値振幅
#cp_diff_amp=cp.zeros(cp_diff.shape,dtype="float32")
cp_diff_amp=cp.where(cp_diff_amp>0.0,cp_diff_amp,0.0)
cp_diff_amp=cp.sqrt(cp_diff_amp)
#cp_diff_amp=cp.asnumpy(cp_diff_amp)
#with mrcfile.new(header + '_cp_diff_amp.mrc', overwrite=True) as mrc:
#	mrc.set_data(cp_diff_amp)
#mrc.close
#exit()

#電子密度の複素数化

cp_dens=cp.array(cp_initial_dens, dtype=cp.complex64)

#SW用ガウシアンの定義
if((iteration!="0") & (OSS_flag==0)):
	def G(x,y,z,ips):
		Z=cp.exp(-((x-row/2)**2 + (y-col/2)**2 + (z-sta/2)**2) / (2 * ips**2)) / (2 * cp.pi * ips**2)
		return Z
	x=cp.arange(0,int(row),1)
	y=cp.arange(0,int(col),1)
	z=cp.arange(0,int(sta),1)
	X,Y,Z=cp.meshgrid(x,y,z)
	SW_ips=float(initial_SW_ips)

#OSS用マスクの定義
if(OSS_flag==1):
	def W(x,y,z,alpha):
		M=cp.exp(-0.5*((x-row/2)**2 + (y-col/2)**2 + (z-sta/2)**2)/alpha**2)
		return M
	x=cp.arange(0,int(row),1)
	y=cp.arange(0,int(col),1)
	z=cp.arange(0,int(sta),1)
	X,Y,Z=cp.meshgrid(x,y,z)
	OSS_alpha=float(row)
	OSS_alpha_step=(float(row)-1.0/float(row)) / (float(iteration)/float(OSS_interval)-1.0)

#重心計算関係

#cx=cp.arange(0,int(row),1)
#cy=cp.arange(0,int(col),1)
#cz=cp.arange(0,int(sta),1)

#numpy配列 ⇒ cupy配列に変換

#cp_diff_amp = cp.asarray(np_diff_amp,dtype="float32")
#cp_sup = cp.asarray(np_sup,dtype="float32")
#cp_initial_dens = cp.asarray(np_initial_dens,dtype="float32")
#cp_dens=cp.asarray(np_dens)

print("iteration scale_factor Rfactor OS_ratio gamma")
with open(log_path, mode='a') as log:
	log.write("iteration scale_factor Rfactor OS_ratio gamma")

for i in range(int(iteration)+int(additional_iteration)):

#	print(i)
	cp_structure_factor = cp.fft.fftn(cp_dens, axes=(0,1,2), norm="ortho")#【フーリエ変換】
#	cp_structure_factor = cp.fft.fftshift(cp_structure_factor)#fftshiftを使ってシフト
	cp_amp = cp.absolute(cp_structure_factor)#絶対値をとる

	#逆空間拘束

	cp_structure_factor[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)]=cp_structure_factor[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)] * cp_diff_amp[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)] / cp_amp[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)]

#	cp_structure_factor_bk=cp_structure_factor
#	cp_structure_factor = cp.fft.ifftshift(cp_structure_factor)
	cp_dens_pre = cp.fft.ifftn(cp_structure_factor, axes=(0,1,2),norm="ortho")
#	cp_dens_pre_real=cp.real(cp_dens_pre)

	#Shrink Wrap
	
	if(OSS_flag!=1):
		if((i+1) <= int(iteration)):
			if((i+1) % int(SW_interval) == 0):
				if(SW_ips>=float(last_SW_ips)):
					SW_ips=float(last_SW_ips)
				
#				print("SW_ips = " + str(SW_ips))
		
				G_kernel=G(X,Y,Z,SW_ips)
				G_kernel=cp.asarray(G_kernel,dtype="float32")
				G_kernel=G_kernel/cp.amax(G_kernel)
				
				cp_structure_factor = cp.fft.fftshift(cp_structure_factor)

				cp_structure_factor.real=G_kernel*cp_structure_factor.real
				cp_structure_factor.imag=G_kernel*cp_structure_factor.imag
				
				cp_structure_factor = cp.fft.ifftshift(cp_structure_factor)
				G_dens = cp.fft.ifftn(cp_structure_factor, axes=(0,1,2),norm="ortho")

				threshold = float(SW_delta)*cp.amax(cp.real(G_dens))
				
				cp_sup=cp.where(cp.real(G_dens)>=threshold,float(1),float(0))
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

	#重心の計算

	if(i+1==int(iteration)+int(additional_iteration)):	
		R_dens=cp_dens_pre
		R_dens.real=R_dens.real*cp_sup
		R_dens.imag[:,:,:]=0
	
	if((i+1==int(iteration)+int(additional_iteration)) & (OSS_flag!=1) & (iteration!="0")):	
		weight_sum=cp.sum(R_dens.real)
		x_axis_sum=cp.sum(R_dens.real, axis=2)
		y_axis_sum=cp.sum(R_dens.real, axis=1)
		z_axis_sum=cp.sum(R_dens.real, axis=0)
	
		x_sum=cp.sum(x_axis_sum * x)
		y_sum=cp.sum(y_axis_sum * y)
		z_sum=cp.sum(z_axis_sum * z)
	
		if(int((x_sum / weight_sum)*10.0) % 10 >= 5):
			x_G = int(x_sum / weight_sum)+1
		else:
			x_G = int(x_sum / weight_sum)
		if(int((y_sum / weight_sum)*10.0) % 10 >= 5):
			y_G = int(y_sum / weight_sum)+1
		else:
			y_G = int(y_sum / weight_sum)
		if(int((z_sum / weight_sum)*10.0) % 10 >= 5):
			z_G = int(z_sum / weight_sum)+1
		else:
			z_G = int(z_sum / weight_sum)
	
		print(str(i+1).zfill(6) + ", " + str(x_G) + ", " + str(y_G) + ", " + str(z_G))

		R_dens=cp.roll(R_dens, int(row/2)-x_G, axis=2)
		R_dens=cp.roll(R_dens, int(col/2)-y_G, axis=1)
		R_dens=cp.roll(R_dens, int(sta/2)-z_G, axis=0)	

		cp_sup=cp.roll(cp_sup, int(row/2)-x_G, axis=2)	
		cp_sup=cp.roll(cp_sup, int(row/2)-y_G, axis=1)
		cp_sup=cp.roll(cp_sup, int(sta/2)-z_G, axis=0)

	#最後の電子密度とパターンの出力

	if(i+1==int(iteration)+int(additional_iteration)):
		R_dens = cp.asnumpy(R_dens)#cupy配列 ⇒ numpy配列に変換
		with mrcfile.new(header + '_final_rdens.mrc', overwrite=True) as mrc:
			mrc.set_data(R_dens.real)
		mrc.close
		R_dens = cp.asarray(R_dens,dtype="float32")

		R_structure_factor = cp.fft.fftn(R_dens, axes=(0,1,2),norm="ortho")
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

		R_amp_square_pos=cp.zeros(cp_diff_amp.shape,dtype="float32")
		R_amp_abs_pos=cp.zeros(cp_diff_amp.shape,dtype="float32")
		R_amp_abs_cp_diff_amp_abs_pos=cp.zeros(cp_diff_amp.shape,dtype="float32")
		cp_diff_amp_pos=cp.zeros(cp_diff_amp.shape,dtype="float32")
		cp_diff_amp_abs_pos=cp.zeros(cp_diff_amp.shape,dtype="float32")
		

		R_amp_square_pos[cp_diff_amp%2>0.0]=R_amp_square[cp_diff_amp%2>0.0]
		R_amp_abs_pos[cp_diff_amp%2>0.0]=R_amp_abs[cp_diff_amp%2>0.0]
		R_amp_abs_cp_diff_amp_abs_pos[cp_diff_amp%2>0.0]=R_amp_abs_cp_diff_amp_abs[cp_diff_amp%2>0.0]
		cp_diff_amp_pos[cp_diff_amp%2>0.0]=cp_diff_amp[cp_diff_amp%2>0.0]
		cp_diff_amp_abs_pos[cp_diff_amp%2>0.0]=cp_diff_amp_abs[cp_diff_amp%2>0.0]

		amp2_sum=cp.sum(R_amp_square_pos)
		amp_x_diff_amp_sum=cp.sum(R_amp_abs_cp_diff_amp_abs_pos)
		diff_amp_sum=cp.sum(cp_diff_amp_abs_pos)
		scale_factor=amp_x_diff_amp_sum/amp2_sum
	
		diff_amp_scale=cp.sum(cp.absolute(cp_diff_amp_abs_pos-scale_factor*R_amp_abs_pos))
		R_factor=diff_amp_scale/diff_amp_sum
	
	#OS比の計算

	if(i+1==int(iteration)+int(additional_iteration)):
		OS_ratio = float(row) * float(col) * float(sta) / cp.sum(cp_sup)
		
	#gammaの計算

	if(i+1==int(iteration)+int(additional_iteration)):
		cp_dens_pre_real=cp_dens_pre.real
		cp_dens_pre_abs=cp.absolute(cp_dens_pre_real)
		rdens_in=cp.where(cp_sup==1, cp_dens_pre_abs,0.0)
		rdens_out=cp.where(cp_sup!=1, cp_dens_pre_abs,0.0)
#		rdens_in=cp.zeros(cp_diff.shape,dtype="float32")
#		rdens_out=cp.zeros(cp_diff.shape,dtype="float32")

#		rdens_in[cp_sup%2==1]=cp_dens_pre_abs[cp_sup%2==1]
#		rdens_in[cp_sup%2!=1]=0.0
#		rdens_out[cp_sup%2==1]=0.0
#		rdens_out[cp_sup%2!=1]=cp_dens_pre_abs[cp_sup%2!=1]

		rdens_in_sum=cp.sum(rdens_in)
		rdens_out_sum=cp.sum(rdens_out)

		print(rdens_in_sum,rdens_out_sum)

		gamma = rdens_out_sum / ((OS_ratio - 1.0 ) * rdens_in_sum)

	#ログファイルに最終のパラメーター書き出し

		print(str(i+1).zfill(6) + " " + str(scale_factor) + " " + str(R_factor) + " " + str(OS_ratio) + " " + str(gamma))
		with open(log_path, mode='a') as log:
			log.write("\n" + str(i+1).zfill(6) + " " + str(scale_factor) + " " + str(R_factor) + " " + str(OS_ratio) + " " + str(gamma))
	
	#実空間拘束

	cp_dens_bk=cp.real(cp_dens)
	
	cp_dens.real[cp_sup%2==1]=cp_dens_pre.real[cp_sup%2==1]
	cp_dens.real[(cp_dens_pre.real%2<0.0) | (cp_sup%2==0)]=cp_dens_bk[(cp_dens_pre.real%2<0.0) | (cp_sup%2==0)]-0.9*cp_dens_pre.real[(cp_dens_pre.real%2<0.0) | (cp_sup%2==0)]

	cp_dens.imag[:,:,:]=0.0
		
	#OSS mask convolution
	
	if((OSS_flag==1) & ((i+1) <= int(iteration))):
		cp_structure_factor = cp.fft.fftn(cp_dens, axes=(0,1,2),norm="ortho")#【フーリエ変換】
		cp_structure_factor = cp.fft.fftshift(cp_structure_factor)#fftshiftを使ってシフト
		
		if(i==0):
			W_kernel=W(X,Y,Z,OSS_alpha)
			W_kernel=cp.asarray(W_kernel,dtype="float32")
			W_kernel=W_kernel / cp.amax(W_kernel)
#			cp_W_kernel = cp.asarray(np_W_kernel,dtype="float32")

		cp_structure_factor.real=cp_W_kernel*cp_structure_factor.real
		cp_structure_factor.imag=cp_W_kernel*cp_structure_factor.imag

		cp_structure_factor = cp.fft.ifftshift(cp_structure_factor)
		W_dens = cp.fft.ifftn(cp_structure_factor, axes=(0,1,2),norm="ortho")

#		print(str(i+1).zfill(6) + " alpha = " + str(OSS_alpha))

		#OSS 実空間拘束

		cp_dens.real[cp_sup%2==0]=W_dens.real[cp_sup%2==0]
		cp_dens.imag[:,:,:]=0.0

	#OSS parameter update

	if(OSS_flag==1):
		if(((i+1) % int(OSS_interval) == 0) & ((i+1) < int(iteration))):
			OSS_alpha=OSS_alpha-OSS_alpha_step

			W_kernel=W(X,Y,Z,OSS_alpha)
			W_kernel=cp.asarray(W_kernel,dtype="float32")
			W_kernel=W_kernel / cp.amax(W_kernel)
#			tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_np_W_kernel.mrc' ,np_W_kernel)

t5=time.time()

print("total time : " + str(t5-t1))
with open(log_path, mode='a') as log:
	log.write("\n" + "total time : " + str(t5-t1))


