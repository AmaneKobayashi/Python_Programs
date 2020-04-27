#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import cupy as cp
import time
import tifffile
import os

from PIL import Image
from skimage import io

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

diff=Image.open(finame)
sup=Image.open(support)
init_dens=Image.open(initial_dens)

np_diff=np.asarray(diff,dtype="float32")
np_sup=np.asarray(sup,dtype="float32")
np_initial_dens=np.asarray(init_dens,dtype="float32")

print("np_diff dtype = " + str(np_diff.dtype))
print("np_sup dtype = " + str(np_sup.dtype))
print("np_initial_dens dtype = " + str(np_initial_dens.dtype))
print("")

print("diff size = " + str(diff.size))
print("sup size = " + str(sup.size))
print("initial_dens size = " + str(init_dens.size))
#print("diff type image size = " + str(type(diff.size)))
#print("sup type image size = " + str(type(sup.size)))
#print("intiial_dens type image size = " + str(type(init_dens.size)))

row=np_diff.shape[0]
col=np_diff.shape[1]
print("row of diff = " + str(row))
print("col of diff = " + str(col))
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
	log.write("program Phase retrieval I0 mode ver.20190315")

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

log_text=log_text + "diff size = " + str(diff.size) +"\n\n"

with open(log_path, mode='a') as log:
	log.write(log_text)

#

t1=time.time()

#実験値振幅
np_diff_amp=np.zeros(diff.size,dtype="float32")
np_diff_amp=np.where(np_diff>0.0,np_diff,0.0)
np_diff_amp=np.sqrt(np_diff_amp)
#np_diff_amp[np_diff%2>=0.0]=np.sqrt(np_diff[np_diff%2>=0.0])
#np_diff_amp=np.where(np_diff>0.0,np.sqrt(np_diff),0.0)
#tifffile.imsave(header + '_np_diff_amp.tif' ,np_diff_amp)

#電子密度の複素数化

np_dens=np.array(np_initial_dens, dtype=np.complex64)

#SW用ガウシアンの定義
if((iteration!="0") & (OSS_flag==0)):
	def G(x,y,ips):
		Z=np.exp(-((x-row/2)**2 + (y-col/2)**2) / (2 * ips**2)) / (2 * np.pi * ips**2)
		return Z
	x=y=np.arange(0,int(row),1)
	X,Y=np.meshgrid(x,y)
	SW_ips=float(initial_SW_ips)

#OSS用マスクの定義
if(OSS_flag==1):
	def W(x,y,alpha):
		M=np.exp(-0.5*((x-row/2)**2 + (y-col/2)**2)/alpha**2)
		return M
	x=y=np.arange(0,int(row),1)
	X,Y=np.meshgrid(x,y)
	OSS_alpha=float(row)
	OSS_alpha_step=(float(row)-1.0/float(row)) / (float(iteration)/float(OSS_interval)-1.0)

#重心計算関係

cx=cy=cp.arange(0,int(row),1)

#numpy配列 ⇒ cupy配列に変換

cp_diff_amp = cp.asarray(np_diff_amp,dtype="float32")
cp_sup = cp.asarray(np_sup,dtype="float32")
cp_initial_dens = cp.asarray(np_initial_dens,dtype="float32")
cp_dens=cp.asarray(np_dens)

print("iteration scale_factor Rfactor OS_ratio gamma")
with open(log_path, mode='a') as log:
	log.write("iteration scale_factor Rfactor OS_ratio gamma")

for i in range(int(iteration)+int(additional_iteration)):

	cp_structure_factor = cp.fft.fft2(cp_dens,norm="ortho")#【フーリエ変換】
	cp_structure_factor = cp.fft.fftshift(cp_structure_factor)#fftshiftを使ってシフト
	cp_amp = cp.absolute(cp_structure_factor)#絶対値をとる

	#逆空間拘束

#	np_amp = cp.asnumpy(cp.square(cp_amp))#cupy配列 ⇒ numpy配列に変換
#	tifffile.imsave('diff.tif' ,np_amp)
#	exit()

	cp_structure_factor[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)]=cp_structure_factor[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)] * cp_diff_amp[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)] / cp_amp[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)]

	cp_structure_factor_bk=cp_structure_factor

	cp_structure_factor = cp.fft.ifftshift(cp_structure_factor)
	cp_dens_pre = cp.fft.ifft2(cp_structure_factor,norm="ortho")
	cp_dens_pre_real=cp.real(cp_dens_pre)

	#重心の計算
	
	if((i+1==int(iteration)+int(additional_iteration)) & (OSS_flag!=1) & (iteration!="0")):	
		weight_sum=cp.sum(cp_dens_pre_real)
		x_axis_sum=cp.sum(cp_dens_pre_real, axis=1)
		y_axis_sum=cp.sum(cp_dens_pre_real, axis=0)
	
		x_sum=cp.sum(x_axis_sum * cx)
		y_sum=cp.sum(y_axis_sum * cy)
	
		if(int((x_sum / weight_sum)*10.0) % 10 >= 5):
			x_G = int(x_sum / weight_sum)+1
		else:
			x_G = int(x_sum / weight_sum)
		if(int((y_sum / weight_sum)*10.0) % 10 >= 5):
			y_G = int(y_sum / weight_sum)+1
		else:
			y_G = int(y_sum / weight_sum)
	
#		print(str(i+1).zfill(6) + ", " + str(x_G) + ", " + str(y_G))

		cp_dens_pre=cp.roll(cp_dens_pre, int(row/2)-x_G, axis=1)
		cp_dens_pre=cp.roll(cp_dens_pre, int(col/2)-y_G, axis=0)	

		cp_sup=cp.roll(cp_sup, int(row/2)-x_G, axis=1)	
		cp_sup=cp.roll(cp_sup, int(row/2)-y_G, axis=0)
	
	#電子密度の出力

	if((i+1) % int(output_interval) == 0):
		np_dens_pre_real = cp.asnumpy(cp_dens_pre_real)#cupy配列 ⇒ numpy配列に変換
		tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_rdens.tif' ,np_dens_pre_real)

	#最後の電子密度とパターンの出力

	if(i+1==int(iteration)+int(additional_iteration)):	
		R_dens=cp_dens_pre
		R_dens_real_sum=cp.sum(R_dens.real)
		R_dens.real=R_dens.real*cp_sup
		back_ground=(R_dens_real_sum-cp.sum(R_dens.real))/(float(row)*float(col))
		print(back_ground)
		R_dens.real[:,:]=R_dens.real[:,:] + back_ground
		R_dens.imag[:,:]=0

		R_structure_factor = cp.fft.fft2(R_dens,norm="ortho")
		R_structure_factor = cp.fft.fftshift(R_structure_factor)

	if(i+1==int(iteration)+int(additional_iteration)):
		np_R_dens = cp.asnumpy(cp.real(R_dens))#cupy配列 ⇒ numpy配列に変換
		tifffile.imsave(header + '_final_rdens.tif' ,np_R_dens)

		R_amp = cp.absolute(R_structure_factor)
		np_R_amp = cp.asnumpy(cp.square(R_amp))#cupy配列 ⇒ numpy配列に変換
		tifffile.imsave(header + '_final_diff.tif' ,np_R_amp)

		np_last_sup = cp.asnumpy(cp_sup)
		tifffile.imsave(header + '_final_sup.tif' ,np_last_sup)

	#R-factorの計算

	if(i+1==int(iteration)+int(additional_iteration)):
		R_amp = cp.absolute(R_structure_factor)
		amp2_sum=cp.sum(cp.square(R_amp[cp_diff_amp%2>0.0]))
		amp_x_diff_amp_sum=cp.sum(cp.absolute(R_amp[cp_diff_amp%2>0.0])*cp.absolute(cp_diff_amp[cp_diff_amp%2>0.0]))
		diff_amp_sum=cp.sum(cp.absolute(cp_diff_amp[cp_diff_amp%2>0.0]))
		scale_factor=amp_x_diff_amp_sum/amp2_sum
	
		diff_amp_scale=cp.sum(cp.absolute(cp.absolute(cp_diff_amp[cp_diff_amp%2>0.0])-scale_factor*cp.absolute(R_amp[cp_diff_amp%2>0.0])))
		R_factor=diff_amp_scale/diff_amp_sum
	
	#OS比の計算

	if(i+1==int(iteration)+int(additional_iteration)):
		OS_ratio = float(row) * float(col) / cp.sum(cp_sup)
		
	#gammaの計算

	if(i+1==int(iteration)+int(additional_iteration)):
		cp_dens_pre_abs=cp.absolute(cp_dens_pre_real)
		rdens_in=cp.where(cp_sup==1, cp_dens_pre_abs,0.0)
		rdens_out=cp.where(cp_sup!=0, cp_dens_pre_abs,0.0)

		rdens_in_sum=cp.sum(rdens_in)
		rdens_out_sum=cp.sum(rdens_out)

		gamma = rdens_out_sum / ((OS_ratio - 1.0 ) * rdens_in_sum)

	#ログファイルに最終のパラメーター書き出し

		print(str(i+1).zfill(6) + " " + str(scale_factor) + " " + str(R_factor) + " " + str(OS_ratio) + " " + str(gamma))
		with open(log_path, mode='a') as log:
			log.write("\n" + str(i+1).zfill(6) + " " + str(scale_factor) + " " + str(R_factor) + " " + str(OS_ratio) + " " + str(gamma))

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
				np_G_kernel=np.asarray(G_kernel,dtype="float32")
				np_G_kernel=np_G_kernel / np.amax(np_G_kernel)
#				tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_G_kernel.tif' ,np_G_kernel)
				cp_G_kernel = cp.asarray(np_G_kernel,dtype="float32")

				sub_cp_dens_pre_real=cp_dens_pre_real-cp.average(cp_dens_pre_real)
				cp_structure_factor_bk = cp.fft.fft2(sub_cp_dens_pre_real,norm="ortho")
				cp_structure_factor_bk = cp.fft.fftshift(cp_structure_factor_bk)				

				cp_structure_factor_bk.real=cp_G_kernel*cp_structure_factor_bk.real
				cp_structure_factor_bk.imag=cp_G_kernel*cp_structure_factor_bk.imag
				
				cp_structure_factor_bk = cp.fft.ifftshift(cp_structure_factor_bk)
				G_dens = cp.fft.ifft2(cp_structure_factor_bk,norm="ortho")

#				np_G_dens = cp.asnumpy(cp.abs(cp_structure_factor_bk))
#				tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_cp_structure_factor_bk.tif' ,np_G_dens)	
	
#				np_G_dens = cp.asnumpy(G_dens.real)
#				tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_G_dens.tif' ,np_G_dens)

				G_dens_real=G_dens.real
#				G_dens_real_average=cp.average(G_dens_real)
#				threshold = float(SW_delta)*(cp.amax(G_dens_real)-G_dens_real_average) + G_dens_real_average
				threshold = float(SW_delta)*cp.amax(G_dens_real)
				
				cp_sup=cp.where(G_dens_real>=threshold,float(1),float(0))
				cp_sup=cp_sup.astype(cp.float32)				
		
				SW_ips=SW_ips*float(SW_ips_step)
		
				if(SW_sup_output_flag==1):
					np_sup = cp.asnumpy(cp_sup)
					tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_sup.tif' ,np_sup)
	
	#実空間拘束

	cp_dens_bk=cp.real(cp_dens)

	cp_dens_pre_real_average=cp.average(cp_dens_pre_real)
	cp_dens_pre_real_sub=cp_dens_pre_real-cp_dens_pre_real_average
	cp_dens_bk_sub=cp_dens_bk-cp_dens_pre_real_average
	
	cp_dens.real[cp_sup%2==1]=cp_dens_pre_real_sub[cp_sup%2==1]
	cp_dens.real[(cp_dens_pre_real_sub%2<0.0) | (cp_sup%2==0)]=cp_dens_bk_sub[(cp_dens_pre_real_sub%2<0.0) | (cp_sup%2==0)]-0.9*cp_dens_pre_real_sub[(cp_dens_pre_real_sub%2<0.0) | (cp_sup%2==0)]
	cp_dens.real[:,:]=cp_dens.real[:,:]+cp_dens_pre_real_average

	cp_dens.imag[:,:]=0.0
		
	#OSS mask convolution
	
	if((OSS_flag==1) & ((i+1) <= int(iteration))):
		cp_structure_factor = cp.fft.fft2(cp_dens,norm="ortho")#【フーリエ変換】
		cp_structure_factor = cp.fft.fftshift(cp_structure_factor)#fftshiftを使ってシフト
		
		if(i==0):
			W_kernel=W(X,Y,OSS_alpha)
			np_W_kernel=np.asarray(W_kernel,dtype="float32")
			np_W_kernel=np_W_kernel / np.amax(np_W_kernel)
			cp_W_kernel = cp.asarray(np_W_kernel,dtype="float32")

		cp_structure_factor.real=cp_W_kernel*cp_structure_factor.real
		cp_structure_factor.imag=cp_W_kernel*cp_structure_factor.imag

		cp_structure_factor = cp.fft.ifftshift(cp_structure_factor)
		W_dens = cp.fft.ifft2(cp_structure_factor,norm="ortho")

#		print(str(i+1).zfill(6) + " alpha = " + str(OSS_alpha))

		#OSS 実空間拘束

		cp_dens.real[cp_sup%2==0]=W_dens.real[cp_sup%2==0]
		cp_dens.imag[:,:]=0.0

	#OSS parameter update

	if(OSS_flag==1):
		if(((i+1) % int(OSS_interval) == 0) & ((i+1) < int(iteration))):
			OSS_alpha=OSS_alpha-OSS_alpha_step

			W_kernel=W(X,Y,OSS_alpha)
			np_W_kernel=np.asarray(W_kernel,dtype="float32")
			np_W_kernel=np_W_kernel / np.amax(np_W_kernel)
			cp_W_kernel = cp.asarray(np_W_kernel,dtype="float32")
#			tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_np_W_kernel.tif' ,np_W_kernel)

t5=time.time()

print("total time : " + str(t5-t1))



