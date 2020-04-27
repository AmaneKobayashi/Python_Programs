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

if (len(sys.argv)==1 | sys.argv[2]=="--help"):
	print("command:python3 HIO-SW.py [-diff] [-sup] [-HIO_iteration] [-header] [-output_interval] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-SW_delta] [-additional_HIO] [-OSS_interval]")
	print("HIO only	: [-diff] [-sup] '''[-HIO_iteration 0]''' [-header] [-additional_HIO]")
	print("HIO-SW	: [-diff] [-sup] [-HIO_iteration] [-header] [-SW_interval] [-initial_SW_ips] [-SW_ips_step] [-last_SW_ips] [-SW_delta]")
	print("OSS	: [-diff] [-sup] [-HIO_iteration] [-header] [-OSS_interval]")
	print("option	: [-output_interval] [-sup_output]")
	exit()

flag_list[:]=0

parameter_name_list[0]="-diff"
parameter_name_list[1]="-sup"
parameter_name_list[2]="-initial_dens"
parameter_name_list[3]="-HIO_iteration"
parameter_name_list[4]="-header"
parameter_name_list[5]="-output_interval"
parameter_name_list[6]="-SW_interval"
parameter_name_list[7]="-initial_SW_ips"
parameter_name_list[8]="-SW_ips_step"
parameter_name_list[9]="-last_SW_ips"
parameter_name_list[10]="-SW_delta"
parameter_name_list[11]="-additional_HIO"
parameter_name_list[12]="-OSS_interval"
n_parameter=13


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
	if(sys.argv[i]=="-HIO_iteration"):
		HIO_iteration=sys.argv[i+1]
		HIO_iteration_flag=1
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
	if(sys.argv[i]=="-additional_HIO"):
		additional_HIO=sys.argv[i+1]
		additional_HIO_flag=1
		flag_list[11]=1
	if(sys.argv[i]=="-OSS_interval"):
		OSS_interval=sys.argv[i+1]
		OSS_flag=1
		flag_list[12]=1
	if(sys.argv[i]=="-sup_output"):
		sup_output_flag=1
		flag_list[13]=1

if(OSS_flag!=1):
	if(HIO_iteration=="0"):
		#HIO only
		for i in range(n_parameter):
			if(flag_list[i]==0 & i != 12 & i !=5 & i != 6 & i !=7 & i != 8 & i != 9 & i != 10 & i != 13):
				print("please input parameter : " + parameter_name_list[i])
				exit()
	else:
		#HIO-SW
		for i in range(n_parameter):
			if(flag_list[i]==0 & i != 12 & i !=5 & i != 13 & i != 11):
				print("please input parameter : " + parameter_name_list[i])
				exit()
else:
	#OSS
	for i in range(n_parameter):
		if(flag_list[i]==0 & i < 5 & i > 11  & i != 13):
			print("please input parameter : " + parameter_name_list[i])	
			exit()

	
#finame=sys.argv[1]
#support=sys.argv[2]
#initial_dens=sys.argv[3]
#HIO_iteration=sys.argv[4]
#header=sys.argv[5]
#output_interval=sys.argv[6]
#SW_interval=sys.argv[7]
#initial_SW_ips=sys.argv[8]
#SW_ips_step=sys.argv[9]
#last_SW_ips=sys.argv[10]
#SW_delta=sys.argv[11]
#additional_HIO=sys.argv[12]



print("finame = " + finame)
print("support = " + support)
print("initial_dens = " + initial_dens)
print("HIO_iteration = " + HIO_iteration)
print("header = " + header)
if(flag_list[5]==0):
	output_interval=10000000000
	print("final output only")
else:
	print("output_interval = " +output_interval)
if(OSS_flag!=1):
	if(HIO_iteration=="0"):
		print("additional_HIO = " + additional_HIO)
		print("HIO only mode")
	else:
		print("SW_interval = " + SW_interval)
		print("initial_SW_ips = " + initial_SW_ips)
		print("SW_ips_step = " + SW_ips_step)
		print("last_SW_ips = " + last_SW_ips)
		print("SW_delta = " + SW_delta)
		if(-additional_HIO!=1):
			additional_HIO="0"
		print("additional_HIO = " + additional_HIO)
		print("HIO-SW mode")
else:
	print("OSS mode")
print("")

log_text="\n\n" + "diff = " + finame + "\n" + "support = " + support + "\n" + "initial_dens = " + initial_dens + "\n" + "HIO_iteration = " + HIO_iteration + "\n" + "header = " + header
if(flag_list[5]==0):
	log_text=log_text + "\n" + "output_interval = final output only"
else:
	log_text=log_text + "\n" + "output_interval = " + output_interval
if(OSS_flag!=1):
	if(HIO_iteration=="0"):
		log_text=log_text + "\n" + "additional_HIO = " + additional_HIO + "\n"
		log_text=log_text + "\n" + "HIO only mode\n"
	else:
		log_text=log_text + "\n" + "SW_interval = " + SW_interval + "\n" + "initial_SW_ips = " + initial_SW_ips + "\n" + "SW_ips_step = " + SW_ips_step + "\n" + "last_SW_ips = " + last_SW_ips
		log_text=log_text + "\n" + "SW_delta = " + SW_delta + "\n" + "additional_HIO = " + additional_HIO + "\n"
		log_text=log_text + "\n" + "HIO-SW mode\n"
else:
	log_text=log_text + "\nOSS mode\n"
t1=time.time()

diff=Image.open(finame)
sup=Image.open(support)
initial_dens=Image.open(initial_dens)

np_diff=np.asarray(diff,dtype="float32")
np_sup=np.asarray(sup,dtype="float32")
np_initial_dens=np.asarray(initial_dens,dtype="float32")

print("np_diff dtype = " + str(np_diff.dtype))
print("np_sup dtype = " + str(np_sup.dtype))
print("np_initial_dens dtype = " + str(np_initial_dens.dtype))
print("")

print("diff size = " + str(diff.size))
print("sup size = " + str(sup.size))
print("initial_dens size = " + str(initial_dens.size))
print("diff type image size = " + str(type(diff.size)))
print("sup type image size = " + str(type(sup.size)))
print("intiial_dens type image size = " + str(type(initial_dens.size)))

row=np_diff.shape[0]
col=np_diff.shape[1]
print("row of diff = " + str(row))
print("col of diff = " + str(col))
print("")

log_text=log_text + "diff size = " + str(diff.size) +"\n\n"

#ディレクトリ作成
os.makedirs(header, exist_ok=True)
if(header.rfind("/") == -1):
	header=header + "/" +header
	log_path=header + "_MPR.log"
else:
	header=header + "/" +header[header.rfind("/")+1:len(header)]
	log_path=header + "_MPR.log"

with open(log_path, mode='w') as log:
	log.write("program HIO-SW ver.20190304")
with open(log_path, mode='a') as log:
	log.write(log_text)
#実験値振幅
np_diff_amp=np.zeros(diff.size,dtype="float32")
np_diff_amp[np_diff%2>=0]=np.sqrt(np_diff[np_diff%2>=0])
#tifffile.imsave(header + '_np_diff_amp.tif' ,np_diff_amp)

#電子密度の複素数化
np_dens=np.array(np_initial_dens, dtype=np.complex64)

#SW用ガウシアンの定義
def G(x,y,ips):
	Z=np.exp(-((x-row/2)**2 + (y-col/2)**2) / (2 * ips**2)) / (2 * np.pi * ips**2)
	return Z
x=y=np.arange(0,int(row),1)
X,Y=np.meshgrid(x,y)
SW_ips=float(initial_SW_ips)

#OSS用マスクの定義
def W(x,y,alpha):
	M=np.exp(-0.5*((x-row/2)**2 + (y-col/2)**2)/alpha**2)
	return M
OSS_alpha=float(row)
OSS_alpha_step=(float(row)-1.0/float(row)) / (float(HIO_iteration)/float(OSS_interval))

#numpy配列 ⇒ cupy配列に変換
cp_diff_amp = cp.asarray(np_diff_amp,dtype="float32")
cp_sup = cp.asarray(np_sup,dtype="float32")
cp_initial_dens = cp.asarray(np_initial_dens,dtype="float32")
cp_dens=cp.asarray(np_dens)

print("iteration scale_factor Rfactor")
with open(log_path, mode='a') as log:
	log.write("iteration scale_factor Rfactor")

for i in range(int(HIO_iteration)+int(additional_HIO)):

	cp_structure_factor = cp.fft.fft2(cp_dens,norm="ortho")#【フーリエ変換】
	cp_structure_factor = cp.fft.fftshift(cp_structure_factor)#fftshiftを使ってシフト
	cp_amp = cp.absolute(cp_structure_factor)#絶対値をとる

	#OSS mask convolution
	
	if(OSS_flag==1 & i>=2):
		W_kernel=W(X,Y,OSS_alpha)
		np_W_kernel=np.asarray(W_kernel,dtype="float32")
		np_W_kernel=np_W_kernel / np.amax(np_W_kernel)
		cp_W_kernel = cp.asarray(np_W_kernel,dtype="float32")

		cp_structure_factor.real=cp_W_kernel*cp_structure_factor.real
		cp_structure_factor.imag=cp_W_kernel*cp_structure_factor.imag

		cp_structure_factor = cp.fft.ifftshift(cp_structure_factor)
		W_dens = cp.fft.ifft2(cp_structure_factor,norm="ortho")

		#OSS 実空間拘束

		cp_dens.real[cp_sup%2==0]=W_dens[cp_sup%2==0]
		cp_dens.imag[:,:]=0.0

	#OSS parameter update

	if((OSS_flag==1) & ((i+1) % int(OSS_interval) == 0)):
		OSS_alpha=OSS_alpha-OSS_alpha_step

	#逆空間拘束

	cp_structure_factor[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)]=cp_structure_factor[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)] * cp_diff_amp[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)] / cp_amp[(cp_diff_amp%2>0.0) & (cp_amp%2!=0.0)]
	
	#電子密度の出力

	cp_structure_factor = cp.fft.ifftshift(cp_structure_factor)
	cp_dens_pre = cp.fft.ifft2(cp_structure_factor,norm="ortho")
	cp_dens_pre_real=cp.real(cp_dens_pre)

	if((i+1) % int(output_interval) == 0):
		np_dens_pre_real = cp.asnumpy(cp_dens_pre_real)#cupy配列 ⇒ numpy配列に変換
		tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_rdens.tif' ,np_dens_pre_real)

	#最後の電子密度とパターンの出力
	
	R_dens=cp_dens_pre
	R_dens.real=R_dens.real*cp_sup
	R_dens.imag[:,:]=0
	R_structure_factor = cp.fft.fft2(R_dens,norm="ortho")
	R_structure_factor = cp.fft.fftshift(R_structure_factor)

	if(i+1==int(HIO_iteration)+int(additional_HIO)):
		np_R_dens = cp.asnumpy(cp.real(R_dens))#cupy配列 ⇒ numpy配列に変換
		tifffile.imsave(header + '_final_rdens.tif' ,np_R_dens)

		R_amp = cp.absolute(R_structure_factor)
		np_R_amp = cp.asnumpy(cp.square(R_amp))#cupy配列 ⇒ numpy配列に変換
		tifffile.imsave(header + '_final_diff.tif' ,np_R_amp)

		np_last_sup = cp.asnumpy(cp_sup)
		tifffile.imsave(header + '_final_sup.tif' ,np_last_sup)

	#R-factorの計算

	R_amp = cp.absolute(R_structure_factor)
	amp2_sum=cp.sum(cp.square(R_amp[cp_diff_amp%2>0.0]))
	amp_x_diff_amp_sum=cp.sum(cp.absolute(R_amp[cp_diff_amp%2>0.0])*cp.absolute(cp_diff_amp[cp_diff_amp%2>0.0]))
	diff_amp_sum=cp.sum(cp.absolute(cp_diff_amp[cp_diff_amp%2>0.0]))
	scale_factor=amp_x_diff_amp_sum/amp2_sum

	diff_amp_scale=cp.sum(cp.absolute(cp.absolute(cp_diff_amp[cp_diff_amp%2>0.0])-scale_factor*cp.absolute(R_amp[cp_diff_amp%2>0.0])))
	R_factor=diff_amp_scale/diff_amp_sum

	print(str(i+1).zfill(6) + " " + str(scale_factor) + " " + str(R_factor))
	with open(log_path, mode='a') as log:
		log.write("\n" + str(i+1).zfill(6) + " " + str(scale_factor) + " " + str(R_factor))

	#Shrink Wrap

	if(((i+1) % int(SW_interval) == 0) & ((i+1) <= int(HIO_iteration))):
		if(SW_ips<=float(last_SW_ips)):
			SW_ips=float(last_SW_ips)
		
		print("SW_ips = " + str(SW_ips))

		G_kernel=G(X,Y,SW_ips)
		np_G_kernel=np.asarray(G_kernel,dtype="float32")
		np_G_kernel=np_G_kernel / np.amax(np_G_kernel)
#		tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_G_kernel.tif' ,np_G_kernel))
		cp_G_kernel = cp.asarray(np_G_kernel,dtype="float32")

		R_structure_factor.real=cp_G_kernel*R_structure_factor.real
		R_structure_factor.imag=cp_G_kernel*R_structure_factor.imag
		
		R_structure_factor = cp.fft.ifftshift(R_structure_factor)
		G_dens = cp.fft.ifft2(R_structure_factor,norm="ortho")

#		np_G_dens = cp.asnumpy(G_dens.real)
#		tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_G_dens.tif' ,np_G_dens)

		cp_sup.real[G_dens.real%2>=float(SW_delta)*cp.amax(G_dens.real)]=1
		cp_sup.real[G_dens.real%2<float(SW_delta)*cp.amax(G_dens.real)]=0
		cp_sup.imag[:,:]=0

		SW_ips=SW_ips*float(SW_ips_step)

		if(sup_output_flag==1):
			np_sup = cp.asnumpy(cp_sup)
			tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_sup.tif' ,np_sup)

	#実空間拘束

	cp_dens_bk=cp.real(cp_dens)
	
	cp_dens.real[cp_sup%2==1]=cp_dens_pre_real[cp_sup%2==1]
	cp_dens.real[(cp_dens_pre_real%2<0.0) | (cp_sup%2==0)]=cp_dens_pre_real[(cp_dens_pre_real%2<0.0) | (cp_sup%2==0)]-0.9*cp_dens_bk[(cp_dens_pre_real%2<0.0) | (cp_sup%2==0)]
#	cp_dens.real[cp_sup%2==0]=cp_dens_pre_real[cp_sup%2==0]-0.9*cp_dens_bk[cp_sup%2==0]

	cp_dens.imag[:,:]=0
		


t5=time.time()

print("total time : " + str(t5-t1))



