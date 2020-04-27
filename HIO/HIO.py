#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import cupy as cp
import time
import tifffile

from PIL import Image
from skimage import io

#if len(sys.argv)!=5:
#	print("command:python3 test_fft_tiff.py finame support fft_iteration header")
#	exit()

finame=sys.argv[1]
support=sys.argv[2]
initial_dens=sys.argv[3]
fft_iteration=sys.argv[4]
header=sys.argv[5]

print("finame = " + finame)
print("support = " + support)
print("initial_dens = " + initial_dens)
print("fft_iteration = " + fft_iteration)
print("header = " + header)
print("")

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
#print("(" + x + " ," + y +") での値 = " + str(np_img[int(y),int(x)]) + " (imageJの表示と同じ)")
print("")

t2=time.time()

#io.imshow(np_img)
#io.show()#表示

#tifffile.imsave(header + 'np_diff.tif' ,np_diff)
np_diff_amp=np.sqrt(np_diff)
#tifffile.imsave(header + 'np_diff_amp.tif' ,np_diff_amp)

np_dens=np.array(np_initial_dens, dtype=np.complex64)
#print(np_dens)
#numpy配列 ⇒ cupy配列に変換
cp_diff_amp = cp.asarray(np_diff_amp,dtype="float32")
cp_sup = cp.asarray(np_sup,dtype="float32")
cp_initial_dens = cp.asarray(np_initial_dens,dtype="float32")
cp_dens=cp.asarray(np_dens)
#print(cp_dens)

t3=time.time()

print("iteration scale_factor Rfactor")

for i in range(int(fft_iteration)):


#	cp_dens_pre_real=cp.real(cp_dens)
#	np_dens_pre_real = cp.asnumpy(cp_dens_pre_real)
#	tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_loop_dens.tif' ,np_dens_pre_real)
#	print(cp_dens)
	cp_structure_factor = cp.fft.fft2(cp_dens,norm="ortho")#【フーリエ変換】
	cp_structure_factor = cp.fft.fftshift(cp_structure_factor)#fftshiftを使ってシフト
	cp_amp = cp.absolute(cp_structure_factor)#絶対値をとる
	
	t4=time.time()

	#逆空間拘束

	cp_structure_factor[cp_amp%2!=0]=cp_structure_factor[cp_amp%2!=0] * cp_diff_amp[cp_amp%2!=0] / cp_amp[cp_amp%2!=0]
#	cp_structure_factor.real[cp_amp%2==0]=cp_diff_amp[cp_amp%2==0] / cp.sqrt(2.0)
#	cp_structure_factor.imag[cp_amp%2==0]=cp_diff_amp[cp_amp%2==0] / cp.sqrt(2.0)
	
#	np_amp=cp.asnumpy(cp_amp)
#	tifffile.imsave(header + "_" + str(i+1) + '_np_amp.tif' ,np_amp)

#	cp_structure_factor_abs=cp.absolute(cp_structure_factor)
#	np_structure_factor=cp.asnumpy(cp_structure_factor_abs)
#	tifffile.imsave(header + "_" + str(i+1) + '_np_structure_factor_abs.tif' ,np_structure_factor)

	#

	cp_structure_factor = cp.fft.ifftshift(cp_structure_factor)
	cp_dens_pre = cp.fft.ifft2(cp_structure_factor,norm="ortho")
	cp_dens_pre_real=cp.real(cp_dens_pre)
	cp_dens_pre_imag=cp.imag(cp_dens_pre)
	cp_dens_pre_abs=cp.absolute(cp_dens_pre)

	#R-factorの計算
	
	R_dens=cp_dens_pre*cp_sup
	R_structure_factor = cp.fft.fft2(R_dens,norm="ortho")
	R_structure_factor = cp.fft.fftshift(R_structure_factor)
	R_amp = cp.absolute(R_structure_factor)

	if(i+1==int(fft_iteration)):
		np_R_amp = cp.asnumpy(cp.square(R_amp))#cupy配列 ⇒ numpy配列に変換
		tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_diff.tif' ,np_R_amp)

	amp2_sum=cp.sum(cp.square(R_amp))
	amp_x_diff_amp_sum=cp.sum(cp.absolute(R_amp)*cp.absolute(cp_diff_amp))
	diff_amp_sum=cp.sum(cp.absolute(cp_diff_amp))
	scale_factor=amp_x_diff_amp_sum/amp2_sum

	diff_amp_scale=cp.sum(cp.absolute(cp.absolute(cp_diff_amp)-scale_factor*cp.absolute(R_amp)))
	R_factor=diff_amp_scale/diff_amp_sum

	print(str(i+1).zfill(6) + " " + str(scale_factor) + " " + str(R_factor))

	#
	if(i+1==int(fft_iteration)):
		np_dens_pre_real = cp.asnumpy(cp_dens_pre_real)#cupy配列 ⇒ numpy配列に変換
		tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_dens.tif' ,np_dens_pre_real)
		np_R_dens = cp.asnumpy(cp.real(R_dens))#cupy配列 ⇒ numpy配列に変換
		tifffile.imsave(header + '_final_rdens.tif' ,np_R_dens)
#		np_dens_pre_imag = cp.asnumpy(cp_dens_pre_imag)#cupy配列 ⇒ numpy配列に変換
#		tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_dens_imag.tif' ,np_dens_pre_imag)
#		np_dens_pre_abs = cp.asnumpy(cp_dens_pre_abs)#cupy配列 ⇒ numpy配列に変換
#		tifffile.imsave(header + "_" + str(i+1).zfill(6) + '_dens_abs.tif' ,np_dens_pre_abs)

	#実空間拘束

	cp_dens_bk=cp.real(cp_dens)
	
	cp_dens.real[cp_sup%2==1]=cp_dens_pre_real[cp_sup%2==1]
	cp_dens.real[cp_dens_pre_real%2<0.0]=cp_dens_pre_real[cp_dens_pre_real%2<0.0]-0.9*cp_dens_bk[cp_dens_pre_real%2<0.0]
	cp_dens.real[cp_sup%2==0]=cp_dens_pre_real[cp_sup%2==0]-0.9*cp_dens_bk[cp_sup%2==0]

	cp_dens.imag[:,:]=0

#	np_dens=cp.asnumpy(cp_dens)
#	tifffile.imsave(header + "_" + str(i+1) + '_np_dens.tif' ,np_dens)


#io.imshow(np_fpow)
#io.show()#表示

t5=time.time()

#print("time open file : " + str(t2-t1))
#print("time numpy配列 ⇒ cupy配列に変換 : " + str(t3-t2))
#print("time フーリエ変換 回数=" + fft_iteration + " : "+ str(t4-t3))
#print("time 絶対値をとる + cupy配列 ⇒ numpy配列に変換 : " + str(t5-t4))
print("total time : " + str(t5-t1))



