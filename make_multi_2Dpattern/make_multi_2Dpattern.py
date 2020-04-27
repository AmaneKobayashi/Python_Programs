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
from skimage import io

if ((len(sys.argv)==1)):
#	print("command:python3 multi_2Dpattern.py [-diff] [-n_diff] [-multiply_2D_data]")
	print("command:python3 multi_2Dpattern.py [-diff] [-n_diff]")
	exit()

n_parameter=2
parameter_name_list=[""]*n_parameter
flag_list=[0]*n_parameter

parameter_name_list[0]="-diff"
parameter_name_list[1]="-n_diff"
#parameter_name_list[2]="-multiply_2D_data"

input_parameter=0

for i in range(len(sys.argv)):
	if(sys.argv[i]=="-diff"):
		finame=sys.argv[i+1]
		flag_list[0]=1
	if(sys.argv[i]=="-n_diff"):
		n_diff=sys.argv[i+1]
		flag_list[1]=1
#	if(sys.argv[i]=="-multiply_2D_data"):
#		multiply_2D_data=sys.argv[i+1]
#		flag_list[2]=1
	if(sys.argv[i]=="--help"):
		print("command:python3 multi_2Dpattern.py [-diff] [-n_diff]")
		exit()

for i in range(n_parameter):
	if(flag_list[i]==0):
		print("please input parameter : [" + parameter_name_list[i] + "]")
		input_parameter=1
if(input_parameter==1):
	exit()

print("diff = " + finame)
print("n_diff = " + n_diff)
#print("n_diff = " + multiply_2D_data)

diff=Image.open(finame)
cp_diff=cp.asarray(diff,dtype="float32")

#m_data=Image.open(multiply_2D_data)
#cp_m_data=cp.asarray(m_data,dtype="float32")

t1=time.time()

multi_pattern=[cp_diff]*int(n_diff)

t2=time.time()
print("make array : " + str(t2-t1))

multi_pattern=cp.asarray(multi_pattern,dtype="float32")

t3=time.time()
print("cupy convert : " + str(t3-t2))

multi_pattern=cp.flip(multi_pattern,axis=1)

print("2D_multi shape = " + str(multi_pattern.shape))

#multi_pattern=multi_pattern*cp_m_data	#全ての階層に適用される。

multi_pattern = cp.asnumpy(multi_pattern)#cupy配列 ⇒ numpy配列に変換
foname = finame[finame.rfind("/")+1:len(finame)-4] + "_n_" + n_diff + ".mrc" 
with mrcfile.new(foname, overwrite=True) as mrc:
	mrc.set_data(multi_pattern)
mrc.close

t5=time.time()
print("total time : " + str(t5-t1))









