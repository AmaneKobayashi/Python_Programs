#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#NORŒvŽZ•û–@‚ÌŒŸ“¢
#
import sys
import numpy as np
import glob
import os
import mrcfile
import tifffile
import pandas as pd
from PIL import Image
#
if(len(sys.argv)!=3):
	print("commnad = python.exe NOR_test.py erode_file autocorrelation_file")
	exit()
#
erode_file=sys.argv[1]
erode_file=erode_file.replace("\\","/")
print("erode_file = " + erode_file)
autocorrelation_file=sys.argv[2]
autocorrelation_file=autocorrelation_file.replace("\\","/")
print("autocorrelation_file = " + autocorrelation_file)
#
with mrcfile.open(autocorrelation_file,permissive=True) as mrc:
	np_autocorrelation=np.asarray(mrc.data,dtype="float32")
mrc.close
#
n=np_autocorrelation.shape[0]
row=np_autocorrelation.shape[1]
col=np_autocorrelation.shape[2]
print("number of autocorrelation image :" + str(n))
#
np_erode=Image.open(erode_file)
np_erode=np.asarray(np_erode,dtype="float32")
stack_np_erode=[np_erode]*n
#
R_amp = np_autocorrelation
R_amp_square=np.square(R_amp)
R_amp_abs=np.absolute(R_amp)
cp_diff_amp_abs=stack_np_erode

R_amp_abs_cp_diff_amp_abs=R_amp_abs*cp_diff_amp_abs

R_amp_square_pos=np.zeros(np_autocorrelation.shape,dtype="float32")
R_amp_abs_pos=np.zeros(np_autocorrelation.shape,dtype="float32")
R_amp_abs_cp_diff_amp_abs_pos=np.zeros(np_autocorrelation.shape,dtype="float32")
cp_diff_amp_pos=np.zeros(np_autocorrelation.shape,dtype="float32")
cp_diff_amp_abs_pos=np.zeros(np_autocorrelation.shape,dtype="float32")

R_amp_square_pos=R_amp_square
R_amp_abs_pos=R_amp_abs
R_amp_abs_cp_diff_amp_abs_pos=R_amp_abs_cp_diff_amp_abs
cp_diff_amp_pos=cp_diff_amp_abs
cp_diff_amp_abs_pos=cp_diff_amp_abs

amp2_sum=np.sum(R_amp_square_pos, axis=(1,2))
amp_x_diff_amp_sum=np.sum(R_amp_abs_cp_diff_amp_abs_pos, axis=(1,2))
diff_amp_sum=np.sum(cp_diff_amp_abs_pos, axis=(1,2))
scale_factor=amp_x_diff_amp_sum/amp2_sum

scale_factor_3D=np.repeat(scale_factor,int(row)*int(col))
scale_factor_3D=scale_factor_3D.reshape(n,int(row),int(col))

diff_amp_scale=np.sum(np.absolute(cp_diff_amp_abs_pos-scale_factor_3D*R_amp_abs_pos), axis=(1,2))
R_factor=diff_amp_scale/diff_amp_sum

n_min_R=np.argmin(R_factor)
min_R=np.min(R_factor)

print(R_factor)
print("n_min_R : " + str(n_min_R) + ", min_R : " + str(min_R))