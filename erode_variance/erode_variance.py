#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#erode画像のぶんさんをけいさん
#値を**_MPR.logから読み込む　→　csvファイルにruntag,varianceなどを書き込む
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
n_parameter=1
parameter_name_list=[""]*n_parameter
flag_list=[0]*n_parameter
#
parameter_name_list[0]="-main_directory"
#
main_directory_flag=0
#
input_parameter=0
for i in range(len(sys.argv)):
	if(sys.argv[i]=="-main_directory"):
		main_directory=sys.argv[i+1]
		main_directory_flag=1
		flag_list[0]=1
	if(sys.argv[i]=="--help"):
		print("command = python.exe rdens_min_NOR.py [-main_directory]")

for i in range(n_parameter):
	if(flag_list[i]==0):
		print("please input parameter : [" + parameter_name_list[i] + "]")
#\→/
main_directory=main_directory.replace("\\","/")
if(main_directory[len(main_directory):len(main_directory)]!="/"):
	main_directory=main_directory + "/"
print("main_directory = " + main_directory)

#ディレクトリ内のrdens_sort.mrcの検索
list_dir_mrc=glob.glob(main_directory + "*/" + "*final_rdens_sort.mrc")
print("number of *final_rdens_sort.mrc = " + str(len(list_dir_mrc)))

#まとめファイル作成
log_path="log_erode_variance.csv"
with open(log_path, mode='w') as log:
	log.write("stack_num, diff, trial, scale_factor, Rfactor, OS_ratio, gamma, target_size, variance")
print("stack_num, diff, trial, scale_factor, Rfactor, OS_ratio, gamma, target_size, variance")
log_text="\n"

#row colの確認
mrc_name=list_dir_mrc[0].replace("\\","/")
with mrcfile.open(mrc_name, permissive=True) as mrc:
	mrc_data=np.asarray(mrc.data,dtype="float32")
mrc.close
row=mrc_data.shape[1]
col=mrc_data.shape[2]
print("row = " + str(row) + ", col = " + str(col))

#diffファイルの定義
diff=[""]*len(list_dir_mrc)

#log_fileからパラメーターを読み込む
for i in range(len(list_dir_mrc)):
	log_file=list_dir_mrc[i][0:len(list_dir_mrc[i])-21] + "_MPR.log"
	log_file=log_file.replace("\\","/")
	df=pd.read_csv(log_file, sep=" ",names=[1,2,3,4,5,6,7])
	diff[i]=df.iat[1,2]
	erode=df.iat[2,2]
	erode_name="E:/SACLA_data/2013_July_exp/" + erode
#	print(str(i) + " " + erode_name)
	np_erode=Image.open(erode_name)
	np_erode=np.asarray(np_erode,dtype="float32")
	
	a=1
	distance=0.0
	for n in range(col):
		for nn in range(row):
			if(np_erode[n,nn]>0.0):
				distance=distance+np.sqrt(np.square(n-col/2)+np.square(nn-row/2))
				a=a+1
	variance=distance/np.float(a)
	
	index=int(df.iat[len(df)-4,0][8:len(df.iat[len(df)-4,0])-1])
	temp=str(i+1) + ", "+ diff[i] + ", " + str(index) +", " + df.iat[index+18,2] + ", " + df.iat[index+18,3] + ", " + df.iat[index+18,4] + ", " + df.iat[index+18,5] + ", " + str(a) +", " + str(variance)
	print(temp)
	log_text=log_text + temp + "\n"
with open(log_path, mode='a') as log:
	log.write(log_text)

