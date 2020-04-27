#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#位相回復計算ディレクトリ中にある**_final_rdens_sort.mrcからNORが0.1ごとに画像をピックアップ →　stack作成
#NORの値を**_MPR.logから読み込む　→　csvファイルにruntag,NORなどを書き込む
#diffractionをスタックにする。
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
		exit()

for i in range(n_parameter):
	if(flag_list[i]==0):
		print("please input parameter : [" + parameter_name_list[i] + "]")
		exit()
#\→/
main_directory=main_directory.replace("\\","/")
if(main_directory[len(main_directory):len(main_directory)]!="/"):
	main_directory=main_directory + "/"
print("main_directory = " + main_directory)

#ディレクトリ内のrdens_sort.mrcの検索
list_dir_mrc=glob.glob(main_directory + "*/" + "*final_rdens_sort.mrc")
print("number of *final_rdens_sort.mrc = " + str(len(list_dir_mrc)))

#row colの確認
mrc_name=list_dir_mrc[0].replace("\\","/")
with mrcfile.open(mrc_name, permissive=True) as mrc:
	mrc_data=np.asarray(mrc.data,dtype="float32")
mrc.close
row=mrc_data.shape[1]
col=mrc_data.shape[2]
print("row = " + str(row) + ", col = " + str(col))

#まとめファイル作成
log_path=[""]*10
for i in range(10):
	log_path[i]="log_" + str(0.1*float(i+1))[0:3] + ".csv"
	with open(log_path[i], mode='w') as log:
		log.write("stack_num, diff, trial, scale_factor, Rfactor, OS_ratio, gamma, NOR_ex" + "\n")

#diffファイルの定義
diff=[[""]*len(list_dir_mrc) for i in range(10)]
#print([len(v) for v in diff])
#print(diff[0][0])

#mrc_file_nameの定義
mrc_name=[[""]*len(list_dir_mrc) for i in range(10)]

#
n=np.zeros(10,np.int)
print(n[0])

#log_fileからパラメーターを読み込む
for i in range(len(list_dir_mrc)):
	log_file=list_dir_mrc[i][0:len(list_dir_mrc[i])-21] + "_MPR.log"
	log_file=log_file.replace("\\","/")
	df=pd.read_csv(log_file, sep=" ",names=[1,2,3,4,5,6,7])

	NOR_ex=int(float(df.iat[len(df)-4,2])*10)
	ind=NOR_ex
	if(ind>9):
		ind=9
	diff[ind][n[ind]]=df.iat[1,2]
	
	index=int(df.iat[len(df)-4,0][8:len(df.iat[len(df)-4,0])-1])
	temp=str(i+1) + ", "+ df.iat[1,2] + ", " + str(index) +", " + df.iat[index+18,2] + ", " + df.iat[index+18,3] + ", " + df.iat[index+18,4] + ", " + df.iat[index+18,5] + ", " + df.iat[len(df)-4,2]
	print(temp)

	with open(log_path[ind], mode='a') as log:
		log.write(temp + "\n")

	mrc_name[ind][n[ind]]=list_dir_mrc[i]

	n[ind]=n[ind]+1

#NORごとの数の確認
for i in range(10):
	print("number of image NOR less than " + str(0.1*(i+1))[0:3] + " :"  + str(n[i]))

i=0
while(i<9):
	if(n[i]==0):
		i=i+1
	#スタックファイル定義
	rdens_min_NOR=np.zeros((n[i],row,col),np.float32)

	#diffraction fileのスタック作成
	for ii in range(n[i]):
		diff_name="E:/SACLA_data/2013_July_exp/" + diff[i][ii]
		print(str(ii) + " " + diff_name)
		np_diff=Image.open(diff_name)
		np_diff=np.asarray(np_diff,dtype="float32")
		rdens_min_NOR[ii,:,:]=np_diff[:,:]

	with mrcfile.new("diff_" + str(0.1*(i+1))[0:3] + ".mrc", overwrite=True) as mrc:
		mrc.set_data(rdens_min_NOR)
	mrc.close

	#rdensの読み込み→stack作成
	for ii in range(n[i]):
		mrc_file=mrc_name[i][ii].replace("\\","/")
		print(str(ii) + " " + mrc_file)
		with mrcfile.open(mrc_file, permissive=True) as mrc:
			mrc_data=np.asarray(mrc.data,dtype="float32")
		mrc.close
		min_dens=np.flipud(mrc_data[0,:,:])
	
		rdens_min_NOR[ii,:,:]=min_dens[:,:]

	with mrcfile.new("rdens_min_NOR_" + str(0.1*(i+1))[0:3] + ".mrc", overwrite=True) as mrc:
		mrc.set_data(rdens_min_NOR)
	mrc.close

	i=i+1

