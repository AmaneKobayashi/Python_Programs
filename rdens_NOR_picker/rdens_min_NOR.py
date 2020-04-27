#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#�ʑ��񕜌v�Z�f�B���N�g�����ɂ���**_final_rdens_sort.mrc����NOR���ł��������摜���s�b�N�A�b�v ���@stack�쐬
#NOR�̒l��**_MPR.log����ǂݍ��ށ@���@csv�t�@�C����runtag,NOR�Ȃǂ���������
#diffraction���X�^�b�N�ɂ���B
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
#\��/
main_directory=main_directory.replace("\\","/")
if(main_directory[len(main_directory):len(main_directory)]!="/"):
	main_directory=main_directory + "/"
print("main_directory = " + main_directory)

#�f�B���N�g������rdens_sort.mrc�̌���
list_dir_mrc=glob.glob(main_directory + "*/" + "*final_rdens_sort.mrc")
print("number of *final_rdens_sort.mrc = " + str(len(list_dir_mrc)))

#�܂Ƃ߃t�@�C���쐬
log_path="log.csv"
with open(log_path, mode='w') as log:
	log.write("stack_num, diff, trial, scale_factor, Rfactor, OS_ratio, gamma, NOR_ex")
print("stack_num, diff, trial, scale_factor, Rfactor, OS_ratio, gamma, NOR_ex")
log_text="\n"

#diff�t�@�C���̒�`
diff=[""]*len(list_dir_mrc)

#log_file����p�����[�^�[��ǂݍ���
for i in range(len(list_dir_mrc)):
	log_file=list_dir_mrc[i][0:len(list_dir_mrc[i])-21] + "_MPR.log"
	log_file=log_file.replace("\\","/")
	df=pd.read_csv(log_file, sep=" ",names=[1,2,3,4,5,6,7])
	diff[i]=df.iat[1,2]
	index=int(df.iat[len(df)-4,0][8:len(df.iat[len(df)-4,0])-1])
	temp=str(i+1) + ", "+ diff[i] + ", " + str(index) +", " + df.iat[index+18,2] + ", " + df.iat[index+18,3] + ", " + df.iat[index+18,4] + ", " + df.iat[index+18,5] + ", " + df.iat[len(df)-4,2]
	print(temp)
	log_text=log_text + temp + "\n"
with open(log_path, mode='a') as log:
	log.write(log_text)

#row col�̊m�F
mrc_name=list_dir_mrc[0].replace("\\","/")
with mrcfile.open(mrc_name, permissive=True) as mrc:
	mrc_data=np.asarray(mrc.data,dtype="float32")
mrc.close
row=mrc_data.shape[1]
col=mrc_data.shape[2]
print("row = " + str(row) + ", col = " + str(col))

#�X�^�b�N�t�@�C���̒�`
rdens_min_NOR=np.zeros((len(list_dir_mrc),row,col),np.float32)

#diffraction file�̃X�^�b�N�쐬
for i in range(len(list_dir_mrc)):
	diff_name="E:/SACLA_data/2013_July_exp/" + diff[i]
	print(str(i) + " " + diff_name)
	np_diff=Image.open(diff_name)
	np_diff=np.asarray(np_diff,dtype="float32")
	rdens_min_NOR[i,:,:]=np_diff[:,:]

with mrcfile.new("diff.mrc", overwrite=True) as mrc:
	mrc.set_data(rdens_min_NOR)
mrc.close


#rdens�̓ǂݍ��݁�stack�쐬
for i in range(len(list_dir_mrc)):
	mrc_name=list_dir_mrc[i].replace("\\","/")
	print(str(i) + " " + mrc_name)
	with mrcfile.open(mrc_name, permissive=True) as mrc:
		mrc_data=np.asarray(mrc.data,dtype="float32")
	mrc.close
	min_dens=np.flipud(mrc_data[0,:,:])
	
	rdens_min_NOR[i,:,:]=min_dens[:,:]

with mrcfile.new("rdens_min_NOR.mrc", overwrite=True) as mrc:
	mrc.set_data(rdens_min_NOR)
mrc.close

