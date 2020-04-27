#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#�ʑ��񕜌v�Z�f�B���N�g�����ɂ���**_final_rdens_sort.mrc����NOR��0.1���Ƃɉ摜���s�b�N�A�b�v ���@stack�쐬
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
		exit()

for i in range(n_parameter):
	if(flag_list[i]==0):
		print("please input parameter : [" + parameter_name_list[i] + "]")
		exit()
#\��/
main_directory=main_directory.replace("\\","/")
if(main_directory[len(main_directory):len(main_directory)]!="/"):
	main_directory=main_directory + "/"
print("main_directory = " + main_directory)

#�f�B���N�g������rdens_sort.mrc�̌���
list_dir_mrc=glob.glob(main_directory + "*/" + "*final_rdens_sort.mrc")
print("number of *final_rdens_sort.mrc = " + str(len(list_dir_mrc)))

#row col�̊m�F
mrc_name=list_dir_mrc[0].replace("\\","/")
with mrcfile.open(mrc_name, permissive=True) as mrc:
	mrc_data=np.asarray(mrc.data,dtype="float32")
mrc.close
row=mrc_data.shape[1]
col=mrc_data.shape[2]
print("row = " + str(row) + ", col = " + str(col))

#�܂Ƃ߃t�@�C���쐬
log_path=[""]*11
for i in range(11):
	if(i==10):
		log_path[i]="log_fail.csv"	
	else:
		log_path[i]="log_" + str(0.1*float(i+1))[0:3] + ".csv"
	with open(log_path[i], mode='w') as log:
		log.write("stack_num, diff, trial, scale_factor, Rfactor, OS_ratio, gamma, NOR_ex, average_distance_autocorrelation, distance_trial" + "\n")
		

#diff�t�@�C���̒�`
diff=[[""]*len(list_dir_mrc) for i in range(11)]
#print([len(v) for v in diff])
#print(diff[0][0])

#mrc_file_name�̒�`
mrc_name=[[""]*len(list_dir_mrc) for i in range(11)]

#
n=np.zeros(11,np.int)
#print(n[0])

#log_file����p�����[�^�[��ǂݍ���
for i in range(len(list_dir_mrc)):
	log_file=list_dir_mrc[i][0:len(list_dir_mrc[i])-21] + "_MPR.log"
	log_file=log_file.replace("\\","/")
#	print(log_file)
	df=pd.read_csv(log_file, sep=" ",names=[1,2,3,4,5,6,7])

	if((df.iat[1,0]!="saturation") & (df.iat[1,0]!="std")):

		average_distance_autocorrelation=df.iat[13,2]
		distance_trial=df.iat[13,0]

		if(np.float(average_distance_autocorrelation) <= 50.0):

			NOR_ex=int(float(df.iat[len(df)-4,2])*10)
			ind=NOR_ex
			if(ind>9):
				ind=9
			diff[ind][n[ind]]=df.iat[1,2]
		
			index=int(df.iat[len(df)-4,0][8:len(df.iat[len(df)-4,0])-1])
			temp=str(i+1) + ", "+ df.iat[1,2] + ", " + str(index) +", " + df.iat[len(df)-4-100+index,2] + ", " + df.iat[len(df)-4-100+index,3] + ", " + df.iat[len(df)-4-100+index,4] + ", " + df.iat[len(df)-4-100+index,5] + ", " + df.iat[len(df)-4,2] + ", "  + average_distance_autocorrelation + ", " + distance_trial
			print(temp)
	
			with open(log_path[ind], mode='a') as log:
				log.write(temp + "\n")
	
			mrc_name[ind][n[ind]]=list_dir_mrc[i][0:len(list_dir_mrc[i])-21] + "_final_rdens_min_NOR.tif"
	
			n[ind]=n[ind]+1
	
		else:
		
			temp=str(i+1) + ", "+ df.iat[1,2] + ", " + "fail" +", " + "fail" + ", " + "fail" + ", " + "fail" + ", " + "fail" + ", " + "fail" + ", "  + average_distance_autocorrelation + ", " + distance_trial
			print(temp)
	
			with open(log_path[10], mode='a') as log:
				log.write(temp + "\n")
	
			mrc_name[10][n[10]]="C:\Python_Programs\rdens_NOR_picker_0.1\fail.tif"
	
			diff[10][n[10]]=df.iat[1,2]
	
			n[10]=n[10]+1

#NOR���Ƃ̐��̊m�F
for i in range(11):
	if(i==10):
		print("number of fail :"  + str(n[i]))
	else:
		print("number of image NOR less than " + str(0.1*(i+1))[0:3] + " :"  + str(n[i]))

i=0
while(i<9):
	if(n[i]==0):
		i=i+1
	#�X�^�b�N�t�@�C����`
	rdens_min_NOR=np.zeros((n[i],row,col),np.float32)

	#diffraction file�̃X�^�b�N�쐬
	for ii in range(n[i]):
		diff_name=main_directory + diff[i][ii]
		diff_name=diff_name.replace("\\","/")
		print(str(ii) + " " + diff_name)
		np_diff=Image.open(diff_name)
		np_diff=np.asarray(np_diff,dtype="float32")
		rdens_min_NOR[ii,:,:]=np_diff[:,:]

	with mrcfile.new("diff_" + str(0.1*(i+1))[0:3] + ".mrc", overwrite=True) as mrc:
		mrc.set_data(rdens_min_NOR)
	mrc.close

	#rdens�̓ǂݍ��݁�stack�쐬
	for ii in range(n[i]):
		mrc_file=mrc_name[i][ii].replace("\\","/")
		print(str(ii) + " " + mrc_file)
		np_diff=Image.open(mrc_file)
		np_diff=np.asarray(np_diff,dtype="float32")
#		with mrcfile.open(mrc_file, permissive=True) as mrc:
#			mrc_data=np.asarray(mrc.data,dtype="float32")
#		mrc.close
#		min_dens=np.flipud(mrc_data[0,:,:])
	
		rdens_min_NOR[ii,:,:]=np_diff[:,:]

	with mrcfile.new("rdens_min_NOR_" + str(0.1*(i+1))[0:3] + ".mrc", overwrite=True) as mrc:
		mrc.set_data(rdens_min_NOR)
	mrc.close

	i=i+1

