#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import tifffile
import pandas as pd
import sys
import os
from PIL import Image
import mrcfile
import shutil
import glob
from time import sleep
import time
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

###input parameter
#main loop
n_loop=10000000
sleep_time=2
move_dir="processed_files"
#Extract
Extract_py="/home/amanekobayashi/work/Extract_EIGER/Extract_EIGER.py"
flat_field="/home/amanekobayashi/work/Extract_EIGER/flat_field.tif"
#mask="/home/amanekobayashi/work/Extract_EIGER/Eiger_mask_Yeast1902-06_6umB_0_80.tif"
#mask="/home/amanekobayashi/work/Extract_EIGER/Eiger_mask_CuCube.tif"
mask="/home/amanekobayashi/work/Extract_EIGER/Eiger_mask_201907_BS_lack_rot.tif"
trimsize=128
#trimsize=512
c_trimsize=str(trimsize)
#phase retrieval
PR_py="/home/amanekobayashi/work/PR_multi_EIGER/PR_multi_EIGER.py"
if(trimsize==128):
	sup="/home/amanekobayashi/work/PR_multi_EIGER/sup_circle_Yeast1902-06_6umB.tif"
	initial_dens="/home/amanekobayashi/work/PR_multi_EIGER/random_128_x_128_x_700.mrc"
elif(trimsize==512):
	sup="/home/amanekobayashi/work/PR_multi_EIGER/sup_circle_Yeast1902-06_6umB-512.tif"
	initial_dens="/home/amanekobayashi/work/make_random_density/random_512_x_512_x_280.mrc"
iteration=1000
SW_interval=10  
initial_SW_ips=5
SW_ips_step=1.1
last_SW_ips=50 
initial_SW_delta=0.01 
last_SW_delta=0.1 
#initial_SW_delta=0.001 
#last_SW_delta=0.005 
n_SW_delta=7  
additional_iteration=100
output_interval=10

c_iteration=str(iteration)
c_SW_interval=str(SW_interval)
c_initial_SW_ips=str(initial_SW_ips)
c_SW_ips_step=str(SW_ips_step) 
c_last_SW_ips=str(last_SW_ips) 
c_initial_SW_delta=str(initial_SW_delta) 
c_last_SW_delta=str(last_SW_delta) 
c_n_SW_delta=str(n_SW_delta)
c_additional_iteration=str(additional_iteration)
c_output_interval=str(output_interval)
# sort and average
Sort_and_average_py="/home/amanekobayashi/work/Sort_and_average/Sort_and_average.py"
shift=20
c_shift=str(shift)
#PCA and clustering
PCA_and_clustering_py="/home/amanekobayashi/work/PCA_and_clustering/PCA_and_clustering.py"
n_class=10
c_n_class=str(n_class)

#color map
cdict1 = {'red'  :((0.0,0.0,0.0),
                  (0.375,0.0,0.0),
                  (0.492,1.0,1.0),
                  (0.75,1.0,1.0),
                  (0.80,0.867,0.867),
                  (0.875,0.867,0.867),
                  (0.895,1.0,1.0),
                  (1.0,1.0,1.0)),
          'green':((0.0,0.0,0.0),
                  (0.129,0.0,0.0),
                  (0.3125,1.0,1.0),
                  (0.4375,1.0,1.0),
                  (0.75,0.0,0.0),
                  (0.8125,0.734,0.734),
                  (0.9375,1.0,1.0),
                  (1.0,1.0,1.0)),
          'blue' :((0.0,0.03,0.03),
                  (0.1875,1.0,1.0),
                  (0.375,1.0,1.0),
                  (0.4375,0.0,0.0),
                  (0.754,0.0,0.0),
                  (0.8125,0.715,0.715),
                  (0.9375,1.0,1.0),
                  (1.0,1.0,1.0))}
royal = matplotlib.colors.LinearSegmentedColormap('wrainbow',cdict1,256)

#sciript start

n_parameter=2
parameter_name_list=[""]*n_parameter
flag_list=[0]*n_parameter

parameter_name_list[0]="-dir"
parameter_name_list[1]="-sleep_time"

dir_flag=0
sleep_time_flag=0

input_parameter=0

for i in range(len(sys.argv)):
	if(sys.argv[i]=="-dir"):
		dir=sys.argv[i+1]
		dir_flag=1
		flag_list[0]=1
	if(sys.argv[i]=="-sleep_time"):
		c_sleep_time=sys.argv[i+1]
		sleep_time=int(c_sleep_time)
		sleep_time_flag=1
		flag_list[1]=1
	if(sys.argv[i]=="--help"):
		print("command:python3 auto_process.py [-dir] [-sleep_time]")
		exit()

for i in range(n_parameter):
	if(flag_list[i]==0):
		print("please input parameter : [" + parameter_name_list[i] + "]")
		input_parameter=1

if(input_parameter==1):
	exit()

print("dir = " + dir)
print("sleep_time = " + str(sleep_time))
print("")

if(dir[len(dir):len(dir)] != "/"):
	dir=dir + "/"

move_dir=dir + move_dir + "/"
os.makedirs(move_dir, exist_ok=True)


for i_loop in range(n_loop):

	list_dir=glob.glob(dir + "Sample*master.h5")
	n_list_dir=len(list_dir)
	if(n_list_dir == 0):
		print("status : idle")
		sleep(sleep_time)
		if(i_loop==n_loop-1):
			print("process stop")
	else:
		for i in range(len(list_dir)):
			#file_name_master
			file_name_master=str(list_dir[i])
			print("Sample master file = " + file_name_master)

			#file_name_data
			file_name_data=file_name_master[0:len(file_name_master)-len("master.h5")] + "data_000001.h5"
			list_dir_data=glob.glob(file_name_data)
			n_list_dir_data=len(list_dir_data)
			if(n_list_dir_data == 0):
				file_name_data_flag=0
				print("Sample data file does not exist.")
				sleep(sleep_time)
			else:
				file_name_data_flag=1

			u1=file_name_master.find('Sample_')
			u2=file_name_master.find('_ang_')
			u3=file_name_master.find('_T_')
			u4=file_name_master.find('_n_')
			u5=file_name_master.find('_Y_')
			u6=file_name_master.find('_Z_')
			u7=file_name_master.find('_dY_')
			u8=file_name_master.find('_dZ_')
			u9=file_name_master[u8+len('_dZ_'):len(file_name_master)].find('_')+u8+len('_dZ_')
			u10=file_name_master[u9+len('_'):len(file_name_master)].find('_')+u9+len('_')

			sample_name=file_name_master[u1+len("Sample_"):u2]
			ang=file_name_master[u2+len('_ang_'):u3]
			T=file_name_master[u3+len('_T_'):u4]
			n=file_name_master[u4+len('_n_'):u5]
			Y=file_name_master[u5+len('_Y_'):u6]
			Z=file_name_master[u6+len('_Z_'):u7]
			dY=file_name_master[u7+len('_dY_'):u8]
			dZ=file_name_master[u8+len('_dZ_'):u9]
			n_EIGER=file_name_master[u9+len("_"):u10]

			#list_dir_BG_master
			list_dir_BG_master=glob.glob(dir + "BG_" + sample_name + "_ang_" + ang + "*master.h5")
			n_list_dir_BG_master=len(list_dir_BG_master)
			if(n_list_dir_BG_master == 0):
				BG_file_name_master_flag=0
				BG_file_name_data_flag=0
				print("BG master file does not exist.")
				sleep(sleep_time)
			elif(n_list_dir_BG_master > 1):
				BG_file_name_master_flag=1
				print("More than two BG master files exist.")

				master_stat_time=os.stat(file_name_master).st_mtime
				BG_stat_time=np.zeros(n_list_dir_BG_master)
				BG_stat_time=BG_stat_time.astype(np.float32)
				for i in range(n_list_dir_BG_master):
					pre_BG_file_name_master=str(list_dir_BG_master[i])
					BG_stat_time=np.absolute(os.stat(pre_BG_file_name_master).st_mtime-BG_stat_time)					
				
				BG_file_name_master=str(list_dir_BG_master[np.argmin(BG_stat_time)])

				#BG_file_name_data
				BG_file_name_data=BG_file_name_master[0:len(BG_file_name_master)-len("master.h5")] + "data_000001.h5"			
				list_dir_BG_data=glob.glob(BG_file_name_data)
				n_list_dir_BG_data=len(list_dir_BG_data)

				if(n_list_dir_BG_data == 0):
					BG_file_name_data_flag=0
					print("BG data file does not exist.")
					sleep(sleep_time)
				else:
					pre_file_size=int(os.path.getsize(BG_file_name_data))
					sleep(sleep_time)
					file_size=int(os.path.getsize(BG_file_name_data))
					if(file_size-pre_file_size == 0):
						BG_file_name_data_flag=1
					else:
						print("file transferring")
						BG_file_name_data_flag=0
						sleep(sleep_time)

			else:
				BG_file_name_master_flag=1
				BG_file_name_master=str(list_dir_BG_master[0])

				#BG_file_name_data
				BG_file_name_data=BG_file_name_master[0:len(BG_file_name_master)-len("master.h5")] + "data_000001.h5"			
				list_dir_BG_data=glob.glob(BG_file_name_data)
				n_list_dir_BG_data=len(list_dir_BG_data)

				if(n_list_dir_BG_data == 0):
					BG_file_name_data_flag=0
					print("BG data file does not exist.")
					sleep(sleep_time)
				else:
					pre_file_size=int(os.path.getsize(BG_file_name_data))
					sleep(sleep_time)
					file_size=int(os.path.getsize(BG_file_name_data))
					if(file_size-pre_file_size == 0):
						BG_file_name_data_flag=1
					else:
						print("file transferring")
						BG_file_name_data_flag=0
						sleep(sleep_time)

			#check file
			if(file_name_data_flag*BG_file_name_master_flag*BG_file_name_data_flag == 1):
				#Extract
				print("------------Extract process--------------")
				subprocess.run(["python3", Extract_py, file_name_master, BG_file_name_master, flat_field, mask, c_trimsize])

				diff=file_name_master[0:len(file_name_master)-len("_master.h5")] + "/Preprocessing/" + file_name_master[file_name_master.rfind("/")+1:len(file_name_master)-len("_master.h5")] + "_sum_comp_trim.tif"
				full_diff=diff[0:len(diff)-9] + ".tif"
				op_full_diff=Image.open(full_diff)
				np_full_diff=np.asarray(op_full_diff,dtype="float32")
				row_full=np_full_diff.shape[0]
				col_full=np_full_diff.shape[1]
				np_full_diff=np.where(np_full_diff > 0, np_full_diff, 1)
				np_full_diff_log=np.log10(np_full_diff)

				X=np.arange(col_full)
				Y=np.arange(row_full)
				Y=np.flip(Y)

				a=plt.figure(figsize=(15,15))
				a=plt.pcolormesh(X,Y,np_full_diff_log[:,:], cmap=royal, vmin=-1.0, vmax=9.0)
				a=plt.gca().set_aspect('equal', adjustable='box')
				a=plt.suptitle(full_diff[full_diff.rfind("/")+1:full_diff.rfind(".")], fontsize=15)
				a=plt.savefig(move_dir + "diff_sum.png")
				a=plt.savefig("/home/amanekobayashi/work/diff_sum.png")

				#phase retrieval
				#exit()
				print("------------PR process--------------")
#				diff=file_name_master[0:len(file_name_master)-len("_master.h5")] + "/Preprocessing/" + file_name_master[file_name_master.rfind("/")+1:len(file_name_master)-len("_master.h5")] + "_sum_comp_trim.tif"
				header=file_name_master[0:len(file_name_master)-len("_master.h5")] + "/HIO-SW/" + file_name_master[file_name_master.rfind("/")+1:len(file_name_master)-len("_master.h5")]
#				print("python3", PR_py, "-diff", diff, "-sup", sup, "-initial_dens", initial_dens, "-iteration", c_iteration, "-header", header, "-SW_interval", c_SW_interval, "-initial_SW_ips", c_initial_SW_ips, "-SW_ips_step", c_SW_ips_step, "-last_SW_ips", c_last_SW_ips, "-initial_SW_delta", c_initial_SW_delta, "-last_SW_delta", c_last_SW_delta, "-n_SW_delta", c_n_SW_delta, "-additional_iteration", c_additional_iteration, "-output_interval", c_output_interval)
				subprocess.run(["python3", PR_py, "-diff", diff, "-sup", sup, "-initial_dens", initial_dens, "-iteration", c_iteration, "-header", header, "-SW_interval", c_SW_interval, "-initial_SW_ips", c_initial_SW_ips, "-SW_ips_step", c_SW_ips_step, "-last_SW_ips", c_last_SW_ips, "-initial_SW_delta", c_initial_SW_delta, "-last_SW_delta", c_last_SW_delta, "-n_SW_delta", c_n_SW_delta, "-additional_iteration", c_additional_iteration])
				#Sort and average
				print("------------Sort and average process--------------")
				stack_file=file_name_master[0:len(file_name_master)-len("_master.h5")] + "/HIO-SW/" + file_name_master[file_name_master.rfind("/")+1:len(file_name_master)-len("_master.h5")] + "_final_rdens.mrc"
#				print("python3 " + Sort_and_average_py + " " + stack_file + " " + c_shift)
				subprocess.run(["python3", Sort_and_average_py, stack_file, c_shift])
				#PCA and clustring
				print("------------PCA and clustring process--------------")
				density_stack_sort=file_name_master[0:len(file_name_master)-len("_master.h5")] + "/HIO-SW/" + file_name_master[file_name_master.rfind("/")+1:len(file_name_master)-len("_master.h5")] + "_final_rdens_sort.mrc"
#				print("python3 "+ PCA_and_clustering_py + " " + density_stack_sort + " " + c_n_class + " " + diff)
				subprocess.run(["python3", PCA_and_clustering_py, density_stack_sort, c_n_class, diff])

				#diffraction file
#				full_diff=diff[0:len(diff)-9] + ".tif"
#				op_full_diff=Image.open(full_diff)
#				np_full_diff=np.asarray(op_full_diff,dtype="float32")
#				row_full=np_full_diff.shape[0]
#				col_full=np_full_diff.shape[1]
#				np_full_diff=np.where(np_full_diff > 0, np_full_diff, 1)
#				np_full_diff_log=np.log10(np_full_diff)

#				X=np.arange(col_full)
#				Y=np.arange(row_full)
#				Y=np.flip(Y)

#				a=plt.figure(figsize=(15,15))
#				a=plt.pcolormesh(X,Y,np_full_diff_log[:,:], cmap=royal, vmin=-1.0, vmax=9.0)
#				a=plt.gca().set_aspect('equal', adjustable='box')
#				a=plt.suptitle(full_diff[full_diff.rfind("/")+1:full_diff.rfind(".")], fontsize=15)
#				a=plt.savefig(move_dir + "diff_sum.png")
#				a=plt.savefig("/home/amanekobayashi/work/diff_sum.png")

				#move files
				shutil.move(file_name_master,move_dir + file_name_master[file_name_master.rfind("/")+1:len(file_name_master)])
				shutil.move(file_name_data,move_dir + file_name_data[file_name_data.rfind("/")+1:len(file_name_data)])
				shutil.move(BG_file_name_master,move_dir + BG_file_name_master[BG_file_name_master.rfind("/")+1:len(BG_file_name_master)])
				shutil.move(BG_file_name_data,move_dir + BG_file_name_data[BG_file_name_data.rfind("/")+1:len(BG_file_name_data)])

				#copy & upload
				copy_target_PCA=move_dir + "PCA_plot.png"
				copy_target_montage=move_dir + "montage.png"
				copy_target_montage_sup=move_dir + "montage_sup.png" 
				PCA_png=file_name_master[0:len(file_name_master)-len("_master.h5")] + "/HIO-SW/" + file_name_master[file_name_master.rfind("/")+1:len(file_name_master)-len("_master.h5")] + "_final_rdens_sort_PCA.png"
				montage_png=file_name_master[0:len(file_name_master)-len("_master.h5")] + "/HIO-SW/" + file_name_master[file_name_master.rfind("/")+1:len(file_name_master)-len("_master.h5")] + "_final_rdens_sort_montage.png"
				montage_png_sup=file_name_master[0:len(file_name_master)-len("_master.h5")] + "/HIO-SW/" + file_name_master[file_name_master.rfind("/")+1:len(file_name_master)-len("_master.h5")] + "_final_rdens_sort_montage_sup.png"
				shutil.copyfile(PCA_png, copy_target_PCA)
				shutil.copyfile(montage_png, copy_target_montage)
				shutil.copyfile(montage_png_sup, copy_target_montage_sup)

				copy_target_PCA="/home/amanekobayashi/work/PCA_plot.png"
				copy_target_montage="/home/amanekobayashi/work/montage.png"
				copy_target_montage_sup="/home/amanekobayashi/work/montage_sup.png" 
				shutil.copyfile(PCA_png, copy_target_PCA)
				shutil.copyfile(montage_png, copy_target_montage)
				shutil.copyfile(montage_png_sup, copy_target_montage_sup)
























