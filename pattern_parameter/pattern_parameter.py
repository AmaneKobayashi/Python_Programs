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
#fixed parameter
small_angle_area=80

#input parameter
print(len(sys.argv))
if(len(sys.argv) != 2):
	print("command : python.exe pattern_parameter.py pattern_stack")
	exit()
pattern_stack=sys.argv[1]
print("pattern_stack : " + str(pattern_stack))
#open mrc file
with mrcfile.open(pattern_stack, permissive=True) as mrc:
	np_pattern_stack=np.asarray(mrc.data,dtype="float32")
mrc.close

#row, col確認
number_stack=np_pattern_stack.shape[0]
row=np_pattern_stack.shape[1]
col=np_pattern_stack.shape[2]
print("number_stack = " + str(number_stack))
print("row = " + str(row))
print("col = " + str(col))

#まとめファイル作成
log_path="pattern_parameter.csv"
log_text="stack_num, sum(full), sum(small), sum(high), average(full), average(small), average(high), std(full), std(small), std(high), number of 0 pixel(full), number of 0 pixel(small)"
with open(log_path, mode='w') as log:
	log.write(log_text)
print(log_text)
log_text="\n"
#配列定義
zero_number_array=np.zeros((row,col), dtype="float32")
std_array=np.zeros((row,col), dtype="float32")
std_array_small=np.zeros((row,col), dtype="float32")
std_array_high=np.zeros((row,col), dtype="float32")

#
for i in range(number_stack):

	zero_number_array[:,:]=0.0
	std_array[:,:]=0.0
	std_array_small[:,:]=0.0
	std_array_high[:,:]=0.0

	zero_number_array=np.where(np_pattern_stack[i,:,:]==0.0, 1.0, 0.0)
	zero_number=np.sum(zero_number_array[:,:])
	zero_number_small=np.sum(zero_number_array[int((row-small_angle_area)/2):int((row+small_angle_area)/2),int((col-small_angle_area)/2):int((col+small_angle_area)/2)])

	sum_full=np.sum(np_pattern_stack[i,:,:])
	sum_small=np.sum(np_pattern_stack[i,int((row-small_angle_area)/2):int((row+small_angle_area)/2),int((col-small_angle_area)/2):int((col+small_angle_area)/2)])
	sum_high=sum_full-sum_small
	
#	zero_number=0
#	zero_number_small=0
#	sum_full=0.0
#	sum_small=0.0
#	for x in range(row):
#		for y in range(col):
#			if(np_pattern_stack[i,x,y]==0.0):
#				zero_number=zero_number+1
#			else:
#				sum_full=sum_full+np_pattern_stack[i,x,y]
#				
#			if((np_pattern_stack[i,x,y]==0.0) & (x > (small_angle_area-row/2)) & (x <= (small_angle_area+row/2)) & (y > (small_angle_area-col/2)) & (y <= (small_angle_area+col/2))):
#				zero_number_small=zero_number_small+1
#
#			if((np_pattern_stack[i,x,y]!=0.0) & (x > (small_angle_area-row/2)) & (x <= (small_angle_area+row/2)) & (y > (small_angle_area-col/2)) & (y <= (small_angle_area+col/2))):
#				sum_small=sum_small+sum_full+np_pattern_stack[i,x,y]

	ave_full=sum_full/float(row*col-zero_number)
	ave_small=sum_small/float(small_angle_area*small_angle_area-zero_number_small)
	ave_high=(sum_full-sum_small)/float((row*col-zero_number)-(small_angle_area*small_angle_area-zero_number_small))
	
	std_array=np.where(np_pattern_stack[i,:,:]!=0.0, np.square(np_pattern_stack[i,:,:]-ave_full), 0.0)
	std_full=np.sqrt(np.sum(std_array[:,:])/float(row*col-zero_number))
	
	std_array_small=np.where(np_pattern_stack[i,:,:]!=0.0, np.square(np_pattern_stack[i,:,:]-ave_small), 0.0)
	std_small=np.sqrt(np.sum(std_array_small[int((row-small_angle_area)/2):int((row+small_angle_area)/2),int((col-small_angle_area)/2):int((col+small_angle_area)/2)])/float(small_angle_area*small_angle_area-zero_number_small))
	
	std_array_high=np.where(np_pattern_stack[i,:,:]!=0.0, np.square(np_pattern_stack[i,:,:]-ave_high), 0.0)
	std_high=np.sqrt(np.sum(std_array_small[int((row-small_angle_area)/2):int((row+small_angle_area)/2),int((col-small_angle_area)/2):int((col+small_angle_area)/2)])/float((row*col-zero_number)-(small_angle_area*small_angle_area-zero_number_small)))

#	std_full=0.0
#	std_small=0.0
#	std_high=0.0
#	n_full=1
#	n_small=1
#	n_high=1
#	for x in range(row):
#		for y in range(col):
#			if(np_pattern_stack[i,x,y]!=0.0):		
#				std_full=std_full+np.square(np_pattern_stack[i,x,y]-ave_full)
#				n_full=n_full+1
#				
#				if((x < (small_angle_area-row/2)) | (x >= (small_angle_area+row/2)) | (y < (small_angle_area-col/2)) | (y >= (small_angle_area+col/2))):
#					std_high=std_high+np.square(np_pattern_stack[i,x,y]-ave_high)
#					n_high=n_high+1
#		
#			if((np_pattern_stack[i,x,y]!=0.0) & (x > (small_angle_area-row/2)) & (x <= (small_angle_area+row/2)) & (y > (small_angle_area-col/2)) & (y <= (small_angle_area+col/2))):
#				std_small=std_small+np.square(np_pattern_stack[i,x,y]-ave_small)
#				n_small=n_small+1
#	
#	std_full=np.sqrt(std_full/float(n_full))
#	std_small=np.sqrt(std_small/float(n_small))
#	std_high=np.sqrt(std_high/float(n_high))
	
	log_text_temp=str(i+1) + "," + str(sum_full) + "," + str(sum_small) + "," + str(sum_full-sum_small) + "," + str(ave_full) + "," + str(ave_small) + "," + str(ave_high) + "," + str(std_full) + "," + str(std_small) + "," + str(std_high) + "," + str(zero_number) + "," + str(zero_number_small)
	log_text=log_text+log_text_temp +"\n"
	print(log_text_temp)

with open(log_path, mode='a') as log:
	log.write(log_text)


