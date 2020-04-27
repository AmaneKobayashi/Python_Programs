#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import tifffile
import pandas as pd
import sys
import os
from PIL import Image

#runの指定
if len(sys.argv)!=2:
	print("command:python3 Extract.py run_number")
	exit()

run=sys.argv[1]
print('run_number = ' + run)

#file名

run_dir_0=run+'-0'

dark=run_dir_0 + '/' + run + '-dark.h5'
geom_h5=run_dir_0 + '/' + run + '-geom.h5'
geom=run_dir_0 + '/' + run + '.geom'

HDF_0=run_dir_0 + '/' + 'run' + run_dir_0 + '.h5'

#sample名
#list(f.keys())
f=h5py.File(HDF_0,"r")
run_comment=str(f['metadata/run_comment'].value)
#print(run_comment)
sample_name=run_comment[run_comment.find(':')+1:run_comment.find(',')]
print('sample_name = ' + sample_name)

#sample名のディレクトリ作成
os.makedirs(sample_name, exist_ok=True)
f.close()

#dark読み込み
d=h5py.File(dark,"r")
dimg=np.array(d['data/data'])
d.close()

#ヒットのみdark減算してtiff出力
for i in range(3):
	HDF=run + '-' + str(i) + '/run' + run + '-' + str(i) + '.h5'
	csv_file=run + '-' + str(i) + '/' + run + '-' + str(i) + '.csv'

	f=h5py.File(HDF,"r")
	csv_df = pd.read_csv(csv_file, index_col=0)
	for row in range(len(csv_df)):
		if csv_df.iat[row,1]==1:
			tag=str(csv_df.iat[row,2][0:13])
			print(tag)
			img=np.array(f[tag + "/data"])
			t_img=img - dimg
			t_img_q1=np.rot90(t_img[0:1024,:],3)

#			tifffile.imsave(sample_name + '/run-' + run +'_' + tag + '_q1.tif' ,t_img_q1)
			t_img_q2=np.rot90(t_img[1024:2048,:],3)
#			tifffile.imsave(sample_name + '/run-' + run +'_' + tag + '_q2.tif' ,t_img_q2)
			t_img_q3=np.rot90(t_img[2048:3072,:],3)
#			tifffile.imsave(sample_name + '/run-' + run +'_' + tag + '_q3.tif' ,t_img_q3)
			t_img_q4=np.rot90(t_img[3072:4096,:],3)
#			tifffile.imsave(sample_name + '/run-' + run +'_' + tag + '_q4.tif' ,t_img_q4)

			t_img_q5=np.rot90(t_img[4096:5120,:])
#			tifffile.imsave(sample_name + '/run-' + run +'_' + tag + '_q5.tif' ,t_img_q5)
			t_img_q6=np.rot90(t_img[5120:6144,:])
#			tifffile.imsave(sample_name + '/run-' + run +'_' + tag + '_q6.tif' ,t_img_q6)
			t_img_q7=np.rot90(t_img[6144:7168,:])
#			tifffile.imsave(sample_name + '/run-' + run +'_' + tag + '_q7.tif' ,t_img_q7)
			t_img_q8=np.rot90(t_img[7168:8192,:])
#			tifffile.imsave(sample_name + '/run-' + run +'_' + tag + '_q8.tif' ,t_img_q8)

			t_img2=np.zeros((2202,2144), dtype='float32')
			t_img2[52:564,14:1038]=t_img_q1
			t_img2[600:1112,14:1038]=t_img_q2
			t_img2[1140:1652,62:1086]=t_img_q3
			t_img2[1680:2192,62:1086]=t_img_q4

			t_img2[0:512,1054:2078]=t_img_q5
			t_img2[544:1056,1054:2078]=t_img_q6
			t_img2[1086:1598,1100:2124]=t_img_q7
			t_img2[1630:2142,1100:2124]=t_img_q8
			tifffile.imsave(sample_name + '/run-' + run +'_' + tag + '.tif' ,t_img2)

	f.close()





