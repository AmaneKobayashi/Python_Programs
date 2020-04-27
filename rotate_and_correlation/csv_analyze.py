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
import pandas as pd
import codecs

print("ver. 20200310")

temp_dens_flag=0

if len(sys.argv)==2:
	csv_list=sys.argv[1]	
	print("csv_list = " + csv_list)
	print("")
else:
	print("command:python3 csv_analyze.py csv_list")
	exit()

t1=time.time()

dlist = pd.read_csv(csv_list,header=None)
print("number of list = " + str(len(dlist)))
#print(dlist)

dcsv = pd.read_csv(dlist.iat[0,0], header=3)
print("number of orientation = " + str(len(dcsv)))

correlation=np.zeros([len(dlist),len(dcsv)])
index_orientation=np.zeros([len(dlist),len(dcsv)])
index_rot=np.zeros([len(dlist),len(dcsv)])
index_x=np.zeros([len(dlist),len(dcsv)])
index_y=np.zeros([len(dlist),len(dcsv)])

for i in range(len(dlist)):
	print(dlist.iat[i,0])
	dcsv = pd.read_csv(dlist.iat[i,0], header=3)
	
	for n in range(len(dcsv)):
		correlation[i,n]=float(dcsv.iat[n,2])
		index_orientation[i,n]=float(dcsv.iat[n,1])
		index_rot[i,n]=float(dcsv.iat[n,3])
		index_x[i,n]=float(dcsv.iat[n,4])
		index_y[i,n]=float(dcsv.iat[n,5])

index=np.argmax(correlation,axis=0)

log_path=csv_list[0:csv_list.rfind(".")] + "_analyze.csv"
print(log_path)

with open(log_path, mode='w') as log:
	log.write("projection_number,index_orientation,correlation,index_rot,index_shift_x,index_shift_y,filename" +"\n")

for n in range(len(dcsv)):
	with open(log_path, mode='a') as log:
		log.write(str(n) + "," + str(index_orientation[index[n],n]) + "," + str(correlation[index[n],n]) + "," + str(index_rot[index[n],n]) + "," + str(index_x[index[n],n]) + "," + str(index_y[index[n],n]) + "," + dlist.iat[index[n],0] + "\n")
		

