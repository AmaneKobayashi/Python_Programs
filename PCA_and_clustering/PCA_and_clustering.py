#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import cupy as cp
import time
import os
import mrcfile
import tifffile
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import subprocess

if len(sys.argv)!=4:
	print("command:python3 PCA_and_clustering.py density_stack_sort n_clus diff")
	exit()

density_stack_sort=sys.argv[1]
n_clus=sys.argv[2]
diff=sys.argv[3]
print('density_stack_sort = ' + density_stack_sort)
print("n_clus = " + n_clus)
print("diff = " + diff)
print("")

t1=time.time()

#mkdir

dir_dens=os.path.dirname(density_stack_sort) + "/" +  "dens"
os.makedirs(dir_dens, exist_ok=True)
dir_ave_dens=os.path.dirname(density_stack_sort) + "/" +  "ave_dens"
os.makedirs(dir_ave_dens, exist_ok=True)
dir_ave_sup=os.path.dirname(density_stack_sort) + "/" +  "ave_sup"
os.makedirs(dir_ave_sup, exist_ok=True)
dir_ave_diff=os.path.dirname(density_stack_sort) + "/" +  "ave_diff"
os.makedirs(dir_ave_diff, exist_ok=True)
dir_PRTF=os.path.dirname(density_stack_sort) + "/" +  "PRTF"
os.makedirs(dir_PRTF, exist_ok=True)
dir_PhaseSta=os.path.dirname(density_stack_sort) + "/" +  "PhaseSta"
os.makedirs(dir_PhaseSta, exist_ok=True)

# open mrc file

with mrcfile.open(density_stack_sort, permissive=True) as mrc:
	np_sort=np.asarray(mrc.data,dtype="float32")
mrc.close

#flip

np_sort=np.fliplr(np_sort)

#

sta_dens=np_sort.shape[0]
row=np_sort.shape[1]
col=np_sort.shape[2]
print("sta_dens = " + str(sta_dens))
print("row = " + str(row))
print("col = " + str(col))
print("")

np_sort_reshape=np.reshape(np_sort,(sta_dens,row*col))

#print(np_sort_reshape.shape)
#print(np_sort[8,0,0],np_sort_reshape[8,0])

pca=PCA(n_components=2)
X_transformed=pca.fit_transform(np_sort_reshape)
#print(pca.explained_variance_ratio_)
#print(pca.components_)

kmeans_model=KMeans(n_clusters=int(n_clus), random_state=10).fit(X_transformed)
labels=kmeans_model.labels_
centers=kmeans_model.cluster_centers_
#print(centers.shape)
#print(labels.dtype)

#PCA plot

colorlist=labels[:]/(float(n_clus)-1.0)

b=fig,ax=plt.subplots(figsize=(8,8))
#b=plt.figure(figsize=(10,10))
b=plt.scatter(X_transformed[:,0],X_transformed[:,1], c=colorlist)
b=plt.scatter(centers[:,0],centers[:,1], s=50, marker="*", c=[0,0,0], label="centroid")

for i,(x,y) in enumerate(zip(centers[:,0],centers[:,1])):
    ax.annotate(str(i),(x,y))
b=plt.xlabel('PC1')
b=plt.ylabel('PC2')
#plt.show(b)
b=plt.title(diff[diff.rfind("/")+1:diff.rfind(".")], fontsize=10)

b=plt.savefig(density_stack_sort[0:len(density_stack_sort)-4] + "_PCA.png")

clus_num=np.zeros(int(n_clus))
clus_num=clus_num.astype(int)
for n in range(int(n_clus)):
	for i in range(sta_dens):
		if(labels[i] == n):
			clus_num[n]=clus_num[n]+1

X=np.arange(row)
#X=np.flip(X)
Y=np.arange(col)
Y=np.flip(Y)

#カラーマップ

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

#open diff
op_diff=Image.open(diff)
np_diff=np.asarray(op_diff,dtype="float32")
np_diff_amp=np.zeros(op_diff.size,dtype="float32")
np_diff_amp=np.where(np_diff>0.0,np_diff,0.0)
np_diff_amp=np.sqrt(np_diff_amp)
cp_diff_amp=cp.asarray(np_diff_amp,dtype="float32")
row=np_diff.shape[0]
col=np_diff.shape[1]

#montage
a=plt.figure(figsize=(20,8))
clus_ave_dens=np.zeros((int(n_clus),row,col))
clus_ave_sup=np.zeros((int(n_clus),row,col))
clus_ave_dens=clus_ave_dens.astype(np.float32)
clus_ave_sup=clus_ave_dens.astype(np.float32)

#log
log_path=density_stack_sort[0:len(density_stack_sort)-4] + "_PCA.log"
with open(log_path, mode='w') as log:
	log.write("class,n_class,RF,Resolution,OS_ratio " + "\n")


for n in range(int(n_clus)):
	clus_dens=np.zeros((clus_num[n],row,col))
	clus_dens=clus_dens.astype(np.float32)
	phase_exp=np.zeros((clus_num[n],row,col))
	phase_exp=phase_exp.astype(np.float32)

	num=0
	for i in range(sta_dens):
		if(labels[i] == n):
			clus_dens[num,:,:]=np_sort[i,:,:]
			num=num+1

	with mrcfile.new(dir_dens + '/class_' + str(n) + '.mrc', overwrite=True) as mrc:
		mrc.set_data(clus_dens)

	mrc.close

	subprocess.run(["python.exe", "C:\Python_Programs\Sort_and_average\Sort_and_average.py", dir_dens + '/class_' + str(n) + '.mrc', "20"])

	with mrcfile.open(dir_dens + '/class_' + str(n) + '_sort.mrc', permissive=True) as mrc:
		clus_dens=np.asarray(mrc.data,dtype="float32")
	mrc.close

	clus_ave_dens[n,:,:]=np.average(clus_dens,axis=0)
	ave_dens=np.average(clus_ave_dens, axis=(1,2))
	clus_ave_sup[n,:,:]=np.where(clus_ave_dens[n,:,:] > ave_dens[n], 1.0, 0.0)

	tifffile.imsave(dir_ave_dens + '/ave_dens_class_' + str(n) + '.tif' ,clus_ave_dens[n,:,:])
	tifffile.imsave(dir_ave_sup  + '/ave_sup_class_' + str(n) + '.tif' ,clus_ave_sup[n,:,:])
	OS_ratio=(row*col)/np.sum(clus_ave_sup[n,:,:])
	c_OS_ratio=str(OS_ratio)
	
	#R-factor

	R_dens=cp.asarray(clus_ave_dens[n,:,:],dtype="float32")
	R_structure_factor = cp.fft.fft2(R_dens,norm="ortho")
	R_structure_factor = cp.fft.fftshift(R_structure_factor)
	R_amp = cp.absolute(R_structure_factor)

	amp2_sum=cp.sum(cp.square(R_amp[cp_diff_amp%2>0.0]))
	amp_x_diff_amp_sum=cp.sum(cp.absolute(R_amp[cp_diff_amp%2>0.0])*cp.absolute(cp_diff_amp[cp_diff_amp%2>0.0]))
	diff_amp_sum=cp.sum(cp.absolute(cp_diff_amp[cp_diff_amp%2>0.0]))
	scale_factor=amp_x_diff_amp_sum/amp2_sum
#	print(scale_factor)
	
	diff_amp_scale=cp.sum(cp.absolute(cp.absolute(cp_diff_amp[cp_diff_amp%2>0.0])-scale_factor*cp.absolute(R_amp[cp_diff_amp%2>0.0])))
	R_factor=diff_amp_scale/diff_amp_sum
	c_R_factor=str(R_factor)
#	print("R_factor = " + str(R_factor))

	R_amp=cp.square(R_amp)
	np_R_amp = cp.asnumpy(cp.square(R_amp))#cupy配列 ⇒ numpy配列に変換
	tifffile.imsave(dir_ave_diff + '/ave_diff_class_' + str(n) + '.tif' ,np_R_amp)

	#PRTF
	
	for i in range(clus_num[n]):
		R_dens=cp.asarray(clus_dens[i,:,:],dtype="float32")
		R_structure_factor = cp.fft.fft2(R_dens,norm="ortho")
		R_structure_factor = cp.fft.fftshift(R_structure_factor)
		cp_phase=cp.angle(R_structure_factor)
		R_structure_factor.real=0.0
		R_structure_factor.imag=cp_phase
		phase_exp[i,:,:] = cp.asnumpy(cp.exp(R_structure_factor))

	sum_phase_exp=np.sum(phase_exp,axis=0)
	sum_phase_exp=np.absolute(sum_phase_exp)
	sum_phase_exp=sum_phase_exp[:,:]/float(clus_num[n])
	tifffile.imsave(dir_PhaseSta  + '/PhaseSta_' + str(n) + '.tif' ,sum_phase_exp)
#	print(sum_phase_exp)

	temp_rad=np.zeros((row,col))		
	temp_rad=temp_rad.astype(int)
	for x in range(row):
		for y in range(col):
			temp_rad[x,y]=int(np.sqrt(np.square(x-row/2)+np.square(y-col/2)))

	rad_num=np.zeros(int(np.max(temp_rad))+1)
	rad_num=rad_num.astype(np.float32)
	rad_value=np.zeros(int(np.max(temp_rad))+1)
	rad_value=rad_value.astype(np.float32)
	for x in range(row):
		for y in range(col):
			rad_num[int(temp_rad[x,y])]=rad_num[int(temp_rad[x,y])]+1.0
			rad_value[int(temp_rad[x,y])]=rad_value[int(temp_rad[x,y])]+sum_phase_exp[x,y]

	PRTF=rad_value[:]/rad_num[:]
#	print("PRTF")
#	print(PRTF)
	tifffile.imsave(dir_PRTF  + '/PRTF_' + str(n) + '.tif' ,PRTF)
#	print(rad_num)
	for i in range((int(np.max(temp_rad)))):
		if(i > 1):
			if(PRTF[i] < 0.5):
				if(PRTF[i-1] > 0.5):
					res_i=i
	res=str(1000.0/(0.14784408*float(res_i)))
#	print(1000.0/(0.057355*float(res_i)),res_i)	
	
#	tifffile.imsave(dir_ave_phase + '/ave_phase_class_' + str(n) + '.tif' ,phase[n,:,:])	


#	plt.imshow(clus_ave_dens[n,:,:])
#	fig1=plt.figure(1)
	a=plt.subplot(2,int(int(n_clus)/2),n+1)
	a=plt.pcolormesh(X,Y,clus_ave_dens[n,:,:], cmap=royal, vmin=0.0)
	a=plt.gca().set_aspect('equal', adjustable='box')
	a=plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
	a=plt.tick_params(bottom=False, left=False, right=False, top=False)
	a=plt.title("class " + str(n) + ', n= ' + str(clus_num[n]),fontsize=15)
	a=plt.xlabel("RF= " + c_R_factor[0:c_R_factor.find(".") + 3] + ", Res= " + res[0:res.find(".") + 2] + " nm" , fontsize=15)
	a=plt.ylabel("OS ratio= " + c_OS_ratio[0:c_OS_ratio.find(".") + 2] , fontsize=15)
	a=plt.suptitle(diff[diff.rfind("/")+1:diff.rfind(".")], fontsize=15)

	#a=plt.colorbar()

	print("class(" + str(n) + ") : n = " + str(clus_num[n]) + ", RF= " + c_R_factor[0:c_R_factor.find(".") + 3] + ", Res= " + res[0:res.find(".") + 2] + " nm" + ", OS_ratio= " + c_OS_ratio[0:c_OS_ratio.find(".") + 2])

	log_text=str(n) + "," + str(clus_num[n]) + "," + c_R_factor + "," + res + "," + str(OS_ratio) + "\n"
	with open(log_path, mode='a') as log:
		log.write(log_text)

a=plt.savefig(density_stack_sort[0:len(density_stack_sort)-4] + "_montage.png")

for n in range(int(n_clus)):
	c=plt.subplot(2,int(int(n_clus)/2),n+1)
	c=plt.pcolormesh(X,Y,clus_ave_sup[n,:,:], cmap=royal, vmin=0.0, vmax=1.0)
	c=plt.gca().set_aspect('equal', adjustable='box')
	c=plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
	c=plt.tick_params(bottom=False, left=False, right=False, top=False)
	c=plt.title("class " + str(n) + ', n= ' + str(clus_num[n]),fontsize=15)
	c=plt.xlabel("RF= " + c_R_factor[0:c_R_factor.find(".") + 3] + ", Res= " + res[0:res.find(".") + 2] + " nm" , fontsize=15)
	c=plt.ylabel("OS ratio= " + c_OS_ratio[0:c_OS_ratio.find(".") + 2] , fontsize=15)
	c=plt.suptitle(diff[diff.rfind("/")+1:diff.rfind(".")], fontsize=15)

c=plt.savefig(density_stack_sort[0:len(density_stack_sort)-4] + "_montage_sup.png")

#plt.show(a)

t2=time.time()
print("calculation time : " + str(t2-t1))










