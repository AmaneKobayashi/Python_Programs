import tifffile
import sub_inputSPEsize
import sub_READSPE
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

finame=sys.argv[1]
print("finame = " + finame)

c_min_range=sys.argv[2]
print("min_range = " + c_min_range)
min_range=int(c_min_range)

c_max_range=sys.argv[3]
print("max_range = " + c_max_range)
max_range=int(c_max_range)

ixpix,iypix=sub_inputSPEsize.inputspesize(finame)
print("ixpix = " + str(ixpix))
print("iypix = " + str(iypix))

if(finame[len(finame)-3:len(finame)] == "SPE"):
    np_diff=sub_READSPE.readspe(ixpix,iypix,finame)
    np_diff=np.where(np_diff > 0, np_diff, 1)


np_diff_log=np.log10(np_diff)
np_diff_log=np.rot90(np_diff_log)
np_diff_log=np.fliplr(np_diff_log)

cdict1 = {'red'  :((0.0 ,0.0  ,0.0),
                  (0.375,0.0  ,0.0),
                  (0.492,1.0  ,1.0),
                  (0.75 ,1.0  ,1.0),
                  (0.80 ,0.867,0.867),
                  (0.875,0.867,0.867),
                  (0.895,1.0  ,1.0),
                  (1.0,  1.0  ,1.0)),
          'green':((0.0  ,0.0  ,0.0),
                  (0.129 ,0.0  ,0.0),
                  (0.3125,1.0  ,1.0),
                  (0.4375,1.0  ,1.0),
                  (0.75  ,0.0  ,0.0),
                  (0.8125,0.734,0.734),
                  (0.9375,1.0  ,1.0),
                  (1.0   ,1.0  ,1.0)),
          'blue' :((0.0   ,0.03 ,0.03),
                  (0.1875,1.0  ,1.0),
                  (0.375 ,1.0  ,1.0),
                  (0.4375,0.0  ,0.0),
                  (0.754 ,0.0  ,0.0),
                  (0.8125,0.715,0.715),
                  (0.9375,1.0  ,1.0),
                  (1.0   ,1.0  ,1.0))}
royal = matplotlib.colors.LinearSegmentedColormap('wrainbow',cdict1,256)

cdict_Red_X=np.zeros([8],dtype="float32")
cdict_Red_Y=np.zeros([8],dtype="float32")
cdict_Green_X=np.zeros([8],dtype="float32")
cdict_Green_Y=np.zeros([8],dtype="float32")
cdict_Blue_X=np.zeros([8],dtype="float32")
cdict_Blue_Y=np.zeros([8],dtype="float32")

cdict_Red_X[0]=0.0 
cdict_Red_X[1]=0.375
cdict_Red_X[2]=0.492
cdict_Red_X[3]=0.75 
cdict_Red_X[4]=0.80 
cdict_Red_X[5]=0.875
cdict_Red_X[6]=0.895
cdict_Red_X[7]=1.0

cdict_Red_Y[0]=0.0  
cdict_Red_Y[1]=0.0  
cdict_Red_Y[2]=1.0  
cdict_Red_Y[3]=1.0  
cdict_Red_Y[4]=0.867
cdict_Red_Y[5]=0.867
cdict_Red_Y[6]=1.0  
cdict_Red_Y[7]=1.0 

cdict_Green_X[0]=0.0  
cdict_Green_X[1]=0.129 
cdict_Green_X[2]=0.3125
cdict_Green_X[3]=0.4375
cdict_Green_X[4]=0.75  
cdict_Green_X[5]=0.8125
cdict_Green_X[6]=0.9375
cdict_Green_X[7]=1.0   

cdict_Green_Y[0]=0.0  
cdict_Green_Y[1]=0.0  
cdict_Green_Y[2]=1.0  
cdict_Green_Y[3]=1.0  
cdict_Green_Y[4]=0.0  
cdict_Green_Y[5]=0.734
cdict_Green_Y[6]=1.0  
cdict_Green_Y[7]=1.0  

cdict_Blue_X[0]=0.0   
cdict_Blue_X[1]=0.1875
cdict_Blue_X[2]=0.375 
cdict_Blue_X[3]=0.4375
cdict_Blue_X[4]=0.754 
cdict_Blue_X[5]=0.8125
cdict_Blue_X[6]=0.9375
cdict_Blue_X[7]=1.0   

cdict_Blue_Y[0]=0.03 
cdict_Blue_Y[1]=1.0  
cdict_Blue_Y[2]=1.0  
cdict_Blue_Y[3]=0.0  
cdict_Blue_Y[4]=0.0  
cdict_Blue_Y[5]=0.715
cdict_Blue_Y[6]=1.0  
cdict_Blue_Y[7]=1.0  


precision=int(10000)
cdict_Red_X[:]=cdict_Red_X[:]*float(precision)
cdict_Green_X[:]=cdict_Green_X[:]*float(precision)
cdict_Blue_X[:]=cdict_Blue_X[:]*float(precision)

cdict_Red_Y[:]=cdict_Red_Y[:]*float(255)
cdict_Green_Y[:]=cdict_Green_Y[:]*float(255)
cdict_Blue_Y[:]=cdict_Blue_Y[:]*float(255)

Red=np.zeros([precision],dtype="float32")
Green=np.zeros([precision],dtype="float32")
Blue=np.zeros([precision],dtype="float32")

Red_index=0
Green_index=0
Blue_index=0
a_Red=(cdict_Red_Y[Red_index+1] - cdict_Red_Y[Red_index]) / (cdict_Red_X[Red_index+1] - cdict_Red_X[Red_index])
b_Red=(cdict_Red_X[Red_index+1] * cdict_Red_Y[Red_index] - cdict_Red_X[Red_index] * cdict_Red_Y[Red_index+1]) / (cdict_Red_X[Red_index+1] - cdict_Red_X[Red_index])  
a_Green=(cdict_Green_Y[Green_index+1] - cdict_Green_Y[Green_index]) / (cdict_Green_X[Green_index+1] - cdict_Green_X[Green_index])
b_Green=(cdict_Green_X[Green_index+1] * cdict_Green_Y[Green_index] - cdict_Green_X[Green_index] * cdict_Green_Y[Green_index+1]) / (cdict_Green_X[Green_index+1] - cdict_Green_X[Green_index])
a_Blue=(cdict_Blue_Y[Blue_index+1] - cdict_Blue_Y[Blue_index]) / (cdict_Blue_X[Blue_index+1] - cdict_Blue_X[Blue_index])
b_Blue=(cdict_Blue_X[Blue_index+1] * cdict_Blue_Y[Blue_index] - cdict_Blue_X[Blue_index] * cdict_Blue_Y[Blue_index+1]) / (cdict_Blue_X[Blue_index+1] - cdict_Blue_X[Blue_index])  

for i in range(precision):
    if(i > int(cdict_Red_X[Red_index+1])):
        Red_index=Red_index+1
        a_Red=(cdict_Red_Y[Red_index+1] - cdict_Red_Y[Red_index]) / (cdict_Red_X[Red_index+1] - cdict_Red_X[Red_index])
        b_Red=(cdict_Red_X[Red_index+1] * cdict_Red_Y[Red_index] - cdict_Red_X[Red_index] * cdict_Red_Y[Red_index+1]) / (cdict_Red_X[Red_index+1] - cdict_Red_X[Red_index])        
    if(i > int(cdict_Green_X[Green_index+1])):
        Green_index=Green_index+1
        a_Green=(cdict_Green_Y[Green_index+1] - cdict_Green_Y[Green_index]) / (cdict_Green_X[Green_index+1] - cdict_Green_X[Green_index])
        b_Green=(cdict_Green_X[Green_index+1] * cdict_Green_Y[Green_index] - cdict_Green_X[Green_index] * cdict_Green_Y[Green_index+1]) / (cdict_Green_X[Green_index+1] - cdict_Green_X[Green_index])                
    if(i > int(cdict_Blue_X[Blue_index+1])):
        Blue_index=Blue_index+1
        a_Blue=(cdict_Blue_Y[Blue_index+1] - cdict_Blue_Y[Blue_index]) / (cdict_Blue_X[Blue_index+1] - cdict_Blue_X[Blue_index])
        b_Blue=(cdict_Blue_X[Blue_index+1] * cdict_Blue_Y[Blue_index] - cdict_Blue_X[Blue_index] * cdict_Blue_Y[Blue_index+1]) / (cdict_Blue_X[Blue_index+1] - cdict_Blue_X[Blue_index])

    Red[i]=a_Red * float(i) + b_Red
    Green[i]=a_Green * float(i) + b_Green
    Blue[i]=a_Blue * float(i) + b_Blue

np_diff_log_X=np.zeros(np_diff_log.shape,dtype="int")
step=(max_range - min_range) / float(precision)
np_diff_log_X=np.array((np_diff_log - min_range) / step,dtype="int")
np_diff_log_X=np.where(np_diff_log <= min_range, 0,np_diff_log_X)
np_diff_log_X=np.where(np_diff_log >= max_range, precision-1, np_diff_log_X)
np_diff_log_X=np.where(np_diff_log == np.nan, 0,np_diff_log_X)

np_diff_log_rgb = np.zeros((ixpix,iypix,3), 'uint8')
for ii in range(iypix):
    for i in range(ixpix):
        #print(np_diff_log_X[i,ii])
        np_diff_log_rgb[i,ii,0]=np.array(Red[np_diff_log_X[i,ii]],dtype="uint8")
        np_diff_log_rgb[i,ii,1]=np.array(Green[np_diff_log_X[i,ii]],dtype="uint8")
        np_diff_log_rgb[i,ii,2]=np.array(Blue[np_diff_log_X[i,ii]],dtype="uint8")

img = Image.fromarray(np_diff_log_rgb)
img.save('myimg.jpeg')

exit()

X=np.arange(ixpix)
Y=np.arange(iypix)
#Y=np.flip(Y)

#print(np_diff_log.size)

#a=plt.figure(dpi=200,figsize=(5,5))
a=plt.axes().set_aspect('equal')
a=plt.pcolormesh(X,Y,np_diff_log[:,:], cmap=royal, vmin=0.0, vmax=5.0)
#a=plt.tick_params(labelbottom=False,
#                labelleft=False,
#                labelright=False,
#                labeltop=False)
#a=plt.tick_params(bottom=False,
#                left=False,
#                right=False,
#                top=False)
plt.axis('off')
#plt.show()

foname=finame[0:len(finame)-4] + ".jpg"
a=plt.savefig(foname, bbox_inches='tight')
