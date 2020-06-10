import tifffile
import os
import sys
sys.path.append('C:\Python_Programs\my_module')
import convert_RGB
from fortran_SPE import sub_inputSPEsize
from fortran_SPE import sub_READSPE
from fortran_RGB import sub_convert_RGB
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib

finame=sys.argv[1]
print("finame = " + finame)

c_min_range=sys.argv[2]
print("min_range = " + c_min_range)
min_range=int(c_min_range)

c_max_range=sys.argv[3]
print("max_range = " + c_max_range)
max_range=int(c_max_range)

if(finame[len(finame)-3:len(finame)] == "SPE"):
    ixpix,iypix=sub_inputSPEsize.inputspesize(finame)
    print("ixpix = " + str(ixpix))
    print("iypix = " + str(iypix))
    np_diff=sub_READSPE.readspe(ixpix,iypix,finame)

if(finame[len(finame)-3:len(finame)] == "tif"):
    img=Image.open(finame)
    np_diff=np.asarray(img,dtype="float32")
    ixpix=np_diff.shape[0]
    iypix=np_diff.shape[1]
    print("ixpix = " + str(ixpix))
    print("iypix = " + str(iypix))

fortran_flag=1
if(fortran_flag == 1):
    np_diff_log_rgb=sub_convert_RGB.convert_rgb(np_diff,min_range,max_range)
    np_diff_log_rgb=np.rot90(np_diff_log_rgb)
    np_diff_log_rgb=np.flipud(np_diff_log_rgb)
else:
    np_diff_log_rgb=convert_RGB.convert_RGB(np_diff,min_range,max_range)
np_diff_log_rgb=np.asarray(np_diff_log_rgb,dtype="uint8")

img = Image.fromarray(np_diff_log_rgb)
img.save('myimg.jpeg')
