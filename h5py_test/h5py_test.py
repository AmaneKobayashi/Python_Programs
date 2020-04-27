#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import tifffile

f=h5py.File("run54459-1.h5","r")

img=np.array(f['tag-581035467/data'])

tifffile.imsave('test3.tif',img)

