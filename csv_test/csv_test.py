#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_csv('../h5py_test/54512-0/54512-0.csv', index_col=0)

#print(df.query('NCRYST==1')['EVENT'])

#print(len(df))

for row in range(len(df)):
	if df.iat[row,1]==1:
		print(df.iat[row,2][0:13])
