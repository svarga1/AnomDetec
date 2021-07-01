#Test file for reading bufr file and writing to nc


import VargaBufr as VB
from  netCDF4 import Dataset
import numpy as np
import ncepbufr






#Create NC skeleton
ncfile=Dataset('test.nc', mode='w', format='NETCDF4')





