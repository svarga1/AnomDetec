#Test to plot the netcdf4 that I made

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

fid=Dataset('test.nc', 'r')
temp=fid.variables['Temp'][0,:,0]

fig=plt.figure()
plt.plot(temp)
plt.savefig('test.png')
plt.close()


import ncepbufr


path=''
bufr=ncepbufr.open(path)

while bufr.load_subset()==-1:
	bufr.advance()
temp=bufr.read_subset('TMDB').squeeze()

plt.plot(temp)
plt.savefig('testbuf')
