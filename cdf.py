#File to read and plot temperature profile from netcdf

import datetime as dt
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from tensorflow import keras


path='/work/noaa/da/cthomas/ens/2020090500/001/gdas.t00z.atmf003.nc'



#Open file
nc_fid=Dataset(path, 'r') 
#nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)


#Get data
#print(nc_fid.variables['grid_xt'].shape)
#print(nc_fid.variables['grid_yt'].shape)

#print(nc_fid.variables['lat'].shape)
#print(nc_fid.variables['tmp'].shape)
#3print(nc_fid.variables['lat'][107, 605])
#3print(nc_fid.variables['lon'][107, 605])
#print(nc_fid.variables['pfull'][:])

#print



#Test plot
#pres=nc_fid.variables['pfull'][:]
temp=nc_fid.variables['tmp'][:]
temp=temp[0,:,107,605]
print(temp)
#plt.plot(temp, pres)
#plt.show()




exit()
#Open file

#ad custom data- make this a function later
training=np.array([])
firstIt=True
z=1
print('starting readin')
for filepath in Path('/work/noaa/da/cthomas/ens/2020090500').glob('*/*.nc'):
	if firstIt:
		nc_fid=Dataset(filepath, 'r')
		pres=nc_fid.variables['pfull'][:]
		temp=nc_fid.variables['tmp'][:]
		training=np.reshape(temp[0,:,107,605], [1,127,1])
		firstIt=False
		nc_fid.close()
		print(z)
		z+=1
	else:
		nc_fid=Dataset(filepath, 'r')
		temp=nc_fid.variables['tmp'][:]
		temp=np.reshape(temp[0,:,107,605], [1,127,1])
		training=np.insert(training, -1, temp, axis=0)
		nc_fid.close()
		print(z)
		z+=1

