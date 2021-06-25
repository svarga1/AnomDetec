#File to read and plot temperature profile from netcdf

import datetime as dt
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt


path='/work/noaa/da/cthomas/ens/2020090500/001/gdas.t00z.atmf003.nc'



#Open file
nc_fid=Dataset(path, 'r') 
#nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)


#Get data
print(nc_fid.variables['grid_xt'].shape)
print(nc_fid.variables['grid_yt'].shape)

print(nc_fid.variables['lat'].shape)
print(nc_fid.variables['tmp'].shape)
print(nc_fid.variables['lat'][107, 605])
print(nc_fid.variables['lon'][107, 605])
#print(nc_fid.variables['pfull'][:])

print



#Test plot
pres=nc_fid.variables['pfull'][:]
temp=nc_fid.variables['tmp'][:]
temp=temp[0,:,107,605]

plt.plot(temp, pres)
plt.show()
