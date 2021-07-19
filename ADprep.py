#Script that creates histogram of bufr file

import VargaBufr as VB
import ncepbufr
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import matplotlib
from numpy import diff
from scipy import interpolate
from random import sample
import scipy
from scipy import stats
from netCDF4 import Dataset
matplotlib.use('agg')

#Make new file
makF=True
if makF:
	fnew=Dataset('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/terpinput.nc', 'w', format='NETCDF4')
	#Create dimensions
	data_dim=fnew.createDimension('data', None)
	prof_dim=fnew.createDimension('profile', None)
	temp_dim=fnew.createDimension('temperature', 1)
	pres_dim=fnew.createDimension('pressure',1)
	wind_dim=fnew.createDimension('wind',1)
	dpt_dim=fnew.createDimension('dewpoint',1)
	##Variables
	tmp=fnew.createVariable('temp', np.float32, ('profile','data','temperature'))
	tmp.units='C'
	pre=fnew.createVariable('pres', np.float32, ('profile' ,'data','pressure'))
	pre.units='hPa'
	wdir=fnew.createVariable('wdir', np.float32, ('profile', 'data', 'wind'))
	wdir.units='Degree heading'
	dpt=fnew.createVariable('dpt', np.float32, ('profile','data','dewpoint'))
	dpt.units='C'
	wspd=fnew.createVariable('wspd', np.float32, ('profile', 'data', 'wind'))
	wspd.units='m/s'



#open file- find length
fid=Dataset('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/big.nc', 'r')
pres=fid.variables['pres'][:,:,:]
temp=fid.variables['temp'][:,:,:]
maxLen=temp.shape[1]
print( maxLen)



#Find pressure values for that length
h=0
while h<pres.shape[0]:
	if len(pres[h,:,:].compressed())==maxLen:
		print(h)
		break
	else:
		h+=1
highPresRes=pres[h,:,:].compressed()

#Loop through- interpolate to highest resolution
i=0
while i<pres.shape[0]:
		longPres=pres[i,:,:].flatten()
		longtemp=temp[i,:,:].flatten()
		mas = np.ma.mask_or(np.ma.getmask(longPres), np.ma.getmask(longtemp))
		longPres.mask=mas
		longtemp.mask=mas
		longPres=longPres.compressed()
		longtemp=longtemp.compressed()
		f=interpolate.interp1d(longPres, longtemp, fill_value='extrapolate') #Feed all points in the profile into interpolation to create function
		tmp[i,:,:]=np.reshape(f(highPresRes), [len(highPresRes),1]) #Create the new points for the old index value
		pre[i,:,:]=np.reshape(highPresRes, [len(highPresRes),1]) #Assign variables	
		i+=1 #Move to the next profile
	
#Save out
if makF:
	fnew.close()
fid.close()
