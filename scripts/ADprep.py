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

fid=Dataset('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/big.nc' , 'r')
pres=fid.variables['pres'][:]
temp=fid.variables['wdir'][:]


strides=[100,110,120,130,140,150,160,170,180,190,200]

i=1
while i<5: #Loop through the profiles I want
	longPres=pres[i,:,:]
	longtemp=temp[i,:,:] #Grab the correct pressure and temperature
	mas=np.ma.mask_or(np.ma.getmask(longPres), np.ma.getmask(longtemp)) #Create a mask that shows where either data is missing
	longPres.mask=mas
	longtemp.mask=mas
	longPres=longPres.compressed()
	longtemp=longtemp.compressed() #apply mask and compress pressure/temperature to 1dim

	#Create file
	fnew=Dataset('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/terpinput{}.nc'.format(i), 'w', format='NETCDF4')
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
	z=0
	for npoints in strides: #Upscale different resolutiions to same size
		#Subset points by striding
		stride=int(np.floor(len(longPres)/npoints))	
		shortPres=longPres[::stride]
		shorttemp=longtemp[::stride]
	
	#Create interpolation function from shortened dataset
		f=interpolate.interp1d(shortPres, shorttemp, fill_value='extrapolate')

	#Interpolate back to original dimensions
		shortnew=f(longPres)
	#Save to file each iteration
		tmp[z,:,:]=np.reshape(shortnew, [len(shortnew), 1])
		pre[z,:,:]=np.reshape(longPres, [len(longPres), 1])
		z+=1
	#Close file
	fnew.close()
	i+=1
fid.close()
