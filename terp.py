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

#Create NC skeleton
#ncfile=Dataset('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/big.nc', 'w', format='NETCDF4')

#Create dimensions
#data_dim=ncfile.createDimension('data', None)
#prof_dim=ncfile.createDimension('profile', None)
#temp_dim=ncfile.createDimension('temperature', 1)
#pres_dim=ncfile.createDimension('pressure',1)
#wind_dim=ncfile.createDimension('wind',1)
#dpt_dim=ncfile.createDimension('dewpoint',1)
##Variables
#temp=ncfile.createVariable('temp', np.float32, ('profile','data','temperature'))
#temp.units='C'
#pre=ncfile.createVariable('pres', np.float32, ('profile' ,'data','pressure'))
#pre.units='hPa'
#wdir=ncfile.createVariable('wdir', np.float32, ('profile', 'data', 'wind'))
#wdir.units='Degree heading'
#dpt=ncfile.createVariable('dpt', np.float32, ('profile','data','dewpoint'))
#dpt.units='C'
#wspd=ncfile.createVariable('wspd', np.float32, ('profile', 'data', 'wind'))
#wspd.units='m/s'
#i=0


#path='/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/gdas.20210601/gdas.t00z.uprair.tm00.bufr_d'

#for filepath in Path('/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/gdas.20210601/').glob('*.bufr_d'):
#	bufr=ncepbufr.open(filepath)
#	while bufr.load_subset()==-1:
#		bufr.advance()
#
#	pres= bufr.read_subset('PRLC').squeeze()/100
#	t=bufr.read_subset('TMDB').squeeze()-273.15
#	d=bufr.read_subset('TMDP').squeeze()-273.15
#	wd=bufr.read_subset('WDIR').squeeze()
#	ws=bufr.read_subset('WSPD').squeeze()
#	if len(pres)>3000:
		#longPres=pres
#		#longtemp=t
#		temp[i,:,:]=np.reshape(t, [len(t),1])
#		pre[i,:,:]=np.reshape(pres, [len(pres),1])
#		wdir[i,:,:]=np.reshape(wd, [len(wd),1])
#		wspd[i,:,:]=np.reshape(ws, [len(ws),1])
#		dpt[i,:,:]=np.reshape(d, [len(d), 1])
#		i+=1
#	while bufr.load_subset()==0:
#		try:
#			pres= bufr.read_subset('PRLC').squeeze()/100
#			t=bufr.read_subset('TMDB').squeeze()-273.15
#			d=bufr.read_subset('TMDP').squeeze()-273.15
#			wd=bufr.read_subset('WDIR').squeeze()
#			ws=bufr.read_subset('WSPD').squeeze()
#			if len(pres)>3000:
#				longPres=pres
##				longtemp=t
#				temp[i,:,:]=np.reshape(t, [len(t),1])
#				pre[i,:,:]=np.reshape(pres, [len(pres),1])
#				wdir[i,:,:]=np.reshape(wd, [len(wd),1])
#				wspd[i,:,:]=np.reshape(ws, [len(ws),1])
#				dpt[i,:,:]=np.reshape(d, [len(d), 1])

#				i+=1
#		
#		except:
#			pass
	
#	while bufr.advance() ==0:
#		while bufr.load_subset()==0:
#			try:
#				pres=bufr.read_subset('PRLC').squeeze()/100
#				t=bufr.read_subset('TMDB').squeeze()-273.15
#				d=bufr.read_subset('TMDP').squeeze()-273.15
#				wd=bufr.read_subset('WDIR').squeeze()
#				ws=bufr.read_subset('WSPD').squeeze()
#				if len(pres)>3000:
#					longPres=pres
#					longtemp=t
#					temp[i,:,:]=np.reshape(t, [len(t),1])
#					pre[i,:,:]=np.reshape(pres, [len(pres),1])
#					wdir[i,:,:]=np.reshape(wd, [len(wd),1])
#					wspd[i,:,:]=np.reshape(ws, [len(ws),1])
#					dpt[i,:,:]=np.reshape(d, [len(d), 1])

#					i+=1

#			except:
#				pass
#
#	bufr.close()
#ncfile.close()



#exit()

suff='terp'
fid=Dataset('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/big.nc', 'r')
pres=fid.variables['pres'][:]
temp=fid.variables['temp'][:]



strides=[10,50,100,150,200]
i=1
while i<5:
	longPres=pres[i,:,:].flatten()			#Need to change how missing data is handled- generate y for all x points, then mask y based on original array? Would require no missing pressure data?
	longtemp=np.cos(temp[i,:,:].flatten()*np.pi/180)
	#longtemp=temp[i,:,:].flatten()
#Data masking
	mas=np.ma.mask_or(np.ma.getmask(longPres),np.ma.getmask(longtemp)) #This returns a mask the same length as longPres and longtemp, evaluated to True where there is missing data in either.
	longPres.mask=mas
	longtemp.mask=mas
	longPres=longPres.compressed()
	longtemp=longtemp.compressed()
	for npoints in strides:

		#Subset points by striding
		stride=int(np.floor(len(longPres)/npoints))
		shortPres=longPres[::stride]
		shorttemp=longtemp[::stride]
		print(shortPres.shape)
		print(shorttemp.shape)
	
		#Subset points by normalization
		ind=np.random.randint(0, len(longPres), npoints)	
		normPres=longPres[ind]
		normtemp=longtemp[ind]
		print(normPres.shape)
		print(normtemp.shape)
	
		#Interpolate back to original resolution
		normf=interpolate.interp1d(normPres, normtemp,fill_value='extrapolate')
		shortf=interpolate.interp1d(shortPres, shorttemp, fill_value='extrapolate')
	
	
		normnew=normf(longPres)
		shortnew=shortf(longPres)
	
		#Observation minus interpolation
#		normIMO=longtemp[:len(longPres.compressed())]-normnew
#		shortIMO=longtemp[:len(longPres.compressed())]-shortnew
		normIMO=longtemp-normnew
		shortIMO=longtemp-shortnew
		print('Stats for normIMO: {0}, {1}, {2}, {3}'.format(np.amin(normIMO), np.amax(normIMO), np.mean(normIMO), np.std(normIMO)))
		print('Stats for shortIMO: {0}. {1}, {2}, {3}'.format(np.amin(shortIMO), np.amax(shortIMO), np.mean(shortIMO), np.std(shortIMO)))

	#Liner Regression
#		normLR=scipy.stats.linregress(longtemp[:len(longPres.compressed())], normnew)
#		shortLR=scipy.stats.linregress(longtemp[:len(longPres.compressed())], shortnew)
		normLR=scipy.stats.linregress(longtemp, normnew)
		shortLR=scipy.stats.linregress(longtemp, shortnew)
		print('For Sampled Data: m={0}, b={1}, r={2},r^2={4}, Se={3}'.format(normLR[0], normLR[1], normLR[2], normLR[4], normLR[2]**2))
		print('For Stride Data: m={0}, b={1}, r={2},r^2={4}, Se={3}'.format(shortLR[0], shortLR[1], shortLR[2], shortLR[4], shortLR[2]**2))

	
#Plot

	#Plot subsets
		fig, ax= plt.subplots(1,3, sharex='row', sharey='row', figsize= (10,10))
		ax[0].scatter(longtemp, longPres, 4)
		ax[0].set_title('Original Data: {} points'.format(len(longPres)))
		ax[1].scatter(shorttemp, shortPres, 4)
		ax[1].set_title('Stride Data: {} points'.format(len(shorttemp)))
		ax[2].scatter(normtemp, normPres, 4)
		ax[2].set_title('Sampled Data: {} points'.format(len(normtemp)))
		ax[0].invert_yaxis()
		ax[0].set_yscale('log')
		ax[0].set_ylabel('Pressure (hPa)')
		ax[0].set_xlabel('Temperature (C)')
		plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/{2}/{0}subset{1}.png'.format(npoints,i,suff))
		plt.close()


		fig, ax= plt.subplots(1,3, sharex='row', sharey='row', figsize=(10,10))
		ax[0].scatter(longtemp, longPres, 4)
		ax[0].set_title('Original Data: {} points'.format(len(longPres)))
		ax[1].scatter(shortnew,longPres,4 )
		ax[1].set_title('Stride Data: {} points'.format(len(shortnew)))
		ax[2].scatter(normnew,longPres, 4 )
		ax[2].set_title('Sampled Data: {} points'.format(len(normnew)))
		ax[0].invert_yaxis()
		ax[0].set_yscale('log')
		ax[0].set_ylabel('Pressure (hPa)')
		ax[0].set_xlabel('Temperature (C)')
		plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/{2}/{0}profile{1}.png'.format(npoints,i,suff))
		plt.close()
	
	#Plot OMI

		fig, ax = plt.subplots(1,2, sharex='row', sharey='row', figsize= (10,10))
		ax[0].hist(shortIMO)
		ax[1].hist(normIMO)
		ax[0].set_title('Stride OMI: {} points'.format(len(shorttemp)))
		ax[1].set_title('Sampled OMI: {} points'.format(len(normtemp)))
		ax[0].set_ylabel('Frequency')
		ax[0].set_xlabel('OMI')
		plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/{2}/{0}hist{1}.png'.format(npoints,i, suff))
		plt.close()

	#OMI Profile 
		fig, ax =plt.subplots(1,2, sharex='row', sharey='row', figsize=(10,10))
		ax[0].scatter(shortIMO, longPres,4)
		ax[1].scatter(normIMO, longPres,4)
		ax[0].set_title('Stride OMI: {} Points'.format(len(shortIMO)))
		ax[1].set_title('Ssmpled OMI: {} Points'.format(len(shortIMO)))
		ax[0].invert_yaxis()
		ax[0].set_yscale('log')
		ax[0].set_ylabel('Pressure (hPa)')
		ax[0].set_xlabel('OMI (C)')
		plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/{2}/{0}OMIprofile{1}.png'.format(npoints,i,suff))
		plt.close()

	#Linear regression
		fig, ax =plt.subplots(1,2, sharex='row', figsize=(10,10))
		ax[0].scatter(longtemp, shortnew, 4)
		ax[0].scatter(longtemp, longtemp, 4, color='r')
		ax[1].scatter(longtemp, normnew, 4)
		ax[1].scatter(longtemp, longtemp, 4, color='r')
		ax[0].set_xlabel('Temperature (C)')
		ax[0].set_ylabel('Interpolated Temperature (C)')		
		ax[0].set_title('Stride: m={0:.4f}, b={1:.4f}, R^2={2:.3f}'.format(shortLR[0], shortLR[1], shortLR[2]**2))
		ax[1].set_title('Sampled: m={0:.4f}, b={1:.4f}, R^2={2:.3f}'.format(normLR[0], normLR[1], normLR[2]**2))
		plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/{2}/{0}reg{1}.png'.format(npoints,i,suff))
		plt.close()

	i+=1
