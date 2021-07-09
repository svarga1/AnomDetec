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

matplotlib.use('agg')


#path='/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/gdas.20210601/gdas.t00z.uprair.tm00.bufr_d'

for filepath in Path('/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/gdas.20210601/').glob('*.bufr_d'):
	bufr=ncepbufr.open(filepath)
	while bufr.load_subset()==-1:
		bufr.advance()

	pres= bufr.read_subset('PRLC').squeeze()/100
	t=bufr.read_subset('TMDB').squeeze()-273.15
	if len(pres)>3000:
		longPres=pres
		longtemp=t
	while bufr.load_subset()==0:
		try:
			pres= bufr.read_subset('PRLC').squeeze()/100
			t=bufr.read_subset('TMDB').squeeze()-273.15
			if len(pres)>3000:
				longPres=pres
				longtemp=t
		
		except:
			pass
	
	while bufr.advance() ==0:
		while bufr.load_subset()==0:
			try:
				pres=bufr.read_subset('PRLC').squeeze()/100
				t=bufr.read_subset('TMDB').squeeze()-273.15
				if len(pres)>3000:
					longPres=pres
					longtemp=t

			except:
				pass

	bufr.close()


print(longtemp.shape)
print(longPres.shape)
print(longtemp)


#fig, ax=plt.subplots()
#plt.scatter(longtemp, longPres)
#ax.invert_yaxis()
#ax.set_yscale('log')
#plt.show()

strides=[10,50,100,150,200]


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
	
	
	normnew=normf(longPres.compressed())
	shortnew=shortf(longPres.compressed())
	
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
	plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/terp/{}subset.png'.format(npoints))
	plt.close()


	fig, ax= plt.subplots(1,3, sharex='row', sharey='row', figsize=(10,10))
	ax[0].scatter(longtemp, longPres, 4)
	ax[0].set_title('Original Data: {} points'.format(len(longPres)))
	ax[1].scatter(shortnew,longPres.compressed(),4 )
	ax[1].set_title('Stride Data: {} points'.format(len(shortnew)))
	ax[2].scatter(normnew, longPres.compressed(),4 )
	ax[2].set_title('Sampled Data: {} points'.format(len(normnew)))
	ax[0].invert_yaxis()
	ax[0].set_yscale('log')
	ax[0].set_ylabel('Pressure (hPa)')
	ax[0].set_xlabel('Temperature (C)')
	plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/terp/{}profile.png'.format(npoints))
	plt.close()
