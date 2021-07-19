import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import matplotlib
matplotlib.use('agg')


oldf=Dataset('big.nc', 'r')
newf=Dataset('terpinput.nc', 'r')

oldPres=oldf.variables['pres'][:,:,:]
oldTemp=oldf.variables['temp'][:,:,:]

newPres=newf.variables['pres'][:,:,:]
newTemp=newf.variables['temp'][:,:,:]




i=0
while i< newPres.shape[0]:
	fig, ax =plt.subplots(1,2, sharex='row', sharey='row', figsize=(10,10))
	ax[0].scatter(oldTemp[i],oldPres[i], 4)
	ax[1].scatter(newTemp[i], newPres[i], 4)
	ax[0].set_title('Original Data')
	ax[1].set_title('Interpolated to max resolution')
	ax[0].invert_yaxis()
	ax[0].set_yscale('log')
	ax[0].set_ylabel('Pressure (hPa)')
	ax[0].set_xlabel('Temperature (C)')
	plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/AD/{}interp.png'.format(i))
	plt.close()
	i+=1


oldf.close()
newf.close()
