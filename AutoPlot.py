#Script that creates histogram of bufr file

import VargaBufr as VB
import ncepbufr
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import matplotlib
from numpy import diff



matplotlib.use('agg')


#path='/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/gdas.20210601/gdas.t00z.uprair.tm00.bufr_d'




for filepath in Path('/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/').glob('gdas.20210601/gdas.t00z.uprair.tm00.bufr_d'):
	bufr=ncepbufr.open(filepath)
	while bufr.load_subset()==-1:
		bufr.advance()

	pres= bufr.read_subset('PRLC').squeeze()/100 #Converts to hPa
	t=bufr.read_subset('LTDS').squeeze() 
	dpdt=diff(pres)/diff(t)
	temp=bufr.read_subset('TMDB').squeeze()-273.15 #converts to celsius
	stationId=str(bufr.read_subset('WMOS').squeeze())
	receiptTime=str(bufr.read_subset('YEAR').squeeze() ) + str(bufr.read_subset('MNTH').squeeze()) + str(bufr.read_subset('DAYS').squeeze()) + str(bufr.read_subset('HOUR').squeeze())
	out=stationId+'.'+receiptTime

	fig,ax =plt.subplots()
	plt.title('{0} points, {1} available'.format(len(temp),len(temp.compressed())))
	plt.xlabel('Dry Bulb Temperature (Celsius)')
	plt.ylabel('Pressure (hPa)')
	ax.invert_yaxis()
	ax.set_yscale('log')


	if any(x>0 for x in dpdt):
#		x=np.count_nonzero(dpdt>0)
#		plt.scatter(temp[:len(temp)-x],pres[:len(temp)-x],3, color='blue')
#		plt.scatter(temp[len(temp)-x:], pres[len(temp)-x:], 3,color='red')				
		plt.scatter(temp[np.insert(dpdt, 0, -1)<0],pres[np.insert(dpdt,0,-1)<0], 3, color='blue')
		plt.scatter(temp[np.insert(dpdt, 0, -1)>0],pres[np.insert(dpdt,0,-1)>0], 3, color='red')

	else:
		plt.scatter(temp, pres, 3,color='blue')

	if len(temp)>500:
		plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/HD/'+out+'.png')
		plt.close()
	else:
		plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/SD/'+out+'.png')
		plt.close()


	while bufr.load_subset()==0:
		try:
			pres= bufr.read_subset('PRLC').squeeze()/100 #Converts to hPa
			t=bufr.read_subset('LTDS').squeeze()
			dpdt=diff(pres)/diff(t)
			temp=bufr.read_subset('TMDB').squeeze()-273.15 #converts to celsius
			stationId=str(bufr.read_subset('WMOS').squeeze())
			receiptTime=str(bufr.read_subset('YEAR').squeeze() ) + str(bufr.read_subset('MNTH').squeeze()) + str(bufr.read_subset('DAYS').squeeze()) + str(bufr.read_subset('HOUR').squeeze())
			out=stationId+'.'+receiptTime
			fig,ax =plt.subplots()
			plt.title('{0} points, {1} available'.format(len(temp),len(temp.compressed())))
			plt.xlabel('Dry Bulb Temperature (Celsius)')
			plt.ylabel('Pressure (hPa)')
			ax.invert_yaxis()
			ax.set_yscale('log')
			if any(x>0 for x in dpdt):
				#x=np.count_nonzero(dpdt>0)
				#plt.scatter(temp[:len(temp)-x],pres[:len(temp)-x], 3,color='blue')
				#plt.scatter(temp[len(temp)-x:], pres[len(temp)-x:],3, color='red')
				plt.scatter(temp[np.insert(dpdt, 0, -1)<0],pres[np.insert(dpdt,0,-1)<0], 3, color='blue')
				plt.scatter(temp[np.insert(dpdt, 0, -1)>0],pres[np.insert(dpdt,0,-1)>0], 3, color='red')

			else:
				plt.scatter(temp, pres, 3, color='blue')
			if len(temp)>500:
				plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/HD/'+out+'.png')
				plt.close()
			else:
				plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/SD/'+out+'.png')
				plt.close()

		except:
			pass
	
	while bufr.advance() ==0:
		while bufr.load_subset()==0:
			try:
				pres= bufr.read_subset('PRLC').squeeze()/100 #Converts to hPa
				t=bufr.read_subset('LTDS').squeeze()
				dpdt=diff(pres)/diff(t)
				temp=bufr.read_subset('TMDB').squeeze()-273.15 #converts to celsius
				stationId=str(bufr.read_subset('WMOS').squeeze())
				receiptTime=str(bufr.read_subset('YEAR').squeeze() ) + str(bufr.read_subset('MNTH').squeeze()) + str(bufr.read_subset('DAYS').squeeze()) + str(bufr.read_subset('HOUR').squeeze())
				out=stationId+'.'+receiptTime
				fig,ax =plt.subplots()
				plt.title('{0} points, {1} available'.format(len(temp),len(temp.compressed())))
				plt.xlabel('Dry Bulb Temperature (Celsius)')
				plt.ylabel('Pressure (hPa)')
				ax.invert_yaxis()
				ax.set_yscale('log')
				if any(x>0 for x in dpdt):
#					x=np.count_nonzero(dpdt>0)
#					plt.scatter(temp[:len(temp)-x],pres[:len(temp)-x],3, color='blue')
#					plt.scatter(temp[len(temp)-x:], pres[len(temp)-x:],3, color='red')
					plt.scatter(temp[np.insert(dpdt, 0, -1)<0],pres[np.insert(dpdt,0,-1)<0], 3, color='blue')
					plt.scatter(temp[np.insert(dpdt, 0, -1)>0],pres[np.insert(dpdt,0,-1)>0], 3, color='red')

				else:
					plt.scatter(temp, pres,3, color='blue')
				if len(temp)>500:
					plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/HD/'+out+'.png')
					plt.close()
				else:
					plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/SD/'+out+'.png')
					plt.close()


			except:
				pass

	bufr.close()

