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
test=np.array([])
lonPres=np.array([])
shortPres=np.array([])
numData=np.array([])
counter=0

for filepath in Path('/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/').glob('*/*.bufr_d'):
	bufr=ncepbufr.open(filepath)
	while bufr.load_subset()==-1:
		bufr.advance()
	lonPres=np.array([])
	shortPres=np.array([])
	pres= bufr.read_subset('PRLC').squeeze()
	if len(pres)<=500:
		shortPres=np.append(shortPres, pres)
	else:
		pass	#lonPres=np.append(lonPres, pres)
	while bufr.load_subset()==0:
		try:
			pres= bufr.read_subset('PRLC').squeeze()
				
			if len(pres)<=500:
				shortPres=np.append(shortPres,pres)
			else:
				pass#				lonPres=np.append(lonPres, pres)
		except:
			pass
	
	while bufr.advance() ==0:
		while bufr.load_subset()==0:
			try:
				pres=bufr.read_subset('PRLC').squeeze()
				
				if len(pres)<=500:
					shortPres=np.append(shortPres, pres)
				else:	
					pass #lonPres=np.append(lonPres, pres)
			except:
				pass

	bufr.close()
	shortPres=shortPres/100
	print(shortPres)
	#lonPres=lonPres/100 #Convers to hPa
	#Plot for each file
	fig, ax = plt.subplots()
	ax.hist(shortPres)
	#ax[0].hist(lonPres)
	#ax.set_xscale('log')
	ax.set_title('HD: {} points'.format(len(shortPres)))
	ax.set_xlabel('Pressure (hPa)')
	#ax[1].set_title('Hd: {} points'.format(len(lonPres)))
	plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/{}short.png'.format(str(filepath).rstrip('.bufr').split('/')[7:]))
	plt.close()




exit()
for filepath in Path('/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/').glob('*/*.bufr_d'):
        test=np.append(test,VB.lenBufrFile(filepath, 'PRLC')) #Also try tmdb, try compressing masked to get rid of missing?

print(counter)
print(numData.shape)

fig=plt.subplots()
ax[0].hist(test)
#ax[1].hist(numData)
ax[0].set_title('No separation')
#ax[1].set_title('Ascent and descent separated (no dyn. all.)')
ax[0].set_xlabel('Points per profile')
ax[0].set_ylabel('frequency')
plt.savefig('histascentdescent')
plt.close()




exit()







for filepath in Path('/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/').glob('*/*.bufr_d'):
	test=np.append(test,VB.lenBufrFile(filepath, 'PRLC')) #Also try tmdb, try compressing masked to get rid of missing?

print(test.shape)


exit()



fig=plt.figure()
plt.hist(test)
plt.xlabel('Number of points per profile')
plt.savefig('pointDistribution')
plt.close()

fig=plt.figure()
plt.hist(test[test>500])
plt.xlabel('Number of points per profile')
plt.savefig('pointDistribution500')





test=np.array([])
for filepath in Path('/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/').glob('*/*.bufr_d'):
        test=np.append(test,VB.lenBufrFile(filepath, 'TMDB')) #Also try tmdb, try compressing masked to get rid of missing?

print(test.shape)

fig=plt.figure()
plt.hist(test)
plt.xlabel('Number of points per profile')
plt.savefig('TMDBpointDistribution')
plt.close()

fig=plt.figure()
plt.hist(test[test>500])
plt.xlabel('Number of points per profile')
plt.savefig('TMDBpointDistribution500')
plt.close()


