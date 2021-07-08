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

shortPresMax=np.array([])
shortPresMin=np.array([])
shortPresMed=np.array([])

lonPresMax=np.array([])
lonPresMin=np.array([])
lonPresMed=np.array([])



for filepath in Path('/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/gdas.20210601/').glob('gdas.t00z.uprair.tm00.bufr_d'):
	bufr=ncepbufr.open(filepath)
	while bufr.load_subset()==-1:
		bufr.advance()


	pres= bufr.read_subset('PRLC').squeeze()/100
	if len(pres)<=500:
		shortPresMax=np.append(shortPresMax,np.amax(pres))
		shortPresMin=np.append(shortPresMin, np.amin(pres))
		shortPresMed=np.append(shortPresMed, np.median(pres))
	else:
		lonPresMax=np.append(lonPresMax,np.amax( pres))
		lonPresMin=np.append(lonPresMin, np.amin(pres))
		lonPresMed=np.append(lonPresMed, np.median(pres))

	while bufr.load_subset()==0:
		try:
			pres= bufr.read_subset('PRLC').squeeze()/100
				
			if len(pres)<=500:
                		shortPresMax=np.append(shortPresMax,np.amax(pres))
		                shortPresMin=np.append(shortPresMin, np.amin(pres))
		                shortPresMed=np.append(shortPresMed, np.median(pres))
			else:
		                lonPresMax=np.append(lonPresMax,np.amax( pres))
		                lonPresMin=np.append(lonPresMin, np.amin(pres))
		                lonPresMed=np.append(lonPresMed, np.median(pres))

		except:
			pass
	
	while bufr.advance() ==0:
		while bufr.load_subset()==0:
			try:
				pres=bufr.read_subset('PRLC').squeeze()/100
				
				if len(pres)<=500:
			                shortPresMax=np.append(shortPresMax,np.amax(pres))
			                shortPresMin=np.append(shortPresMin, np.amin(pres))
			                shortPresMed=np.append(shortPresMed, np.median(pres))
				else:
			                lonPresMax=np.append(lonPresMax,np.amax( pres))
			                lonPresMin=np.append(lonPresMin, np.amin(pres))
			                lonPresMed=np.append(lonPresMed, np.median(pres))

			except:
				pass

	bufr.close()


#Plot for each file
fig, ax = plt.subplots(1,2,  sharey='row')
shortX=np.arange(1, len(shortPresMax)+1)
longX=np.arange(1,len(lonPresMax)+1)
ax[0].scatter(shortX, shortPresMax)
ax[0].scatter(shortX, shortPresMin)
ax[0].scatter(shortX, shortPresMed)
ax[1].scatter(longX, lonPresMax)
ax[1].scatter(longX, lonPresMin)
ax[1].scatter(longX, lonPresMed)
#ax[0].set_yscale('log')
ax[0].invert_yaxis()
ax[0].set_title('Non-HD')
ax[1].set_title('HD')
ax[0].set_ylabel('Pressure (hPa)')
plt.savefig('3num3.png')
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


