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



numData=np.array([])
counter=0
x=np.array([])
duration=np.array([])
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
	plt.scatter(temp, pres)
	plt.title('{} points'.format(len(temp)))
	plt.xlabel('Dry Bulb Temperature (Celsius)')
	plt.ylabel('Pressure (hPa)')
	ax.invert_yaxis()
	ax.set_yscale('log')


	if len(temp)>500:
		plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/HD/'+out+'.png')
		plt.close()
	else:
		plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/SD/'+out+'.png')
		plt.close()


	exit()

	if any(x>0 for x in dpdt):
		pass		
	else:
		numData=np.append(numData, len(dpdt))
		duration=np.append(duration, np.amax(t))
	while bufr.load_subset()==0:
		try:
			pres= bufr.read_subset('PRLC').squeeze()
			t=bufr.read_subset('LTDS').squeeze()
			dpdt=diff(pres)/diff(t)
			
			if any(x>0 for x in dpdt):
				counter+=1
#				numData=np.append(numData, len(dpdt[dpdt>0]))
#				duration=np.append(duration, np.amax(t[dpdt>0])-np.amin(t[dpdt>0]))
#				numData=np.append(numData, len(dpdt[dpdt<=0]))
#				duration=np.append(duration, np.amax(t[dpdt<=0]))
				numData=np.append(numData, len(dpdt))
				duration=np.append(duration, np.amax(t))
			else:
				numData=np.append(numData, len(dpdt))
				duration=np.append(duration, np.amax(t))
		except:
			pass
	
	while bufr.advance() ==0:
		while bufr.load_subset()==0:
			try:
				pres=bufr.read_subset('PRLC').squeeze()
				t=bufr.read_subset('LTDS').squeeze()
				dpdt=diff(pres)/diff(t)
				
				if any(x>0 for x in dpdt):
					counter+=1
#					numData=np.append(numData, len(dpdt[dpdt>0]))
#					duration=np.append(duration, np.amax(t[dpdt>0])-np.amin(t[dpdt>0]))
#					numData=np.append(numData, len(dpdt[dpdt<=0]))
#					duration=np.append(duration, np.amax(t[dpdt<=0]))
					numData=np.append(numData, len(dpdt))
					duration=np.append(duration,np.amax( t))
				else:	
					numData=np.append(numData, len(dpdt))
					duration=np.append(duration, np.amax( t))
			except:
				pass

	bufr.close()

duration=duration/60 #convert from seconds to minutes

print(numData.shape)
print(duration.shape)
print(np.amax(duration))
print(np.amin(duration))
print(np.mean(duration))
print(np.median(duration))
print(np.std(duration))



#fig=plt.figure()
#plt.scatter(numData,duration)
#plt.title('Duration vs Number of Observations (Split Ascent/Descent)')
#plt.xlabel('Number of Observations')
#plt.ylabel('Time (Minutes)')
#plt.savefig('durationscatter.png')
#plt.close()

fig=plt.figure()
plt.hist(duration, bins=np.arange(0,315,15))
plt.title('Duration of flight')
plt.xlabel('Duration (Minutes)')
plt.savefig('durationhist.png')
plt.close()


exit()

for filepath in Path('/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/').glob('*/*.bufr_d'):
        test=np.append(test,VB.lenBufrFile(filepath, 'PRLC')) #Also try tmdb, try compressing masked to get rid of missing?

print(counter)
print(numData.shape)

fig,ax=plt.subplots(1,2,sharex='row',sharey='row')
ax[0].hist(test)
ax[1].hist(numData)

ax[0].set_title('No separation')
ax[1].set_title('Ascent and descent separated (no dyn. all.)')
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


