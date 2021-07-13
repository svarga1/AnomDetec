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
for filepath in Path('/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/').glob('*/*.bufr_d'):
	bufr=ncepbufr.open(filepath)
	while bufr.load_subset()==-1:
		bufr.advance()

	pres= bufr.read_subset('PRLC').squeeze()
	t=bufr.read_subset('LTDS').squeeze()
	dpdt=diff(pres)/diff(t)
	

	if any(x>0 for x in dpdt):
#		numData=np.append(numData, len(dpdt[dpdt>0]))
#		duration=np.append(duration, np.amax(t[dpdt>0])-np.amin(t[dpdt>0]))
#		numData=np.append(numData, len(dpdt[dpdt<=0]))
#		duration=np.append(duration, np.amax(t[dpdt<=0]))
		duration=np.append(duration, t.max())
	else:
		numData=np.append(numData, len(dpdt))
		duration=np.append(duration, t.max())
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
				duration=np.append(duration, t.max())
			else:
				numData=np.append(numData, len(dpdt))
				duration=np.append(duration, t.max())
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
					duration=np.append(duration, t.max())
				else:	
					numData=np.append(numData, len(dpdt))
					duration=np.append(duration, t.max())
			except:
				pass

	bufr.close()

duration=duration/60 #convert from seconds to minutes

print(numData.shape)
print(duration.shape)
print(np.amax(duration[duration>0]))
print(np.amin(duration[duration>0]))
print(np.mean(duration[duration>0]))
print(np.median(duration[duration>0]))
print(np.std(duration[duration>0]))



#fig=plt.figure()
#plt.scatter(numData,duration)
#plt.title('Duration vs Number of Observations (Split Ascent/Descent)')
#plt.xlabel('Number of Observations')
#plt.ylabel('Time (Minutes)')
#plt.savefig('durationscatter.png')
#plt.close()

fig=plt.figure()
plt.hist(duration[duration>0],  bins=np.arange(0,315,15))
plt.title('Duration of flight')
plt.xlabel('Duration (Minutes)')
plt.savefig('0compresseddurationhist.png')
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


