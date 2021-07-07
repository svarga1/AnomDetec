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
for filepath in Path('/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/').glob('*/*.bufr_d'):
	bufr=ncepbufr.open(filepath)
	while bufr.load_subset()==-1:
		bufr.advance()

	pres= bufr.read_subset('PRLC').squeeze()
	t=bufr.read_subset('LTDS').squeeze()
	dpdt=diff(pres)/diff(t)
	numData=np.append(numData, np.mean(dpdt))
	x=np.append(x,len(dpdt))
	#if any(x>0 for x in dpdt):
	#	counter+=1
#		numData=np.append(numData, len(dpdt[dpdt>0]))
#		numData=np.append(numData, len(dpdt[dpdt<=0]))
#	else:
#		numData=np.append(numData, len(dpdt))
	while bufr.load_subset()==0:
		try:
			pres= bufr.read_subset('PRLC').squeeze()
			t=bufr.read_subset('LTDS').squeeze()
			dpdt=diff(pres)/diff(t)
			numData=np.append(numData, np.mean(dpdt))
			x=np.append(x, len(dpdt))

#			if any(x>0 for x in dpdt):
#				counter+=1
#				numData=np.append(numData, len(dpdt[dpdt>0]))
#				numData=np.append(numData, len(dpdt[dpdt<=0]))
#			else:
#				numData=np.append(numData, len(dpdt))
		except:
			pass
	
	while bufr.advance() ==0:
		while bufr.load_subset()==0:
			try:
				pres=bufr.read_subset('PRLC').squeeze()
				t=bufr.read_subset('LTDS').squeeze()
				dpdt=diff(pres)/diff(t)
				numData=np.append(numData, np.mean(dpdt))
				x=np.append(x, len(dpdt))
#				if any(x>0 for x in dpdt):
#					counter+=1
#					numData=np.append(numData, len(dpdt[dpdt>0]))
#					numData=np.append(numData, len(dpdt[dpdt<=0]))
#				else:	
#					numData=np.append(numData, len(dpdt))
			except:
				pass

	bufr.close()

fig=plt.figure()
plt.scatter(x, numData)
plt.savefig('dpdtscatter.png')
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


