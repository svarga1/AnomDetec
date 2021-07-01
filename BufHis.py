#Script that creates histogram of bufr file

import VargaBufr as VB
import ncepbufr
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import matplotlib

matplotlib.use('agg')


#path='/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/gdas.20210601/gdas.t00z.uprair.tm00.bufr_d'
test=np.array([])


for filepath in Path('/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/').glob('*/*.bufr_d'):
	test=np.append(test,VB.lenBufrFile(filepath, 'PRLC')) #Also try tmdb, try compressing masked to get rid of missing?

print(test.shape)

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


