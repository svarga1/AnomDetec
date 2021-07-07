#Test file for reading bufr file and writing to nc


import VargaBufr as VB
from  netCDF4 import Dataset
import numpy as np
import ncepbufr






#Create NC skeleton
ncfile=Dataset('test.nc', mode='w', format='NETCDF4')

#Create dimensions

data_dim=ncfile.createDimension('data', None)
prof_dim=ncfile.createDimension('profile', None)
temp_dim=ncfile.createDimension('temperature', 1)
time_dim=ncfile.createDimension('time', 1)
#Define variables

temp=ncfile.createVariable('Temp', np.float32, ('profile','data','temperature'))
temp.units='K'
temp.standard_name='dry_bulb_temperature'

#Write Variables

path='/work/noaa/stmp/Cory.R.Martin/svarga/hd_sondes/gdas.20210601/gdas.t00z.uprair.tm00.bufr_d' #Probably want to automate this at some point, but for now I can just use 1:1 file


bufr=ncepbufr.open(path) #open file
ind=0
while bufr.load_subset()==-1:
	bufr.advance()
bufTem=bufr.read_subset('TMDB').squeeze() #read data, reshape, and add to file
bufTem=np.reshape(bufTem, [len(bufTem),1])
temp[ind,:,:]=bufTem
ind+=1


while bufr.load_subset()==0:
	try:
		bufTem=bufr.read_subset('TMDB').squeeze()
		bufTem=np.reshape(bufTem, [len(bufTem),1])
		temp[ind,:,:]=bufTem
		ind+=1
	except:
		pass
while bufr.advance()==0: #Step through every message
	while bufr.load_subset()==0: #Step through every subset
		try:
			bufTem=bufr.read_subset('TMDB').squeeze()
			bufTem=np.reshape(bufTem, [len(bufTem),1])
			temp[ind,:,:]=bufTem
			ind+=1
		except:
			pass
print(ncfile.variables['Temp'][:].shape)
exit()

bufr.close() #Close the file
ncfile.close() #Close the netCDF file

