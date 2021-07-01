#Module for functions relating to bufrfiles

#Required Modules
import ncepbufr
import numpy as np


#Function that opens a bfur file, advances to the first message, and loads the first subset
def ready(FILE):
	bufr=ncepbufr.open(FILE) #Opens the Bufr file
	while bufr.load_subset() == -1: #Load_subset returns -1 when it is unable to load. A succesful load will break the loop
		bufr.advance() #When the subset fails to load, the next message is checked
	return bufr #The bufr object is returned with the first subset loaded


#Function that describes a bufr file: Number of messages, Number of subsets, and number of data points:
def descBufrFile(FILE):
	numMes=1
	bufr=ready(FILE)
	
	numSub=np.float(bufr.subsets)
	numData=len(bufr.read_subset('PRLC').squeeze()) #Uses pressure as a default, not a great thing. Only works for these files

	while bufr.advance()==0:
		numSub+=np.float(bufr.subsets) #Tracks the number of subsets
		while bufr.load_subset()==0 :
			try:
				numData+=len(bufr.read_subset('PRLC').squeeze()) #Tracks the number of data points
			except:
				pass
		numMes+=1 #Tracks the total number of messages
	bufr.close() #Closes the file
	return numMes, numSub, numData


#Function that returns a 1d array of the number of data points in each subset and message in a file

def lenBufrFile(FILEPATH,KEY):
	bufr=ready(FILEPATH)
	
	numData=np.array([len(bufr.read_subset(KEY).squeeze())])


	while bufr.load_subset()==0:
		try:
			numData=np.append(numData, len(bufr.read_subset(KEY).squeeze()))
		except:
			pass

	while bufr.advance() ==0:
		while bufr.load_subset()==0:
			try:
				numData=np.append(numData, len(bufr.read_subset(KEY).squeeze()))
			except:
				pass
	bufr.close()
	return numData

def numBufrFile(FILEPATH,KEY):
        bufr=ready(FILEPATH)

        numData=np.array([bufr.read_subset(KEY).squeeze()])


        while bufr.load_subset()==0:
                try:
                        numData=np.append(numData, bufr.read_subset(KEY).squeeze())
                except:
                        pass

        while bufr.advance() ==0:
                while bufr.load_subset()==0:
                        try:
                                numData=np.append(numData, bufr.read_subset(KEY).squeeze())
                        except:
                                pass
        bufr.close()
        return numData

