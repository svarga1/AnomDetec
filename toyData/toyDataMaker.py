#Program to create fake sinusoidal data for testing


import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
#Create 40  sets of 100  datapoints between 0 and 4pi
i=0
while i<40:
	s=0
	x=[]
	while s<100:
		x=np.append(x,random.uniform(0,4*np.pi))
		s+=1
	y=random.uniform(1,4)*np.sin(x)
	np.savetxt('toyData{}.csv'.format(i),np.transpose([x,y]), delimiter=',')
	if i==0 or i%5==0:
		fig=plt.figure()
		plt.scatter(x,y)
		plt.savefig('toyData{}'.format(i))
	i+=1
	
	
