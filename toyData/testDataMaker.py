#Program to create fake sinusoidal data for testing


import numpy as np
import random
import matplotlib.pyplot as plt
#Create 40  sets of 100  datapoints between 0 and 4pi
i=0
while i<6:
	s=0
	x=[]
	while s<100:
		x=np.append(x,random.uniform(0,4*np.pi))
		s+=1
	if i==0: #Sign flip
		y=random.uniform(-1,-4)*np.sin(x)
	if i==1: #Piecewise Linear
		x=np.linspace(0,4*np.pi,100)
		y=random.uniform(1,2)*np.sin(x)
		slope=(y[60]-y[40])/(x[60]-x[40])
		b=y[40]-slope*x[40]
		#z=40	
		#while z<61:
		#	y[z]=slope*x[z]+b
		#	z+=1
		y[40:61]=slope*x[40:61]+b
	if i ==2:	#Frequency change
		x=np.linspace(0,4*np.pi,100)
		y=np.sin(x)
		y[40:61]=np.sin(3*x[40:61])
	if i==3: #amplitude reduction
		y=random.uniform(0.3,0.7)*np.sin(x)
	if i==4: #absolute value
		y=abs(np.sin(x))
	if i==5:
		y=random.uniform(0.3,0.7)*np.cos(x)
	fig=plt.figure()
	plt.scatter(x,y)
	plt.savefig('test/testData{}'.format(i))
	np.savetxt('test/testData{}.csv'.format(i),np.transpose([x,y]), delimiter=',')
	i+=1
	

