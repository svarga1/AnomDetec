#Script that detects anomalies in upscaled profiles

#Modules
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import matplotlib
from netCDF4 import Dataset
from scipy import interpolate
from pathlib import Path


#Orion flags
matplotlib.use('agg')

#Post
suff='chunk'
testf=Dataset('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/big.nc', 'r')
#testf=Dataset('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/terpinput/ens/test.nc')


#Fix this
testpres=testf.variables['pres'][:,:,:]
testtemp=testf.variables['temp'][:,:,:]


#Read in Data
for filepath in Path('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/terpinput').glob('*2.nc'): #Use the interpolated as training, high res as test
	it=str(filepath)[-4]
	fid=Dataset(filepath, 'r')
	pres=fid.variables['pres'][:,:,:]
	temp=fid.variables['temp'][:,:,:]

	#Chunk- Drop the first and last 100-- fix this
	temp=temp[:,1500:int(temp.shape[1]-1500),:]
	pres=pres[:,1500:int(pres.shape[1]-1500),:]

	#ML Preprocessing
	training_mean=temp.mean()
	training_std=temp.std()
	x_train=temp
	z=0
	while z<pres.shape[0]:
		x_train[z,:,:]=(x_train[z,:,:]-training_mean)/training_std
		z+=1
#Build Model


	model=keras.Sequential(
  [
    layers.Input(shape=pres.shape[1:]),
    layers.Conv1D(
      filters=64, kernel_size=7, padding='same', strides=2, activation='relu'),
    layers.Dropout(rate=0.2),
    layers.Conv1D(filters=32, kernel_size=7, padding ='same', strides=2, activation='relu'), 
    layers.Conv1DTranspose( 
      filters=32, kernel_size=7, padding='same', strides=2, activation='relu'),
    layers.Dropout(rate=0.2),
    layers.Conv1DTranspose( 
     filters=64, kernel_size=7, padding='same', strides=2, activation ='relu'), 
    layers.Conv1DTranspose(filters=1,kernel_size=7,padding='same'), 
layers.Cropping1D(cropping=(1,0))

  ]
)
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')


	history=model.fit(x_train, x_train, epochs=100, batch_size=127, validation_split=0.1) #Train the model
	
	#Save the model
	model.save('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/models/model{}'.format(it))
#	
#	#Plot training and validation loss
	fig, ax =plt.subplots()
	ax.plot(history.history['loss'], label='Training Loss')
	ax.plot(history.history['val_loss'], label='Validation Loss')
	plt.legend()
	plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/AD/loss{}{}'.format(suff, it))
	plt.close()

	#Get training MAE loss
	fig=plt.figure()
	train_mae_loss=np.mean(np.abs(model.predict(x_train)-x_train), axis=1)
	plt.hist(train_mae_loss)
	plt.xlabel('Train MAE Loss')
	plt.ylabel('Number of Samples')
	plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/AD/ma3Hist{}{}'.format(suff, it))
	plt.close()
	#Loss threshold
	threshold=np.max(train_mae_loss)
	print(threshold)
	print(train_mae_loss.shape)

	#test data preprocessing
	x_test=testtemp[int(it),:,:]
	x_test_pres=testpres[int(it),:,:]
	mas=np.ma.mask_or(np.ma.getmask(x_test), np.ma.getmask(x_test_pres))
	x_test.mask=mas
	x_test_pres.mask=mas
	x_test=x_test.compressed()
	x_test_pres=x_test_pres.compressed()

	x_test_pres=np.reshape(x_test_pres, [1, len(x_test_pres), 1])
	x_test=np.reshape(x_test, [1,len(x_test), 1])
	x_test=x_test[:, 1500:int(x_test.shape[1])-1500,:]
	x_test_pres=x_test_pres[:,1500:int(x_test_pres.shape[1])-1500,:]
	
	x_test_norm=(x_test-training_mean)/training_std

	#Test predictions
	x_test_pred=model.predict(x_test_norm)


	#Detect anomalies
	anomalies=np.abs(x_test_pred-x_test_norm)>threshold
	print(x_test_pred.shape)
	print(x_test_norm.shape)
	print(anomalies)
	print(anomalies[0,:,0].shape)


	#Outlier ID plot
	fig, ax = plt.subplots(1,2, sharex='row', sharey='row', figsize=(10,10))
	ax[0].scatter(x_test, x_test_pres, 4, color='blue')
	ax[1].scatter(x_test[0,anomalies[0,:,0],0], x_test_pres[0,anomalies[0,:,0],0], 4, color='red')
	ax[0].set_xlabel('Temperature (C)')
	ax[0].set_ylabel('Pressure (hPa)')
	ax[0].invert_yaxis()
	ax[0].set_yscale('log')
	plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/AD/outlier1D{}{}'.format(suff,it))
	plt.close()

	#Model reconstruction
	fig, ax = plt.subplots(1,2, sharex='row', sharey='row', figsize=(10,10))
	ax[0].scatter(x_test, x_test_pres, 4, color='blue')
	ax[1].scatter(x_test_pred[0,:,0]*training_std+training_mean,x_test_pres, 4, color='r')
	ax[0].set_title('Original Data')
	ax[1].set_title('Model Reconstruction')
	ax[0].set_ylabel('Pressure (hPa)')
	ax[1].set_xlabel('Temperature (C)')
	ax[0].invert_yaxis()
	ax[0].set_yscale('log')
	plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/AD/recon{}{}.png'.format(suff, it))
	plt.close()

	#Plot OMI profile
	predUN=x_test_pred[0,:,0]*training_std+training_mean
	OMR=x_test-predUN
	fig, ax = plt.subplots()	
	ax.scatter(OMR[0,:,0], x_test_pres[0,:,0], 4, color= 'blue')
	ax.scatter(OMR[0,anomalies[0,:,0],0], x_test_pres[0,anomalies[0,:,0],0], 4, color='red')
	ax.set_title('OMR Profile')
	ax.set_ylabel('Pressure (hPa)')
	ax.set_xlabel('OMR (C)')
	ax.set_yscale('log')
	ax.invert_yaxis()
	plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/AD/OMIprofile{}{}.png'.format(suff, it))
	plt.close()

	#OMI Hist
	fig, ax = plt.subplots(1,2, sharex='row', sharey='row')
	ax[0].hist(OMR[0,:,0])
	ax[1].hist(OMR[0,anomalies[0,:,0],0])
	ax[0].set_title('Full OMR')
	ax[1].set_title('Anomaly OMR')
	ax[0].set_xlabel('OMR (C)')
	plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/AD/OMIhist{}{}.png'.format(suff, it))
	plt.close()

	

	#Plot 1D temp
	fig, ax =plt.subplots()
	ax.plot(x_test_pred[0,:,0]*training_std+training_mean)
	ax.set_title('Temperature Series')
	ax.set_xlabel('Index')
	ax.set_ylabel('Temperature (C)')
	plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/AD/tempseries{}{}.png'.format(suff, it))
	plt.close()


testf.close()
fid.close()
