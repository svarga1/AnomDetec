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
suff='init'
testf=Dataset('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/big.nc', 'r')
testpres=testf.variables['pres'][:,:,:]
testtemp=testf.variables['temp'][:,:,:]


#Read in Data
for filepath in Path('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/terpinput').glob('*.nc'): #Use the interpolated as training, high res as test
	it=str(filepath)[-4]
	fid=Dataset(filepath, 'r')
	pres=fid.variables['pres'][:,:,:]
	temp=fid.variables['temp'][:,:,:]

	#ML Preprocessing
	training_mean=temp.mean()
	training_std=temp.std()
	x_train=(temp-training_mean)/training_std
	
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
layers.Cropping1D(cropping=(0,1))

  ]
)
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')


	history=model.fit(x_train, x_train, epochs=100, batch_size=127, validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')]) #Train the model

	
	#Plot training and validation loss
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
	plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/AD/maeHist{}{}'.format(suff, it))
	plt.close()
	#Loss threshold
	threshold=np.max(train_mae_loss)


	#test data preprocessing
	x_test=testtemp[int(it),:,:]
	x_test_pres=testpres[int(it),:,:]
	mas=np.ma.mask_or(np.ma.getmask(x_test), np.ma.getmask(x_test_pres))
	x_test.mask=mas
	x_test_pres.mask=mas
	x_test=x_test.compressed()
	x_test_pres=x_test_pres.compressed()
	x_test_norm=(x_test-training_mean)/training_std

	#Test predictions
	x_test_pred=model.predict(np.reshape(x_test_norm, [1, len(x_test),1]))

	#Test MAE loss
	test_mae_loss=np.mean(np.abs(x_test_pred-np.reshape(x_test_norm, [1, len(x_test_norm),1])), axis=1)
	fig=plt.figure()
	plt.hist(test_mae_loss)
	plt.xlabel('Test MAE Loss')
	plt.ylabel('Number of Samples')
	plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/AD/testMAE{}{}'.format(suff, it))
	plt.close()

	#Detect anomalies
	anomalies=np.abs(x_test_pred-x_test_norm)>threshold
	
	#Outlier ID plot
	fig, ax = plt.subplots(1,2, sharex='row', sharey='row', figsize=(10,10))
	ax[0].scatter(x_test, x_test_pres, 4, color='blue')
	ax[1].scatter(x_test[anomalies[0,:,0].flatten()], x_test_pres[anomalies[0,:,0].flatten()], 4, color='red')
	ax[0].set_xlabel('Temperature (C)')
	ax[0].set_ylabel('Pressure (hPa)')
	ax[0].invert_yaxis()
	ax[0].set_yscale('log')
	plt.savefig('/work/noaa/da/svarga/anomDetec/AnomDetecBufr/pics/AD/outlierID{}{}'.format(suff,it))
	plt.close()



testf.close()
fid.close()