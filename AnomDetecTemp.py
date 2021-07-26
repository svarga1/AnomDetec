#Test for keras to make sure anomaly detection works applied to time series
#Steps from https://keras.io/examples/timeseries/timeseries_anomaly_detection/

#Setup
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from pathlib import Path
import matplotlib
import netCDF4
from netCDF4 import Dataset




matplotlib.use('agg')
#Sufffix
suff='crop03split'

#Load custom data- make this a function later


training=np.array([])
firstIt=True
z=1
print('starting readin')
for filepath in Path('/work/noaa/da/cthomas/ens/2020090500').glob('*/*003.nc'):
	if firstIt:
		nc_fid=Dataset(filepath, 'r')
		pres=nc_fid.variables['pfull'][:]
		temp=nc_fid.variables['tmp'][0,:,107,605]
		training=np.reshape(temp, [1,127,1])
		firstIt=False
		nc_fid.close()
	else:
		nc_fid=Dataset(filepath, 'r')
		temp=nc_fid.variables['tmp'][0,:,107,605]
		temp=np.reshape(temp, [1,127,1])
		training=np.insert(training, -1, temp, axis=0)
		nc_fid.close()
		print(z)
		z+=1


print(training.shape)
	
print('Normalizing training data')
test=np.reshape(training[-1,:,:]-273.15, [1,127,1])

training=training[0:-1,:,:]-273.15


#Prepare training data
#Normalize and save the mean and std for normalizing test data
training_mean=training[:,:,0].mean()
training_std=training[:,:,0].std()
training=(training-training_mean)/(training_std)
x_train=training



#build a model



x_train=np.reshape(x_train, [z-1,127,1])
print(x_train.shape)

model=keras.Sequential(
  [
    layers.Input(shape=(127,1)),
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
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse') #0.001 and mse
model.summary()

print(x_train.shape)
#exit()


#Train the model
history=model.fit(
  x_train, x_train, epochs=100, batch_size=127, validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')],)

#Plot training and validation loss
fig=plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Reconstruction Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('loss{}'.format(suff))
plt.close()


#Anomaly detection

#Get training MAE loss
x_train_pred=model.predict(x_train)
train_mae_loss=np.mean(np.abs(x_train_pred-x_train),axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel('Train MAE Loss')
plt.ylabel('Number of samples')
plt.savefig('maeHist{}'.format(suff))

##Reconstruction loss threshould
threshold=np.max(train_mae_loss)



print(x_train[0].shape)
print(x_train_pred[0].shape)


#Compare reconstruction
#fig=plt.figure()
#plt.plot(x_train[0,:,0], pres) #Put x-train to plot normalized data, training to plot original data
#plt.plot( x_train_pred[0,:,0], pres,  color='r')
#ax=plt.gca()
#ax.set_ylim(ax.get_ylim()[::-1])
#plt.xlabel('Normalized Temperature ')
#plt.ylabel('Pressure (mb)')
#plt.savefig('reconstruction{}'.format(suff))



#Test data
#test=np.array([])
#firstIt=True
#for filepath in Path('toyData').glob('test/*.csv'):
#	if firstIt:
#		data=np.asarray(pd.read_csv(filepath, names=['x','y']).values)
#		data=data[data[:,0].argsort()]
#		test=np.reshape(data, [1,100,2]) #Not a good fix, won't work for  data where i don't know the size
#		firstIt=False
#		paths=[filepath]
#	else:
#		data=np.asarray(pd.read_csv(filepath, names=['x','y']).values)
#		data=data[data[:,0].argsort()]
#		test=np.insert(test, -1, data, axis=0)
#		paths=np.insert(paths,-1, filepath, axis=0)


print(test.shape)

#Process test data
test=(test-training_mean)/(training_std)
x_test=test
x_test=np.reshape(x_test, [1,127,1])


#Predictions
x_test_pred=model.predict(x_test)

#Plot predictions
#i=0
#while i<1:
#	fig=plt.figure()
#	plt.plot(x_test[i,:,0])
#	plt.plot(x_test_pred[i,:,0], color='r')
#	#plt.title(paths[i])
#	plt.savefig('test{}'.format(i))
#	plt.close()
#	i+=1








#Get test MAE loss
test_mae_loss=np.mean(np.abs(x_test_pred-x_test),axis=1)
test_mae_loss=test_mae_loss.reshape((-1))
plt.hist(test_mae_loss, bins=50)
plt.xlabel('test MAE loss')
plt.ylabel('Number of samples')
plt.savefig('testmae{}'.format(suff))
plt.close()
#Detect all samples that are anomalies
anomalies=np.abs(x_test_pred-x_test)>threshold  #Used 0.5 by threshold
print(np.count_nonzero(anomalies))
print(anomalies.shape)

#Plot anomalies


#Overlay anomalies

#subset=
#Model Reconstruction vs Test data
#plt.figure()
fig, ax=plt.subplots()
#ax[0].plot(x_test_pred[0,:,0],pres, color='r')
#ax[0].plot(x_test[0,:,0], pres, color= 'blue')
#axi=plt.gca()
#axi.set_ylim(ax.get_ylim()[::-1])
#ax[0].invert_yaxis()
#ax[0].set_xlabel('Normalized Temperature ')
#ax[0].set_ylabel('Pressure (mb)')
#plt.savefig('modelvtest')
#plt.close()


#Outlier Id plot
#fig=plt.figure()
ax.scatter(x_test[0,:,0]*training_std+training_mean,pres, color='b')
ax.scatter(x_test[anomalies]*training_std+training_mean,pres[np.reshape(anomalies,[127])], color='r')
ax.set_yscale('log')
ax.invert_yaxis()
#ax=plt.gca()
#ax.set_ylim(ax.get_ylim()[::-1])

ax.set_xlabel('Temperature (C) ')
ax.set_ylabel('Pressure (hPa)')
plt.savefig('overlay')
plt.close()

#Reconsutrction
fig, ax = plt.subplots()
ax.plot(x_test_pred[0,:,0]*training_std+training_mean, pres, color='r', label='Reconstruction')
ax.plot(x_test[0,:,0]*training_std+training_mean, pres, color= 'blue', label='Test Data')
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Pressure (hPa)')
ax.set_yscale('log')
ax.invert_yaxis()
plt.legend()
plt.savefig('reconstruction')
plt.close()

#Model res scatter
fig, ax =plt.subplots()
ax.scatter(x_test[0,:,0]*training_std+training_mean, pres, 4)
ax.set_yscale('log')
ax.invert_yaxis()
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Pressure (hPa)')
ax.set_title('Model Output')
plt.savefig('modelscatter')
plt.close()

#Model Res Scatter - limited pressure
fig, ax =plt.subplots()
ax.scatter(x_test[0,:,0][pres>=10]*training_std+training_mean, pres[pres>=10], 4)
ax.set_yscale('log')
ax.invert_yaxis()
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Pressure (hPa)')
ax.set_title('Model Resolution: {} points'.format(len(pres)))
plt.savefig('modelpresscatter')
plt.close() 
