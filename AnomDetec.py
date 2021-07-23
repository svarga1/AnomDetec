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
import time




matplotlib.use('agg')
#Sufffix
suff='full'
start_time= time.time()
#Load custom data- make this a function later


training=np.array([])
firstIt=True
z=1
print('starting readin')
for filepath in Path('/work/noaa/da/cthomas/ens/2020090500').glob('*/*.nc'):
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
steptime=time.time()
print('The Amount of time since starting is: {}'.format(time.time()-start_time))

print('Normalizing training data')


#Prepare training data
#Normalize and save the mean and std for normalizing test data
training_mean=training[:,:,0].mean()
training_std=training[:,:,0].std()
exit()
training=(training-training_mean)/(training_std)
x_train=training



print('The Amount of time since starting is: {}'.format(time.time()-start_time))
print('The amount of time on this step is: {}'.format(time.time()-steptime))
steptime=time.time()





#build a model

print('Building model')

x_train=np.reshape(x_train, [z,127,1])
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

#Save the model
history.save('fullmodel')
print(time.time()-start_time)



#Plot training and validation loss
fig=plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Reconstruction Loss')
plt.legend()
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
fig=plt.figure()
plt.plot(x_train[0,:,0], pres) #Put x-train to plot normalized data, training to plot original data
plt.plot( x_train_pred[0,:,0], pres,  color='r')
ax=plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
plt.xlabel('Normalized Temperature ')
plt.ylabel('Pressure (mb)')
plt.savefig('reconstruction{}'.format(suff))



exit()


#Test data
test=np.array([])
firstIt=True
for filepath in Path('toyData').glob('test/*.csv'):
	if firstIt:
		data=np.asarray(pd.read_csv(filepath, names=['x','y']).values)
		data=data[data[:,0].argsort()]
		test=np.reshape(data, [1,100,2]) #Not a good fix, won't work for  data where i don't know the size
		firstIt=False
		paths=[filepath]
	else:
		data=np.asarray(pd.read_csv(filepath, names=['x','y']).values)
		data=data[data[:,0].argsort()]
		test=np.insert(test, -1, data, axis=0)
		paths=np.insert(paths,-1, filepath, axis=0)


print(test.shape)

#Process test data
print(test[0,:,1])
test[:,:,1]=(test[:,:,1]-training_mean)/(training_std)
x_test=test[:,:,1]
x_test=np.reshape(x_test, [6,100,1])


#Predictions
x_test_pred=model.predict(x_test)

#Plot predictions
i=0
while i<6:
	fig=plt.figure()
	plt.plot(x_test[i,:,0])
	plt.plot(x_test_pred[i,:,0], color='r')
	plt.title(paths[i])
	plt.savefig('test{}'.format(i))
	plt.close()
	i+=1










#create sequence from test values.
#x_test = create_sequences(df_test_value.values)

#Get test MAE loss
test_mae_loss=np.mean(np.abs(x_test_pred-x_test),axis=1)
test_mae_loss=test_mae_loss.reshape((-1))
plt.hist(test_mae_loss, bins=50)
plt.xlabel('test MAE loss')
plt.ylabel('Number of samples')
plt.savefig('testmae{}'.format(suff))

#Detect all samples that are anomalies
anomalies=test_mae_loss>threshold

#Plot anomalies
#Data i is an anomaly if samples [(i-timestamps+1) to (i)] are anomalies
anomalous_data_indices=[]
for data_idx in range(TIME_STEPS-1, len(x_test)-TIME_STEPS+1):
  if np.all(anomalies[data_idx-TIME_STEPS+1 : data_idx]):
    anomalous_data_indices.append(data_idx)
    
#Overlay anomalies
#df_subset=df_daily_jumpsup.iloc[anomalous_data_indices]
print(len(anomalous_data_indices))
#subset=
#fig, ax = plt.subplots()
#df_daily_jumpsup.plot(legend=False, ax=ax)
#df_subset.plot(legend=False, ax=ax, color='r')
#plt.savefig('overlay{}'.format(suff))



