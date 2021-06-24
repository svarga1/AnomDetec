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

matplotlib.use('agg')
#Sufffix
suff=''



#Load custom data- make this a function later
training=np.array([])
firstIt=True
for filepath in Path('toyData').glob('train2/*.csv'):
  if firstIt:
     data=np.asarray(pd.read_csv(filepath, names=['x','y']).values)
     data=data[data[:,0].argsort()]
     training=np.reshape(data, [1,100,2]) #Not a good fix, won't work for  data where i don't know the size
     firstIt=False
  else:
    data=np.asarray(pd.read_csv(filepath, names=['x','y']).values)
    data=data[data[:,0].argsort()]
    training=np.insert(training, -1, data, axis=0)


print(training.shape)
	



#Prepare training data
#Normalize and save the mean and std for normalizing test data
training_mean=training[:,:,1].mean()
training_std=training[:,:,1].std()
training[:,:,1]=(training[:,:,1]-training_mean)/(training_std)
x_train=training[:,:,1]

#Create sequences

TIME_STEPS=50



def create_sequences(values, time_steps=TIME_STEPS):
  output=[]
  for i in range(len(values)-time_steps+1):
    output.append(values[i:(i+time_steps)])
  return np.stack(output)
#i=0
#while i<training.shape[0]:
#	if i!=0:	
#		x_train=np.append(x_train, create_sequences(training[i,:,1]))
#	else:
#		x_train=[create_sequences(training[i,:,1])]
#	i+=1
#
#x_train=create_sequences(training[0,:,1])     #df_training_value.values)


print(x_train.shape)
x_train=np.reshape(x_train, [40,100,1])
#exit()

#x_train=x_train[:,:,1]

#build a model
#x_train=training 		#Change sigmoid to relu

model=keras.Sequential(
  [
    layers.Input(shape=(100,1)),
    layers.Conv1D(
      filters=32, kernel_size=7, padding='same', strides=2, activation='sigmoid'),
    layers.Dropout(rate=0.2),
    layers.Conv1D(filters=16, kernel_size=7, padding ='same', strides=2, activation='sigmoid'), 
    layers.Conv1DTranspose( 
      filters=16, kernel_size=7, padding='same', strides=2, activation='sigmoid'),
    layers.Dropout(rate=0.2),
    layers.Conv1DTranspose( 
     filters=32, kernel_size=7, padding='same', strides=2, activation ='sigmoid'), 
    layers.Conv1DTranspose(filters=1,kernel_size=7,padding='same'),
  ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse') #0.001 and mse
model.summary()

#Train the model
history=model.fit(
  x_train, x_train, epochs=100, batch_size=128, validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')],)

#Plot training and validation loss
fig=plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.savefig('loss{}'.format(suff))
plt.close()


#Anomaly detection

#Get training MAE loss
x_train_pred=model.predict(x_train)
#train_mae_loss=np.mean(np.abs(x_train_pred-x_train),axis=1)

#plt.hist(train_mae_loss, bins=50)
#plt.xlabel('Train MAE Loss')
#plt.ylabel('Number of samples')
#plt.savefig('maeHist{}'.format(suff))

##Reconstruction loss threshould
#threshold=np.max(train_mae_loss)



print(x_train[0].shape)
print(x_train_pred[0].shape)


#Compare reconstruction
fig=plt.figure()
plt.plot(x_train[0,:,0] ) #Put x-train to plot normalized data, training to plot original data
plt.plot(x_train_pred[0,:,0], color='r')
plt.savefig('reconstruction{}'.format(suff))

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
#x_test_pred=model.predict(x_test)
#test_mae_loss=np.mean(np.abs(x_test_pred-x_test),axis=1)
#test_mae_loss=test_mae_loss.reshape((-1))
#
#plt.hist(test_mae_loss, bins=50)
#plt.xlabel('test MAE loss')
#plt.ylabel('Number of samples')
#plt.savefig('testmae{}'.format(suff))

#Detect all samples that are anomalies
#anomalies=test_mae_loss>threshold

#Plot anomalies
#Data i is an anomaly if samples [(i-timestamps+1) to (i)] are anomalies
#anomalous_data_indices=[]
#for data_idx in range(TIME_STEPS-1, len(df_test_value)-TIME_STEPS+1):
#  if np.all(anomalies[data_idx-TIME_STEPS+1 : data_idx]):
#    anomalous_data_indices.append(data_idx)
    
#Overlay anomalies
#df_subset=df_daily_jumpsup.iloc[anomalous_data_indices]
#fig, ax = plt.subplots()
#df_daily_jumpsup.plot(legend=False, ax=ax)
#df_subset.plot(legend=False, ax=ax, color='r')
#plt.savefig('overlay{}'.format(suff))




