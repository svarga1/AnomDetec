#Test for keras to make sure anomaly detection works applied to time series
#Steps from https://keras.io/examples/timeseries/timeseries_anomaly_detection/

#Setup
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

#Sufffix
suff='kernel10'


#Load data
url='https://raw.githubusercontent.com/numenta/NAB/master/data/'
df_small_noise_url=url+'artificialNoAnomaly/art_daily_small_noise.csv'
df_small_noise=pd.read_csv(df_small_noise_url, parse_dates=True, index_col='timestamp') #Normal data
df_daily_jumpsup_url=url+'artificialWithAnomaly/art_daily_jumpsup.csv'
df_daily_jumpsup=pd.read_csv(df_daily_jumpsup_url, parse_dates=True, index_col='timestamp') #Anomalous data

#Data Visualization
#fig, ax =plt.subplots()
#df_small_noise.plot(legend=False, ax=ax)
#plt.savefig('normaldata')
#fig, ax = plt.subplots()
#df_daily_jumpsup.plot(legend=False,ax=ax)
#plt.savefig('anomdata')

#Prepare training data
#Normalize and save the mean and std for normalizing test data
training_mean=df_small_noise.mean()
training_std=df_small_noise.std()
df_training_value=(df_small_noise-training_mean)/training_std

#Create sequences

TIME_STEPS=288

def create_sequences(values, time_steps=TIME_STEPS):
  output=[]
  for i in range(len(values)-time_steps+1):
    output.append(values[i:(i+time_steps)])
  return np.stack(output)
x_train=create_sequences(df_training_value.values)

#build a model-- change 16 to 32 and 8 to 16, change kernel to 7
model=keras.Sequential(
  [
    layers.Input(shape=(x_train.shape[1],x_train.shape[2])),
    layers.Conv1D(
      filters=32, kernel_size=10, padding='same', strides=2, activation='relu'),
    layers.Dropout(rate=0.2),
    layers.Conv1D(filters=16, kernel_size=10, padding ='same', strides=2, activation='relu'), 
    layers.Conv1DTranspose( 
      filters=16, kernel_size=10, padding='same', strides=2, activation='relu'),
    layers.Dropout(rate=0.2),
    layers.Conv1DTranspose( 
     filters=32, kernel_size=10, padding='same', strides=2, activation ='relu'), 
    layers.Conv1DTranspose(filters=1,kernel_size=7,padding='same'),
  ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model.summary()

#Train the model
history=model.fit(
  x_train, x_train, epochs=15, batch_size=128, validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')],)

#Plot training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.savefig('loss{}'.format(suff))

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

#Compare reconstruction
plt.plot(x_train[0])
plt.plot(x_train_pred[0])
plt.savefig('test{}'.format(suff))

#Test data
df_test_value=(df_daily_jumpsup-training_mean) / training_std
fig, ax = plt.subplots()
df_test_value.plot(legend=False, ax=ax)
plt.savefig('testdata{}'.format(suff))

#create sequence from test values.
x_test = create_sequences(df_test_value.values)

#Get test MAE loss
x_test_pred=model.predict(x_test)
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
for data_idx in range(TIME_STEPS-1, len(df_test_value)-TIME_STEPS+1):
  if np.all(anomalies[data_idx-TIME_STEPS+1 : data_idx]):
    anomalous_data_indices.append(data_idx)
    
#Overlay anomalies
df_subset=df_daily_jumpsup.iloc[anomalous_data_indices]
fig, ax = plt.subplots()
df_daily_jumpsup.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color='r')
plt.savefig('overlay{}'.format(suff))




