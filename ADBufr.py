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


#Orion flags
matplotlib.use('agg')

#Post
suff='init'


#Read in Data-- Come back to this entire section
fid=Dataset('terpinput.nc', 'r')
pres=fid.variables['pres'][:,:,:]
temp=fid.variables['temp'][:,:,:]
mas=np.ma.mask_or(temp==np.inf, np.isnan(temp))
pres.mask=mas
temp.mask=mas

#Train test split
pres_train=pres[5:,:,:]
temp_train=temp[5:,:,:]
pres_test=pres[:5,:,:]
temp_test=temp[:5,:,:]
#ML Preprocessing

train_mean=np.nanmean(temp_train)
train_std=temp_train.std()
print([train_mean, train_std])

exit()


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
layers.Cropping1D(cropping=(0,2))

  ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')


history=model.fit(longtemp, longtemp, epochs=100, batch_size=127, validation_split=0.1) #Since I can specifiy the target, I should be able to train the low-res upscale on the high-res upscaled to find points?






