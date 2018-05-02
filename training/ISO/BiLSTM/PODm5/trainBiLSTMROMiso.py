
# coding: utf-8

# In[1]:

import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import optimizers
from keras.models import load_model

f = h5py.File('PODm5_isotropicTurb32BoxALL.mat')
data = f.get('data')
data = np.transpose(data)

# Set User-defined params
n_cells=250
lr=0.005
batch_size = 32
epochs = 75
modelfilename = 'isoturb32boxROM_PODm5_c2.1.h5'
lossfilename = 'isoturb32boxROM_PODm5_c2.1_res'

length = 10
output = 10


# generate input and output pairs of sequences
def create_sequences(data, length, output):
    nsignals = data.shape[1]
    siglen = data.shape[0]
    sampX=[]
    sampy=[]
    indx = siglen - output - length
    for j in range(nsignals):
        sig = data[:,j]
        for i in range(indx):
            tempX = sig[i:length+i]
            tempy = sig[i+length:length+i+output]
            sampX.append(tempX)
            sampy.append(tempy)
    nsamples = len(sampX)        
    X = np.array(sampX).reshape(nsamples, length, 1)
    y = np.array(sampy).reshape(nsamples, output, 1) 
    return X, y  

#Split training and test datasets
def define_test_dataset(X, y, n_patterns, ntestsigs):
    testindex = int(np.floor(ntestsigs*n_patterns))
    X_train = X[:-testindex,:,:]
    y_train = y[:-testindex,:,:]
    X_test = X[-testindex:,:,:]
    y_test = y[-testindex:,:,:]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test

# configure problem
nsignals = data.shape[1]
siglen = data.shape[0]

# Extract sequences
inputdata = data[:,0:6]
X, y = create_sequences(inputdata, length, output)
#np.random.shuffle(X)
#np.random.shuffle(y)
ntestpatterns = siglen - length - output
ntestsigs = 1
X_train, y_train, X_test, y_test = define_test_dataset(X, y, ntestpatterns, ntestsigs)
X_train.shape


# define model
model = Sequential()
model.add(Bidirectional(LSTM(n_cells, return_sequences=True), input_shape=(length, 1)))
model.add(TimeDistributed(Dense(1)))
adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='mae', optimizer='adam')
print(model.summary())

# fit model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

print('Saving weights..')
#save weights for analysis
model.save(modelfilename)

loss_history =history.history['loss']


# Save results to file
print('Saving results')
np.savez_compressed(lossfilename, batch_size=batch_size, epochs=epochs, loss_history=loss_history)

