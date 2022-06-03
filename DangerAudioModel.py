#import os
import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
#from scipy.io import wavfile as wav
import librosa
import librosa.display
import numpy as np
import pandas as pd
#import random
#import struct
#import matplotlib.pyplot as plt
#import IPython.display as ipd
#import progressbar
#import time

data = pd.read_csv('metadata/audioDataset.csv')
valid_data = data[['slice_file_name', 'fold' ,'classID', 'class']][ data['end']-data['start'] > 0 ]
valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')


from tqdm import tqdm_notebook

D=[]
        
for row in tqdm_notebook(valid_data.itertuples()): 
    y1, sr1 = librosa.load("audio/" + row.path, duration=2.97)  
    ps = librosa.feature.melspectrogram(y=y1, sr=sr1)
    #print(ps)
    #print(ps.shape)
    
    if ps.shape != (128, 128): 
        print(row.path + "!=128,128")
        continue
        
    print(row.path)
    #print(row.classID)    
    D.append( (ps, row.classID) )
    
    

dataset = D
print('dataset:', len(dataset))


#from keras.utils import np_utils
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split([i[0] for i in D],
                                                   [i[1] for i in D],
                                                   test_size=0.2,
                                                   random_state=0)

#X_train, y_train = zip(*train)
#X_test, y_test = zip(*test)

X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
X_test = np.array([x.reshape( (128, 128, 1) ) for x in X_test])
#print(y_train)
#print(y_test)
y_train = np.array(keras.utils.np_utils.to_categorical(y_train, 10))
y_test = np.array(keras.utils.np_utils.to_categorical(y_test, 10))
#print(y_train)
#print(y_test)


####
model = Sequential()
input_shape=(128, 128, 1)

model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(10))
model.add(Activation('softmax'))


####

model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=['accuracy'])

model.fit(
    x=X_train, 
    y=y_train,
    epochs=12,
    batch_size=128,
    validation_data= (X_test, y_test))

score = model.evaluate(
    x=X_test,
    y=y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('success')

####

predict = np.argmax(model.predict(X_test), axis=-1)

print(y_test)
print(predict)

model.save('DangerAudioModel.h5') 
print('Model exported and finished')

