import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
import librosa
import librosa.display
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from keras.callbacks import TensorBoard

#import prediction

def importData():
    data = pd.read_csv('metadata/audioDataset.csv')
    
    # Get data over 3 seconds long
    valid_data = data[['slice_file_name', 'fold' ,'classID', 'class']][ data['end']-data['start'] >= 3 ]
    valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')
    
    print('===========Import data begin===========')

    D=[]
            
    for row in valid_data.itertuples(): 
        y, sr = librosa.load("audio/" + row.path, duration=2.97)  
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        #print(ps)
        #print(ps.shape)
        
        if ps.shape != (128, 128): 
            print(row.path + "!=128,128")
            continue
        
        D.append( (ps, row.classID) )
        print(row.path)
        
    print('===========Import data finish===========')
    return D
    

def trainData(dataset):
    
    print('dataset:', len(dataSet))
    
    X_train,X_test, y_train, y_test = train_test_split([i[0] for i in dataset],
                                                       [i[1] for i in dataset],
                                                       test_size=0.2,
                                                       random_state=0)
    
    # Reshape for CNN input
    X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
    X_test = np.array([x.reshape( (128, 128, 1) ) for x in X_test])
    
    # One-Hot encoding for classes
    y_train = np.array(keras.utils.np_utils.to_categorical(y_train, 10))
    y_test = np.array(keras.utils.np_utils.to_categorical(y_test, 10))
    
    model = Sequential()
    
    input_shape=(128, 128, 1)
    
    # 24 depths 128 - 5 + 1 = 124 x 124 x 24
    model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
    
    # 31 x 62 x 24
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))
    
    # 27 x 58 x 48
    model.add(Conv2D(48, (5, 5), padding="valid"))
    
    # 6 x 29 x 48
    model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    model.add(Activation('relu'))
    
    # 2 x 25 x 48
    model.add(Conv2D(48, (5, 5), padding="valid"))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dropout(rate=0.5))
    
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    
    #Output
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    
    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=['accuracy'])
    
    
    #tensorboard = TensorBoard(log_dir="logs/", histogram_freq=0,
    #                     write_graph=True, write_images=True)
    
    model.fit(
        x=X_train, 
        y=y_train,
        epochs=12,
        batch_size=128,
        validation_data= (X_test, y_test)#,
        #callbacks=[tensorboard]
        )
    
    score = model.evaluate(
        x=X_test,
        y=y_test)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Model exported and finished')
    model.save('DangerAudioModel.h5') 
        
    return model
    
    
    
    

if __name__ == '__main__':
    dataSet = importData()
    model = trainData(dataSet)

