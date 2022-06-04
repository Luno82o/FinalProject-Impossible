import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
import librosa
import librosa.display
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

import ExtractMFCC

data = pd.read_csv('metadata/audioDataset.csv')
valid_data = data[['slice_file_name', 'fold' ,'classID', 'class']][ data['end']-data['start'] > 0 ]
valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')


def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)
     
    return mfccs_processed

def build_model_graph(input_shape=(40,)):
    model = Sequential()
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model


if __name__ == '__main__':

    features=[]
            
    for row in tqdm_notebook(valid_data.itertuples()): 
        print(row.path)
        filename = "audio/" + row.path
        mfccs_processed = extract_features(filename)
        features.append( (mfccs_processed, row.classID) )
        
    featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
        
        
    np.save("DangerAudioDataset", featuresdf, allow_pickle=True)
    
    dataset = featuresdf
    print('dataset:', len(featuresdf))
    
    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())
    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))
    
    # split the dataset 
    from sklearn.model_selection import train_test_split 
    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 127)
    
    num_labels = yy.shape[1]
    filter_size = 2
    
    #model = build_model_graph()



    
    input_shape=(40,0)
    
    model = Sequential()
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    model.build(input_shape)
    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    # Display model architecture summary 
    model.summary()
    # Calculate pre-training accuracy 
    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = 100*score[1]
    
    print("Pre-training accuracy: %.4f%%" % accuracy)
    
    num_epochs = 100
    num_batch_size = 32
    model.fit(x_train, y_train, batch_size=num_batch_size,
              epochs=num_epochs, validation_data=(x_test, y_test),
              verbose=1)
    
    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: {0:.2%}".format(score[1]))
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: {0:.2%}".format(score[1]))

    """
    #from keras.utils import np_utils
    #from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    
    X_train,X_test, y_train, y_test = train_test_split([i[0] for i in featuresdf],
                                                       [i[1] for i in featuresdf],
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
    """
    
    
    model.save('DangerAudioModel.h5') 
    print('Model exported and finished')

