from keras.models import load_model
import librosa
import numpy as np


def getPridict(filename):
    model = load_model('DangerAudioModel.h5')
    data = getFeature_Mel(filename)
    prediction = np.argmax(model.predict(data), axis=-1)
    return prediction

def getFeature_Mel(filename):
    y1, sr1 = librosa.load(filename, duration=2.97)  
    ps = librosa.feature.melspectrogram(y=y1, sr=sr1)
    data = np.array([ps.reshape( (128, 128, 1) )])
    return data