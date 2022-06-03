from keras.models import load_model
import librosa
import numpy as np


def getPridict(filename):
    # 載入模型
    model = load_model('DangerAudioModel.h5')
    
    y1, sr1 = librosa.load(filename, duration=2.97)  
    ps = librosa.feature.melspectrogram(y=y1, sr=sr1)
    data = np.array([ps.reshape( (128, 128, 1) )])
    
    prediction = np.argmax(model.predict(data), axis=-1)
    return(prediction)