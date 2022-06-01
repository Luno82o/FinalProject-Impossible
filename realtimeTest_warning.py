from keras.models import load_model
import librosa
import librosa.display
import numpy as np

import PyAudioRecord
import translate

def load_data(path):
    y1, sr1 = librosa.load(path, duration=2.97)  
    ps = librosa.feature.melspectrogram(y=y1, sr=sr1)
    ps = np.array([ps.reshape( (128, 128, 1) )])
    return ps

# 載入模型
model = load_model('ScreamDemo1.h5')

print("start?(Y/N)")
filename = PyAudioRecord.recordAudio()
text = translate.Voice_To_Text_wav(filename)    #將音檔轉換成文字
print(text)

data = load_data(filename)                      #

prediction = np.argmax(model.predict(data), axis=-1)

print(prediction)
if(prediction == 0):
    print("someone is screaming!")
else:
    if(prediction == 1):
        print("someone needs help!")
    else:
        print("nothing")
        