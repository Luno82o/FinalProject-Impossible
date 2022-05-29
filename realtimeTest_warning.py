from keras.models import load_model
import librosa
import librosa.display
import numpy as np

import pyaudio
import wave
import time

def load_data(path):
    y1, sr1 = librosa.load(path, duration=2.97)  
    ps = librosa.feature.melspectrogram(y=y1, sr=sr1)
    ps = np.array([ps.reshape( (128, 128, 1) )])
    return ps

# 載入模型
model = load_model('ScreamDemo1.h5')

print("record?(Y/N):")
record = input()
num = 0

while record == 'Y' :
    #time.sleep(1)
    num = num + 1
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 3
    
    
    localtime = time.localtime()
    result = time.strftime("%y-%m-%d_%H-%M-%S", localtime)
    WAVE_OUTPUT_FILENAME = "test" + result + ".wav"
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("* recording")
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("* done recording")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    #path = "audio/fold5/LunScream22-05-15_21-53-08.wav"
    data = load_data(WAVE_OUTPUT_FILENAME)
    
    prediction = np.argmax(model.predict(data), axis=-1)
    print(prediction)
    if(prediction == 0):
        print("safe")
    if(prediction == 1):
        print("someone is screaming!")
    if(prediction == 2):
        print("someone needs help!")
        
    print("record?(Y/N):")
    record = input()