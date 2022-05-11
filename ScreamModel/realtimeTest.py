"""
PyAudio example: Record a few seconds of audio and save to a WAVE file.
https://people.csail.mit.edu/hubert/pyaudio/
"""

import pyaudio
import wave

import librosa
import numpy as np


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

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

#X, sample_rate = librosa.load(WAVE_OUTPUT_FILENAME, res_type='kaiser_fast')
#mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
#feature = mfccs

y1, sr1 = librosa.load(WAVE_OUTPUT_FILENAME, duration=2.97)  
ps = librosa.feature.melspectrogram(y=y1, sr=sr1)
print(ps)
print(type(ps))


#data, labels = load_data(<the path of the data>)
predict = model.predict(ps)
