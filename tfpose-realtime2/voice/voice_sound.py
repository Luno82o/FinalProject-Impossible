import librosa
import numpy as np
import os
import pyaudio
import math
import struct
import wave
import time
import glob
import scipy.io.wavfile as wav

from tensorflow.keras.models import load_model
from pydub import AudioSegment

import scipy.io.wavfile as wav
#import argparse


#--------------------------------------
#record voice 相關
Threshold = 10

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
swidth = 2

TIMEOUT_LENGTH = 3

class Recorder:

    @staticmethod
    def rms(frame):
        count = len(frame) / swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self,filepath):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=chunk)
        self.filepath=filepath
    
    def getcurrenttime(self):
        localtime = time.localtime()
        result = time.strftime("%y-%m-%d_%H-%M-%S", localtime)
        return result
    
    def getsound(self):
        data = self.stream.read(chunk)
        return data
            
    def ifnoise(self, data):
        rms_val = self.rms(data)
        if rms_val > Threshold:
            return True
        else:
            return False

    def write(self, recording, time):
        #n_files = len(os.listdir(self.filepath))
        filename = os.path.join(self.filepath, '{}.wav'.format(time))
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        print('Written to file: {}'.format(filename))
        return filename
    
    def merge_files(self, path_read_folder, path_write_wav_file):
    
        #files = os.listdir(path_read_folder)
        merged_signal = []
        i = 0
        for filename in glob.glob(os.path.join(path_read_folder, '*.wav')):
            if (i==0):
                mergedfilename = filename
                i = i+1
            print(filename)
            sr, signal = wav.read(filename)
            merged_signal.append(signal)
        merged_signal = np.hstack(merged_signal)
        merged_signal = np.asarray(merged_signal, dtype=np.int16)
        
        result = os.path.split(mergedfilename)[1]
        path_write_wav_file = os.path.join(path_write_wav_file, '{}'.format(result))
        print("path_write_wav_file : " + path_write_wav_file)
        wav.write(path_write_wav_file, sr, merged_signal)
        
        for filename in glob.glob(os.path.join(path_read_folder, '*.wav')):
            os.remove(filename)    
    
#--------------------------------------
#將聲音切成數個兩秒的音檔，每個片段之間重疊一秒
def cutwav(file_path):
    audio = AudioSegment.from_file(file_path, "wav")
    audio_time = len(audio)                                             #獲取待切割音频的時間長度，单位是毫秒
    cut_parameters = np.arange(2, audio_time/1000, 1)                   #np.arange()函数第一个参数为起点，第二个参数为终点，第三个参数为步长（2秒）

    #print("audio_time=", audio_time, "cut_parameters=", cut_parameters)
    
    file = os.path.split(file_path)[1]
    file = os.path.splitext(file)[0]
    file = os.path.join(".\\record\\2second", file)
    
    #切割音檔
    start_time = int(0)                                                 #开始时间设为0
    filelist = []
    for t in cut_parameters:
        stop_time = int(t * 1000)                                       #pydub以毫秒为单位工作
        audio_chunk = audio[start_time:stop_time]                       #音频切割按开始时间到结束时间切割
        
        outputfile = file+ "_" +str(int(t-1)) +".wav"
        audio_chunk.export(outputfile, format="wav")                    #保存音频文件
        start_time = stop_time - 1000                                   #下一個開始時間設為结束時間的前1秒(也就是疊加上一段音頻的最後1秒
        
        print(outputfile)
        filelist.append(outputfile)
        
    print('finish')
    return filelist

#--------------------------------------
#取得聲音片段的mel特徵值
def getmel(file_path):
    y,sr = librosa.load(file_path, sr=22050, duration=2)
    data = librosa.feature.melspectrogram(y=y, sr=sr)
    data = np.array([data.reshape( (128, 87, 1) )])
    return data

#--------------------------------------
#用mel特徵值放入模型預測
def getPridict(data, modelpath):
    model = load_model(modelpath)
    
    prediction = np.argmax(model.predict(data), axis=-1)
    return prediction

#--------------------------------------
#用預測結果判斷是否為危險聲音
def is_danger_sound(prediction):
    if(prediction == 0) or (prediction == 1): #0=scream 1=help 2=haha
        return True
    else:
        return False