import pyaudio
import math
import struct
import wave
import time
import os

import glob
import numpy as np
import scipy.io.wavfile as wav


Threshold = 20

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

    
    def record(self):
        print('Noise detected, recording beginning')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH

        while current <= end:

            data = self.stream.read(chunk)
            if self.rms(data) >= Threshold: end = time.time() + TIMEOUT_LENGTH

            current = time.time()
            rec.append(data)
        return rec

    def write(self, recording, time):
        #n_files = len(os.listdir(self.filepath))

        filename = os.path.join(self.filepath, '{}.wav'.format(time))

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        print('[寫入聲音至/tmp]: {}'.format(filename))
        print("",self.getcurrenttime())
        return filename


    def listen(self):
        print('Listening beginning')
        while(True):
            input = self.stream.read(chunk)
            rms_val = self.rms(input)
            if rms_val > Threshold:
                filename = self.record()
                break
        return filename

    def recordnoise(self):
        print('silence recording beginning')
        rec = []
        
        localtime = time.localtime()
        result = time.strftime("%y-%m-%d_%H-%M-%S", localtime)
        
        while(True):
            data = self.stream.read(chunk)                      #收錄聲音
            rms_val = self.rms(data)                            #計算音量
            
            if rms_val > Threshold:                             #判斷音量是否大於設定值
                file = self.write(b''.join(rec),result)         #是，將安靜的聲音存檔
                

                localtime = time.localtime()
                result = time.strftime("%y-%m-%d_%H-%M-%S", localtime)

                rec = self.record()                             #錄製聲音
                file = self.write(b''.join(rec),result)         #將聲音存檔
                rec = []                                        #list初始化
                break
            else:
                rec.append(data)
        return file
    
    def merge_files(self, path_read_folder, path_write_wav_file):
    
        #files = os.listdir(path_read_folder)
        merged_signal = []
        i = 0
        for filename in glob.glob(os.path.join(path_read_folder, '*.wav')):
            if (i==0):
                mergedfilename = filename
                i = i+1
            print("合併檔案：",filename)
            sr, signal = wav.read(filename)
            merged_signal.append(signal)
        merged_signal = np.hstack(merged_signal)
        merged_signal = np.asarray(merged_signal, dtype=np.int16)
        
        result = os.path.split(mergedfilename)[1]
        path_write_wav_file = os.path.join(path_write_wav_file, '{}'.format(result))
        print("合併後檔案位置： " + path_write_wav_file)
        wav.write(path_write_wav_file, sr, merged_signal)
        
        for filename in glob.glob(os.path.join(path_read_folder, '*.wav')):
            os.remove(filename)
            
    def getsound(self):
        data = self.stream.read(chunk)
        return data
        
    def ifnoise(self, data):
        rms_val = self.rms(data)
        if rms_val > Threshold:
            return True
        else:
            return False
        
    def getcurrenttime(self):
        localtime = time.localtime()
        result = time.strftime("%y-%m-%d_%H-%M-%S", localtime)
        return result


if __name__ == '__main__':
    a = Recorder()
    filename = a.listen()