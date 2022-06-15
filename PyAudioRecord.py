"""
PyAudio example: Record a few seconds of audio and save to a WAVE file.
https://people.csail.mit.edu/hubert/pyaudio/
"""

import pyaudio
import wave
import time

import argparse

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--folder', '-f', default='PyAudioOutput', type=str, required=False,  help='patn for wav')
    parser.add_argument('--lable', '-l', default='test', type=str, required=False,  help='lable for wav')
    return parser.parse_args()

def recordAudio(folder='PyAudioOutput', lable='test'):
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 3
    
    localtime = time.localtime()
    result = time.strftime("%y-%m-%d_%H-%M-%S", localtime)
    WAVE_OUTPUT_FILENAME = folder + "\\" + lable + result + ".wav"
    
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
    print("save at", WAVE_OUTPUT_FILENAME)
    return WAVE_OUTPUT_FILENAME



if __name__ == '__main__':
    args = parse_args()
    value = input("record?(Y/N)\n")
    while (value == 'Y'):
        path = recordAudio(args.folder, args.lable)
        print(path)
        value = input("record?(Y/N)\n")
    