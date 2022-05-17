"""
PyAudio example: Record a few seconds of audio and save to a WAVE file.
https://people.csail.mit.edu/hubert/pyaudio/
"""

import pyaudio
import wave
import time

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
    RECORD_SECONDS = 4
    
    
    localtime = time.localtime()
    result = time.strftime("%y-%m-%d_%H-%M-%S", localtime)
    WAVE_OUTPUT_FILENAME = "helpC" + result + ".wav"
    
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
    
    print("record?(Y/N):")
    record = input()