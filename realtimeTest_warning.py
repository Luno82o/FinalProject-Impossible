from keras.models import load_model
import librosa
import librosa.display
import numpy as np

import PyAudioRecord
import translate
import argparse


def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--path', '-p', type=str, required=False,  help='patn for wav')
    return parser.parse_args()


def load_data(path):
    y1, sr1 = librosa.load(path, duration=2.97)  
    ps = librosa.feature.melspectrogram(y=y1, sr=sr1)
    ps = np.array([ps.reshape( (128, 128, 1) )])
    return ps

if __name__ == '__main__':
    # 載入模型
    model = load_model('ScreamDemo1.h5')
    args = parse_args()
    
    begin = input("start?(Y/N)\n")
    
    if(begin == "Y"):
        
        if(args.path):
            text = translate.Voice_To_Text_wav(args.path)    #將音檔轉換成文字
            data = load_data(args.path)
        else:
            filename = PyAudioRecord.recordAudio()
            text = translate.Voice_To_Text_wav(filename)    #將音檔轉換成文字
            data = load_data(filename)
        print(text)
        
        prediction = np.argmax(model.predict(data), axis=-1)
        print(prediction)
        
        if(prediction == 0):
            print("someone is screaming!")
        else:
            if(prediction == 1):
                print("someone needs help!")
            else:
                print("nothing")
    else:
        print("bye~")