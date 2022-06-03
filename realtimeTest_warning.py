import argparse

import PyAudioRecord
import translate
import prediction


def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--path', '-p', type=str, required=False,  help='patn for wav')
    return parser.parse_args()


if __name__ == '__main__':
    
    args = parse_args()
    
    begin = input("start?(Y/N)\n")
    
    if(begin == "Y"):
        
        if(args.path==""):
            filename = PyAudioRecord.recordAudio()
            text = translate.Voice_To_Text_wav(filename)    #將音檔轉換成文字
            prediction = prediction.audioPridict(filename)
        else:
            text = translate.Voice_To_Text_wav(args.path)    #將音檔轉換成文字
            prediction = prediction.audioPridict(args.path)
            
        print(text)
        
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