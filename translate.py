"""
https://markjong001.pixnet.net/blog/post/246140004
"""
import speech_recognition
import argparse


def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--path', '-p', type=str, required=False,  help='patn for wav')
    return parser.parse_args()


# 將wav檔的聲音轉成文字的function
def Voice_To_Text_wav(WAVE_FILENAME):
    print("讀取" + WAVE_FILENAME)
    wavFile = speech_recognition.AudioFile(WAVE_FILENAME)
    r = speech_recognition.Recognizer()
    with wavFile as source: 
        print("轉換中......")
        r.adjust_for_ambient_noise(source)
        audio = r.record(source)
        #print(type (audio))
    try:
        Text = r.recognize_google(audio, language="zh-TW")
    except:
        Text = "無法翻譯"
    return Text

# 將聲音轉成文字的function
def Voice_To_Text():
    r = speech_recognition.Recognizer()
    with speech_recognition.Microphone() as source: 
        print("請開始說話:")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        Text = r.recognize_google(audio, language="zh-TW")
    except:
        Text = "無法翻譯"
    return Text

if __name__ == '__main__':
    args = parse_args()
    text=Voice_To_Text_wav(args.path)
    print(text)
