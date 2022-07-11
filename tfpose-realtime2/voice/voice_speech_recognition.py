import sys
import os
import wave

#--------------------------------------------------------
#set path
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)

#--------------------------------------------------------
#import /lib
import lib.realtime_recognize_google as realtime_recognize_google


#--------------------------------------
#speech recognition
def recognition(filepath):
    
    audio = realtime_recognize_google.recordStatementwav(filepath)
    query = realtime_recognize_google.recognizeCommand(audio)
        
    return audio,query

#--------------------------------------
#is danger 
def is_danger_text(path,text):
   
   dangertext = []
   try:
       with open(path,"r",encoding='utf-8') as f:
           for line in f.readlines():
               dangertext = line.split(',')
               
   except IOError:
       print('ERROR: can not found ' + path)
   
   for word in dangertext:
       if word in text:
           return True
   return False

#--------------------------------------
#save audio
def saveAudio(audio):
    print("")

