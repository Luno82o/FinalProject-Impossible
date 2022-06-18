# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 23:36:12 2022

@author: USER
"""

import speech_recognition as sr
#from pydub import AudioSegment

# Capture Voice
# takes command through microphone
def recordStatement():
	r = sr.Recognizer()
	with sr.Microphone() as source:
		print("listening.....")
		r.pause_threshold = 0.5
		audio = r.listen(source)
	return audio

def recognizeCommand(audio):
	r = sr.Recognizer()
	try:
		print("Recognizing.....")
		query = r.recognize_google(audio, language='zh-tw')
	except Exception as e:
		print("say that again please.....")
		return "None"
	return query

def takecommand():
    audio = recordStatement()
    query = recognizeCommand(audio)
    print(f"{query}\n")
    return query

if __name__ == '__main__':
    # Input from user
    # Make input to lowercase
    while (True):
        audio = recordStatement()
        query = recognizeCommand(audio)
        if(query == "None"):
            continue
        elif(query == "離開"):
            print("bye bye ~")
            break
        
        print(f"你是不是說: {query}\n")