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

def recordStatementwav(filename):
    r = sr.Recognizer()
    with sr.WavFile(filename) as source:
        audio = r.record(source)
    return audio