import speech_recognition as sr


# --------------------------------------
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


# --------------------------------------
# speech recognition
def recognition(filepath):
    audio = recordStatementwav(filepath)
    query = recognizeCommand(audio)

    return audio, query
