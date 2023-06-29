import pyttsx3
from pygame import mixer

class Talk:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('rate', 160)
        self.engine.setProperty('voice', self.voices[0].id)

    def talk(self, text):
        mixer.init()
        self.engine.say(text)
        self.engine.runAndWait()




