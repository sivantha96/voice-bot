#!/usr/bin/env python3

from vosk import Model, KaldiRecognizer
import pyaudio
import json
import pickle
from collections import namedtuple
from keras.models import load_model
import json
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import pyttsx3
import os
engine = pyttsx3.init()

if not os.path.exists("model"):
    print("Model does not exist in the path ./")
    exit(1)


model = load_model('voicebot_model.h5')
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def custom_object_decoder(dict):
    return namedtuple('X', dict.keys())(*dict.values())


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_array, intents_json):
    tag = intents_array[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    intents_array = predict_class(msg, model)
    res = get_response(intents_array, intents)
    return res


model_speech = Model("model")
rec = KaldiRecognizer(model_speech, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1,
                rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

while True:

    data = stream.read(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result(), object_hook=custom_object_decoder)
        if len(result.text) == 0:
            pass
        else:
            stream.stop_stream()
            print(result.text)
            response = chatbot_response(result.text)
            print(response)
            engine.say(response)
            engine.runAndWait()
            stream.start_stream()
