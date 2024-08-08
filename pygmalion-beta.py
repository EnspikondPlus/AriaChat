import PySimpleGUI as psg
import torch
import soundfile as sf
import pyaudio
import wave
import contextlib
import speech_recognition as sr

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, pipeline
from datasets import load_dataset
from pygame import mixer
from time import sleep


class Recorder(object):
    def __init__(self, channels=1, rate=44100, frames_per_buffer=1024):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer

    def open(self, gname, mode='wb'):
        return RecordingFile(gname, mode, self.channels, self.rate,
                             self.frames_per_buffer)


class RecordingFile(object):
    def __init__(self, gname, mode, channels,
                 rate, frames_per_buffer):
        self.fname = gname
        self.mode = mode
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self._pa = pyaudio.PyAudio()
        self.wavefile = self._prepare_file(self.fname, self.mode)
        self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        self.close()

    def record(self, sduration):
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                     channels=self.channels,
                                     rate=self.rate,
                                     input=True,
                                     frames_per_buffer=self.frames_per_buffer)
        for _ in range(int(self.rate / self.frames_per_buffer * sduration)):
            saudio = self._stream.read(self.frames_per_buffer)
            self.wavefile.writeframes(saudio)
        return None

    def start_recording(self):
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                     channels=self.channels,
                                     rate=self.rate,
                                     input=True,
                                     frames_per_buffer=self.frames_per_buffer,
                                     stream_callback=self.get_callback())
        self._stream.start_stream()
        return self

    def stop_recording(self):
        self._stream.stop_stream()
        return self

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            self.wavefile.writeframes(in_data)
            return in_data, pyaudio.paContinue

        return callback

    def close(self):
        self._stream.close()
        self._pa.terminate()
        self.wavefile.close()

    def _prepare_file(self, gname, mode='wb'):
        wavefile = wave.open(gname, mode)
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.rate)
        return wavefile


generator = pipeline('text-generation', model='PygmalionAI/pygmalion-1.3b')

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
outputmodel = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

recognizer = sr.Recognizer()

aria = psg.Image('Images/idle.png', key='-ARIA-', size=(300, 300))
l1 = psg.Text('Talk with Aria:', key='-TITLE-', font=('Arial Bold', 20), expand_x=True, justification='left')
t1 = psg.Input('', enable_events=True, key='-INPUT-', font=('Arial Bold', 20), expand_x=True, justification='left')
b1 = psg.Button('Record', key='-RECORD-', font=('Arial Bold', 20))
b2 = psg.Button('Stop', key='-STOP-', font=('Arial Bold', 20), disabled=True)
b3 = psg.Button('Talk', key='-TALK-', font=('Arial Bold', 20), disabled=True)
b4 = psg.Button('Exit', key='-EXIT-', font=('Arial Bold', 20))
tt = psg.Text('', key='-STATUS-', font=('Arial Bold', 20), expand_x=True, justification='left', text_color='red')
m1 = psg.Multiline("Aria's Persona: Aria is a friendly girl who likes to talk about everything and anything.",
                   key='-CONVO-', font=('Arial Bold', 15), expand_x=True, justification='left', size=(50, 8),
                   autoscroll=True)
layout = [[aria], [l1], [b1, b2, b3, b4], [tt], [m1]]
window = psg.Window('Aria Chatbot', layout, size=(750, 750))

step = 0
rec = Recorder(channels=2)
mixer.init()

while True:
    event, values = window.read()
    if event == '-RECORD-':
        with rec.open("Sound/talk.wav", 'wb') as recfile2:
            window.refresh()
            recfile2.start_recording()
            window['-RECORD-'].update(disabled=True)
            window['-STOP-'].update(disabled=False)
            window['-STATUS-'].update("Recording...")
            window.refresh()
            while True:
                event, values = window.read()
                if event == '-STOP-':
                    recfile2.stop_recording()
                    window['-STOP-'].update(disabled=True)
                    window['-TALK-'].update(disabled=False)
                    window['-RECORD-'].update(disabled=False)
                    break
        window['-STATUS-'].update("")
        window.refresh()

    if event == '-TALK-':
        window['-RECORD-'].update(disabled=True)
        window['-TALK-'].update(disabled=True)
        window.refresh()
        speech = sr.AudioFile("Sound/talk.wav")
        with speech as source:
            audio = recognizer.record(source)

        input_text = recognizer.recognize_google(audio)
        window['-CONVO-'].update(window['-CONVO-'].get() + "\nYou: " + input_text)
        window.refresh()

        output = generator(window['-CONVO-'].get(), do_sample=True, max_new_tokens=60)

        clean_output = output[0]['generated_text'].split(input_text, 1)[1].split("\n")[1].split("Aria: ")[1]

        window['-CONVO-'].update(window['-CONVO-'].get() + "\nAria: " + clean_output)
        window.refresh()

        inputs = processor(text=clean_output, return_tensors="pt")
        speech = outputmodel.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        sf.write("Sound/reply.wav", speech.numpy(), samplerate=16000)

        fname = "Sound/reply.wav"
        with contextlib.closing(wave.open(fname, 'r')) as f:
            frames = f.getnframes()
            duration = frames / float(16000)

        speech = mixer.Sound("Sound/reply.wav")
        speechcnl = mixer.Channel(2)
        speechlen = speech.get_length()
        speechcnl.play(speech)
        for i in range(int(speechlen*5)):
            sleep(0.1)
            window['-ARIA-'].update("Images/open.png")
            window.refresh()
            sleep(0.1)
            window['-ARIA-'].update("Images/shut.png")
            window.refresh()

        step += 1
        window['-RECORD-'].update(disabled=False)
        window['-ARIA-'].update("Images/idle.png")

    if event == psg.WIN_CLOSED or event == '-EXIT-':
        window.close()
        exit(0)
