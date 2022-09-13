#IMPORTAMOS LIBRERIAS NECESARIAS
from email.mime import image
import pyaudio
import wave
import librosa
from librosa.display import specshow
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

#DEFINIMOS PARAMETROS
FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=44100
CHUNK=1024
duracion=10
archivo="test_record_d.wav"

#INICIAMOS "pyaudio"
audio=pyaudio.PyAudio()

#INICIAMOS GRABACIÓN
stream=audio.open(format=FORMAT,channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Grabando...")
frames=[]

for i in range(0, int(RATE/CHUNK*duracion)):
    data=stream.read(CHUNK)
    frames.append(data)
print("Grabación terminada")

#DETENEMOS GRABACIÓN
stream.stop_stream()
stream.close()
audio.terminate()

#CREAMOS/GUARDAMOS EL ARCHIVO DE AUDIO
waveFile = wave.open(archivo, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

#CARGAMOS EL ARCHIVO GRABADO Y LO CONVERTIMOS EN ESPECTROGRAMA
y, sr = librosa.load(archivo)
figure, ax = plt.subplots()
figure.set_size_inches(4, 3)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
librosa.display.waveshow(y, sr=sr)
log_mel_spectrogram = librosa.power_to_db(S)
img = librosa.display.specshow(log_mel_spectrogram, x_axis='time', y_axis='mel', sr=sr,fmax=8000, ax=ax)
figure.colorbar(img, ax=ax, format='%+2.0f dB') 
plt.savefig("Espectrograma_d.jpg")
print("Espectrograma guardado")

#CARGAR ESPECTROGRAMA, CARGAR MODELO Y PREDECIR
model_enfermo = keras.models.load_model("modelo_enfermo_hp.h5")
model_enfermedad = keras.models.load_model("modelo_enfermedades_hp.h5")
imagen = cv2.imread("Espectrograma_d.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow('Imagen espectrograma paciente.',imagen)
imagen_pm = cv2.resize(imagen, (288,216), interpolation = cv2.INTER_AREA)
imagen_pm = imagen_pm.reshape(1,288,216,3)