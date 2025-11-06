#
# Requiere tener instalado ffmpeg: https://www.gyan.dev/ffmpeg/builds/
# Para que torchaudio funcione se requiere unas versiones concretas de torch y torchudio, para que sean compatibles
# Yo (David) estoy usando python 3.11.9 y he instalado:  
#        pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
#        pip install torchcodec
#        pip install "numpy<2" --force-reinstall
# De momento Torchaudio-cpu, pero conviene usar gpu

import os
import shutil
import time
import numpy as np
import torch
import torchaudio

# --- Funciones auxiliares ---

def apply_fade(signal, sr, fade_time=0.05):
    """Aplica fade in/out al audio para suavizar transitorios."""
    fade_samples = int(sr * fade_time)
    fade_in = torch.linspace(0, 1, fade_samples)
    fade_out = torch.linspace(1, 0, fade_samples)
    signal[:, :fade_samples] *= fade_in
    signal[:, -fade_samples:] *= fade_out
    return signal

def wav_to_spectrogram_tensor(signal, sample_rate):
    """Convierte señal en espectrograma (tensor en dB)."""
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=1024,
        hop_length=512,
        power=2.0
    )(signal)

    spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    return spectrogram_db

# --- Directorios del dataset ---
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sube un nivel desde scripts/
DATASETWAV_DIR = os.path.join(MAIN_DIR, "Datasets", "datasetFMwav")
DATASETFM_DIR = os.path.join(MAIN_DIR, "Datasets", "datasetFMespec_torchaudio")

# Si la carpeta de salida ya existe, borrarla y recrearla
if os.path.exists(DATASETFM_DIR):
    shutil.rmtree(DATASETFM_DIR)
os.makedirs(DATASETFM_DIR)

# --- Medición de tiempo ---
start_time = time.time()

# --- Procesamiento ---
wav_files = [f for f in os.listdir(DATASETWAV_DIR) if f.endswith('.wav')]
print(f"Encontrados {len(wav_files)} archivos WAV en {DATASETWAV_DIR}")

for wav_file in wav_files:
    file_path = os.path.join(DATASETWAV_DIR, wav_file)
    signal, sample_rate = torchaudio.load(file_path)  # waveform: [channels, time]

    # Aplicar fade
    signal = apply_fade(signal, sample_rate)

    # Convertir a espectrograma tensor
    spectrogram_db = wav_to_spectrogram_tensor(signal, sample_rate)

    # Guardar tensor en disco
    output_filename = os.path.splitext(wav_file)[0] + ".pt"
    output_path = os.path.join(DATASETFM_DIR, output_filename)
    torch.save(spectrogram_db, output_path)

# --- Fin de medición ---
end_time = time.time()
elapsed = end_time - start_time

print("Conversión completada. Tensores guardados en", DATASETFM_DIR)
print(f"Tiempo total de ejecución: {elapsed:.2f} segundos")
