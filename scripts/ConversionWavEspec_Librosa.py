import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import shutil
import time

# ===========================================================================================================
#  Script para convertir los archivos .wav del dataset en .png de espectrogramas y guardarlos en disco.
# ===========================================================================================================

# --- Funciones auxiliares ---

def apply_fade(signal, sr, fade_time=0.05):
    """Aplica fade in/out al audio para suavizar transitorios."""
    fade_samples = int(sr * fade_time)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    signal[:fade_samples] *= fade_in
    signal[-fade_samples:] *= fade_out
    return signal

def wav_to_spectrogram_and_save(signal, sample_rate, output_path):
    """Convierte señal en espectrograma y guarda como imagen PNG."""
    stft = librosa.stft(signal)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)

    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(
        spectrogram_db,
        y_axis='log',
        x_axis='time',
        sr=sample_rate,
        cmap='inferno',
        ax=ax
    )
    ax.axis('off')
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return output_path

# --- Directorios del dataset ---
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sube un nivel desde scripts/
DATASETWAV_DIR = os.path.join(MAIN_DIR, "Datasets", "datasetFMwav")
DATASETFM_DIR = os.path.join(MAIN_DIR, "Datasets", "datasetFMespec_librosa")

# Si la carpeta de salida ya existe, borrarla y recrearla
if os.path.exists(DATASETFM_DIR):
    shutil.rmtree(DATASETFM_DIR)
os.makedirs(DATASETFM_DIR)

# --- Medición de tiempo ---
start_time = time.time()

# --- Procesamiento ---
wav_files = [f for f in os.listdir(DATASETWAV_DIR) if f.endswith('.wav')]
print(f"Encontrados {len(wav_files)} archivos WAV")

for wav_file in wav_files:
    file_path = os.path.join(DATASETWAV_DIR, wav_file)
    signal, sample_rate = librosa.load(file_path, sr=None)

    # Aplicar fade
    signal = apply_fade(signal, sample_rate)

    # Nombre de salida
    output_filename = os.path.splitext(wav_file)[0] + ".png"
    output_path = os.path.join(DATASETFM_DIR, output_filename)

    # Guardar espectrograma
    wav_to_spectrogram_and_save(signal, sample_rate, output_path)

# --- Fin de medición ---
end_time = time.time()
elapsed = end_time - start_time

print("Conversión completada. Espectrogramas guardados en", DATASETFM_DIR)
print(f"Tiempo total de ejecución: {elapsed:.2f} segundos")

