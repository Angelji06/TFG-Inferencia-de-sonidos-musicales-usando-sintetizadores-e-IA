# logica.py
# Placeholders: implementa aquí tu lógica real

import time
import os

# logica.py
import os
import shutil
import time
import math
import itertools
import csv
import glob
import torch
import torchaudio
from torch.utils.data import DataLoader

# importa tus componentes (ajusta los nombres/paths según tu proyecto)
from SpectrogramTensorDataset4 import SpectrogramTensorDataset
from Prototipo4 import CNNRegressor4, HybridLoss 
# pyo debe importarse DESPUÉS porque inicializa servidor de audio
from pyo import Server, Sig, FM, Pattern

# IMPORTANTE: importa tu función espectrograma centralizada
from SpectrogramTensorDataset4 import waveform_to_spectrogram_tensor

#==============================================================================================================
#=================================== GENERACION DE DATASET ====================================================
#==============================================================================================================

# 1. GENERACIÓN DE WAVs FM + CSV DE ETIQUETAS CON BARRIDO PYO
def generar_wavs_FM():
    """
    Genera dataset FM (.wav) + labels.csv usando PYO (offline) SIN Pattern.
    Funciona dentro de funciones (pyo offline estable).
    """

    print("=== GENERACIÓN WAV FM (fase 1) ===")

    # Directorios
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(script_dir)
    datasets_dir = os.path.join(main_dir, "Datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    DATASETWAV_DIR = "datasetFMwav"
    out_path = os.path.join(datasets_dir, DATASETWAV_DIR)

    # Crear / limpiar
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    print("Carpeta creada:", out_path)

    # CSV
    csv_path = os.path.join(out_path, "labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "carrier", "ratio", "index"])

    # Servidor offline
    s = Server(audio="offline", nchnls=1, sr=44100).boot()

    # Parámetros del barrido
    params = {
        "carrier": (100, 2000, 100),
        "ratio":   (0.05, 2, 0.05),
        "index":   (1, 10, 0.5)
    }

    # Crear señales dinámicas
    carrier = Sig(100)
    ratio = Sig(0.05)
    index = Sig(1)

    synth = FM(carrier=carrier, ratio=ratio, index=index, mul=1, add=0)
    synth.out()

    TIME = 0.5  # duración por wav

    # Generadores de valores
    import numpy as np

    carrier_vals = np.arange(*params["carrier"])
    ratio_vals   = np.arange(*params["ratio"])
    index_vals   = np.arange(*params["index"])

    g = 0

    # BARRIDO MANUAL 100% SEGURO
    for c in carrier_vals:
        for r in ratio_vals:
            for i in index_vals:

                g += 1
                fname = f"fm_{g}.wav"
                file_path = os.path.join(out_path, fname)

                # Setear valores
                carrier.setValue(float(c))
                ratio.setValue(float(r))
                index.setValue(float(i))

                # Registrar en CSV
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([fname, c, r, i])

                # Render offline
                s.recordOptions(
                    dur=TIME,
                    filename=file_path,
                    fileformat=0,
                    sampletype=3
                )
                s.start()  # ← AQUI SE GENERA EL WAV

    print(f"WAVs generados: {g}")
    return out_path


# 2. CONVERSIÓN WAV → TENSORES PYTORCH
def convertir_wavs_a_tensores(wav_folder):
    """
    Convierte los .wav en `wav_folder` a tensores de espectrograma (.pt).
    Devuelve la ruta a la carpeta final de tensores.
    """

    print("=== CONVERSIÓN WAV → TENSOR (fase 2) ===")

    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_folder = os.path.join(main_dir, "Datasets", "datasetFMespec_torchaudio")

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)

    wav_files = [f for f in os.listdir(wav_folder) if f.endswith(".wav")]
    print("WAV encontrados:", len(wav_files))

    for wav_file in wav_files:
        file_path = os.path.join(wav_folder, wav_file)

        waveform, sr = torchaudio.load(file_path)

        # Fade (del script original)
        fade_samples = int(sr * 0.05)
        if fade_samples * 2 < waveform.shape[-1]:
            fade_in = torch.linspace(0, 1, fade_samples)
            fade_out = torch.linspace(1, 0, fade_samples)
            waveform[:, :fade_samples] *= fade_in
            waveform[:, -fade_samples:] *= fade_out

        # Espectrograma
        spec = waveform_to_spectrogram_tensor(waveform, sr)

        # Guardar tensor
        out_name = wav_file.replace(".wav", ".pt")
        torch.save(spec, os.path.join(out_folder, out_name))

    print("Conversión completada:", out_folder)
    return out_folder


# FUNCIÓN PRINCIPAL (LA QUE USA TKINTER)
def generar_dataset():
    """
    Función que llama Tkinter cuando pulsas 'Generar dataset'.
    Ejecuta:
        1. Generación FM (WAV+CSV)
        2. Conversión a tensores (.pt)
    Devuelve dict con la carpeta final de tensores.
    """
    start = time.time()

    wav_folder = generar_wavs_FM()
    tensor_folder = convertir_wavs_a_tensores(wav_folder)

    end = time.time()
    print(f"=== DATASET COMPLETO GENERADO en {end-start:.2f}s ===")

    # Para que el front lo trate igual que cargar_dataset
    return {
        "tipo": "carpeta_tensores",
        "ruta": tensor_folder,
        "tensores": [f for f in os.listdir(tensor_folder) if f.endswith(".pt")]
    }

#==============================================================================================================
#=============================== CARGA DATASET YA EXISTENTE ===================================================
#==============================================================================================================

def cargar_dataset(path):
    import os
    if not os.path.isdir(path):
        raise ValueError("Se esperaba una carpeta, no un archivo.")

    tensores = [
        f for f in os.listdir(path)
        if f.endswith((".pt", ".pth", ".tensor"))
    ]

    if not tensores:
        raise ValueError("No se encontraron tensores en la carpeta.")

    return {"tipo": "carpeta_tensores", "ruta": path, "tensores": tensores}

#==============================================================================================================
#==================================== ENTRENAMIENTO MODELO ====================================================
#==============================================================================================================

def entrenar_modelo(nombreModelo,
                    dataset_obj,
                    epochs=10,
                    batch_size=16,
                    lr=1e-3,
                    device=None,
                    n_params=3,
                    input_channels=1,
                    base_filters=32,
                    save_dir="models",
                    num_workers=0,
                    pin_memory=False,
                    print_every_batches=100):
    """
    Entrena un CNNRegressor sobre el dataset dado.

    Args:
        dataset_obj: dict con al menos la clave 'ruta' apuntando al directorio de tensores (.pt),
                     o una cadena con la ruta al directorio de tensores.
        epochs, batch_size, lr: hiperparámetros.
        device: 'cpu'|'cuda' o None (se elige automáticamente si None).
        n_params, input_channels, base_filters: parámetros del modelo.
        save_dir: carpeta donde se guardará el .pth (se crea si no existe).
        save_name: nombre de archivo (si None se genera con timestamp).
        num_workers, pin_memory: argumentos para DataLoader.
        print_every_batches: se pasa a model.fit para prints intermedios.

    Retorna:
        path completo al archivo .pth guardado (string).
    """


    tensors_dir = dataset_obj.get("ruta") 

    # --- Device ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Dataset y DataLoader ---
    dataset = SpectrogramTensorDataset(tensors_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers, pin_memory=pin_memory)

    print("Instanciando modelo!")
    # --- Instanciar modelo ---
    model = CNNRegressor4(n_params=n_params, input_channels=input_channels, base_filters=base_filters)

    print("Entrenando modelo!")
    # --- Entrenamiento: el método fit está definido en la clase ---
    history = model.fit(train_loader,
                        device=device,
                        epochs=epochs,
                        lr=lr,
                        print_every_batches=print_every_batches)

    # --- Guardar modelo ---
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, nombreModelo)

    # Guardar state_dict (método save de la clase o torch.save directo)
    try:
        # si la clase implementa .save(path) lo usamos
        model.save(save_path)
    except Exception:
        # fallback: guardar state_dict directamente
        torch.save(model.state_dict(), save_path)

    print(f"Entrenamiento finalizado. Modelo guardado en: {save_path}")

    # devuelve la ruta completa para que la UI la use como pathModelo
    return save_path
