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
import numpy as np
import soundfile as sf
import sounddevice as sd

# importa tus componentes (ajusta los nombres/paths según tu proyecto)
from SpectrogramTensorDataset4 import SpectrogramTensorDataset
from Prototipo4 import CNNRegressor4, HybridLoss 

#==============================================================================================================
#=================================== GENERACION DE DATASET ====================================================
#==============================================================================================================

# 1. GENERACIÓN DE WAVs FM + CSV DE ETIQUETAS CON BARRIDO CON NUMPY
def generar_wavs_FM():
    # dirs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(script_dir)
    out_path = os.path.join(main_dir, "Datasets", "datasetFMwav")
    if os.path.exists(out_path): shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)

    # params (misma semántica que la versión con pyo)
    params = {"carrier": (100,2000,100), "ratio": (0.05,2,0.05), "index": (1,10,0.5)}
    SR, TIME = 44100, 1
    t = np.linspace(0, TIME, int(SR*TIME), endpoint=False)
    # csv header
    csv_path = os.path.join(out_path, "labels.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["filename","carrier","ratio","index"])
   

    print("=== GENERACIÓN DE WAVS (fase 1/2) ===")
    g = 0
    for c in np.arange(*params["carrier"]):
        for r in np.arange(*params["ratio"]):
            for I in np.arange(*params["index"]):
                g += 1
                fname = f"fm_{g}.wav"
                print(f"Generado fm_{g}.wav")
                file_path = os.path.join(out_path, fname)
                fm = c * r
                mod = np.sin(2*np.pi*fm*t)
                x = np.sin(2*np.pi*c*t + I*mod).astype(np.float32)
                sf.write(file_path, x, SR, subtype='PCM_16')
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([fname, float(c), float(r), float(I)])

    print(f"WAVs generados: {g}; carpeta: {out_path}")
    return out_path

# 2. CONVERSIÓN WAV → TENSORES PYTORCH
def convertir_wavs_a_tensores(wav_folder, device):
    print("=== CONVERSIÓN WAV → TENSOR (fase 2/2) ===")

    # Directorios y carpetas
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_folder = os.path.join(main_dir, "Datasets", "datasetFMespec_torchaudio")
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    wav_files = [f for f in os.listdir(wav_folder) if f.endswith(".wav")]
    print("WAV encontrados:", len(wav_files))

    # Crear el transform UNA VEZ (STFT)
    spec_transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None, return_complex=True).to(device)

    # Transformación
    for wav_file in wav_files:
        file_path = os.path.join(wav_folder, wav_file)

        # Carga el tensor onda en CPU
        waveform, sr = torchaudio.load(file_path)  

        # Fade
        fade_samples = int(sr * 0.05)
        if fade_samples * 2 < waveform.shape[-1]:
            fade_in = torch.linspace(0, 1, fade_samples)
            fade_out = torch.linspace(1, 0, fade_samples)
            waveform[:, :fade_samples] *= fade_in
            waveform[:, -fade_samples:] *= fade_out

        # Tensor espectrograma
        spec = waveform_to_spectrogram_tensor(waveform, sr, device, spec_transform)

        # Guardar tensor espectrograma
        out_name = wav_file.replace(".wav", ".pt")
        torch.save(spec, os.path.join(out_folder, out_name))
        print(f"generado {out_name}")

    print("Conversión completada en:", out_folder)
    return out_folder

# FUNCIÓN PRINCIPAL
def generar_dataset(device):
    start = time.time()

    wav_folder = generar_wavs_FM()
    tensor_folder = convertir_wavs_a_tensores(wav_folder, device)   #Le paso el device para acelerar la transformacion a tensor

    end = time.time()
    print(f"=== DATASET COMPLETO GENERADO en {end-start:.2f}s ===")

    # Para que el front lo trate igual que cargar_dataset
    return {
        "tipo": "carpeta_tensores",
        "ruta": tensor_folder,
        "tensores": [f for f in os.listdir(tensor_folder) if f.endswith(".pt")]
    }

# Función que pasa una onda a un tensor de espectrograma
def waveform_to_spectrogram_tensor(waveform, sr, device, spec_transform):
    # Normalización: evita variaciones grandes de volumen
    waveform = waveform / waveform.abs().max().clamp(min=1e-8)
    waveform = waveform.to(device)

    # Espectrograma complejo (STFT)
    spec = spec_transform(waveform)   

    # Conversión a escala logarítmica (dB): comprime el rango dinámico y facilita el aprendizaje REVISAR ESTO
    mag = spec.abs()  # Magnitud lineal
    db = torchaudio.transforms.AmplitudeToDB(stype='amplitude',top_db=80.0).to(device)(mag)

    return db

#==============================================================================================================
#=============================== CARGA DATASET YA EXISTENTE ===================================================
#==============================================================================================================

# Busca la carpeta y se asegura de que contiene tensores
def check_dataset(path):
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

# Función encargada de instanciar y entrenar el modelo
def entrenar_modelo(nombreModelo, dataset_obj, epochs=10, batch_size=16, lr=1e-3, device="cuda", print_every_batches=100):
    tensors_dir = dataset_obj.get("ruta") 

    # --- Dataset y DataLoader ---
    dataset = SpectrogramTensorDataset(tensors_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Instanciando modelo!")
    # --- Instanciar modelo ---
    model = CNNRegressor4(3,1,32)

     # --- Entrenamiento ---
    print(f"Entrenando modelo!       Usando {device}")
    history = model.fit(train_loader, device=device, epochs=epochs, lr=lr, print_every_batches=print_every_batches)

    # --- Guardar modelo ---
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    if not nombreModelo.lower().endswith(".pth"):  # asegurar extensión .pth
        nombreModelo = nombreModelo + ".pth"
    save_path = os.path.join(save_dir , nombreModelo)
    torch.save(model.state_dict(), save_path)       # Guardar state_dict

    print(f"Entrenamiento finalizado. Modelo guardado en: {save_path}")

    return save_path  #Retorna: path completo al archivo .pth guardado (string).

#==============================================================================================================
#=========================================== PRUEBA MODELO ====================================================
#==============================================================================================================
def hacer_inferencia(ruta_modelo, ruta_wav, device="cpu"):
    """
    Carga el modelo, procesa el wav y devuelve los parámetros predichos (C, R, I).
    Adapada para CNNRegressor4 que devuelve (params, recon).
    """
    import torch
    import torchaudio
    import numpy as np
    from Prototipo4 import CNNRegressor4
    from SpectrogramTensorDataset4 import waveform_to_spectrogram_tensor

    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError("No se encuentra el archivo del modelo.")

    # 1. Configurar dispositivo
    device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    # 2. Instanciar arquitectura
    # Importante: n_params debe coincidir con tu entrenamiento (por defecto 3)
    model = CNNRegressor4(n_params=3) 
    
    # 3. Cargar pesos
    # Usamos weights_only=True para evitar el warning de seguridad
    try:
        state_dict = torch.load(ruta_modelo, map_location=device, weights_only=True)
    except:
        # Fallback por si tu versión de torch es antigua
        state_dict = torch.load(ruta_modelo, map_location=device)
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Procesar Audio
    waveform, sr = torchaudio.load(ruta_wav)
    # Convertir a espectrograma
    spec = waveform_to_spectrogram_tensor(waveform, sr)
    # Añadir dimensión de Batch (1, 1, H, W) y mover al device
    spec = spec.unsqueeze(0).to(device)

    # 5. Inferencia
    with torch.no_grad():
        # TU MODELO DEVUELVE: (params, recon)
        resultado = model(spec)
        
        # Separamos la tupla
        pred_params = resultado[0]  # Nos quedamos con los parámeteros
        # resultado[1] sería la 'recon' (imagen), la ignoramos
    
    # 6. Limpieza y conversión a lista plana de Python
    # .detach() saca el tensor del grafo de gradientes
    # .flatten() convierte [[c, r, i]] en [c, r, i]
    lista_valores = pred_params.detach().cpu().numpy().flatten().tolist()
    
    return lista_valores

def fm_synthesize(carrier, ratio, index, duration=1.0, sr=44100):
    """
    Genera la señal de audio sintética usando fórmulas FM.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mod = np.sin(2 * np.pi * (carrier * ratio) * t)
    car = np.sin(2 * np.pi * carrier * t + index * mod)
    
    # sounddevice prefiere float32 para el audio
    return car.astype(np.float32), sr

def play_audio(waveform, sr):
    """
    Reproduce un array de Numpy directamente usando sounddevice.
    """
    # sd.play es asíncrono (el código seguiría corriendo), 
    # por lo que añadimos sd.wait() para asegurarnos de que se escucha todo
    # antes de hacer otra cosa (opcional, puedes quitarlo si quieres).
    sd.play(waveform, sr)
    sd.wait()

def reproducir_wav(path):
    """
    Lee un archivo con soundfile y lo reproduce con sounddevice.
    """
    if os.path.exists(path):
        # Leemos el audio y el sample rate del archivo
        data, sr = sf.read(path)
        play_audio(data, sr)
    else:
        print(f"Error: No se encuentra el archivo {path}")

def reproducir_prediccion(params):
    """
    Genera el audio en tiempo real basado en los parámetros predichos y lo reproduce.
    """
    # Desempaquetar parámetros (asegurando floats de Python)
    carrier = float(params[0])
    ratio = float(params[1])
    index = float(params[2])
    
    print(f"Reproduciendo predicción: C={carrier:.2f}, R={ratio:.2f}, I={index:.2f}")
    
    # 1. Sintetizar (generamos 2 segundos para escucharlo bien)
    waveform, sr = fm_synthesize(carrier, ratio, index, duration=2.0)
    
    # 2. Reproducir
    play_audio(waveform, sr)