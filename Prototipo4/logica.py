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
    t_start = time.time()  
    # dirs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(script_dir)
    out_path = os.path.join(main_dir, "Datasets", "datasetFMwav")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)
    
    # params (misma semántica que la versión con pyo)
    params = {"carrier": (100,2000,100), "ratio": (0.05,2,0.05), "index": (1,10,0.5)}
    SR, TIME = 44100, 1

    # t en float32 para evitar casts en cada iteración
    t = np.linspace(0, TIME, int(SR*TIME), endpoint=False).astype(np.float32)

    # csv header (abrimos el archivo una sola vez y escribimos en buffer por bloques)
    csv_path = os.path.join(out_path, "labels.csv")
    csv_buffer = []
    buffer_flush = 1000  # escribir cada 1000 filas (REVISAR NUMERO)

    with open(csv_path, "w", newline="") as f_header:
        csv.writer(f_header).writerow(["filename","carrier","ratio","index"])

    print("=== GENERACIÓN DE WAVS (fase 1/2) ===")
    print("Generando...")
    g = 0
    # iteración principal
    for c in np.arange(*params["carrier"]):
        for r in np.arange(*params["ratio"]):
            for I in np.arange(*params["index"]):
                g += 1
                fname = f"fm_{g}.wav"
                file_path = os.path.join(out_path, fname)

                # cálculo en float32
                fm = float(c) * float(r)
                mod = np.sin(2.0 * np.pi * fm * t).astype(np.float32)
                x = np.sin(2.0 * np.pi * float(c) * t + float(I) * mod).astype(np.float32)

                # escribir wav
                sf.write(file_path, x, SR, subtype='PCM_16')

                # acumular fila en buffer
                csv_buffer.append([fname, float(c), float(r), float(I)])
                if len(csv_buffer) >= buffer_flush:
                    # volcar buffer al CSV en bloque
                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerows(csv_buffer)
                    csv_buffer = []

                # prints reducidos para evitar overhead de I/O en consola
                if g % 1000 == 0:
                    print(f"Generados {g} wavs...")

    # flush final del buffer restante
    if csv_buffer:
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerows(csv_buffer)

    # tiempo final
    t_end = time.time()
    elapsed = t_end - t_start
    per_file = elapsed / g if g > 0 else 0.0
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"WAVs generados: {g}; carpeta: {out_path}")
    print(f"Tiempo total generación WAVs: {int(h)}h {int(m)}m {s:.2f}s  |  media por wav: {per_file:.4f}s")

    return out_path


# 2. CONVERSIÓN WAV → TENSORES PYTORCH
def convertir_wavs_a_tensores(wav_folder, device):
    print("=== CONVERSIÓN WAV → TENSOR (fase 2/2) ===")
    t_start = time.time()  # inicio temporizador

    # Directorios y carpetas
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_folder = os.path.join(main_dir, "Datasets", "datasetFMespec_torchaudio")
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    wav_files = [f for f in os.listdir(wav_folder) if f.endswith(".wav")]
    n_wavs = len(wav_files)
    print("WAV encontrados:", n_wavs)

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

    # tiempo total y por fichero
    t_end = time.time()
    elapsed = t_end - t_start
    per_file = elapsed / n_wavs if n_wavs > 0 else 0.0
    print(f"Conversión completada en: {out_folder}")
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"Tiempo conversión total: {int(h)}h {int(m)}m {s:.2f}s  |  media por wav: {per_file:.4f}s")
    return out_folder

# FUNCIÓN PRINCIPAL
def generar_dataset(device):
    start = time.time()

    wav_folder = generar_wavs_FM()
    #wav_folder = r"C:\Users\David\Documents\GitHub\TFG-Inferencia-de-sonidos-musicales-usando-sintetizadores-e-IA\Datasets\datasetFMwav"
    tensor_folder = convertir_wavs_a_tensores(wav_folder, device)   #Le paso el device para acelerar la transformacion a tensor

    end = time.time()
    elapsed = end - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"=== DATASET COMPLETO GENERADO en {int(h)}h {int(m)}m {s:.2f}s ===")

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
    start = time.time()  
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

    # -- Guardar stats (REVISAR no estoy seguro de que se haga asi) ---
    stats = dataset.get_stats()  # {'means': array, 'stds': array}
    checkpoint = {
        'state_dict': model.state_dict(),
        'param_means': stats['means'],
        'param_stds': stats['stds']
    }
    torch.save(checkpoint, save_path)

    end = time.time()
    elapsed = end - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"Entrenamiento finalizado. Modelo guardado en: {save_path}")
    print(f"Tiempo de entrenamiento: {int(h)}h {int(m)}m {s:.2f}s")

    return save_path  #Retorna: path completo al archivo .pth guardado (string).

#==============================================================================================================
#=========================================== PRUEBA MODELO ====================================================
#==============================================================================================================
def hacer_inferencia(ruta_modelo, ruta_wav, device="cpu"):
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError("No se encuentra el archivo del modelo.")

    # 1) Normalizar device
    device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    # 2) Instanciar arquitectura y cargar checkpoint
    model = CNNRegressor4(n_params=3)
    ckpt = torch.load(ruta_modelo, map_location=device)

    # soporte ambos formatos: checkpoint con 'state_dict' o antiguo state_dict directo
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        means = ckpt.get('param_means', None)
        stds = ckpt.get('param_stds', None)
    else:
        # archivo antiguo que contenía solo state_dict
        state_dict = ckpt
        means = None
        stds = None

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3) Procesar audio: leer WAV y calcular espectrograma como en entrenamiento
    waveform, sr = torchaudio.load(ruta_wav)           # tensor en CPU
    # normalizar peak igual que en pipeline de training
    waveform = waveform / waveform.abs().max().clamp(min=1e-8)

    # crear transform igual que en training
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=1024, hop_length=256, power=None, return_complex=True
    ).to(device)

    # mover waveform al device antes de transform y calcular magnitud dB
    waveform = waveform.to(device)
    spec_c = spec_transform(waveform)                  # compleja
    mag = spec_c.abs()
    db = torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=80.0).to(device)(mag)

    # asegurar shape (1, H, W) y batch dim (1, C, H, W)
    if db.dim() == 2:
        db = db.unsqueeze(0)    # (1, H, W)
    spec = db.unsqueeze(0).to(device)  # (1, 1, H, W)

    # 4) Inferencia
    with torch.no_grad():
        out = model(spec)
        if isinstance(out, (tuple, list)):
            pred_params = out[0]
        else:
            pred_params = out

    pred = pred_params.detach().cpu().numpy().flatten()  # (3,)

    # 5) Desnormalizar usando stats guardadas en el checkpoint
    if (means is None) or (stds is None):
        raise RuntimeError("El checkpoint no contiene 'param_means'/'param_stds'. Reentrena guardando stats en el checkpoint.")

    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)
    pred_raw = pred * stds + means

    return pred_raw.tolist()

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
    arr = np.asarray(waveform, dtype=np.float32)
    sd.play(arr, sr)
    sd.wait()

# Lee un archivo con soundfile y lo reproduce con sounddevice.
def reproducir_wav(path):
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