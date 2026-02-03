import time
import os
import shutil
import time
import math
import itertools
import csv
import glob
import torch
import torchaudio
from torch.utils.data import DataLoader, random_split
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa         
import librosa.display

# importa tus componentes (ajusta los nombres/paths según tu proyecto)
from SpectrogramTensorDataset4 import SpectrogramTensorDataset
from Prototipo4 import CNNRegressor4, HybridLoss 

#==============================================================================================================
#=================================== GENERACION DE DATASET ====================================================
#==============================================================================================================

GEN_PARAMS = {
    "carrier": (100, 2000, 100), 
    "ratio": (0.05, 2, 0.05), 
    "index": (1, 10, 0.5)
}

def get_gen_params():
    return GEN_PARAMS.copy()

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
    
    # params
    params = GEN_PARAMS
    SR, TIME = 44100, 1

    # Extraemos los pasos (step) para calcular el límite del Jitter
    step_c = params["carrier"][2]
    step_r = params["ratio"][2]
    step_i = params["index"][2]

    # t en float32 para evitar casts en cada iteración
    t = np.linspace(0, TIME, int(SR*TIME), endpoint=False).astype(np.float32)

    # csv header
    csv_path = os.path.join(out_path, "labels.csv")
    csv_buffer = []
    buffer_flush = 1000 

    with open(csv_path, "w", newline="") as f_header:
        # Importante: Las etiquetas ahora serán decimales (floats), no enteros exactos
        csv.writer(f_header).writerow(["filename","carrier","ratio","index"])

    print("=== GENERACIÓN DE WAVS (CON JITTER) (fase 1/2) ===")
    print(f"Jitter aplicado: Carrier ~±{step_c/2}Hz, Ratio ~±{step_r/2}, Index ~±{step_i/2}")
    print("Generando...")
    
    g = 0
    
    # --- ITERACIÓN PRINCIPAL ---
    # Usamos la rejilla como base, pero sintetizamos valores desviados
    for c_grid in np.arange(*params["carrier"]):
        for r_grid in np.arange(*params["ratio"]):
            for I_grid in np.arange(*params["index"]):
                g += 1
                fname = f"fm_{g}.wav"
                file_path = os.path.join(out_path, fname)

                # --- APLICACIÓN DEL JITTER ---  Esto convierte la rejilla discreta en una cobertura continua
                jitter_c = np.random.uniform(-step_c/2.0, step_c/2.0)
                jitter_r = np.random.uniform(-step_r/2.0, step_r/2.0)
                jitter_I = np.random.uniform(-step_i/2.0, step_i/2.0)

                # Valores finales para la síntesis 
                c_real = float(c_grid + jitter_c)
                r_real = float(r_grid + jitter_r)
                I_real = float(I_grid + jitter_I)
                
                # Protección mínima: evitamos frecuencias negativas o ratios/indices <= 0 extremos
                if c_real < 10: c_real = 10.0
                if r_real < 0.01: r_real = 0.01
                if I_real < 0: I_real = 0.0

                # --- SÍNTESIS (Usando los valores reales con Jitter) ---
                # Fm = Fc * Ratio
                fm = c_real * r_real
                
                # Moduladora
                mod = np.sin(2.0 * np.pi * fm * t).astype(np.float32)
                
                # Portadora modulada
                # x = sin(2*pi*Fc*t + I*mod)
                x = np.sin(2.0 * np.pi * c_real * t + I_real * mod).astype(np.float32)

                # escribir wav
                sf.write(file_path, x, SR, subtype='PCM_16')

                # --- GUARDAR ETIQUETA ---
                csv_buffer.append([fname, c_real, r_real, I_real])
                
                if len(csv_buffer) >= buffer_flush:
                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerows(csv_buffer)
                    csv_buffer = []

                if g % 1000 == 0:
                    print(f"Generados {g} wavs...")

    # flush final
    if csv_buffer:
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerows(csv_buffer)

    t_end = time.time()
    elapsed = t_end - t_start
    per_file = elapsed / g if g > 0 else 0.0
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"WAVs generados: {g}; carpeta: {out_path}")
    print(f"Tiempo total: {int(h)}h {int(m)}m {s:.2f}s  |  media: {per_file:.4f}s")

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)          # subir un nivel (carpeta del proyecto)
    save_dir = os.path.join(root_dir, "models")     # carpeta estable

    os.makedirs(save_dir, exist_ok=True)

    # --- Dataset y DataLoader ---
    dataset = SpectrogramTensorDataset(tensors_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Instanciando modelo!")
    # --- Instanciar modelo ---
    model = CNNRegressor4(3,1,32)

     # --- Entrenamiento ---
    print(f"Entrenando modelo!       Usando {device}")

    # Vamos usar el 80% de los parámetros para entrenar y el 20% restante lo usamos para la validación
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    # Lo dividimos aleatoriamente
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    history = model.fit(train_loader, val_loader=val_loader, device=device, epochs=epochs, lr=lr, print_every_batches=print_every_batches)

    # --- Guardar modelo ---
    save_path = os.path.join(save_dir , nombreModelo)

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

# Las dos funciones siguientes son la anterior dividida en dos, para hacer más eficiente el bucle de generación de predicciones
def cargar_modelo_para_inferencia(ruta_modelo, device="cpu"):
    """
    Carga el modelo y las estadísticas UNA SOLA VEZ.
    Devuelve el objeto model listo y las medias/desviaciones.
    """
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
    
    # Preparamos las stats como numpy arrays aquí para no hacerlo en cada vuelta
    if means is not None:
        means = np.asarray(means, dtype=np.float32)
    if stds is not None:
        stds = np.asarray(stds, dtype=np.float32)

    print("Modelo cargado exitosamente en:", device)
    
    # Devolvemos todo lo que necesita la siguiente función
    return model, means, stds, device

def hacer_inferencia_rapida(model, means, stds, ruta_wav, device):
    """
    Procesa un solo WAV usando un modelo YA cargado.
    """
    # 3) Procesar audio: leer WAV y calcular espectrograma como en entrenamiento
    waveform, sr = torchaudio.load(ruta_wav)           # tensor en CPU
    
    # normalizar peak igual que en pipeline de training
    waveform = waveform / waveform.abs().max().clamp(min=1e-8)

    # crear transform igual que en training
    # (Nota: Podríamos sacarlo fuera también, pero crearlo aquí es rápido y seguro)
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

    # 5) Desnormalizar usando stats QUE NOS HAN PASADO
    if (means is None) or (stds is None):
        raise RuntimeError("Error: 'means' o 'stds' son None. Revisa el checkpoint.")

    pred_raw = pred * stds + means

    return pred_raw.tolist()


# Genera la señal de audio sintética usando fórmulas FM.
def fm_synthesize(carrier, ratio, index, duration=1.0, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mod = np.sin(2 * np.pi * (carrier * ratio) * t)
    car = np.sin(2 * np.pi * carrier * t + index * mod)
    
    # sounddevice prefiere float32 para el audio
    return car.astype(np.float32), sr

# Reproduce audio usando soundevice
def play_audio(waveform, sr):
    arr = np.asarray(waveform, dtype=np.float32)
    sd.play(arr, sr)
    sd.wait()

# Lee un archivo con soundfile y lo reproduce con sounddevice (se usa para reproducir el wav original)
def reproducir_wav(path):
    if os.path.exists(path):
        data, sr = sf.read(path)
        play_audio(data, sr)
    else:
        print(f"Error: No se encuentra el archivo {path}")

# Genera el audio en tiempo real basado en los parámetros predichos y lo reproduce
def reproducir_prediccion(params):
    # Desempaquetar parámetros (asegurando floats de Python)
    carrier = float(params[0])
    ratio = float(params[1])
    index = float(params[2])
    
    print(f"Reproduciendo predicción: C={carrier:.2f}, R={ratio:.2f}, I={index:.2f}")
    
    # 1. Sintetizar (generamos 2 segundos para escucharlo bien)
    waveform, sr = fm_synthesize(carrier, ratio, index, duration=2.0)
    
    # 2. Reproducir
    play_audio(waveform, sr)

# Funciones para mostrar espectrogramas

def mostrar_espectrograma(wav, sample_rate, title):
    stft = librosa.stft(wav)
    spectrogram = np.abs(stft)

    # 3. Convertir a dB NORMALIZADO
    # ref=np.max es la CLAVE: Hace que el sonido más fuerte sea 0 dB
    S_db = librosa.amplitude_to_db(spectrogram)

    # 4. VISUALIZACIÓN    
    fig, ax = plt.subplots(figsize=(10, 4))

    librosa.display.specshow(
       S_db,
        y_axis='log',
        x_axis='time',
        sr=sample_rate,
        cmap='inferno',
        ax=ax
    )
    ax.axis('off')

    plt.title(title)
    plt.show()

def prediccion_multiples_wav(path_modelo, path_entrada, path_salida):
    os.makedirs(path_salida, exist_ok=True)


    lista_wavs = glob.glob(os.path.join(path_entrada, "*.wav"))
    print(f"hay {len(lista_wavs)} wavs. Empezando a procesar...")
    model, means, stds, device = cargar_modelo_para_inferencia(path_modelo, device="cuda")

    for ruta_wav_original in lista_wavs:
        
        prediccion = hacer_inferencia_rapida(model, means, stds, ruta_wav_original, device)

        p_carrier = prediccion[0]
        p_ratio   = prediccion[1]
        p_index   = prediccion[2]

        audio_prediccion, sr = fm_synthesize(p_carrier, p_ratio, p_index)

        nombre_archivo = os.path.basename(ruta_wav_original) 
        ruta_guardado = os.path.join(path_salida, nombre_archivo)

        sf.write(ruta_guardado, audio_prediccion, sr)
