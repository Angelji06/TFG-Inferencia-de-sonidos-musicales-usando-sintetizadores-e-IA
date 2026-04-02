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

from SpectrogramTensorDataset5 import SpectrogramTensorDataset
from Prototipo5 import CNNRegressor5, HybridLoss 

# Función que convierte una onda en espectrograma
def procesar_espectrograma(waveform, sr=44100, device="cpu", spec_transform=None, db_transform=None):
    """
    Abstracción total: Normalización -> STFT -> Magnitud -> DB.
    Garantiza que el tensor tenga forma (1, Freq, Tiempo) para el modelo.
    """
    # 1. Asegurar formato Tensor y Device
    if not torch.is_tensor(waveform):
        waveform = torch.from_numpy(waveform).float()
    
    waveform = waveform.to(device)

    # 2. Normalización de pico (Peak Normalization)
    waveform = waveform / waveform.abs().max().clamp(min=1e-8)

    # 3. Configuración de Transformada idéntica a entrenamiento (Instanciada solo si no se proporcionan por parámetro)
    if spec_transform is None:
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=1024, hop_length=256, power=None, return_complex=True
        ).to(device)
    
    if db_transform is None:
        db_transform = torchaudio.transforms.AmplitudeToDB(
            stype='amplitude', top_db=80.0
        ).to(device)

    # 4. Cálculo
    spec_complex = spec_transform(waveform)
    mag = spec_complex.abs()
    spec_db = db_transform(mag) # Resultado en dB

    # 5. Ajuste de dimensiones para CNN (Batch=1, Canal=1, F, T)
    if spec_db.dim() == 2: # Si es (F, T)
        spec_db = spec_db.unsqueeze(0) 
    if spec_db.dim() == 3: # Si es (C, F, T)
        spec_db = spec_db.unsqueeze(0)

    return spec_db # Retorna (1, 1, Freq, Tiempo)

#==============================================================================================================
#=================================== GENERACION DE DATASET ====================================================
#==============================================================================================================

GEN_PARAMS = {
    "carrier":      (100, 2000),   # Frecuencia portadora
    "ratio":        (0.05, 2),     # Relación de frecuencias entre la portadora y la moduladora
    "index":        (1, 10),       # Indice de modulación
    "amp_attack":   (0.015, 1.9),  # Envolvente amplitud  (att + sus + dec libre entre 0.3s y 2.0s)
    "amp_sustain":  (0.015, 1.9),
    "amp_decay":    (0.015, 1.9),
    "mod_attack":   (0.01, 1.9),   # Envolvente modulación (att + dec libre entre 0.2s y 2.0s)
    "mod_decay":    (0.01, 1.9)
}

def get_gen_params():
    return GEN_PARAMS.copy()

# Funcion para generar una envolvente con attack, sustain y decay
def generar_envolvente(t, attack, sustain, decay):
    env = np.zeros_like(t, dtype=np.float32)

    # Fase de Ataque (0 → 1)
    idx_a = t <= attack
    if attack > 0:
        env[idx_a] = t[idx_a] / attack
    else:
        env[idx_a] = 1.0

    # Fase de Sustain (mantiene 1.0)
    idx_s = (t > attack) & (t <= attack + sustain)
    env[idx_s] = 1.0

    # Fase de Decaimiento (1 → 0)
    idx_d = (t > attack + sustain) & (t <= attack + sustain + decay)
    if decay > 0:
        env[idx_d] = 1.0 - (t[idx_d] - attack - sustain) / decay

    return env

# Genera la señal de audio sintética usando fórmulas FM.
def fm_synthesize(carrier, ratio, index, a_att, a_sus, a_dec, m_att, m_dec, duration=2, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False).astype(np.float32)

    # Envolventes
    amp_env = generar_envolvente(t, a_att, a_sus, a_dec)      # amplitud: attack + sustain + decay
    mod_env = generar_envolvente(t, m_att, 0.0, m_dec)        # modulación: sin sustain

    # Sintesis
    fm = carrier * ratio
    mod = np.sin(2 * np.pi * fm * t)
    car = amp_env * np.sin(2 * np.pi * carrier * t + (index * mod_env) * mod)

    # sounddevice prefiere float32 para el audio
    return car.astype(np.float32), sr

# 1. GENERACIÓN DE WAVs FM + CSV DE ETIQUETAS CON BARRIDO CON NUMPY
def generar_wavs_FM(num_muestras=30000):   # conviene que este valor se pueda ajustar en un futuro desde la gui
    # dirs
    t_start = time.time()  
    script_dir = os.path.dirname(os.path.abspath(__file__))   
    main_dir = os.path.dirname(script_dir)
    out_path = os.path.join(main_dir, "Datasets", "datasetFMwav_v5")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)
    csv_path = os.path.join(out_path, "labels.csv")          
    csv_buffer = []         # Buffer para el guardado en el csv
    buffer_flush = 1000 
    with open(csv_path, "w", newline="") as f_header:
        csv.writer(f_header).writerow(["filename","carrier","ratio","index","amp_attack","amp_sustain","amp_decay","mod_attack","mod_decay"])

    # params
    params = GEN_PARAMS
    SR, TIME = 44100, 2 #subo un poquito la duración para que se noten bien las envolventes
    
    # --- ITERACIÓN PRINCIPAL ---  El prototipo5 ya no funciona con rejilla + jitter, pues esto no es viable con 7 parámetros. Asi que se generan aleatoriamente.
    print(f"=== GENERACIÓN DE {num_muestras} WAVS ALEATORIOS (fase 1/2) ===")
    g = 0
    for g in range(1, num_muestras + 1):
        fname = f"fm_{g}.wav"
        file_path = os.path.join(out_path, fname)

        # Generación aleatoria uniforme dentro de los límites
        c_real = np.random.uniform(*params["carrier"])
        r_real = np.random.uniform(*params["ratio"])
        i_real = np.random.uniform(*params["index"])

        # Envolvente amplitud: duración total libre (0.3s – TIME), el resto del clip queda en silencio
        total_amp = np.random.uniform(0.3, TIME)
        a_fracs = np.random.uniform(0.05, 1.0, 3)  # 3 pesos aleatorios, mínimo 0.05 para evitar fases nulas
        a_fracs /= a_fracs.sum()                    # normalizar → proporciones que suman 1
        a_att, a_sus, a_dec = a_fracs * total_amp   # escalar al tiempo total: att + sus + dec = total_amp

        # Envolvente modulación: duración total libre (0.2s – TIME)
        total_mod = np.random.uniform(0.2, TIME)
        m_frac = np.random.uniform(0.05, 0.95)  # proporción del attack; con 2 fases basta un número
        m_att = m_frac * total_mod               # att = fracción del total
        m_dec = (1.0 - m_frac) * total_mod      # dec = resto; att + dec = total_mod

        # Sintesis
        x, _ = fm_synthesize(c_real, r_real, i_real, a_att, a_sus, a_dec, m_att, m_dec, duration=TIME, sr=SR)

        # Guardado wav
        sf.write(file_path, x, SR, subtype='PCM_16')

        # Registrado en csv
        csv_buffer.append([fname, c_real, r_real, i_real, a_att, a_sus, a_dec, m_att, m_dec])
        if len(csv_buffer) >= buffer_flush:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerows(csv_buffer)
            csv_buffer = []

        if g % 2000 == 0:
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
    # dirs
    t_start = time.time()  
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_folder = os.path.join(main_dir, "Datasets", "datasetFMespec_torchaudio_v5")
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    wav_files = [f for f in os.listdir(wav_folder) if f.endswith(".wav")]
    n_wavs = len(wav_files)

    print("=== CONVERSIÓN WAV → TENSOR (fase 2/2) ===")
    print("WAV encontrados:", n_wavs)

    # Crear el transform UNA VEZ (STFT)
    spec_transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, power=None, return_complex=True).to(device)
    db_transform = torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=80.0).to(device)

    # Transformación
    for i, wav_file in enumerate(wav_files, start=1):
        # Carga el tensor onda en CPU
        waveform, sr = torchaudio.load(os.path.join(wav_folder, wav_file))

        # Fade
        fade_samples = int(sr * 0.05)
        if fade_samples * 2 < waveform.shape[-1]:
            fade_in = torch.linspace(0, 1, fade_samples)
            fade_out = torch.linspace(1, 0, fade_samples)
            waveform[:, :fade_samples] *= fade_in
            waveform[:, -fade_samples:] *= fade_out

        # Guardamos solo el tensor (C, F, T), quitando la dimensión de batch
        spec = procesar_espectrograma(waveform, sr, device, spec_transform, db_transform).squeeze(0)

        # Guardar tensor espectrograma
        out_name = wav_file.replace(".wav", ".pt")
        torch.save(spec, os.path.join(out_folder, out_name))

        if i % 2000 == 0:
            print(f"Convertidos {i}/ tensores...")

    # Copiar labels.csv a la carpeta de tensores
    src_csv = os.path.join(wav_folder, "labels.csv")
    if os.path.exists(src_csv):
        shutil.copy(src_csv, os.path.join(out_folder, "labels.csv"))

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
def entrenar_modelo(nombreModelo, dataset_obj, epochs=10, batch_size=16, lr=1e-3, device="cuda", print_every_batches=100, spec_w=1.0, sc_w=0.5, param_w=0.05):
    # dirs
    start = time.time()  
    tensors_dir = dataset_obj.get("ruta") 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)          
    save_dir = os.path.join(root_dir, "models")    # Te crea una carpeta models un nivel arriba (carpeta del proyecto)
    os.makedirs(save_dir, exist_ok=True)

    # --- Dataset y DataLoader ---
    dataset = SpectrogramTensorDataset(tensors_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Instanciar modelo ---
    print("Instanciando modelo!")
    model = CNNRegressor5(8,1,32)

    criterion = HybridLoss(spec_weight=spec_w, sc_weight=sc_w, param_weight=param_w) 

    # --- Entrenamiento ---
    print(f"Entrenando modelo!       Usando {device}")
    train_size = int(len(dataset) * 0.8)    # 80% train, 20% val
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # Lo dividimos aleatoriamente
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    history = model.fit(train_loader, val_loader=val_loader, device=device, epochs=epochs, lr=lr, print_every_batches=print_every_batches, criterion=criterion)

    # --- Guardar modelo ---
    save_path = os.path.join(save_dir , nombreModelo)

    # -- Guardar stats ---
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
# ------------ INFERENCIA ------------
# Carga el modelo y las estadísticas de normalización desde el checkpoint.
# Se llama una sola vez; el resultado se reutiliza en cada llamada a hacer_inferencia().
def cargar_modelo_para_inferencia(ruta_modelo, device="cpu"):
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError("No se encuentra el archivo del modelo.")

    device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    model = CNNRegressor5(n_params=8)
    ckpt = torch.load(ruta_modelo, map_location=device)

    # Soporte para checkpoints nuevos {state_dict, means, stds} y antiguos (solo state_dict)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        means = np.asarray(ckpt['param_means'], dtype=np.float32)
        stds  = np.asarray(ckpt['param_stds'],  dtype=np.float32)
    else:
        state_dict = ckpt
        means = stds = None

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Modelo cargado exitosamente en:", device)
    return model, means, stds, device

# Dado un modelo ya cargado y la ruta de un WAV, devuelve los 8 parámetros FM en escala real.
def hacer_inferencia(model, means, stds, ruta_wav, device):
    if means is None or stds is None:
        raise RuntimeError("El checkpoint no contiene 'param_means'/'param_stds'. Reentrena guardando stats en el checkpoint.")

    waveform, sr = torchaudio.load(ruta_wav)
    spec = procesar_espectrograma(waveform, sr, device)

    with torch.no_grad():
        out = model(spec)
        pred_params = out[0] if isinstance(out, (tuple, list)) else out

    pred_raw = pred_params.detach().cpu().numpy().flatten() * stds + means
    return pred_raw.tolist()

# --------- REPRODUCCIÓN DE AUDIO ---------
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
    ratio   = float(params[1])
    index   = float(params[2])
    a_att   = float(params[3])
    a_sus   = float(params[4])
    a_dec   = float(params[5])
    m_att   = float(params[6])
    m_dec   = float(params[7])

    print(f"Reproduciendo predicción: C={carrier:.2f}, R={ratio:.2f}, I={index:.2f}, AmpAtt={a_att:.2f}, AmpSus={a_sus:.2f}, AmpDec={a_dec:.2f}, ModAtt={m_att:.2f}, ModDec={m_dec:.2f}")

    # 1. Sintetizar (generamos 2 segundos para escucharlo bien)
    waveform, sr = fm_synthesize(carrier, ratio, index, a_att, a_sus, a_dec, m_att, m_dec, duration=2.0)
    
    # 2. Reproducir
    play_audio(waveform, sr)

def prediccion_multiples_wav(path_modelo, path_entrada, path_salida):
    os.makedirs(path_salida, exist_ok=True)


    lista_wavs = glob.glob(os.path.join(path_entrada, "*.wav"))
    print(f"hay {len(lista_wavs)} wavs. Empezando a procesar...")
    model, means, stds, device = cargar_modelo_para_inferencia(path_modelo, device="cuda")

    for ruta_wav_original in lista_wavs:

        prediccion = hacer_inferencia(model, means, stds, ruta_wav_original, device)

        p_carrier, p_ratio, p_index, p_a_att, p_a_sus, p_a_dec, p_m_att, p_m_dec = prediccion

        audio_prediccion, sr = fm_synthesize(p_carrier, p_ratio, p_index, p_a_att, p_a_sus, p_a_dec, p_m_att, p_m_dec)

        nombre_archivo = os.path.basename(ruta_wav_original) 
        ruta_guardado = os.path.join(path_salida, nombre_archivo)

        sf.write(ruta_guardado, audio_prediccion, sr)

def generar_carpeta_prueba(tensor_folder, n=100):
    """
    Copia los primeros n tensores (.pt) y el labels.csv de tensor_folder
    a una nueva carpeta <tensor_folder>_test<n>.
    Devuelve la ruta de la carpeta generada.
    """
    files = sorted([f for f in os.listdir(tensor_folder) if f.endswith('.pt')])[:n]
    out_folder = tensor_folder + f"_test{n}"
    os.makedirs(out_folder, exist_ok=True)
    for f in files:
        shutil.copy(os.path.join(tensor_folder, f), os.path.join(out_folder, f))
    src_csv = os.path.join(tensor_folder, "labels.csv")
    if os.path.exists(src_csv):
        shutil.copy(src_csv, os.path.join(out_folder, "labels.csv"))
    print(f"Carpeta de prueba generada: {out_folder} ({len(files)} tensores)")
    return out_folder

# Evalúa el modelo sobre todos los tensores de tensor_folder. Usa el método evaluate de CNNRegressor5
def evaluar_modelo(ruta_modelo, tensor_folder, device="cpu"):
    model, _, _, device = cargar_modelo_para_inferencia(ruta_modelo, device)
    dataset  = SpectrogramTensorDataset(tensor_folder)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    metrics = model.evaluate(test_loader, device=str(device))
    return metrics

# Función que muestra los 4 espectrogramas
def comparar_espectrogramas_4en1(wav_orig, sr_orig, params_pred, device="cpu"):
    # 1. Sintetizar la predicción
    wav_pred, sr_pred = fm_synthesize(*[float(p) for p in params_pred], duration=2.0)

    # 2. Obtener matrices numéricas
    s_orig_tensor = procesar_espectrograma(wav_orig, sr_orig, device).cpu().squeeze()
    s_pred_tensor = procesar_espectrograma(wav_pred, sr_pred, device).cpu().squeeze()

    s_orig = s_orig_tensor.numpy()
    s_pred = s_pred_tensor.numpy()

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Comparativa de Espectrogramas: Original vs Predicción', fontsize=16, fontweight='bold')

    # --- FILA 0: ORIGINAL ---
    # Librosa Log
    librosa.display.specshow(s_orig, y_axis='log', x_axis='time', sr=sr_orig, hop_length=256, cmap='inferno', ax=axs[0, 0])
    axs[0, 0].set_title("Original: Escala Logarítmica")
    axs[0, 0].set_xlabel("Tiempo (s)") # <--- Unidad: Segundos
    axs[0, 0].set_ylabel("Frecuencia (Hz)")

    # Torchaudio Lineal
    im1 = axs[0, 1].imshow(s_orig, origin='lower', aspect='auto', cmap='inferno')
    axs[0, 1].set_title("Original: Escala Lineal")
    axs[0, 1].set_xlabel("Tiempo (Frames / STFT Windows)") # <--- Unidad: Frames
    axs[0, 1].set_ylabel("Bins de Frecuencia (0-512)")
    fig.colorbar(im1, ax=axs[0, 1], format='%+2.0f dB')

    # --- FILA 1: PREDICCIÓN ---
    # Librosa Log
    librosa.display.specshow(s_pred, y_axis='log', x_axis='time', sr=sr_pred, hop_length=256, cmap='inferno', ax=axs[1, 0])
    axs[1, 0].set_title("Predicción: Escala Logarítmica")
    axs[1, 0].set_xlabel("Tiempo (s)")
    axs[1, 0].set_ylabel("Frecuencia (Hz)")

    # Torchaudio Lineal
    im2 = axs[1, 1].imshow(s_pred, origin='lower', aspect='auto', cmap='inferno')
    axs[1, 1].set_title("Predicción: Escala Lineal")
    axs[1, 1].set_xlabel("Tiempo (Frames / STFT Windows)")
    axs[1, 1].set_ylabel("Bins de Frecuencia (0-512)")
    fig.colorbar(im2, ax=axs[1, 1], format='%+2.0f dB')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()