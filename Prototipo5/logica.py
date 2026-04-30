import csv
import glob
import os
import shutil
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, random_split, Subset

from dataset import SpectrogramTensorDataset
from losses import HybridLoss, MultiScaleSpectralLoss
from models import CNNRegressor5, CNNRegressorSimple

# Función que convierte una onda en espectrograma
def procesar_espectrograma(waveform, sr=44100, device="cpu", spec_transform=None, db_transform=None, mode='stft'):
    """
    Normalización -> Espectrograma (STFT lineal o Mel) -> dB.
    Garantiza que el tensor tenga forma (1, 1, Freq, Tiempo) para el modelo.

    mode='stft' : 513 bins lineales (0–22050 Hz)
    mode='mel'  : 128 bandas Mel (escala logarítmica perceptual)
    """
    # 1. Asegurar formato Tensor y Device
    if not torch.is_tensor(waveform):
        waveform = torch.from_numpy(waveform).float()
    waveform = waveform.to(device)

    # 2. Fade in/out (50 ms) para suavizar bordes y evitar artefactos espectrales
    fade_samples = int(sr * 0.05)
    if fade_samples * 2 < waveform.shape[-1]:
        fade_in = torch.linspace(0, 1, fade_samples, device=device)
        fade_out = torch.linspace(1, 0, fade_samples, device=device)
        waveform = waveform.clone()
        waveform[..., :fade_samples] *= fade_in
        waveform[..., -fade_samples:] *= fade_out

    # 3. Normalización de pico (Peak Normalization)
    waveform = waveform / waveform.abs().max().clamp(min=1e-8)

    # 3. Transformada a dB (instanciada solo si no se proporciona por parámetro)
    if db_transform is None:
        db_transform = torchaudio.transforms.AmplitudeToDB(
            stype='amplitude', top_db=80.0
        ).to(device)

    if mode == 'mel':
        if spec_transform is None:
            spec_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=1024, hop_length=256, n_mels=128, power=1.0
            ).to(device)
        spec_db = db_transform(spec_transform(waveform))   # (mel_bins, T) -> dB
    else:   # stft
        if spec_transform is None:
            spec_transform = torchaudio.transforms.Spectrogram(
                n_fft=1024, hop_length=256, power=None, return_complex=True
            ).to(device)
        spec_db = db_transform(spec_transform(waveform).abs())  # (freq_bins, T) -> dB

    # 4. Ajuste de dimensiones para CNN (Batch=1, Canal=1, F, T)
    if spec_db.dim() == 2:
        spec_db = spec_db.unsqueeze(0)
    if spec_db.dim() == 3:
        spec_db = spec_db.unsqueeze(0)

    return spec_db  # (1, 1, Freq, Tiempo)

# ──────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE DATASET
# ──────────────────────────────────────────────────────────────────────────────

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
    
    # --- ITERACIÓN PRINCIPAL ---  El prototipo5 ya no funciona con rejilla + jitter, pues esto no es viable con 8 parámetros. Asi que se generan aleatoriamente.
    print(f"=== GENERACIÓN DE {num_muestras} WAVS ALEATORIOS (fase 1/2) ===")
    g = 0
    for g in range(1, num_muestras + 1):
        fname = f"fm_{g}.wav"
        file_path = os.path.join(out_path, fname)

        # Generación aleatoria uniforme dentro de los límites
        c_real = np.exp(np.random.uniform(np.log(params["carrier"][0]), np.log(params["carrier"][1])))
        r_real = np.random.uniform(*params["ratio"])
        i_real = np.random.uniform(*params["index"])

        # Envolvente amplitud: cada fase uniforme, rescalada si la suma excede TIME
        min_a = params["amp_attack"][0]
        a_att = np.random.uniform(*params["amp_attack"])
        a_sus = np.random.uniform(*params["amp_sustain"])
        a_dec = np.random.uniform(*params["amp_decay"])
        amp_sum = a_att + a_sus + a_dec
        if amp_sum > TIME:
            factor = (TIME - 3 * min_a) / (amp_sum - 3 * min_a)
            a_att = min_a + (a_att - min_a) * factor
            a_sus = min_a + (a_sus - min_a) * factor
            a_dec = min_a + (a_dec - min_a) * factor

        # Envolvente modulación: cada fase uniforme, rescalada si la suma excede TIME
        min_m = params["mod_attack"][0]
        m_att = np.random.uniform(*params["mod_attack"])
        m_dec = np.random.uniform(*params["mod_decay"])
        mod_sum = m_att + m_dec
        if mod_sum > TIME:
            factor = (TIME - 2 * min_m) / (mod_sum - 2 * min_m)
            m_att = min_m + (m_att - min_m) * factor
            m_dec = min_m + (m_dec - min_m) * factor

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
    per_file = elapsed / g
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"WAVs generados: {g}; carpeta: {out_path}")
    print(f"Tiempo total: {int(h)}h {int(m)}m {s:.2f}s  |  media: {per_file:.4f}s")

    return out_path

# 2. CONVERSIÓN WAV → TENSORES PYTORCH
def convertir_wavs_a_tensores(wav_folder, device, mode='stft'):
    # dirs
    t_start = time.time()
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_folder = os.path.join(main_dir, "Datasets", f"datasetFMespec_torchaudio_v5_{mode}")
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    wav_files = [f for f in os.listdir(wav_folder) if f.endswith(".wav")]
    n_wavs = len(wav_files)

    print(f"=== CONVERSIÓN WAV → TENSOR {mode.upper()} (fase 2/2) ===")
    print("WAV encontrados:", n_wavs)

    # Crear el transform UNA VEZ según el modo
    db_transform = torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=80.0).to(device)
    if mode == 'mel':
        spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100, n_fft=1024, hop_length=256, n_mels=128, power=1.0
        ).to(device)
    else:
        spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=1024, hop_length=256, power=None, return_complex=True
        ).to(device)

    # Transformación
    for i, wav_file in enumerate(wav_files, start=1):
        waveform, sr = torchaudio.load(os.path.join(wav_folder, wav_file))

        # Guardamos solo el tensor (C, F, T), quitando la dimensión de batch
        spec = procesar_espectrograma(waveform, sr, device, spec_transform, db_transform, mode=mode).squeeze(0)

        out_name = wav_file.replace(".wav", ".pt")
        torch.save(spec, os.path.join(out_folder, out_name))

        if i % 2000 == 0:
            print(f"Convertidos {i} tensores...")

    # Copiar labels.csv y guardar spec_mode.txt
    src_csv = os.path.join(wav_folder, "labels.csv")
    if os.path.exists(src_csv):
        shutil.copy(src_csv, os.path.join(out_folder, "labels.csv"))
    with open(os.path.join(out_folder, "spec_mode.txt"), "w") as f:
        f.write(mode)

    t_end = time.time()
    elapsed = t_end - t_start
    per_file = elapsed / n_wavs if n_wavs > 0 else 0.0
    print(f"Conversión completada en: {out_folder}")
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"Tiempo conversión total: {int(h)}h {int(m)}m {s:.2f}s  |  media por wav: {per_file:.4f}s")
    return out_folder

# FUNCIÓN PRINCIPAL
def generar_dataset(device, mode='stft'):
    start = time.time()

    wav_folder = generar_wavs_FM()
    tensor_folder = convertir_wavs_a_tensores(wav_folder, device, mode=mode)

    end = time.time()
    elapsed = end - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"=== DATASET COMPLETO GENERADO ({mode.upper()}) en {int(h)}h {int(m)}m {s:.2f}s ===")

    return {
        "tipo": "carpeta_tensores",
        "ruta": tensor_folder,
        "tensores": [f for f in os.listdir(tensor_folder) if f.endswith(".pt")],
        "spec_mode": mode
    }

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE DATASET EXISTENTE
# ──────────────────────────────────────────────────────────────────────────────

# Busca la carpeta y se asegura de que contiene tensores
def check_dataset(path):
    if not os.path.isdir(path):
        raise ValueError("Se esperaba una carpeta, no un archivo.")

    tensores = [
        f for f in os.listdir(path)
        if f.endswith((".pt", ".pth", ".tensor"))
    ]

    if not tensores:
        raise ValueError("No se encontraron tensores en la carpeta.")

    # Leer el modo de espectrograma (datasets antiguos no tienen este fichero → stft por defecto)
    mode_file = os.path.join(path, "spec_mode.txt")
    spec_mode = 'stft'
    if os.path.exists(mode_file):
        with open(mode_file) as f:
            spec_mode = f.read().strip()

    return {"tipo": "carpeta_tensores", "ruta": path, "tensores": tensores, "spec_mode": spec_mode}

# ──────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO DEL MODELO
# ──────────────────────────────────────────────────────────────────────────────

# Función encargada de instanciar y entrenar el modelo
def entrenar_modelo(nombreModelo, dataset_obj, epochs=10, batch_size=16, lr=1e-3, device="cuda", print_every_batches=100, spec_w=1.0, sc_w=0.5, param_w=0.05, arch='full', loss_fn='hybrid'):
    # dirs
    start = time.time()  
    tensors_dir = dataset_obj.get("ruta") 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)          
    save_dir = os.path.join(root_dir, "models")    # Te crea una carpeta models un nivel arriba (carpeta del proyecto)
    os.makedirs(save_dir, exist_ok=True)

    # --- Dataset ---
    dataset = SpectrogramTensorDataset(tensors_dir)

    # --- Instanciar modelo según arquitectura ---
    print(f"Instanciando modelo ({arch})!")
    if arch == 'simple':
        model = CNNRegressorSimple(8, 1, 32)
        criterion = torch.nn.SmoothL1Loss()
    else:
        model = CNNRegressor5(8, 1, 32)
        if loss_fn == 'multiscale':
            criterion = MultiScaleSpectralLoss(param_weight=param_w)
        else:
            criterion = HybridLoss(spec_weight=spec_w, sc_weight=sc_w, param_weight=param_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Entrenamiento ---
    print(f"Entrenando modelo!       Usando {device}")
    # Split 70/15/15 con semilla fija para reproducibilidad
    generator = torch.Generator().manual_seed(42)
    train_size = int(len(dataset) * 0.70)
    val_size   = int(len(dataset) * 0.15)
    test_size  = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    history = model.fit(train_loader, val_loader=val_loader, device=device, epochs=epochs, print_every_batches=print_every_batches, criterion=criterion, optimizer=optimizer)

    # --- Guardar modelo ---
    save_path = os.path.join(save_dir, nombreModelo)

    # -- Guardar CHECKPOINT stats, modo de espectrograma y arquitectura ---
    stats = dataset.get_stats()
    spec_mode = dataset_obj.get('spec_mode', 'stft')
    checkpoint = {
        'state_dict': model.state_dict(),
        'param_means': stats['means'],
        'param_stds': stats['stds'],
        'spec_mode': spec_mode,
        'arch': arch,
        'test_indices': list(test_dataset.indices),
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint guardado  |  espectrograma: {spec_mode}  |  arquitectura: {arch}")

    end = time.time()
    elapsed = end - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"Entrenamiento finalizado. Modelo guardado en: {save_path}")
    print(f"Tiempo de entrenamiento: {int(h)}h {int(m)}m {s:.2f}s")

    return save_path  #Retorna: path completo al archivo .pth guardado (string).

# ──────────────────────────────────────────────────────────────────────────────
# INFERENCIA
# ──────────────────────────────────────────────────────────────────────────────
# Carga el modelo y las estadísticas de normalización desde el checkpoint.
# Se llama una sola vez; el resultado se reutiliza en cada llamada a hacer_inferencia().
def cargar_modelo_para_inferencia(ruta_modelo, device="cpu"):
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError("No se encuentra el archivo del modelo.")

    device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    ckpt = torch.load(ruta_modelo, map_location=device)

    state_dict = ckpt['state_dict']
    means      = np.asarray(ckpt['param_means'], dtype=np.float32)
    stds       = np.asarray(ckpt['param_stds'],  dtype=np.float32)
    spec_mode  = ckpt.get('spec_mode', 'stft')
    arch       = ckpt.get('arch', 'full')

    # Instanciar la arquitectura correcta
    if arch == 'simple':
        model = CNNRegressorSimple(n_params=8)
    else:
        model = CNNRegressor5(n_params=8)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Modelo cargado en: {device}  |  espectrograma: {spec_mode}  |  arquitectura: {arch}")
    return model, means, stds, device, spec_mode

# Dado un modelo ya cargado y la ruta de un WAV, devuelve los 8 parámetros FM en escala real.
def hacer_inferencia(model, means, stds, ruta_wav, device, mode='stft'):
    waveform, sr = torchaudio.load(ruta_wav)
    spec = procesar_espectrograma(waveform, sr, device, mode=mode)

    with torch.no_grad():
        out = model(spec)
        pred_params = out[0] if isinstance(out, (tuple, list)) else out

    pred_raw = pred_params.detach().cpu().numpy().flatten() * stds + means
    return pred_raw.tolist()

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
    model, means, stds, device, spec_mode = cargar_modelo_para_inferencia(path_modelo, device="cuda")

    for ruta_wav_original in lista_wavs:

        prediccion = hacer_inferencia(model, means, stds, ruta_wav_original, device, mode=spec_mode)

        p_carrier, p_ratio, p_index, p_a_att, p_a_sus, p_a_dec, p_m_att, p_m_dec = prediccion

        audio_prediccion, sr = fm_synthesize(p_carrier, p_ratio, p_index, p_a_att, p_a_sus, p_a_dec, p_m_att, p_m_dec)

        nombre_archivo = os.path.basename(ruta_wav_original) 
        ruta_guardado = os.path.join(path_salida, nombre_archivo)

        sf.write(ruta_guardado, audio_prediccion, sr)

# ──────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN
# ──────────────────────────────────────────────────────────────────────────────

# Evalúa el modelo sobre el test set guardado en el checkpoint. Usa el método evaluate de CNNRegressor5
def evaluar_modelo(ruta_modelo, tensor_folder, device="cpu"):
    ''' BOTÓN EVALUAR '''
    # Carga modelo para obtener los indices de test
    model, means, stds, device, _ = cargar_modelo_para_inferencia(ruta_modelo, device)
    ckpt = torch.load(ruta_modelo, map_location='cpu')
    test_indices = ckpt['test_indices']
    dataset = SpectrogramTensorDataset(tensor_folder)
    test_dataset = Subset(dataset, test_indices)
    print(f"Evaluando sobre test set ({len(test_indices)} muestras, separadas antes del entrenamiento)")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Llama a funcion evaluate del modelo
    metrics = model.evaluate(test_loader, device=str(device), means=means, stds=stds, synth_fn=fm_synthesize)
    return metrics

# Sonidos FM representativos que cubren el espacio tímbrico del sintetizador
# Carriers (graves/agudos), ratios (armónicos/inarmónicos), índices (limpio/ruidoso) y envolventes (percusivo/pad/pluck).
BENCHMARK_SOUNDS = [
    {"name": "bajo_limpio",       "params": [100,  1.0, 1.0, 0.02, 0.50, 0.50, 0.01, 0.50]},
    {"name": "bajo_rico",         "params": [100,  2.0, 8.0, 0.02, 0.40, 0.60, 0.01, 0.40]},
    {"name": "medio_armonico",    "params": [440,  1.0, 3.0, 0.05, 0.40, 0.50, 0.05, 0.40]},
    {"name": "medio_inarmonico",  "params": [440,  1.5, 5.0, 0.05, 0.30, 0.60, 0.05, 0.30]},
    {"name": "agudo_limpio",      "params": [1200, 1.0, 1.0, 0.02, 0.30, 0.40, 0.01, 0.30]},
    {"name": "agudo_ruidoso",     "params": [1200, 1.9, 9.5, 0.02, 0.20, 0.30, 0.01, 0.20]},
    {"name": "percusivo",         "params": [300,  1.0, 5.0, 0.02, 0.05, 0.20, 0.01, 0.10]},
    {"name": "pad_lento",         "params": [300,  1.0, 2.0, 0.80, 0.80, 0.40, 0.60, 0.40]},
    {"name": "pluck",             "params": [600,  2.0, 4.0, 0.02, 0.10, 0.80, 0.01, 0.60]},
    {"name": "muy_bajo",          "params": [100,  0.5, 3.0, 0.02, 0.40, 0.50, 0.02, 0.40]},
    {"name": "muy_agudo",         "params": [1800, 1.0, 2.0, 0.02, 0.30, 0.40, 0.02, 0.30]},
    {"name": "indice_maximo",     "params": [500,  1.0, 9.5, 0.05, 0.40, 0.50, 0.05, 0.40]},
]

PARAM_NAMES = ["carrier", "ratio", "index", "amp_att", "amp_sus", "amp_dec", "mod_att", "mod_dec"]

# Calcula Mel L1 y MCD entre dos señales de audio
def _audio_metrics(audio_orig, audio_pred, sr):
    # Mel L1: L1 sobre log del espectrograma mel de amplitud (power=1.0, coherente con el entrenamiento).
    mel_o = librosa.feature.melspectrogram(y=audio_orig, sr=sr, n_fft=1024, hop_length=256, n_mels=128, power=1.0)
    mel_p = librosa.feature.melspectrogram(y=audio_pred, sr=sr, n_fft=1024, hop_length=256, n_mels=128, power=1.0)
    mel_l1 = float(np.mean(np.abs(np.log1p(mel_o) - np.log1p(mel_p))))

    # MCD: Mel-Cepstral Distortion en dB, coeficientes 1-12
    mfcc_o = librosa.feature.mfcc(y=audio_orig, sr=sr, n_mfcc=13)[1:]  
    mfcc_p = librosa.feature.mfcc(y=audio_pred, sr=sr, n_mfcc=13)[1:]
    min_t  = min(mfcc_o.shape[1], mfcc_p.shape[1])
    diff_c = mfcc_o[:, :min_t] - mfcc_p[:, :min_t]
    mcd    = float((10 / np.log(10)) * np.mean(np.sqrt(2 * np.sum(diff_c ** 2, axis=0))))

    return mel_l1, mcd

# Evalúa 12 sonidos de referencia con Mel L1, MCD y param_mae.
# Genera WAVs original/predicho en benchmark/ y benchmark_metrics.png.
def evaluar_benchmark_diverso(ruta_modelo, device="cpu"):
    ''' BOTÓN BENCHMARK DIVERSO '''
    model, means, stds, device, spec_mode = cargar_modelo_para_inferencia(ruta_modelo, device)

    script_dir    = os.path.dirname(os.path.abspath(__file__))
    benchmark_dir = os.path.join(script_dir, "benchmark")
    if os.path.exists(benchmark_dir):
        shutil.rmtree(benchmark_dir)
    os.makedirs(benchmark_dir)

    # Clamp a rangos válidos: coherentes con el dominio de entrenamiento (GEN_PARAMS)
    param_mins = np.array([100.0, 0.05, 1.0, 0.015, 0.015, 0.015, 0.01, 0.01], dtype=np.float32)
    param_maxs = np.array([2000.0, 2.0, 10.0, 1.9, 1.9, 1.9, 1.9, 1.9], dtype=np.float32)

    results = []
    print(f"\n=== BENCHMARK DIVERSO ({len(BENCHMARK_SOUNDS)} sonidos) ===\n")
    print(f"  {'Sonido':<22} {'Mel L1':>8}  {'MCD (dB)':>10}")
    print("  " + "-" * 44)

    for idx, sound in enumerate(BENCHMARK_SOUNDS, start=1):
        true_params = sound["params"]
        name        = f"{idx:02d}_{sound['name']}"

        # Sintetizar original e inferir
        audio_orig, sr = fm_synthesize(*true_params, duration=2.0)
        waveform       = torch.from_numpy(audio_orig).unsqueeze(0)
        spec           = procesar_espectrograma(waveform, sr, str(device), mode=spec_mode)
        with torch.no_grad():
            out       = model(spec)
            pred_norm = out[0] if isinstance(out, (tuple, list)) else out
        pred_params = np.clip(pred_norm.cpu().numpy().flatten() * stds + means, param_mins, param_maxs).tolist()

        # Re-sintetizar con params predichos
        audio_pred, sr_pred = fm_synthesize(*[float(p) for p in pred_params], duration=2.0)

        sf.write(os.path.join(benchmark_dir, f"{name}_original.wav"),   audio_orig, sr,      subtype='PCM_16')
        sf.write(os.path.join(benchmark_dir, f"{name}_prediccion.wav"), audio_pred, sr_pred, subtype='PCM_16')

        mel_l1, mcd = _audio_metrics(audio_orig, audio_pred, sr)
        param_mae = {n: abs(t - p) for n, t, p in zip(PARAM_NAMES, true_params, pred_params)}

        results.append({
            "name": name, "true_params": true_params, "pred_params": pred_params,
            "mel_l1": mel_l1, "mcd": mcd, "param_mae": param_mae,
        })
        print(f"  {name:<22} {mel_l1:>8.4f}  {mcd:>10.4f}")

    mel_l1_vals = [r["mel_l1"] for r in results]
    mcd_vals    = [r["mcd"]    for r in results]
    print("  " + "-" * 44)
    print(f"  {'MEDIA':<22} {np.mean(mel_l1_vals):>8.4f}  {np.mean(mcd_vals):>10.4f}")

    # Plot: barras Mel L1 y MCD por sonido
    names  = [r["name"].split("_", 1)[1] for r in results]
    x      = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].bar(x, mel_l1_vals, color='steelblue', edgecolor='black')
    axes[0].axhline(np.mean(mel_l1_vals), color='red', linestyle='--', label=f'media={np.mean(mel_l1_vals):.3f}')
    axes[0].set_ylabel("Mel L1")
    axes[0].set_title("Benchmark — Mel L1 por sonido")
    axes[0].legend()

    axes[1].bar(x, mcd_vals, color='darkorange', edgecolor='black')
    axes[1].axhline(np.mean(mcd_vals), color='red', linestyle='--', label=f'media={np.mean(mcd_vals):.3f}')
    axes[1].set_ylabel("MCD (dB)")
    axes[1].set_title("Benchmark — MCD por sonido")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha='right')
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(benchmark_dir, "benchmark_metrics.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()

    print(f"\nAudios y gráfica guardados en: {benchmark_dir}")
    return results, benchmark_dir

# Función que muestra los 4 espectrogramas
def comparar_espectrogramas_4en1(wav_orig, sr_orig, params_pred, device="cpu", mode='stft'):
    # 1. Sintetizar la predicción
    wav_pred, sr_pred = fm_synthesize(*[float(p) for p in params_pred], duration=2.0)

    # 2. Obtener matrices numéricas (mismo modo que el modelo)
    s_orig_tensor = procesar_espectrograma(wav_orig, sr_orig, device, mode=mode).cpu().squeeze()
    s_pred_tensor = procesar_espectrograma(wav_pred, sr_pred, device, mode=mode).cpu().squeeze()

    mode_label = "Mel (log freq)" if mode == 'mel' else "STFT (freq lineal)"

    s_orig = s_orig_tensor.numpy()
    s_pred = s_pred_tensor.numpy()

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'Comparativa de Espectrogramas: Original vs Predicción  [{mode_label}]', fontsize=16, fontweight='bold')

    # --- FILA 0: ORIGINAL ---
    # Eje de frecuencia logarítmico (perceptual)
    librosa.display.specshow(s_orig, y_axis='log', x_axis='time', sr=sr_orig, hop_length=256, cmap='inferno', ax=axs[0, 0])
    axs[0, 0].set_title("Original: eje frecuencia logarítmico (Hz)")
    axs[0, 0].set_xlabel("Tiempo (s)")
    axs[0, 0].set_ylabel("Frecuencia (Hz)")

    # Eje de frecuencia lineal (bins)
    im1 = axs[0, 1].imshow(s_orig, origin='lower', aspect='auto', cmap='inferno')
    axs[0, 1].set_title("Original: eje frecuencia lineal (bins)")
    axs[0, 1].set_xlabel("Tiempo (frames STFT)")
    axs[0, 1].set_ylabel("Bins de frecuencia")
    fig.colorbar(im1, ax=axs[0, 1], format='%+2.0f dB')

    # --- FILA 1: PREDICCIÓN ---
    librosa.display.specshow(s_pred, y_axis='log', x_axis='time', sr=sr_pred, hop_length=256, cmap='inferno', ax=axs[1, 0])
    axs[1, 0].set_title("Predicción: eje frecuencia logarítmico (Hz)")
    axs[1, 0].set_xlabel("Tiempo (s)")
    axs[1, 0].set_ylabel("Frecuencia (Hz)")

    im2 = axs[1, 1].imshow(s_pred, origin='lower', aspect='auto', cmap='inferno')
    axs[1, 1].set_title("Predicción: eje frecuencia lineal (bins)")
    axs[1, 1].set_xlabel("Tiempo (frames STFT)")
    axs[1, 1].set_ylabel("Bins de frecuencia")
    fig.colorbar(im2, ax=axs[1, 1], format='%+2.0f dB')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()