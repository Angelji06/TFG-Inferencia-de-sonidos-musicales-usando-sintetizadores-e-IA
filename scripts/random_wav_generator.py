import os
import numpy as np
import soundfile as sf
import pandas as pd
import random
from tqdm import tqdm

# ================= CONFIGURACIÓN =================
CANTIDAD = 2000            # Cantidad suficiente para FAD
CARPETA_SALIDA = "./Datasets/dataset_random"

# IMPORTANTE: Igual que tu entrenamiento
SAMPLE_RATE = 44100        
DURATION = 1.0             

# --- TUS PARÁMETROS EXACTOS ---
# carrier: (100, 2000, 100) -> Min 100, Max 2000
RANGO_CARRIER = (100.0, 2000.0)

# ratio: (0.05, 2, 0.05) -> Min 0.05, Max 2
RANGO_RATIO   = (0.05, 2.0)

# index: (1, 10, 0.5) -> Min 1, Max 10
RANGO_INDEX   = (1.0, 10.0)
# =================================================

def fm_synthesize(carrier, ratio, index, duration, sr):
    # Vector de tiempo
    t = np.linspace(0, duration, int(sr * duration), endpoint=False).astype(np.float32)
    
    # Cálculo de FM
    fm = carrier * ratio
    modulator = np.sin(2 * np.pi * fm * t)
    
    # Señal final
    audio = np.sin(2 * np.pi * carrier * t + index * modulator)
    
    return audio

def main():
    # Limpiar/Crear carpeta
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)
    
    data_log = [] 
    
    print(f"Generando {CANTIDAD} archivos para Test Ciego...")
    print(f"Rangos: Carrier={RANGO_CARRIER}, Ratio={RANGO_RATIO}, Index={RANGO_INDEX}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    
    for i in tqdm(range(CANTIDAD)):
        # 1. Aleatorio Uniforme (Cualquier valor decimal dentro del rango)
        # Esto prueba si el modelo interpola bien, no solo si memoriza los pasos.
        c = random.uniform(*RANGO_CARRIER)
        r = random.uniform(*RANGO_RATIO)
        idx = random.uniform(*RANGO_INDEX)
        
        # 2. Sintetizar
        audio = fm_synthesize(c, r, idx, DURATION, SAMPLE_RATE)
        
        # 3. Guardar WAV
        # Usamos subtype PCM_16 para máxima compatibilidad (igual que tu entrenamiento)
        nombre_archivo = f"blind_test_{i:04d}.wav"
        ruta_completa = os.path.join(CARPETA_SALIDA, nombre_archivo)
        sf.write(ruta_completa, audio, SAMPLE_RATE, subtype='PCM_16')
        
        # 4. Guardar datos (Ground Truth)
        data_log.append({
            "filename": nombre_archivo,
            "carrier": c,
            "ratio": r,
            "index": idx
        })
        
    # Guardar CSV
    df = pd.DataFrame(data_log)
    df.to_csv(os.path.join(CARPETA_SALIDA, "metadata.csv"), index=False)
    
    print("¡Generación completada!")
    print(f"Carpeta lista para FAD: {CARPETA_SALIDA}")

if __name__ == "__main__":
    main()