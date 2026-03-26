import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Importamos tu función de síntesis (asegúrate de que el nombre coincida)
from logica import fm_synthesize 

def calcular_error_mfcc(mfcc_objetivo_mean, audio_generado, sr, n_mfcc=13):
    """
    Extrae los MFCCs del audio generado, los promedia en el tiempo 
    y calcula el Error Cuadrático Medio (MSE) respecto al objetivo.
    """
    # 1. Extraer MFCCs (devuelve una matriz de n_mfcc x frames_de_tiempo)
    mfcc_gen = librosa.feature.mfcc(y=audio_generado, sr=sr, n_mfcc=n_mfcc)
    
    # 2. Promediar en el eje del tiempo (axis=1) para obtener el "perfil tímbrico" global
    mfcc_gen_mean = np.mean(mfcc_gen, axis=1)
    
    # 3. Calcular la distancia (MSE) entre los dos perfiles
    error = np.mean((mfcc_objetivo_mean - mfcc_gen_mean) ** 2)
    return error

def grid_search_fm_mfcc(ruta_audio_objetivo, sr=44100, duration=1.0):
    print(f"🎙️ Analizando: {os.path.basename(ruta_audio_objetivo)}")
    
    # 1. Cargar el audio real (Ground Truth)
    y_target, _ = librosa.load(ruta_audio_objetivo, sr=sr, duration=duration)
    
    # Pre-calcular el perfil MFCC del objetivo UNA SOLA VEZ
    mfcc_target = librosa.feature.mfcc(y=y_target, sr=sr, n_mfcc=13)
    mfcc_target_mean = np.mean(mfcc_target, axis=1) # Vector de 13 valores
    
    # 2. Definir tu espacio de búsqueda
    carriers = np.arange(100, 2000 + 100, 100)
    ratios = np.arange(0.05, 2.0 + 0.05, 0.05)
    indexes = np.arange(1.0, 10.0 + 0.5, 0.5)
    
    total_combinaciones = len(carriers) * len(ratios) * len(indexes)
    print(f"🔍 Iniciando fuerza bruta con MFCC: {total_combinaciones} combinaciones...")
    
    mejores_candidatos = [] 
    
    # 3. El Triple Bucle
    for c in tqdm(carriers, desc="Explorando Carriers"):
        for r in ratios:
            for i in indexes:
                
                # A. Generar el audio
                audio_test, _ = fm_synthesize(c, r, i, duration=duration, sr=sr)
                
                # B. Comparar usando nuestra nueva métrica MFCC
                error = calcular_error_mfcc(mfcc_target_mean, audio_test, sr, n_mfcc=13)
                
                # C. Guardar el resultado
                mejores_candidatos.append((error, c, r, i))

    # 4. Ordenar los resultados de menor a mayor error
    mejores_candidatos.sort(key=lambda x: x[0])
    
    return mejores_candidatos[:3] # Devolvemos el Top 3

def main():
    # --- Configurar la ventana de selección de archivo ---
    root = tk.Tk()
    root.withdraw() 
    
    print("Abriendo ventana para seleccionar el audio...")
    ruta_audio = filedialog.askopenfilename(
        title="Selecciona el WAV a evaluar (MFCC)",
        filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")]
    )
    
    if not ruta_audio:
        print("❌ Operación cancelada. No se seleccionó ningún archivo.")
        return

    # --- Ejecutar la Búsqueda ---
    top_3 = grid_search_fm_mfcc(ruta_audio)

    # --- Guardar y mostrar resultados ---
    carpeta_salida = "./resultados_grid_search_mfcc"
    os.makedirs(carpeta_salida, exist_ok=True)
    
    print("\n" + "="*50)
    print("🏆 ¡BÚSQUEDA COMPLETADA! TOP 3 MEJORES COMBINACIONES (MFCC)")
    print("="*50)
    
    nombre_base = os.path.splitext(os.path.basename(ruta_audio))[0]
    
    for rank in range(3):
        err, c, r, idx = top_3[rank]
        print(f"#{rank+1} -> Error MFCC: {err:.4f} | Carrier: {c:.1f}, Ratio: {r:.2f}, Index: {idx:.1f}")
        
        # Generamos el audio ganador para guardarlo
        audio_ganador, sr = fm_synthesize(c, r, idx, duration=1.0, sr=44100)
        
        # Guardamos en disco
        ruta_guardado = os.path.join(carpeta_salida, f"{nombre_base}_top{rank+1}_C{c:.0f}_R{r:.2f}_I{idx:.1f}_mfcc.wav")
        sf.write(ruta_guardado, audio_ganador, sr, subtype='PCM_16')

    print("\n✅ Los audios ganadores se han guardado en la carpeta:")
    print(f"   {os.path.abspath(carpeta_salida)}")

if __name__ == "__main__":
    main()