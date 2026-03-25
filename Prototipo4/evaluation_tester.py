import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Importamos tu función de síntesis (asegúrate de que el nombre coincida con el de tu logica.py)
from logica import fm_synthesize 

def calcular_error_mel(mel_objetivo, audio_generado, sr):
    """
    Calcula el Error Cuadrático Medio (MSE) entre los Espectrogramas Mel en decibelios.
    """
    mel_gen = librosa.feature.melspectrogram(y=audio_generado, sr=sr, n_mels=128)
    mel_gen_db = librosa.power_to_db(mel_gen, ref=np.max)
    error = np.mean((mel_objetivo - mel_gen_db) ** 2)
    return error

def grid_search_fm(ruta_audio_objetivo, sr=44100, duration=1.0):
    print(f"🎙️ Analizando: {os.path.basename(ruta_audio_objetivo)}")
    
    # 1. Cargar el audio real (Ground Truth)
    y_target, _ = librosa.load(ruta_audio_objetivo, sr=sr, duration=duration)
    
    # Pre-calcular el Mel del objetivo UNA SOLA VEZ
    mel_target = librosa.feature.melspectrogram(y=y_target, sr=sr, n_mels=128)
    mel_target_db = librosa.power_to_db(mel_target, ref=np.max)
    
    # 2. Definir tu espacio de búsqueda
    carriers = np.arange(100, 2000 + 100, 100)
    ratios = np.arange(0.05, 2.0 + 0.05, 0.05)
    indexes = np.arange(1.0, 10.0 + 0.5, 0.5)
    
    total_combinaciones = len(carriers) * len(ratios) * len(indexes)
    print(f"🔍 Iniciando fuerza bruta: {total_combinaciones} combinaciones...")
    
    mejores_candidatos = [] 
    
    # 3. El Triple Bucle
    for c in tqdm(carriers, desc="Explorando Carriers"):
        for r in ratios:
            for i in indexes:
                
                # A. Generar el audio
                # (Ajusta la llamada si tu fm_synthesize devuelve cosas distintas)
                audio_test, _ = fm_synthesize(c, r, i, duration=duration, sr=sr)
                
                # B. Comparar usando nuestra métrica
                error = calcular_error_mel(mel_target_db, audio_test, sr)
                
                # C. Guardar el resultado
                mejores_candidatos.append((error, c, r, i))

    # 4. Ordenar los resultados de menor a mayor error
    mejores_candidatos.sort(key=lambda x: x[0])
    
    return mejores_candidatos[:3] # Devolvemos solo el Top 3

def main():
    # --- Configurar la ventana de selección de archivo ---
    root = tk.Tk()
    root.withdraw() # Oculta la ventana principal vacía de Tkinter
    
    print("Abriendo ventana para seleccionar el audio...")
    ruta_audio = filedialog.askopenfilename(
        title="Selecciona el WAV a evaluar",
        filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")]
    )
    
    if not ruta_audio:
        print("❌ Operación cancelada. No se seleccionó ningún archivo.")
        return

    # --- Ejecutar la Búsqueda ---
    top_3 = grid_search_fm(ruta_audio)

    # --- Guardar y mostrar resultados ---
    carpeta_salida = "./resultados_grid_search"
    os.makedirs(carpeta_salida, exist_ok=True)
    
    print("\n" + "="*50)
    print("🏆 ¡BÚSQUEDA COMPLETADA! TOP 3 MEJORES COMBINACIONES")
    print("="*50)
    
    nombre_base = os.path.splitext(os.path.basename(ruta_audio))[0]
    
    for rank in range(3):
        err, c, r, idx = top_3[rank]
        print(f"#{rank+1} -> Error Mel: {err:.4f} | Carrier: {c:.1f}, Ratio: {r:.2f}, Index: {idx:.1f}")
        
        # Generamos el audio ganador para guardarlo
        audio_ganador, sr = fm_synthesize(c, r, idx, duration=1.0, sr=44100)
        
        # Guardamos en disco
        ruta_guardado = os.path.join(carpeta_salida, f"{nombre_base}_top{rank+1}_C{c:.0f}_R{r:.2f}_I{idx:.1f}.wav")
        sf.write(ruta_guardado, audio_ganador, sr, subtype='PCM_16')

    print("\n✅ Los audios ganadores se han guardado en la carpeta:")
    print(f"   {os.path.abspath(carpeta_salida)}")

if __name__ == "__main__":
    main()