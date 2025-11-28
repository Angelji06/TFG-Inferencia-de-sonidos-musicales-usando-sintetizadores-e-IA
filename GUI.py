# Para utilizar esta GUI resulta muy útil tener guardado el modelo preentrenado desde el notebook y así no tener que estar entrenando todo el rato desde aquí 

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import logicaParaGUI as LG 
import librosa

model = None  # Variable global para guardar el modelo entrenado
current_wav = None
last_pred = None
wav_cargado = None
sample_rate = 44100

def entrenar_o_cargar_modelo():
    global model
    btn_train.config(state="disabled")
    lbl_status.config(text="Procesando...")

    usar_guardado = var_guardar.get()  # True si checkbox está activado

    def train_thread():
        global model
        model = LG.iniciar_modelo(usar_modelo_guardado=usar_guardado, epochs=5)
        lbl_status.config(text="Modelo listo.")
        btn_train.config(state="normal")
    
    threading.Thread(target=train_thread).start()

def cargar_wav_y_predecir():
    global model, current_wav, last_pred, wav_cargado, sample_rate

    if model is None:
        messagebox.showerror("Error", "Primero entrena el modelo.")
        return
    
    wav_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    
    if wav_path:
        current_wav = wav_path
        lbl_status.config(text=f"Prediciendo para: {wav_path}")
        pred = LG.predict_wav(model, wav_path)
        last_pred = pred  # Guardamos la predicción para poder reproducirla
        lbl_result.config(text=f"Predicción: {pred}")
        wav_cargado, sample_rate = librosa.load(current_wav, sr=None)


# --- Interfaz ---
root = tk.Tk()
root.title("Interfaz IA FM")
btn_train = tk.Button(root, text="Entrenar o cargar modelo", command=entrenar_o_cargar_modelo, width=30)
btn_train.pack(pady=10)

# Checkbox para usar modelo guardado
var_guardar = tk.BooleanVar(value=True)
chk_guardar = tk.Checkbutton(root, text="Usar modelo guardado si existe", variable=var_guardar)
chk_guardar.pack(pady=5)

btn_predict = tk.Button(root, text="Cargar WAV y predecir", command=cargar_wav_y_predecir, width=30)
btn_predict.pack(pady=10)

lbl_status = tk.Label(root, text="Estado: Esperando acción...")
lbl_status.pack(pady=10)

lbl_result = tk.Label(root, text="Predicción: N/A")
lbl_result.pack(pady=10)

# botones para la reproducción de sonidos y comparación auditiva

btn_play_original = tk.Button(root, text="Escuchar WAV original", command=lambda: LG.reproducir_wav(current_wav))
btn_play_original.pack(pady=5)

btn_play_pred = tk.Button(root, text="Escuchar predicción FM", command=lambda: LG.reproducir_prediccion(last_pred))
btn_play_pred.pack(pady=5)

btn_watch_spec_original = tk.Button(root, text="Ver espectrograma original", command=lambda: LG.mostrar_espectrograma(wav_cargado, sample_rate))
btn_watch_spec_original.pack(pady=5)

btn_watch_spec_prediccion = tk.Button(root, text="Ver espectrograma prediccion", command=lambda: LG.mostrar_espectrograma_prediccion(last_pred))
btn_watch_spec_prediccion.pack(pady=5)

root.mainloop()