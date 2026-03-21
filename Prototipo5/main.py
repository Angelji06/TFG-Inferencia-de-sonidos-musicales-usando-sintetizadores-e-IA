import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
from logica import get_gen_params, generar_dataset, check_dataset, entrenar_modelo, reproducir_wav,reproducir_prediccion, hacer_inferencia, prediccion_multiples_wav, comparar_espectrogramas_4en1

from Prototipo5 import CNNRegressor5
import librosa
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa.display
#import simpleaudio as sa
 
class App:
    def __init__(self, root):
        self.root = root
        root.title("Predictor de parámetros de sintetizador FM")
        
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        w = int(screen_w * 0.9)
        h = int(screen_h * 0.75)
        root.geometry(f"{w}x{h}")

        # Estado compartido
        self.dataset_obj = None
        self.pathDataset = None
        self.model_trained = False
        self.nombreModelo = None
        self.pathModelo = None

        # Parámetros de entrenamiento por defecto
        self.train_epochs = 5
        self.train_lr = 1e-3
        self.train_batch_size = 32
        self.train_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_print_every = 50

        # Páginas
        self.page_inicio = tk.Frame(root)
        self.page_entrenamiento = tk.Frame(root)
        self.page_test = tk.Frame(root)

        self._build_inicio()
        self._build_entrenamiento()
        self._build_test()

        self.show_page(self.page_inicio)        

    def show_page(self, page):
        self.refresh_inicio_status()
        self.refresh_entrenamiento_page()
        if hasattr(self, "refresh_test_page"):
            self.refresh_test_page()
        # ocultar todas y mostrar la solicitada
        for p in (self.page_inicio, self.page_entrenamiento, self.page_test):
            p.pack_forget()
        page.pack(fill="both", expand=True, padx=12, pady=12)

    # ================================ Página Inicio ================================
    def _build_inicio(self):
        p = self.page_inicio

        tk.Label(p, text="Inicio", font=("Arial", 18)).pack(pady=(8,12))
        tk.Label(p, text="¿Qué quieres hacer?", font=("Arial", 12)).pack(pady=(0,8))

        btn_frame = tk.Frame(p)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame,text="Usar modelo existente",width=28,command=self._seleccionar_modelo_existente).grid(row=0, column=0, padx=8, pady=6)

        tk.Button(btn_frame,text="Entrenar modelo",width=28,command=lambda: self._ir_entrenamiento()).grid(row=0, column=1, padx=8, pady=6)

        # Crear dos StringVar independientes (se crean una vez)
        self.dataset_status_var = tk.StringVar()
        self.modelo_status_var  = tk.StringVar()

        # Etiquetas que mostrarán esos StringVar
        tk.Label(p, textvariable=self.dataset_status_var).pack(pady=(16,0))
        tk.Label(p, textvariable=self.modelo_status_var).pack(pady=(0,0))

        # Botón Test
        self.btn_inicio_ir_test = tk.Button(
            p,
            text="Ir a TEST",
            state="disabled",
            command=lambda: self.show_page(self.page_test)
        )
        self.btn_inicio_ir_test.pack(pady=(10, 0))

        # Inicializar su contenido
        self.refresh_inicio_status()

    def refresh_inicio_status(self):
        """Actualiza el texto de las dos etiquetas de inicio."""
        etiquetaDataset = self.pathDataset if self.dataset_obj is not None else "Ninguno"
        etiquetaModelo  = getattr(self, "nombreModelo", None) if self.model_trained else None
        etiquetaModelo  = etiquetaModelo or "Ninguno"

        self.dataset_status_var.set(f"Dataset: {etiquetaDataset}")
        self.modelo_status_var.set(f"Modelo: {etiquetaModelo}")

        # Activar o desactivar botón TEST
        if self.model_trained:
            self.btn_inicio_ir_test.config(state="normal")
        else:
            self.btn_inicio_ir_test.config(state="disabled")

    def _ir_entrenamiento(self):
        self.show_page(self.page_entrenamiento)

    def _seleccionar_modelo_existente(self):
        path = filedialog.askopenfilename(
            title="Selecciona un modelo",
            filetypes=[("Modelos PyTorch", "*.pth"), ("Todos los archivos", "*.*")]
        )
        if not path:
            return  # usuario canceló

        self.pathModelo = path
        self.nombreModelo = os.path.basename(path)

        # Guardamos la selección como modelo cargado
        self.model_trained = True  # para permitir acceso a TEST

        self.refresh_inicio_status()

        # Ir a TEST
        self.show_page(self.page_test)

    # ================================ Página Entrenamiento ================================
    def _build_entrenamiento(self):
        p = self.page_entrenamiento

        tk.Label(p, text="Entrenamiento", font=("Arial", 16)).pack(pady=(6,10))

        # Seccion dispositivo
        device_frame = tk.LabelFrame(p, text="Device", padx=8, pady=8)
        device_frame.pack(fill="x", padx=6, pady=(4,10))

        tk.Label(device_frame, text="Device:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.device_var = tk.StringVar(value=self.train_device)
        device_menu = tk.OptionMenu(device_frame, self.device_var, "cuda", "cpu")
        device_menu.config(width=8)
        device_menu.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        # Sección superior: generar / cargar dataset
        top_frame = tk.LabelFrame(p, text="Dataset (generar o cargar)", padx=8, pady=8)
        top_frame.pack(fill="x", padx=6, pady=(4,10))

        self.btn_generar_ds = tk.Button(top_frame, text="Generar dataset", width=20, command=self._generar_dataset)
        self.btn_generar_ds.grid(row=0, column=0, padx=6, pady=6)

        self.btn_cargar_ds = tk.Button(top_frame, text="Cargar dataset", width=20,command=self._cargar_dataset)
        self.btn_cargar_ds.grid(row=0, column=1, padx=6, pady=6)

        tk.Label(top_frame, textvariable=self.dataset_status_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=(6,0))

        # Controles de hiperparámetros (nuevo)
        params_frame = tk.LabelFrame(p, text="Parámetros de entrenamiento", padx=8, pady=8)
        params_frame.pack(fill="x", padx=6, pady=(4,8))

         # Nombre opcional del modelo
        tk.Label(params_frame, text="Nombre modelo (.pth):").grid(row=4, column=0, sticky="w")
        self.hp_name_var = tk.StringVar(value="prueba.pth")   
        tk.Entry(params_frame, textvariable=self.hp_name_var, width=20).grid(row=4, column=1, padx=4)

        # Epochs
        tk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.entry_epochs = tk.Spinbox(params_frame, from_=1, to=1000, width=8)
        self.entry_epochs.delete(0, "end")
        self.entry_epochs.insert(0, str(self.train_epochs))
        self.entry_epochs.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        # Learning rate
        tk.Label(params_frame, text="LR:").grid(row=0, column=2, sticky="e", padx=4, pady=4)
        self.entry_lr = tk.Entry(params_frame, width=12)
        self.entry_lr.insert(0, str(self.train_lr))
        self.entry_lr.grid(row=0, column=3, sticky="w", padx=4, pady=4)

        # Batch size
        tk.Label(params_frame, text="Batch size:").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        self.entry_batch = tk.Spinbox(params_frame, from_=1, to=1024, width=8)
        self.entry_batch.delete(0, "end")
        self.entry_batch.insert(0, str(self.train_batch_size))
        self.entry_batch.grid(row=1, column=1, sticky="w", padx=4, pady=4)

        # print_every_batches
        tk.Label(params_frame, text="Print every (batches):").grid(row=2, column=0, sticky="e", padx=4, pady=4)
        self.entry_print_every = tk.Spinbox(params_frame, from_=0, to=10000, width=8)
        self.entry_print_every.delete(0, "end")
        self.entry_print_every.insert(0, str(self.train_print_every))
        self.entry_print_every.grid(row=2, column=1, sticky="w", padx=4, pady=4)

        # Sección inferior: entrenar con dataset existente
        bottom_frame = tk.LabelFrame(p, text="Entrenar modelo (requiere dataset)", padx=8, pady=8)
        bottom_frame.pack(fill="both", expand=True, padx=6, pady=(4,10))

        self.btn_entrenar = tk.Button(bottom_frame, text="Entrenar modelo", state="disabled",width=20, command=self._entrenar_modelo)
        self.btn_entrenar.pack(pady=(8,6))

        tk.Label(bottom_frame, textvariable=self.modelo_status_var).pack(pady=(4,0))

        # Botón para ir a Test (solo habilitado cuando model_trained=True)
        self.btn_ir_test = tk.Button(p, text="Ir a TEST", state="disabled", command=lambda: self.show_page(self.page_test))
        self.btn_ir_test.pack(pady=(6,4))

        # Botón volver
        tk.Button(p, text="Volver", command=lambda: self.show_page(self.page_inicio)).pack(side="bottom", pady=8)
        self.refresh_entrenamiento_page()

    def refresh_entrenamiento_page(self):
        """
        Actualiza los textos y el estado de botones de la página de entrenamiento
        en función del estado actual de self.dataset_obj, self.pathDataset,
        self.model_trained, self.nombreModelo, etc.
        """
        # --- Estado dataset ---
        if self.dataset_obj is None:
            self.dataset_status_var.set("Dataset: pendiente")
            self.btn_entrenar.config(state="disabled")
        else:
            self.pathDataset = self.dataset_obj.get('ruta') 
            self.dataset_status_var.set(f"Dataset: {self.pathDataset}")
            self.btn_entrenar.config(state="normal")

        # --- Estado entrenamiento/modelo ---
        if not getattr(self, "model_trained", False):
            self.modelo_status_var.set("Modelo: Ninguno")
            self.btn_ir_test.config(state="disabled")
        else:
            self.modelo_status_var.set(f"Modelo listo: {self.nombreModelo }")
            self.btn_ir_test.config(state="normal")

    def _generar_dataset(self):
        try:
            ds = generar_dataset(self.device_var.get())
            self.dataset_obj = ds
            self.pathDataset = ds.get('ruta')

            self.refresh_entrenamiento_page()
            self.btn_entrenar.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar dataset:\n{e}")

    def _cargar_dataset(self):
        path = filedialog.askdirectory(title="Selecciona carpeta con tensores")
        self.pathDataset = path
        if not path:
            return

        try:
            ds = check_dataset(path)
            self.dataset_obj = ds
            self.pathDataset = path

            self.refresh_entrenamiento_page()

            self.btn_entrenar.config(state="normal")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la carpeta:\n{e}")

    def _entrenar_modelo(self):
        if self.dataset_obj is None:
            messagebox.showwarning("Aviso", "No hay dataset para entrenar.")
            return

        # Leer y validar hiperparámetros desde la GUI
        try:
            nombre = str(self.hp_name_var.get())
            epochs = int(self.entry_epochs.get())
            lr = float(self.entry_lr.get())
            batch_size = int(self.entry_batch.get())
            device = str(self.device_var.get())
            print_every = int(self.entry_print_every.get())
        except Exception as e:
            messagebox.showerror("Error", f"Parámetros inválidos: {e}")
            return

        # Guardar en atributos por si quieres reutilizarlos
        self.nombreModelo = nombre
        self.train_epochs = epochs
        self.train_lr = lr
        self.train_batch_size = batch_size
        self.train_device = device
        self.train_print_every = print_every

        # Desactivar botón mientras entrena (bloqueante en este hilo)
        self.btn_entrenar.config(state="disabled")
        self.btn_generar_ds.config(state="disabled")
        self.btn_cargar_ds.config(state="disabled")

        try:
            result = entrenar_modelo(self.nombreModelo, self.dataset_obj, epochs=epochs, lr=lr, batch_size=batch_size, device=device, print_every_batches=print_every)
            
            self.pathModelo = result
            self.nombreModelo = os.path.basename(result)

            self.model_trained = True

            # Actualizar etiquetas y botones
            self.refresh_entrenamiento_page()
            self.refresh_inicio_status()

            messagebox.showinfo("Entrenamiento", "Entrenamiento finalizado correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el entrenamiento:\n{e}")
        finally:
            # volver a activar botones
            self.btn_entrenar.config(state="normal")
            self.btn_generar_ds.config(state="normal")
            self.btn_cargar_ds.config(state="normal")

    # ================================ Página Test ================================

    def _build_test(self):
        p = self.page_test
        tk.Label(p, text="Test", font=("Arial", 18)).pack(pady=(12,8))
        tk.Label(p, text="Selecciona un WAV, pulsa 'Predecir WAV' y luego reproduce.", font=("Arial", 12)).pack(pady=(0,12))

        frame = tk.Frame(p)
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        # 1. Info Modelo
        model_frame = tk.Frame(frame)
        model_frame.pack(fill="x", pady=6)
        tk.Label(model_frame, text="Modelo cargado:").pack(side="left", padx=(0,4))
        # Usamos un Label dinámico para poder actualizar su texto si cambia el modelo
        self.label_model_test_var = tk.StringVar(value="Ninguno")
        tk.Label(model_frame, textvariable=self.label_model_test_var).pack(side="left", padx=4)

        # 2. Selección de WAV
        wav_frame = tk.Frame(frame)
        wav_frame.pack(fill="x", pady=6)
        tk.Button(wav_frame, text="Seleccionar WAV", command=self._seleccionar_wav).pack(side="left", padx=6)
        self.label_wav_selected = tk.Label(wav_frame, text="WAV: Ninguno")
        self.label_wav_selected.pack(side="left", padx=6)

        # Llama a la función para generar el lote de wavs para FAD
        tk.Button(wav_frame, text="Generar Eval Set (FAD)", bg="#e1f5fe", command=self._predecir_2000_wavs).pack(side="right", padx=6)

        # 4. Botones de Acción
        action_frame = tk.Frame(frame)
        action_frame.pack(fill="x", pady=(8,6))
        
        self.btn_predict = tk.Button(action_frame, text="Predecir WAV", state="disabled", command=self._predecir_wav)
        self.btn_predict.pack(side="left", padx=6)
        
        tk.Button(action_frame, text="Reproducir original", command=self._play_original).pack(side="left", padx=6)
        tk.Button(action_frame, text="Reproducir síntesis", command=self._play_synth).pack(side="left", padx=6)

        tk.Button(action_frame, text="Visualizar Espectrogramas", command=self._ver_comparativa_espectrogramas,).pack(side="left", padx=6)
        
        tk.Button(action_frame, text="Volver", command=lambda: self.show_page(self.page_inicio)).pack(side="right", padx=6)

        # 5. Área de Resultados
        res_frame = tk.LabelFrame(frame, text="Resultado")
        res_frame.pack(fill="both", expand=True, pady=6)
        self.result_text = tk.Text(res_frame, height=12)
        self.result_text.pack(fill="both", expand=True)

        # Estado interno para reproducción
        self.test_wav_path = None
        self.last_prediction_params = None

        # Actualizar nombre del modelo al entrar (por si acaso)
        self.refresh_test_page()

    def refresh_test_page(self):
        """Actualiza la etiqueta del modelo en la página de test."""
        nombre = getattr(self, "nombreModelo", "Ninguno")
        if hasattr(self, "label_model_test_var"):
            self.label_model_test_var.set(nombre)

    def _seleccionar_wav(self):
        path = filedialog.askopenfilename(title="Selecciona WAV", filetypes=[("WAV files", "*.wav"), ("All", "*.*")])
        if not path:
            return
        self.test_wav_path = path
        self.label_wav_selected.config(text=f"WAV: {os.path.basename(path)}")
        self.btn_predict.config(state="normal")
        
        # Limpiar resultados anteriores
        self.last_prediction_params = None
        self.result_text.delete("1.0", tk.END)

    def _predecir_wav(self):
        # UI Feedback
        self.btn_predict.config(state="disabled") 
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Procesando inferencia...\n")
        
        # Refresco visual
        if hasattr(self, 'root'): self.root.update_idletasks() 

        try:
            # 1. Hacer inferencia
            params = hacer_inferencia(self.pathModelo, self.test_wav_path, device = self.device_var.get())
              
            # 2. GUARDAR DATOS PARA EL BOTÓN DE ESPECTROGRAMA
            self.last_prediction_params = params
            self.ultimo_wav_orig, self.ultimo_sr_orig = librosa.load(self.test_wav_path, sr=44100)

            # Comprobar si el audio está dentro del conjunto de entrenamiento
            nombre = os.path.basename(self.test_wav_path) 
            if nombre[:3] == "fm_":  # De momento simplemente revisando el nombre del archivo
                ruta_csv = os.path.join(os.path.dirname(os.getcwd()), 'Datasets', 'datasetFMwav_v5', 'labels.csv')
                df_labels = pd.read_csv(ruta_csv) 
                fila = df_labels[df_labels['filename'] == nombre]

                val_fc = fila.iloc[0]['carrier']
                val_fm = fila.iloc[0]['ratio']
                val_i  = fila.iloc[0]['index']
                val_aa = fila.iloc[0]['amp_attack']
                val_ad = fila.iloc[0]['amp_decay']
                val_ma = fila.iloc[0]['mod_attack']
                val_md = fila.iloc[0]['mod_decay']
                
                texto_ori = (
                    f"Audio en Dataset Entrenamiento! Valores Originales: \n"
                    f"Carrier (fc): {val_fc:.2f}\n"
                    f"Ratio (fm/fc): {val_fm:.2f}\n"
                    f"Index (I):    {val_i:.2f}\n"
                    f"Amp Attack:   {val_aa:.3f}\n"
                    f"Amp Decay:    {val_ad:.3f}\n"
                    f"Mod Attack:   {val_ma:.3f}\n"
                    f"Mod Decay:    {val_md:.3f}\n\n"
                )
                self.result_text.insert(tk.END, texto_ori)
            else:
                GENparams = get_gen_params() # Devuelve la copia de GEN_PARAMS

                c_min, c_max = GENparams['carrier'][0], GENparams['carrier'][1]
                r_min, r_max = GENparams['ratio'][0], GENparams['ratio'][1]
                i_min, i_max = GENparams['index'][0], GENparams['index'][1]
                aa_min, aa_max = GENparams['amp_attack'][0], GENparams['amp_attack'][1]
                ad_min, ad_max = GENparams['amp_decay'][0], GENparams['amp_decay'][1]
                ma_min, ma_max = GENparams['mod_attack'][0], GENparams['mod_attack'][1]
                md_min, md_max = GENparams['mod_decay'][0], GENparams['mod_decay'][1]

                texto_ori = (
                    f"Audio NO en Dataset Entrenamiento!\n"
                    f"Recuerda el Dominio (Rango de entrenamiento):\n"
                    f"Carrier (Fc): {c_min} - {c_max} Hz\n"
                    f"Ratio (h):    {r_min} - {r_max}\n"
                    f"Index (I):    {i_min} - {i_max}\n"
                    f"Amp Attack:   {aa_min} - {aa_max} s\n"
                    f"Amp Decay:    {ad_min} - {ad_max} s\n"
                    f"Mod Attack:   {ma_min} - {ma_max} s\n"
                    f"Mod Decay:    {md_min} - {md_max} s\n\n"
                )

                self.result_text.insert(tk.END, texto_ori)

            # Mostrar texto en pantalla
            c, r, i, a_att, a_dec, m_att, m_dec = params
            texto_pre = (
                f"--- PREDICCIÓN ---\n"
                f"Carrier (fc):  {c:.2f} Hz\n"
                f"Ratio (fm/fc): {r:.2f}\n"
                f"Index (I):     {i:.2f}\n"
                f"Amp Attack:    {a_att:.3f} s\n"
                f"Amp Decay:     {a_dec:.3f} s\n"
                f"Mod Attack:    {m_att:.3f} s\n"
                f"Mod Decay:     {m_dec:.3f} s\n"
                f"\n(Nota: Si los valores son extraños, recuerda que pueden estar fuera del dominio de entrenamiento)"
            )
            self.result_text.insert(tk.END, texto_pre)

        except Exception as e:
            self.result_text.insert(tk.END, f"ERROR:\n{str(e)}\n")
            print(f"ERROR: {e}")
        finally:
            self.btn_predict.config(state="normal")

    def _ver_comparativa_espectrogramas(self):
        if hasattr(self, 'last_prediction_params'):
            comparar_espectrogramas_4en1(self.ultimo_wav_orig, self.ultimo_sr_orig, self.last_prediction_params, self.device_var.get())
        else:
            messagebox.showwarning("Aviso", "Primero debes predecir un audio.")

    def _play_synth(self):
        reproducir_prediccion(self.last_prediction_params)

    def _play_original(self):
        if not self.test_wav_path:
            return
            
        try:
            reproducir_wav(self.test_wav_path)
        except Exception as e:
            messagebox.showerror("Error Audio", f"No se pudo reproducir el original:\n{e}")

    # bucle para la predicción de 2000 wavs. Los usaremos para evaluar el modelo con FAD
    def _predecir_2000_wavs(self):
        carpeta_input = filedialog.askdirectory(
            title="Selecciona la carpeta con los wavs"
        )

        carpeta_output = carpeta_input + "_predicciones"


        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Iniciando generación masiva para FAD...\nMira la consola para ver el progreso.\n")
        self.root.update_idletasks()

        try:
            # Llamamos a tu lógica pasando la ruta CORRECTA del modelo
            # Nota: Asumo que prediccion_multiples_wav gestiona internamente las carpetas de entrada/salida
            # o que pregunta por ellas. Si necesita argumentos, añádelos aquí.
            prediccion_multiples_wav(self.pathModelo, carpeta_input, carpeta_output) 
            
            messagebox.showinfo("Éxito", "Generación de audios completada.")
            self.result_text.insert(tk.END, "¡Generación completada!\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Falló la generación masiva:\n{e}")
            self.result_text.insert(tk.END, f"Error: {e}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
