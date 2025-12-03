import torch
import torchaudio
import soundfile
import sounddevice
import matplotlib
import pandas

print("--- DIAGNÓSTICO DEL SISTEMA ---")
print(f"CUDA Disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo GPU: {torch.cuda.get_device_name(0)}")
print("Librerías cargadas correctamente.")