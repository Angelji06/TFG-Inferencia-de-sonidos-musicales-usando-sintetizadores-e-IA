import os

ruta_tensores = r"C:\Users\David\Documents\GitHub\TFG-Inferencia-de-sonidos-musicales-usando-sintetizadores-e-IA\Datasets\datasetFMespec_torchaudio_v5"

archivos = os.listdir(ruta_tensores)
corruptos = 0

for f in archivos:
    if f.endswith('.pt'):
        ruta_completa = os.path.join(ruta_tensores, f)
        # Si el archivo pesa 0 bytes, está corrupto
        if os.path.getsize(ruta_completa) == 0:
            print(f"Borrando archivo corrupto: {f}")
            os.remove(ruta_completa)
            corruptos += 1

print(f"Limpieza completada. Se borraron {corruptos} archivos corruptos.")