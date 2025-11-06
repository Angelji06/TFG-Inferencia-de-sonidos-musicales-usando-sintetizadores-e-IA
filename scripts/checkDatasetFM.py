import pandas as pd

# -----------------------------------------------------------------------
#  SCRIPT PARA COMPROBAR ELEMENTOS QUE FALTAN EN EL DATASETFM
# -----------------------------------------------------------------------

# Ruta al CSV
csv_path = 'C:/Users/dvcen/Documents/GitHub/TFG-Inferencia-de-sonidos-musicales-usando-sintetizadores-e-IA/Datasets/datasetwav/labels.csv'

# Cargar el CSV
df = pd.read_csv(csv_path)

# Extraer y normalizar los nombres
nombres_csv = df['filename'].astype(str).str.strip().str.lower()

# Generar la lista esperada con extensión .wav
esperados = [f'pru_{i}.wav' for i in range(1, 15201)]

# Detectar faltantes
faltantes = [nombre for nombre in esperados if nombre not in nombres_csv.values]

# Mostrar resultados
print(f"Total faltantes: {len(faltantes)}")
for nombre in faltantes:
    print(f"FALTA {nombre}")
print(f"Total faltantes: {len(faltantes)}")
