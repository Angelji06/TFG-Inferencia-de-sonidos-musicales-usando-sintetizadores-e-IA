import pandas as pd
import os

# =============================================
#   Archivo para pequeñas utilidades varias
# =============================================

def get_true_labels(wav_filename, labels_csv_path="Datasets/datasetFMwav/labels.csv"):
    """
    Busca los valores reales de carrier, ratio e index para un archivo .wav dado.
    """
    # Cargar CSV
    df = pd.read_csv(labels_csv_path)

    # Normalizar nombre del archivo
    wav_filename = os.path.basename(wav_filename).strip().lower()

    # Buscar fila correspondiente
    match = df[df["filename"].str.lower() == wav_filename]

    if match.empty:
        raise ValueError(f"No se encontró '{wav_filename}' en {labels_csv_path}")

    row = match.iloc[0]
    return row["carrier"], row["ratio"], row["index"]
