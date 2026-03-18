import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import numpy as np

# Clase hija de Dataset (pytorch)
# Normaliza los valores para evitar que el carrier (el parámetro con valores más grandes) domine la pérdida
class SpectrogramTensorDataset(Dataset):
    def __init__(self, tensors_dir, param_cols=('carrier', 'ratio', 'index','amp_attack', 'amp_decay', 'mod_attack', 'mod_decay')):
        self.tensors_dir = tensors_dir
        self.param_cols = list(param_cols)

        # Localizar CSV (se asume la estructura: parent_dir/datasetFMwav/labels.csv)
        labels_csv = os.path.join(os.path.dirname(tensors_dir), "datasetFMwav_v5", "labels.csv")
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"labels.csv no encontrado en {labels_csv}")
        df = pd.read_csv(labels_csv)

        # Calcular valores para normalización
        vals = df[self.param_cols].astype(np.float32).values
        self.param_means = vals.mean(axis=0).astype(np.float32)  # Se hace la media de cada parametro
        stds = vals.std(axis=0).astype(np.float32)               # Se calcula la desviación estándar de cada parámetro
        stds[stds == 0] = 1.0   # evitar división por 0
        self.param_stds = stds

        # Itera los nombres de los archivos, los guarda en raw y luego normaliza
        self.labels = {}
        for _, row in df.iterrows():
            key = os.path.splitext(str(row["filename"]).strip())[0].lower()
            raw = np.array([float(row[c]) for c in self.param_cols], dtype=np.float32)
            norm = (raw - self.param_means) / self.param_stds

            self.labels[key] = norm


        # listar .pt en tensors_dir
        files = [f for f in os.listdir(tensors_dir) if f.lower().endswith('.pt')]
        files.sort()
        if len(files) == 0:
            raise RuntimeError(f"No se encontraron archivos .pt en {tensors_dir}")
        self.files = files

    def __len__(self):
        return len(self.files)

    # Devuelve el tensor espectrograma y otro con sus parametros normalizados
    def __getitem__(self, idx):
        # En base al indice que recibe, busca su nombre
        fname = self.files[idx]
        key = os.path.splitext(fname)[0].lower()  

        # Construye su ruta al .pt original y lo carga 
        spec_path = os.path.join(self.tensors_dir, fname)
        spectrogram = torch.load(spec_path, map_location="cpu").float()  #Es mejor tenerlos en CPU en este momento, luego pasaremos a GPU

        # Asegurar forma (C,H,W) donde C es 1, importante pues waveform_to_spectrogram_tensor() devuelve un tensor (freq_bins, time_frames), sin canal!!
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)  

        # Obtener parámetros normalizados
        if key not in self.labels:
            raise KeyError(f"Etiqueta no encontrada para {key}")
        params = torch.tensor(self.labels[key], dtype=torch.float32)

        return spectrogram, params 

    # Desnormaliza un vector de parámetros
    def denormalize(self, params_norm):
        if isinstance(params_norm, torch.Tensor):   # Si es tensorr pytorch
            device = params_norm.device
            dtype = params_norm.dtype
            mean_t = torch.from_numpy(self.param_means).to(device=device, dtype=dtype)
            std_t = torch.from_numpy(self.param_stds).to(device=device, dtype=dtype)
            return params_norm * std_t + mean_t
        else:                                       # Si es array numpy
            arr = np.asarray(params_norm, dtype=np.float32)
            return arr * self.param_stds + self.param_means

    # Obtener las medias y desviaciones estándar 
    def get_stats(self):
        return {'means': self.param_means.copy(), 'stds': self.param_stds.copy()}