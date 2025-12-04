import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import numpy as np

class SpectrogramTensorDataset(Dataset):
    def __init__(self, tensors_dir, transform=None, target_transform=None, param_cols=('carrier', 'ratio', 'index'), normalize=True):
        self.tensors_dir = tensors_dir
        self.transform = transform
        self.target_transform = target_transform
        self.param_cols = list(param_cols)
        self.normalize = normalize

        # localizar CSV (se asume la estructura: parent_dir/datasetFMwav/labels.csv)
        labels_csv = os.path.join(os.path.dirname(tensors_dir), "datasetFMwav", "labels.csv")
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"labels.csv no encontrado en {labels_csv}")
        df = pd.read_csv(labels_csv)

        #2) Calcular stats (solo memoria)
        vals = df[self.param_cols].astype(np.float32).values
        self.param_means = vals.mean(axis=0).astype(np.float32)
        stds = vals.std(axis=0).astype(np.float32)
        stds[stds == 0] = 1.0   # evitar división por 0
        self.param_stds = stds


        # 3) Mapear filename → params normalizados
        self.labels = {}
        for _, row in df.iterrows():
            key = os.path.splitext(str(row["filename"]).strip())[0].lower()
            raw = np.array([float(row[c]) for c in self.param_cols], dtype=np.float32)
            if self.normalize:
                norm = (raw - self.param_means) / self.param_stds
            else:
                norm = raw
            self.labels[key] = norm


        # listar .pt en tensors_dir
        files = [f for f in os.listdir(tensors_dir) if f.lower().endswith('.pt')]
        files.sort()
        if len(files) == 0:
            raise RuntimeError(f"No se encontraron archivos .pt en {tensors_dir}")
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        key = os.path.splitext(fname)[0].lower()  # ej: "pru_1"

        # 1) Cargar espectrograma .pt
        spec_path = os.path.join(self.tensors_dir, fname)
        spectrogram = torch.load(spec_path).float()

        # asegurar forma (C,H,W) donde C es 1 
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)  # (1, H, W)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        # 2) Obtener parámetros desde el CSV
        if key not in self.labels:
            raise KeyError(f"Etiqueta no encontrada para {key}")
        carrier, ratio, index = self.labels[key]
        params = torch.tensor(self.labels[key], dtype=torch.float32)

        if self.target_transform:
            params = self.target_transform(params)

        # 4) Devolver SOLO (spec, params)
        return spectrogram, params
