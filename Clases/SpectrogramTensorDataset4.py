import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class SpectrogramTensorDataset(Dataset):
    """
    Dataset que devuelve (spectrogram_tensor, params_tensor).
    No carga ni requiere archivos .wav.
    """

    def __init__(self, tensors_dir, transform=None, target_transform=None):
        self.tensors_dir = tensors_dir
        self.transform = transform
        self.target_transform = target_transform

        # localizar CSV (se asume la estructura: parent_dir/datasetFMwav/labels.csv)
        labels_csv = os.path.join(os.path.dirname(tensors_dir), "datasetFMwav", "labels.csv")
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"labels.csv no encontrado en {labels_csv}")
        df = pd.read_csv(labels_csv)

        # mapping filename (sin extensión) -> (carrier, ratio, index)
        self.labels = {}
        for _, row in df.iterrows():
            key = os.path.splitext(str(row['filename']).strip())[0].lower()
            self.labels[key] = (float(row['carrier']), float(row['ratio']), float(row['index']))

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
        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"No se encontró el tensor: {spec_path}")
        spectrogram = torch.load(spec_path).float()

        # asegurar forma (C,H,W) donde C es 1 en tu caso
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)  # (1, H, W)

        # 2) Obtener parámetros desde el CSV
        if key not in self.labels:
            raise KeyError(f"Etiqueta no encontrada para {key}")
        carrier, ratio, index = self.labels[key]
        params = torch.tensor([carrier, ratio, index], dtype=torch.float32)

        # 3) Aplicar transforms si los hay
        if self.transform:
            spectrogram = self.transform(spectrogram)
        if self.target_transform:
            params = self.target_transform(params)

        # 4) Devolver SOLO (spec, params)
        return spectrogram, params



def waveform_to_spectrogram_tensor(waveform, sample_rate):
    """
    Convierte un waveform en un espectrograma compatible con el modelo.
    Usa la misma configuración que los espectrogramas .pt del dataset.
    """
    # Convertir a mono si tiene más de un canal
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Crear transformador de espectrograma (ajusta parámetros si tu dataset usa otros)
    transform = torchaudio.transforms.Spectrogram(
        n_fft=1024,
        win_length=None,
        hop_length=512,
        power=2.0
    )

    # Aplicar transformación
    spectrogram = transform(waveform)

    # Normalizar (ns si hace falta)
    spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)

    # Asegurar tipo y forma
    spectrogram = spectrogram.float()
    if spectrogram.dim() == 2:
        spectrogram = spectrogram.unsqueeze(0)

    return spectrogram
