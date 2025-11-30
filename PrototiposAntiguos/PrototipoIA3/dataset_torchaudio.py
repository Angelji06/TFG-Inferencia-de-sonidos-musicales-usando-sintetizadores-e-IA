import os
import torch
import torchaudio
from torch.utils.data import Dataset
import pandas as pd

# =============================================================================================================
#  Clase que convierte la carpeta llena de tensores .pt en un objeto que hereda de la clase Dataset de PyTorch
# =============================================================================================================

class SpectrogramTensorDataset(Dataset):
    """Dataset para tensores .pt de espectrogramas generados con torchaudio.

    Cada archivo en la carpeta debe ser `pru_<n>.pt` y existe un CSV con labels
    en `Datasets/datasetFMwav/labels.csv` con las columnas: filename,carrier,ratio,index.

    Devuelve (tensor, target) donde tensor es FloatTensor (1,H,W) y target es FloatTensor([carrier, ratio, index]).
    """

    def __init__(self, tensors_dir, transform=None, target_transform=None):
        self.tensors_dir = tensors_dir
        self.transform = transform
        self.target_transform = target_transform

        # localizar CSV y leerlo
        labels_csv = os.path.join(os.path.dirname(tensors_dir), 'datasetFMwav', 'labels.csv')
        df = pd.read_csv(labels_csv)

        # Crear mapping filename w/o extension -> (carrier, ratio, index)
        self.labels = {}
        for _, row in df.iterrows():
            name = os.path.splitext(str(row['filename']).strip())[0]  # e.g. pru_1
            self.labels[name.lower()] = (float(row['carrier']), float(row['ratio']), float(row['index']))

        # list of available .pt files
        files = [f for f in os.listdir(tensors_dir) if f.lower().endswith('.pt')]
        files.sort()
        self.files = files
        self.tensors_dir = tensors_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.tensors_dir, fname)
        # cargar tensor
        x = torch.load(path)
        # asegurar FloatTensor
        x = x.float()
        # el tensor parece tener shape [1, H, W], si falta canal añadir
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # obtener label por nombre sin extensión
        key = os.path.splitext(fname)[0].lower()
        if key not in self.labels:
            raise KeyError(f"Etiqueta no encontrada para {key}")
        carrier, ratio, index = self.labels[key]
        y = torch.tensor([carrier, ratio, index], dtype=torch.float32)

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y


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

if __name__ == '__main__':
    # prueba rápida
    ds = SpectrogramTensorDataset(
        tensors_dir=os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'datasetFMespec_torchaudio')
    )
    
    print('len dataset', len(ds))
    x,y = ds[0]
    print('sample x', type(x), x.shape, 'y', y)
