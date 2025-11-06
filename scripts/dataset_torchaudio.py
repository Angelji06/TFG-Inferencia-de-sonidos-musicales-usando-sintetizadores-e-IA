import os
import torch
from torch.utils.data import Dataset
import pandas as pd


class SpectrogramTensorDataset(Dataset):
    """Dataset para tensores .pt de espectrogramas generados con torchaudio.

    Cada archivo en la carpeta debe ser `pru_<n>.pt` y existe un CSV con labels
    en `Datasets/datasetFMwav/labels.csv` con las columnas: filename,carrier,ratio,index.

    Devuelve (tensor, target) donde tensor es FloatTensor (1,H,W) y target es FloatTensor([carrier, ratio, index]).
    """

    def __init__(self, tensors_dir, labels_csv=None, transform=None, target_transform=None):
        self.tensors_dir = tensors_dir
        self.transform = transform
        self.target_transform = target_transform

        # localizar CSV por defecto
        if labels_csv is None:
            labels_csv = os.path.join(os.path.dirname(os.path.dirname(tensors_dir)), 'datasetFMwav', 'labels.csv')

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


if __name__ == '__main__':
    # prueba rápida
    ds = SpectrogramTensorDataset(
        tensors_dir=os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'datasetFMespec_torchaudio')
    )
    print('len dataset', len(ds))
    x,y = ds[0]
    print('sample x', type(x), x.shape, 'y', y)
