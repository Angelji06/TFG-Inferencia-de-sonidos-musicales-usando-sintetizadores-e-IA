import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import pandas as pd
import numpy as np
import simpleaudio as sa
import torchaudio

def fm_synthesize(carrier, ratio, index, duration=0.5, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mod = np.sin(2 * np.pi * (carrier * ratio) * t)
    car = np.sin(2 * np.pi * carrier * t + index * mod)
    return car.astype(np.float32), sr

def play_audio(waveform, sr):
    # Normalizamos a int16 para reproducir
    audio = (waveform * 32767).astype(np.int16)
    sa.play_buffer(audio, 1, 2, sr)

def reproducir_wav(path):
    waveform, sr = torchaudio.load(path)
    waveform = waveform[0].numpy()  # Mono
    play_audio(waveform, sr)
    
def reproducir_prediccion(params):
    carrier, ratio, index = params
    waveform, sr = fm_synthesize(carrier, ratio, index, duration=1.0)
    play_audio(waveform, sr)

from Clases.SpectrogramTensorDataset4 import SpectrogramTensorDataset
from scripts.train_regression import SmallCNNRegressor
from scripts.PaddingTensores import PadOrCrop

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "Datasets/datasetFMespec_torchaudio"
BATCH_SIZE = 32
LR = 1e-3

def crear_columna_tensor():
    # Leer CSV de etiquetas
    labels_df = pd.read_csv("Datasets\datasetFMwav\labels.csv")
    print(labels_df.head())

    # Crear nueva columna con el nombre del tensor 
    labels_df["tensor_name"] = labels_df["filename"].str.replace(".wav", ".pt")
    print(labels_df.head())

def cargar_dataset():
    # Usaremos la implementación de SpectrogramTensorDataset ya definida en `scripts/dataset_torchaudio.py`
    # Esta carga los .pt y devuelve (tensor, [carrier, ratio, index])
    transform = PadOrCrop((513, 173))
    dataset = SpectrogramTensorDataset(
        tensors_dir=DATASET_PATH,
        transform=transform
    )

    # Dividir en train/test (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    return dataset, train_loader, test_loader

def create_model():
    model = SmallCNNRegressor(out_dim=3)
    model = model.to(DEVICE)
    return model

def train_model(model, train_loader, epochs=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    return model

def load_model(path="cnn_spectrogram.pth"):
    model = SmallCNNRegressor(out_dim=3) 
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  
    return model

def predict_tensor(model, tensor):
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(DEVICE)
        out = model(tensor.unsqueeze(0))     # batch=1
        return out.cpu().numpy()[0]
    
def predict_wav(model, wav_path):
    import torchaudio
    from Clases.SpectrogramTensorDataset4 import waveform_to_spectrogram_tensor

    waveform, sr = torchaudio.load(wav_path)

    spec = waveform_to_spectrogram_tensor(waveform, sr)

    return predict_tensor(model, spec)

def iniciar_modelo(usar_modelo_guardado=True, epochs=5):
    import os
    path_modelo = "cnn_spectrogram.pth"
    
    if usar_modelo_guardado and os.path.exists(path_modelo):
        print("Cargando modelo guardado...")
        model = load_model(path_modelo)
    else:
        print("Entrenando modelo desde cero...")
        dataset, train_loader, _ = cargar_dataset()
        model = create_model()
        model = train_model(model, train_loader, epochs)
        torch.save(model.state_dict(), path_modelo)
        print("Modelo entrenado y guardado.")
    
    return model

