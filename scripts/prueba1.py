import torch
import torchaudio
import torch.nn as nn
import matplotlib.pyplot as plt

# Define el modelo tal como fue entrenado
class SmallCNNRegressor(nn.Module):
    def __init__(self, in_channels=1, out_dim=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

# Cargar modelo entrenado
model = SmallCNNRegressor()
checkpoint = torch.load("cnn_spectrogram.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

# Cargar archivo .wav
waveform, sample_rate = torchaudio.load("pru_14930.wav")

# Transformar a espectrograma Mel (ajusta si usaste STFT u otro tipo)
transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
spec = transform(waveform)  # (1, n_mels, time)
spec = spec[:, :, :128]     # Recorta o interpola si es necesario
spec = spec.unsqueeze(0)    # Añade dimensión batch

# Normaliza si fue parte del entrenamiento
spec = (spec - spec.mean()) / spec.std()

# Inferencia
with torch.no_grad():
    output = model(spec)
    carrier, ratio, index = output[0]

print(f"Frecuencia portadora estimada: {carrier.item():.2f} Hz")
print(f"Ratio de modulación estimado: {ratio.item():.2f}")
print(f"Índice de modulación estimado: {index.item():.2f}")
