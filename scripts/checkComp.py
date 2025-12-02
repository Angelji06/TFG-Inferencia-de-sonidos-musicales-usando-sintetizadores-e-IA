import torch
import pyo
import numpy as np
import torchvision
import torchaudio
import torchcodec
import simpleaudio

print("numpy", np.__version__)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
print("torchcodec", getattr(torchcodec, '__version__', 'desconocido'))
print("pyo", pyo.__version__)
print("simpleaudio", simpleaudio.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))