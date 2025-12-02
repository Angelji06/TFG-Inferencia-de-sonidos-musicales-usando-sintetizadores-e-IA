import torch
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)    # versión CUDA con la que fue compilado (o None)
print("cuda available:", torch.cuda.is_available())
try:
    import subprocess, sys
    # comprueba nvidia-smi (sólo si tienes GPU NVIDIA)
    res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print("nvidia-smi salida (si existe):\n", res.stdout if res.returncode==0 else res.stderr)
except Exception as e:
    print("nvidia-smi no ejecutable o no presente:", e)