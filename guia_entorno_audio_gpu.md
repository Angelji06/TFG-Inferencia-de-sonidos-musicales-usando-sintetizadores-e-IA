# Guía de Configuración del Entorno de Desarrollo (Audio GPU)

Esta guía documenta los pasos necesarios para replicar el entorno de ejecución del proyecto. Se utiliza una configuración híbrida: **Conda** para el núcleo (PyTorch/CUDA) y **Pip** para librerías que presentan conflictos de binarios en Windows (como `soundfile` o `matplotlib`).

## 1. Gestión del Entorno Virtual (Conda)

### 1.1 Crear el entorno
Creamos un entorno limpio especificando Python 3.10 para asegurar compatibilidad con las librerías de audio.

```bash
conda create -n audio_gpu python=3.10 -y
```

### 1.2 Ver entornos existentes
Para confirmar que el entorno se ha creado o ver la ruta donde se encuentra:

```bash
conda env list
```

(El entorno activo aparecerá marcado con un asterisco *).

### 1.3 Activar el entorno  
IMPORTANTE: Ejecuta este comando siempre antes de instalar nada o correr el programa.

```bash
conda activate audio_gpu
```

## 2. Instalación del Núcleo (Vía Conda)

Usamos conda para instalar PyTorch y las librerías base que dependen de drivers del sistema (como NVIDIA CUDA y FFMPEG).  
Ejecutar en orden:

```bash
# 1. PyTorch con soporte GPU (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Si usas cpu
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 2. Herramientas de procesamiento de audio y matemáticas base
# Usamos conda-forge para obtener las versiones compiladas más estables de ffmpeg y librosa
conda install -c conda-forge ffmpeg librosa numpy -y

# 3. Codecs opcionales (recomendado)
conda install -c conda-forge torchcodec -y
```

## 3. Instalación de Dependencias Conflictivas (Vía Pip)

Las siguientes librerías me dieron problemas al instalarse vía Conda en Windows. Se instalan vía pip para obtener los wheels oficiales que incluyen los binarios necesarios.

```bash
pip install soundfile pandas openpyxl matplotlib scikit-learn sounddevice
```

Consejo: Si soundfile sigue dando error, ejecutar:

```bash
conda remove soundfile --force
pip install soundfile
```

## 4. Verificación de la Instalación

Ejecutar el archivo `scipts/check.py`:

## 5. Solución de Problemas Comunes

### A. FutureWarning: weights_only=False

```python
torch.load(ruta_modelo, map_location=device, weights_only=True)
```

### B. AttributeError: 'tuple' object has no attribute...

```python
prediction = model(spec)
if isinstance(prediction, tuple):
    prediction = prediction[0]
```

### C. El audio no suena (o valores muy bajos)

```python
carrier = params[0] * 2000
reproducir_prediccion([carrier, ...])
```
