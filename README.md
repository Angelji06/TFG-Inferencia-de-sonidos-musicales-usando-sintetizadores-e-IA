# Inferencia de sonidos musicales usando sintetizadores e IA

**Autores:** David Cendejas Rodríguez y Ángel Jiménez Izquierdo  
**Tutores:** Miguel Gómez-Zamalloa Gil y Jaime Sánchez Hernández  

Este proyecto (aun en desarrollo) explora diferentes aproximaciones al **procesamiento, identificación y reproducción de un timbre** a partir de una muestra de audio. Para ello, se han desarrollado varios prototipos incrementales, cada uno aplicando mejoras en dataset, arquitectura y metodología.

---
## Prototipo 1 — Clasificación básica  

**Notebook:** `PrototipoIA1.ipynb`

Primera aproximación al aprendizaje automático aplicado al audio.

### Dataset
- Tamaño reducido (≈ 50 muestras)  
- Archivos `.wav` de 1 segundo  
- Generados sintéticamente con:
  - Frecuencias aleatorias  
  - Formas de onda: *sine, square, sawtooth, triangle, noise*  
- Construidos usando **NumPy**

### Entrenamiento
- Implementación con **TensorFlow**  
- Extracción de características mediante **MFCC**  
- Asociación de MFCC → tipo de onda (clasificación)

### Modelo
- Tipo: **Clasificación**  
- No convolucional  
- Arquitectura simple y poco optimizada  
- **Precisión baja**

---

## Prototipo 2 — Clasificación con CNN preentrenada  
**Notebook:** `PrototipoIA2.ipynb`

Optimización del primer prototipo introduciendo modelos convolucionales y mejor tratamiento del dataset.

### Dataset
- Basado en el Prototipo 1  
- Convertido a **espectrogramas `.png`** usando *librosa*  
- Menor coste computacional y de almacenamiento  
- Permite usar modelos de visión

### Entrenamiento
- Uso de **ResNet34 preentrenada** vía fastAI  
- Adaptación del modelo mediante **DataBlocks**

### Modelo
- Tipo: **Clasificación**  
- Convolucional  
- Arquitectura externa (ResNet34)  
- **Buena precisión**, especialmente en ondas puras dentro del rango entrenado

---

## Prototipo 3 — Regresión con CNN propia  
**Notebook:** `PrototipoIA4_Regresion.ipynb`

Avance hacia la **síntesis paramétrica**, no solo identificación.

### Dataset
- Tamaño mediano (≈ 15.000 muestras)  
- Generado mediante barrido de parámetros en un **sintetizador FM de pyo**  
- Convertido a tensores de espectrogramas con **torchaudio**

### Entrenamiento
- Implementado en **PyTorch**  
- Dataset definido en `SpectrogramTensorDataset`  
- Arquitectura definida en `SmallCNNRegressor`  
- Entrenamiento en **5 etapas**  
- Función de pérdida: **MSELoss**

### Modelo
- Tipo: **Regresión**  
- Convolucional  
- Arquitectura propia  
- **Precisión limitada**, aún por optimizar

---
