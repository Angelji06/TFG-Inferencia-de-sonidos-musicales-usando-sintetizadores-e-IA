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
**Notebook:** `PrototipoIA3_Regresion.ipynb`

Avance hacia la **síntesis paramétrica**, no solo identificación.

### Dataset
- Tamaño mediano (≈ 15.000 muestras)  
- Generado mediante barrido de parámetros en un **sintetizador FM de pyo**
- Ahora con .csv con las etiquetas de los valores carrier, ratio e index de cada muestra
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
## Prototipo 4 — Regresión con función de pérdida híbrida
**Notebook:** `PrototipoIA4_Regresion.ipynb`

MSE sobre parámetros es incorrecto debido a nula inyectividad, se plantea una función de pérdida hibrida que sigue teniendo en cuenta el MSE sobre parámetros (ponderado a 0.1) y sobretodo compara el espectrograma predicho con el original. Ahora la arquitectura de la CNN ya no es arbitraria.

### Dataset
- Igual que prototipo 3

### Entrenamiento
- 10 etapas, con función de pérdida hibrida

### Modelo
- Tipo: **Regresión**  
- Convolucional  
- Arquitectura propia:
  - Encoder: 3 capas
  - Bottleneck
  - Global pooling
  - Decoder: 3 capas
  - Recon head

---
## Prototipo 5 — FINAL: Modelo de sound matching de síntesis FM de 7 parámetros con 

**Notebook:** `PrototipoIA4_Regresion.ipynb`

MSE sobre parámetros es incorrecto debido a nula inyectividad, se plantea una función de pérdida hibrida que sigue teniendo en cuenta el MSE sobre parámetros (ponderado a 0.1) y sobretodo compara el espectrograma predicho con el original. Ahora la arquitectura de la CNN ya no es arbitraria.

### Dataset
- Modificado la función de generación, al subir el número de parámetros ya no se podía hacer un barrido, por lo que ahora se recibe el numero de muestras deseadas y se genera aleatoriamente dentro de unos rangos.

### Modelo
Se mantiene la filosofía de la función de pérdida hibrida y la red de doble cabeza con reconstrucción de espectrogramas.

- Tipo: **Regresión**  
- Convolucional  
- Arquitectura propia:
  - Encoder: 3 capas
  - Bottleneck
  - Global pooling
  - Decoder: 3 capas
  - Recon head