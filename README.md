# 🎵 Arquitectura del Proyecto de Síntesis FM

## 🧠 `Prototipo5.py` — Modelo y Pérdida
Define la arquitectura de la red neuronal y la función de costo.

* **`HybridLoss`**: Función de pérdida combinada que integra L1 Espectral, Convergencia Espectral (*Spectral Convergence*) y *SmoothL1* aplicado a los parámetros.
* **`CNNRegressor5` (Encoder)**: Extrae características mediante 3 bloques compuestos por `Conv` + `BatchNorm` + `ReLU` + `MaxPool`, conectados a un *bottleneck*.
* **`CNNRegressor5` (Cabezas de salida)**: El modelo se bifurca en dos ramas. La rama `fc_params` aplica `GlobalAvgPool` y una capa `FC` para predecir 7 parámetros FM. La rama `decoder` emplea 3 capas `ConvTranspose` para la reconstrucción del espectrograma.
* **Métodos principales**: Proporciona la interfaz base del modelo con `fit()`, `evaluate()` y `load()`.

## 📊 `SpectrogramTensorDataset5.py` — Dataset
Responsable de la carga de datos y el acondicionamiento para el entrenamiento.

* **Carga de datos**: Consume tensores `.pt` precomputados y extrae las etiquetas de parámetros desde un archivo CSV.
* **Normalización**: Aplica estandarización *Z-score* por cada parámetro utilizando la media y desviación estándar global del dataset.
* **`denormalize()`**: Método integrado para invertir matemáticamente la normalización y recuperar los valores absolutos de los parámetros sintetizadores.

## ⚙️ `logica.py` — Lógica
El motor computacional que maneja la señal de audio, las transformaciones y las métricas.

* **Síntesis FM (`fm_synthesize`)**: Generador de audio que incorpora envolventes de ataque (*attack*) y decaimiento (*decay*) tanto para la amplitud como para la modulación.
* **Pipeline de datos**: Transforma audios WAV aleatorios en tensores de espectrograma aplicando STFT (con `n_fft=1024` y `hop=256`) seguido de una conversión `AmplitudeToDB`.
* **Inferencia**: Incluye rutinas para predecir sobre un único archivo y versiones optimizadas para lotes que mantienen el modelo cargado en memoria.
* **`prediccion_multiples_wav`**: Automatiza la generación de WAVs sintetizados para facilitar la evaluación de calidad de audio mediante métricas FAD.
* **`comparar_espectrogramas_4en1`**: Utilidad de visualización que genera una matriz 2x2 para comparar visualmente el audio original frente a la predicción en escalas lineal y logarítmica.

## 🖥️ `main.py` — Interfaz Gráfica (Tkinter)
Aplicación de escritorio interactiva estructurada en 3 páginas principales para el control del flujo de trabajo.

* **Página de Inicio**: Punto de entrada que permite al usuario cargar un modelo previamente entrenado o navegar directamente al panel de entrenamiento.
* **Página de Entrenamiento**: Consola de control para generar o cargar el dataset, con ajustes detallados de hiperparámetros (épocas, *Learning Rate*, *batch size* y ponderación de la función de pérdida).
* **Página de Test**: Entorno de validación para realizar predicciones sobre un único WAV, con herramientas para reproducir el audio generado, visualizar sus espectrogramas y exportar lotes de prueba para análisis FAD.