# Detalles Técnicos para la Memoria — Prototipo 5

---

## 1. Síntesis FM

- Fórmula de síntesis FM de 2 operadores (portadora + moduladora)
- 8 parámetros del sintetizador: carrier, ratio, index, amp_attack, amp_sustain, amp_decay, mod_attack, mod_decay
- Envolvente de amplitud de 3 fases (attack/sustain/decay) lineal por tramos
- Envolvente de modulación de 2 fases (attack/decay, sin sustain)
- Duración fija de audio: 2 segundos, 44100 Hz, PCM 16-bit
- Problema de no-inyectividad FM (distintos parámetros pueden producir el mismo sonido)

## 2. Generación del dataset

- 30.000 muestras con muestreo aleatorio uniforme (no rejilla — inviable con 8 dimensiones)
- Carrier muestreado en escala logarítmica (`np.exp(np.random.uniform(np.log(...)))`)
- Rangos definidos en `GEN_PARAMS`
- Rescalado proporcional de envolventes si la suma de fases excede la duración del audio
- Guardado de etiquetas en CSV con buffer de escritura (flush cada 1000 muestras)
- Pipeline en dos fases: generación WAV → conversión a tensores `.pt`

## 3. Procesamiento de espectrogramas

- Dos modos: STFT lineal (513 bins) y Mel (128 bandas)
- Parámetros STFT: n_fft=1024, hop_length=256
- AmplitudeToDB con top_db=80
- Peak normalization del waveform antes de la transformada
- Fade in/out de 50 ms para evitar artefactos de borde
- Tensores con forma (1, 1, F, T) para la CNN

## 4. Normalización de parámetros

- Z-score por parámetro (μ y σ calculadas sobre todo el dataset)
- Motivación: evitar que el carrier (100–2000 Hz) domine la pérdida frente a parámetros de rango [0.01, 1.9]
- μ y σ almacenadas en el checkpoint para desnormalización en inferencia

## 5. Arquitecturas de red

- **CNNRegressorSimple**: encoder (3 bloques Conv2d+BN+ReLU+MaxPool2d) → bottleneck → AdaptiveAvgPool2d → FC(256→64→8)
- **CNNRegressor5**: mismo encoder + decoder con ConvTranspose2d×3 como regularizador
- Decoder como regularizador: fuerza al encoder a retener información completa del espectrograma
- Interpolación bilineal para ajustar dimensiones si MaxPool introduce asimetría
- Filtros base: 32 → 64 → 128 → 256 (bottleneck)
- Cabeza de regresión con AdaptiveAvgPool2d(1,1) (media global por canal)

## 6. Funciones de pérdida

- **SmoothL1Loss** (solo parámetros, para arquitectura simple)
- **HybridLoss**: L1 espectral + Spectral Convergence (norma de Frobenius relativa) + SmoothL1 paramétrico
- **MultiScaleSpectralLoss**: inspirada en DDSP (Engel et al., 2020), average pooling temporal con 6 escalas (1,2,4,8,16,32), término lineal + término logarítmico
- Pesos por defecto: spec_w=1.0, sc_w=0.5, param_w=0.05
- Conversión dB→magnitud lineal dentro de la loss (10^(dB/20))

## 7. Entrenamiento

- Split 70/15/15 (train/val/test) con semilla fija (42) para reproducibilidad
- Optimizador Adam
- Early stopping implícito: se guarda el state_dict de la mejor época (menor val_loss) y se restaura al final
- Índices del test set almacenados en el checkpoint
- Hiperparámetros configurables desde GUI: epochs, lr, batch_size, pesos de loss, print_every

## 8. Checkpoint

- Contenido: state_dict, param_means, param_stds, spec_mode, arch, test_indices
- Retrocompatibilidad con checkpoints antiguos (defaults: stft, full)

## 9. Inferencia

- Carga de modelo con detección automática de arquitectura y modo de espectrograma
- Clampeo de predicciones a rangos válidos del dominio de entrenamiento
- Desnormalización: param_real = param_norm × σ + μ

## 10. Métricas de evaluación

- **Mel L1**: mean(|log1p(mel_pred) − log1p(mel_true)|) sobre audio re-sintetizado
- **MCD** (Mel-Cepstral Distortion): coeficientes MFCC 1–12, fórmula estándar en dB
- **MSE, RMSE, MAE** por parámetro (diagnóstico, afectadas por no-inyectividad)
- **L1 espectral del decoder** (solo arquitectura full)
- Evaluación sobre hasta 500 muestras re-sintetizadas con semilla fija
- Generación de eval set para FAD externo

## 11. Benchmark diverso

- 12 sonidos de referencia predefinidos cubriendo el espacio tímbrico (graves/agudos, armónicos/inarmónicos, limpios/ruidosos, percusivos/pads/plucks)
- Generación de WAVs original vs predicción para escucha comparativa
- Gráfica de barras Mel L1 y MCD por sonido

## 12. Visualización

- Scatter true vs pred por parámetro con diagonal de predicción perfecta
- Histogramas de distribución de Mel L1 y MCD
- Comparación de espectrogramas original vs predicción en 4 paneles (eje log y lineal)
- Comparación target vs reconstrucción del decoder (primeros 5 ejemplos)

## 13. Espacio experimental (6 pipelines válidos)

- Matriz 2×2×2 (espectrograma × arquitectura × loss) con restricción: simple solo admite SmoothL1
- 6 combinaciones válidas (P1–P6) con preguntas de investigación asociadas

## 14. Dataset como clase PyTorch

- `SpectrogramTensorDataset` hereda de `torch.utils.data.Dataset`
- Carga lazy de tensores `.pt` (map_location CPU)
- Normalización vectorizada en constructor
- Emparejamiento tensor–etiqueta por nombre de fichero

## 15. GUI (Tkinter)

- Tres páginas: Inicio, Entrenamiento, Test
- Configuración de hiperparámetros, arquitectura, loss y modo de espectrograma desde la interfaz
- Detección automática de audio del dataset de entrenamiento (por nombre de fichero) para mostrar parámetros originales
- Integración con sintetizador interactivo FMSynth8 (ventana secundaria)
- Selector de device (CUDA/CPU)

## 16. Bibliotecas y dependencias

- PyTorch, torchaudio, librosa, numpy, sounddevice, soundfile, matplotlib, tkinter, ffmpeg (backend torchaudio)
