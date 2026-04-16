# Esquema Técnico — Prototipo 5

---

## 1. Tratamiento del dato

### 1.1 Generación del dataset (`logica.py → generar_wavs_FM`)

- **30.000 muestras** generadas aleatoriamente (no rejilla — inviable con 8 parámetros)
- Cada muestra: 8 parámetros FM muestreados con distribución **uniforme** dentro de rangos fijos (`GEN_PARAMS`)
- **Envolvente de amplitud**: se sortea una duración total `[0.3s, 2.0s]`, luego 3 pesos aleatorios normalizados → `att + sus + dec = total`
- **Envolvente de modulación**: misma idea con 2 fases (`att + dec`, sin sustain)
- Audio: 2 segundos, 44100 Hz, PCM 16-bit (`.wav`)
- Etiquetas guardadas en `labels.csv`

### 1.2 Procesamiento de espectrogramas (`logica.py → procesar_espectrograma`)

Dos modos seleccionables, ambos terminan en **dB** (amplitud logarítmica):

| Paso | STFT (lineal) | Mel (perceptual) |
|---|---|---|
| Transformada | `Spectrogram(n_fft=1024, hop=256)` → complejo | `MelSpectrogram(n_fft=1024, hop=256, n_mels=128, power=1.0)` |
| Magnitud | `.abs()` | integrado en MelSpectrogram |
| Escala amplitud | `AmplitudeToDB(top_db=80)` | `AmplitudeToDB(top_db=80)` |
| Bins frecuencia | 513 (0–22050 Hz, lineales) | 128 (escala Mel, logarítmica) |
| Shape tensor | `(1, 513, T)` | `(1, 128, T)` |

Pasos previos aplicados a ambos:
- **Peak normalization**: `waveform / max(|waveform|)` → rango `[-1, 1]`
- **Fade in/out** (50 ms) al convertir WAV → tensor, para evitar clicks

### 1.3 Normalización de parámetros (`SpectrogramTensorDataset5.py`)

Normalización **Z-score por parámetro** calculada sobre el dataset completo:

```
param_norm = (param_raw - μ) / σ
```

- `μ` y `σ` se calculan en el constructor del dataset y se guardan en el checkpoint
- En inferencia: `param_real = param_norm * σ + μ`
- Motivo: el carrier (`100–2000 Hz`) dominaría la pérdida sin normalizar frente a parámetros en rango `[0.01, 1.9]`

---

## 2. Guardado del modelo y checkpoints

Formato del checkpoint `.pth` (`logica.py → entrenar_modelo`):

```python
{
  'state_dict' : OrderedDict,    # pesos del modelo
  'param_means': np.array[8],    # μ de cada parámetro (desnormalización)
  'param_stds' : np.array[8],    # σ de cada parámetro (desnormalización)
  'spec_mode'  : 'stft'|'mel',   # modo de espectrograma con el que se entrenó
  'arch'       : 'full'|'simple' # arquitectura usada
}
```

- **Mejor época**: durante el entrenamiento se guarda el `state_dict` de la época con menor `val_loss`; al finalizar se restaura automáticamente
- **Retrocompatibilidad**: `cargar_modelo_para_inferencia` detecta checkpoints antiguos (sin `spec_mode`/`arch`) y asigna valores por defecto (`stft`, `full`)

---

## 3. Arquitecturas del modelo (`Prototipo5.py`)

Dos arquitecturas comparables, mismo encoder:

### `CNNRegressor5` (full)

```
Input (1, F, T)
  → Encoder: Conv2d×3 + BN + ReLU + MaxPool2d  →  (256, F/8, T/8)
  → Bottleneck: Conv2d + BN + ReLU
  ┌─ GlobalAvgPool → Flatten → FC(256→64→8)     →  params (8,)
  └─ Decoder: ConvTranspose2d×3 → Conv2d        →  recon (1, F, T)
```

### `CNNRegressorSimple` (simple)

```
Input (1, F, T)
  → Encoder: Conv2d×3 + BN + ReLU + MaxPool2d  →  (256, F/8, T/8)
  → Bottleneck: Conv2d + BN + ReLU
  └─ GlobalAvgPool → Flatten → FC(256→64→8)     →  params (8,)
```

---

## 4. Función de pérdida

### `full`: HybridLoss (tres términos)

```
L = w_spec · L1(spec_pred, spec_real)
  + w_sc   · ‖spec_pred − spec_real‖_F / ‖spec_real‖_F
  + w_param · SmoothL1(params_pred, params_real)

Pesos por defecto: w_spec=1.0, w_sc=0.5, w_param=0.05
```

### `simple`: solo pérdida paramétrica

```
L = SmoothL1(params_pred, params_real)
```

---

## 5. Métricas de evaluación (`Prototipo5.py → evaluate`)

Calculadas **por parámetro** sobre el conjunto de test:

| Métrica | Fórmula | Qué mide |
|---|---|---|
| MSE | `mean((pred − true)²)` | Error cuadrático medio |
| RMSE | `√MSE` | Error en unidades del parámetro |
| MAE | `mean(\|pred − true\|)` | Error absoluto medio (robusto a outliers) |
| L1 espectral | `mean(\|spec_pred − spec_real\|)` | Calidad de reconstrucción (solo `full`) |

Salidas generadas automáticamente:
- `preds_vs_trues.csv` — todas las predicciones vs valores reales
- `scatter_params.png` — scatter true vs pred por parámetro con diagonal perfecta
- `spectrogram_example.png` — primeros 5 pares target/reconstrucción (solo `full`)

---

## 6. Combinaciones posibles para comparar

Con los mismos datos de entrenamiento, misma arquitectura base y único factor variable:

| Modelo | Espectrograma | Arquitectura | Pérdida |
|---|---|---|---|
| A | STFT | full | HybridLoss |
| B | STFT | simple | SmoothL1 |
| C | Mel | full | HybridLoss |
| D | Mel | simple | SmoothL1 |

---

## 7. Bibliotecas principales

| Biblioteca | Versión | Dónde se usa |
|---|---|---|
| `torch` | 2.5.1 | Definición del modelo, capas CNN, funciones de pérdida, optimizador Adam, backpropagation y guardado de checkpoints |
| `torchaudio` | 2.5.1 | Carga de WAV, transformadas `Spectrogram`, `MelSpectrogram`, `AmplitudeToDB` |
| `librosa` | 0.11.0 | Carga de audio en inferencia, visualización de espectrogramas con eje logarítmico (`specshow`) |
| `numpy` | 2.2.6 | Síntesis FM, generación de envolventes, operaciones con arrays en evaluación y desnormalización |
| `sounddevice` | 0.5.3 | Reproducción de audio en tiempo real y stream del sintetizador interactivo |
| `soundfile` | 0.13.1 | Escritura de WAVs del dataset y lectura para reproducción |
| `matplotlib` | 3.10.8 | Gráficas de evaluación y ventana de comparación de espectrogramas |
| `scikit-learn` | 1.7.2 | Dependencia transitiva de librosa (no usado directamente en el código principal) |
| `tkinter` | 8.6.13 | GUI completa: páginas de inicio, entrenamiento y test; sintetizador interactivo |
| `ffmpeg` | ≥8.0 | Backend de torchaudio para decodificación de audio; instalado vía conda para garantizar compatibilidad en Windows |

---

## 8. Funciones más relevantes

### `logica.py`

| Función | Descripción |
|---|---|
| `generar_envolvente(t, attack, sustain, decay)` | Genera una envolvente ADSR como array numpy. Usada tanto en la síntesis del dataset como en el sintetizador interactivo en tiempo real. |
| `fm_synthesize(carrier, ratio, index, a_att, a_sus, a_dec, m_att, m_dec)` | Síntesis FM completa: calcula las dos envolventes y genera la señal. Punto central del proyecto — define el espacio sonoro que el modelo aprende a invertir. |
| `procesar_espectrograma(waveform, sr, device, ..., mode)` | Convierte una forma de onda en tensor dB listo para el modelo. Soporta modos `stft` (513 bins lineales) y `mel` (128 bandas perceptuales). Garantiza forma `(1, 1, F, T)`. |
| `generar_wavs_FM(num_muestras)` | Genera el dataset completo de WAVs con parámetros aleatorios uniformes y guarda `labels.csv`. |
| `convertir_wavs_a_tensores(wav_folder, device, mode)` | Convierte cada WAV a tensor espectrograma y lo guarda como `.pt`. Guarda también `spec_mode.txt` para que el dataset sea autocontenido. |
| `entrenar_modelo(nombreModelo, dataset_obj, ..., arch)` | Orquesta el entrenamiento completo: instancia modelo y pérdida según `arch`, divide train/val, llama a `fit()` y guarda el checkpoint. |
| `cargar_modelo_para_inferencia(ruta_modelo, device)` | Carga el checkpoint, detecta arquitectura y modo de espectrograma, e instancia la clase correcta. Retrocompatible con checkpoints antiguos. |
| `hacer_inferencia(model, means, stds, ruta_wav, device, mode)` | Carga un WAV, lo convierte al espectrograma correcto y devuelve los 8 parámetros FM en escala real (desnormalizados). |
| `comparar_espectrogramas_4en1(wav_orig, sr_orig, params_pred, device, mode)` | Genera 4 paneles: original y predicción con eje de frecuencia logarítmico y lineal. El título indica el modo de espectrograma usado. |
| `evaluar_modelo(ruta_modelo, tensor_folder, device)` | Evalúa el modelo sobre un conjunto de tensores precalculados. Delega en `CNNRegressor5.evaluate()`. |

### `Prototipo5.py`

**`CNNRegressor5`** — arquitectura completa con reconstrucción espectral. El encoder comprime el espectrograma en tres etapas (Conv2d + BN + ReLU + MaxPool2d) hasta `(256, F/8, T/8)`. Desde el bottleneck se bifurca en dos cabezas: una de regresión (GlobalAvgPool + FC → 8 parámetros) y un decoder simétrico (ConvTranspose2d ×3) que reconstruye el espectrograma original. La reconstrucción actúa como regularización: obliga al encoder a capturar toda la estructura del espectrograma, no solo lo mínimo para predecir los parámetros.

**`CNNRegressorSimple`** — arquitectura reducida sin decoder. Mismo encoder y bottleneck que `CNNRegressor5`, pero sin las capas de reconstrucción. Solo existe la cabeza de regresión. Al eliminar la señal de gradiente espectral, el encoder aprende únicamente lo necesario para minimizar el error paramétrico, lo que puede llevar a representaciones menos ricas.

**`HybridLoss`** — función de pérdida de tres términos usada exclusivamente con `CNNRegressor5`:

```
L = w_spec · L1(spec_pred, spec_real)
  + w_sc   · ‖spec_pred − spec_real‖_F / ‖spec_real‖_F
  + w_param · SmoothL1(params_pred, params_real)
```

El término de **spectral convergence** mide el error relativo en magnitud (norma de Frobenius), capturando la estructura global del espectro independientemente de su escala absoluta. Los pesos por defecto (`w_spec=1.0`, `w_sc=0.5`, `w_param=0.05`) priorizan la fidelidad espectral sobre la precisión paramétrica, bajo la hipótesis de que espectrogramas similares implican parámetros similares.

### `SpectrogramTensorDataset5.py`

**`SpectrogramTensorDataset`** — clase `Dataset` de PyTorch que sirve pares `(espectrograma, parámetros)` al DataLoader durante el entrenamiento. Al inicializarse, lee `labels.csv`, calcula la media y desviación estándar de cada uno de los 8 parámetros sobre todo el dataset y normaliza las etiquetas en memoria (Z-score). Los tensores `.pt` se cargan del disco bajo demanda en `__getitem__`, asegurando forma `(1, F, T)`. Expone `get_stats()` para que los valores de normalización se guarden en el checkpoint y estén disponibles en inferencia.
