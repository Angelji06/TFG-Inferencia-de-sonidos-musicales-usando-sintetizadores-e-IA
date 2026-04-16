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
