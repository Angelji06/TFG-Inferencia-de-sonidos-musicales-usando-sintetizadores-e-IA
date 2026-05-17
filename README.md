# Estimación automática de Parámetros de Síntesis FM para la obtención de Timbres Musicales mediante Redes Neuronales}

Trabajo de Fin de Grado, Universidad Complutense de Madrid

## Autores

- David Cendejas Rodríguez, Grado en Ingeniería de Computadores
- Ángel Jiménez Izquierdo, Grado en Ingeniería Informática

## Tutores

- Miguel Gómez-Zamalloa Gil
- Jaime Sánchez Hernández

---

## Descripción

Este proyecto desarrolla un sistema capaz de inferir los parámetros de un sintetizador FM a partir de una muestra de audio. Dado un sonido de entrada, el modelo predice los 8 parámetros que lo describen: frecuencia portadora, ratio de modulación, índice de modulación y las envolventes de amplitud y modulación (ataque, sustain y decaimiento).

El pipeline completo abarca tres fases:

1. **Generación del dataset**: se sintetizan miles de sonidos FM con parámetros aleatorios usando una implementación propia del sintetizador. Cada audio se convierte a espectrograma mel y se almacena como tensor de PyTorch.

2. **Entrenamiento**: una red neuronal convolucional (CNN) recibe los espectrogramas y aprende a predecir los parámetros del sintetizador. La arquitectura incluye un encoder, un bottleneck, una cabeza de regresión y un decoder que reconstruye el espectrograma como señal de regularización.

3. **Inferencia**: dado un archivo WAV externo, el sistema genera su espectrograma, lo procesa con el modelo entrenado y devuelve los parámetros FM predichos, pudiendo reproducir el audio sintetizado resultante para compararlo con el original.

---

## Ejecución

### Requisitos previos

Se necesita tener instalado [Miniconda](https://docs.anaconda.com/miniconda/) o Anaconda. Si no se dispone de él, los scripts de instalación lo descargan e instalan automáticamente.

### Windows

Ejecutar el instalador haciendo doble clic sobre `instalar.bat` o desde la terminal:

```
instalar.bat
```

Una vez completada la instalación, lanzar la aplicación con:

```
lanzar.bat
```

### Linux

Dar permisos de ejecución al instalador y ejecutarlo:

```bash
chmod +x instalar.sh
./instalar.sh
```

Una vez completada la instalación, lanzar la aplicación con:

```bash
bash lanzar.sh
```

Si no encuentra conda utilizar el siguiente comando (si está instalado en la ruta por defecto)

```bash
~/miniconda3/bin/conda init bash
```

### CPU (opcional)

Por defecto el entorno instala PyTorch con soporte CUDA 12.1, lo que requiere una tarjeta NVIDIA compatible. Para usar únicamente CPU, sustituir en `environment.yml` la línea `pytorch-cuda=12.1` por:

```yaml
- cpuonly
```

---

## Estructura del proyecto

El código final del proyecto se encuentra en la carpeta `Prototipo5/`.

```
Prototipo5/
├── main.py                    Interfaz gráfica (Tkinter). Punto de entrada de la aplicación.
├── logica.py                  Orquestación general: generación del dataset, entrenamiento,
│                              inferencia, reproducción de audio y visualización.
├── models.py                  Arquitecturas de red neuronal:
│                                - CNNRegressor5: encoder + decoder + cabeza de parámetros
│                                - CNNRegressorSimple: solo encoder + cabeza de parámetros
├── losses.py                  Funciones de pérdida:
│                                - HybridLoss: L1 espectral + Spectral Convergence + SmoothL1
│                                - MultiScaleSpectralLoss: pérdida multi-escala inspirada en DDSP
├── dataset.py                 Dataset de PyTorch que carga los tensores .pt y normaliza
│                              los parámetros con Z-score.
└── FMsynth8.py                Sintetizador FM interactivo con interfaz gráfica propia,
                               permite explorar y ajustar parámetros en tiempo real.
```

Los prototipos anteriores (pruebas de concepto y experimentos intermedios) están en `PrototiposAntiguos/`.
