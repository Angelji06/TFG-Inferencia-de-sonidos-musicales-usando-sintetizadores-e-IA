"""
audio_utils.py
Utilidades para:
 - carga de audio
 - guardado wav
 - reproducción simple
 - generación de espectrograma (PIL.Image)
 - wrapper genérico para predicción con modelo PyTorch

Instala dependencias (ejemplo):
pip install numpy librosa matplotlib pillow soundfile sounddevice simpleaudio torch

Nota: predict_with_model es genérico: intenta adaptar la entrada al shape que el modelo espera.
"""
import os
import io
import numpy as np

# Dependencias opcionales
try:
    import librosa
except Exception as e:
    raise ImportError("Instala 'librosa' para usar audio_utils: pip install librosa") from e

try:
    import soundfile as sf
except Exception as e:
    raise ImportError("Instala 'soundfile' para usar audio_utils: pip install soundfile") from e

# Reproducción: preferimos sounddevice (streaming). Si no está, fallback a simpleaudio (solo WAV).
_HAS_SD = True
try:
    import sounddevice as sd
except Exception:
    _HAS_SD = False
    try:
        import simpleaudio as sa
    except Exception:
        sa = None

# Matplotlib y PIL para espectrogramas
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import mlab
except Exception as e:
    raise ImportError("Instala 'matplotlib' para generar espectrogramas.") from e

try:
    from PIL import Image
except Exception:
    raise ImportError("Instala 'Pillow' (PIL) para generar imágenes.") from e

# PyTorch para predicción (opcional si no usas modelos PyTorch)
try:
    import torch
except Exception:
    torch = None

# ---------------------------
# Funciones de I/O y util
# ---------------------------
def load_audio_file(path, target_sr=None, mono=True):
    """
    Carga archivo de audio usando librosa (devuelve numpy float32 in range -1..1).
    target_sr: int o None para mantener sr nativo.
    """
    arr, sr = librosa.load(path, sr=target_sr, mono=mono)
    arr = arr.astype(np.float32)
    return arr, sr

def save_wav(path, arr, sr):
    """
    Guarda array float32 (-1..1) como WAV mediante soundfile.
    """
    # asegurar tipo
    arr = np.asarray(arr)
    # soundfile escribirá floats correctamente
    sf.write(path, arr, sr)
    return path

def play_audio(arr, sr):
    """
    Reproduce audio bloqueante. Intenta sounddevice.play; si no disponible usa simpleaudio.
    arr: 1D numpy float32 (-1..1) o 2D (channels, samples) o (samples, channels)
    """
    arr = np.asarray(arr)
    # normalizar si fuera necesario
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32)

    # convertir shape a (samples, channels)
    if arr.ndim == 1:
        out = arr
    elif arr.ndim == 2:
        # si shape (channels, samples) lo transponemos
        if arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
            out = arr.T
        else:
            out = arr
    else:
        raise ValueError("Formato de array con más de 2 dims no soportado")

    if _HAS_SD:
        sd.play(out, sr)
        sd.wait()  # bloqueante
        return
    else:
        # fallback a simpleaudio: necesita WAV PCM16
        if sa is None:
            raise RuntimeError("No se encontró librería para reproducir audio (instala sounddevice o simpleaudio).")
        # convertir a int16 PCM
        ints = np.int16(np.clip(out, -1, 1) * 32767)
        if ints.ndim == 2:
            # simpleaudio requiere interleaved bytes
            interleaved = ints.flatten()
        else:
            interleaved = ints
        play_obj = sa.play_buffer(interleaved, 1 if ints.ndim == 1 else ints.shape[1], 2, sr)
        play_obj.wait_done()
        return

# ---------------------------
# Espectrogramas
# ---------------------------
def spectrogram_image(arr, sr, n_fft=1024, hop_length=None, cmap="viridis"):
    """
    Genera un espectrograma (mel o STFT en dB) y lo devuelve como PIL.Image.
    - arr: numpy 1D
    - sr: sample rate
    """
    arr = np.asarray(arr)
    if arr.ndim > 1:
        # mezclar canales si hay
        arr = np.mean(arr, axis=0)

    if hop_length is None:
        hop_length = n_fft // 4

    # STFT y dB
    S = np.abs(librosa.stft(arr, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Dibujar con matplotlib en un canvas y devolver como PIL
    fig = plt.figure(figsize=(6,4), dpi=150)
    ax = fig.add_subplot(111)
    img = ax.imshow(S_db, origin='lower', aspect='auto')
    ax.axis('off')
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.canvas.print_png(buf)
    plt.close(fig)
    buf.seek(0)
    pil = Image.open(buf).convert("RGB")
    buf.close()
    return pil
def predict_with_model(model_path_or_obj, in_array, in_sr, device="cpu", model_constructor=None, strict=True):
    """
    Wrapper genérico para realizar predicción con un modelo PyTorch.
    - model_path_or_obj: ruta a .pth/.pt o un torch.nn.Module ya instanciado.
      Si la ruta contiene únicamente un state_dict (o checkpoint con 'state_dict'/'model_state_dict'),
      debes proporcionar model_constructor (callable o (callable, args, kwargs)) para instanciar la arquitectura.
    - in_array: numpy 1D (float32)
    - in_sr: sample rate
    - device: "cpu" o "cuda"
    - model_constructor: opcional. Puede ser:
        * callable() -> torch.nn.Module
        * (callable, args_tuple)
        * (callable, args_tuple, kwargs_dict)
    - strict: pasado a load_state_dict(..., strict=strict)
    Devuelve: (out_array numpy float32, sr)
    """
    if torch is None:
        raise RuntimeError("Torch no disponible: instala pytorch para usar predict_with_model")

    model = None

    # Si nos pasan un path, intentamos cargarlo
    if isinstance(model_path_or_obj, str):
        path = model_path_or_obj
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo no encontrado: {path}")

        loaded = torch.load(path, map_location=device)

        # Caso 1: el archivo contiene directamente un Module serializado
        if isinstance(loaded, torch.nn.Module):
            model = loaded

        # Caso 2: el archivo es un dict (checkpoint o state_dict)
        elif isinstance(loaded, dict):
            # Intentar localizar un state_dict en formas comunes
            if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                state_dict = loaded["state_dict"]
            elif "model_state_dict" in loaded and isinstance(loaded["model_state_dict"], dict):
                state_dict = loaded["model_state_dict"]
            else:
                # Heurística: si los valores del dict son tensores, considerarlo state_dict
                vals = list(loaded.values())
                if len(vals) > 0 and isinstance(vals[0], torch.Tensor):
                    state_dict = loaded
                else:
                    # No parece contener state_dict reconocible -> error instructivo
                    raise RuntimeError(
                        "El checkpoint no contiene un state_dict reconocible. Si el archivo no es un modelo "
                        "serializado completo, proporciona 'model_constructor' para instanciar la clase del modelo."
                    )

            # Ahora tenemos un state_dict: necesitamos model_constructor para instanciar la arquitectura
            if model_constructor is None:
                raise RuntimeError(
                    "El archivo contiene únicamente un state_dict. Debes pasar 'model_constructor' para "
                    "instanciar la arquitectura antes de cargar los pesos.\n\n"
                    "Ejemplos:\n"
                    "  # sin args\n"
                    "  app.model_constructor = MyModelClass\n\n"
                    "  # con args/kwargs\n"
                    "  app.model_constructor = (MyModelClass, (arg1, arg2), {'dropout':0.1})"
                )

            # Resolver model_constructor a callable + args/kwargs
            if callable(model_constructor):
                factory = model_constructor
                args = ()
                kwargs = {}
            elif isinstance(model_constructor, (tuple, list)) and len(model_constructor) >= 1:
                factory = model_constructor[0]
                args = model_constructor[1] if len(model_constructor) >= 2 and model_constructor[1] is not None else ()
                kwargs = model_constructor[2] if len(model_constructor) >= 3 and model_constructor[2] is not None else {}
            else:
                raise RuntimeError("model_constructor inválido: debe ser callable o (callable, args, kwargs)")

            if not callable(factory):
                raise RuntimeError("model_constructor[0] debe ser callable (la clase/función que construye el modelo).")

            # Instanciar y cargar state_dict
            model = factory(*args, **kwargs)
            if not isinstance(model, torch.nn.Module):
                raise RuntimeError("La fábrica proporcionada no devolvió un torch.nn.Module.")

            try:
                model.load_state_dict(state_dict, strict=strict)
            except Exception as e:
                raise RuntimeError(f"Error al cargar state_dict en el modelo instanciado: {e}") from e

        else:
            raise RuntimeError("Formato de archivo de modelo no reconocido (ni Module ni dict).")

    else:
        # Si no es path, asumimos que han pasado un Module ya creado
        if isinstance(model_path_or_obj, torch.nn.Module):
            model = model_path_or_obj
        else:
            raise RuntimeError("model_path_or_obj debe ser ruta a fichero o un torch.nn.Module instanciado.")

    # A estas alturas 'model' es un torch.nn.Module listo
    model.to(device)
    model.eval()

    # Preparar la entrada (waveform 1D)
    wav = np.asarray(in_array, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=0)

    maxv = np.max(np.abs(wav)) if wav.size else 1.0
    if maxv > 0:
        wav = wav / maxv

    tensor = torch.from_numpy(wav).float().to(device)

    # Intentar varias formas de entrada comunes
    with torch.no_grad():
        out = None
        last_err = None

        # Intento 1: (1,1,N)
        try:
            x = tensor.unsqueeze(0).unsqueeze(0)
            out = model(x)
        except Exception as e1:
            last_err = e1
            # Intento 2: (1,N)
            try:
                x = tensor.unsqueeze(0)
                out = model(x)
            except Exception as e2:
                last_err = e2
                # Intento 3: tensor plano
                try:
                    out = model(tensor)
                except Exception as e3:
                    last_err = e3
                    raise RuntimeError(
                        "No se pudo ejecutar el modelo con formas de entrada comunes (1,1,N), (1,N) o (N,). "
                        f"Último error: {last_err}"
                    ) from last_err

    # Extraer numpy de la salida
    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, torch.Tensor):
        out_np = out.detach().cpu().numpy()
    else:
        out_np = np.array(out)

    # Normalizar / deshacer batch/channel dims
    if out_np.ndim > 1 and out_np.shape[0] == 1:
        out_np = out_np.squeeze(0)
    if out_np.ndim == 2:
        # (channels, samples) -> mezclar
        if out_np.shape[0] == 1:
            out_np = out_np[0]
        elif out_np.shape[1] == 1:
            out_np = out_np[:, 0]
        else:
            out_np = np.mean(out_np, axis=0)

    out_np = out_np.astype(np.float32)
    m = np.max(np.abs(out_np)) if out_np.size else 1.0
    if m > 0:
        out_np = out_np / m

    return out_np, in_sr
