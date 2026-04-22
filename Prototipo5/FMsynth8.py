import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sounddevice as sd

from logica import generar_envolvente, GEN_PARAMS

# ─────────────────────────── Constantes ────────────────────────────────────

SR         = 44100   # Frecuencia de muestreo (Hz)
BLOCK_SIZE = 512     # Tamaño de bloque del callback de audio (muestras)

# Tecla QWERTY → semitono relativo a C4 (do central, MIDI 60)
#
#  Teclas negras:  W  E     T  Y  U        O  P
#  Teclas blancas: A  S  D  F  G  H  J  K  L
#  Notas:          C  D  E  F  G  A  B  C5 D5
#
KEY_TO_SEMITONE: dict[str, int] = {
    'a':  0,   # C4
    'w':  1,   # C#4
    's':  2,   # D4
    'e':  3,   # D#4
    'd':  4,   # E4
    'f':  5,   # F4
    't':  6,   # F#4
    'g':  7,   # G4
    'y':  8,   # G#4
    'h':  9,   # A4
    'u': 10,   # A#4
    'j': 11,   # B4
    'k': 12,   # C5
    'o': 13,   # C#5
    'l': 14,   # D5
    'p': 15,   # D#5
}

C4_HZ = 261.6255653005986   # Frecuencia de C4 en Hz


def key_to_freq(key: str, octave_shift: int = 0) -> float:
    """Convierte una tecla QWERTY a frecuencia Hz, con desplazamiento de octava."""
    semitone = KEY_TO_SEMITONE[key]
    return C4_HZ * (2.0 ** ((semitone + octave_shift * 12) / 12.0))


# ─────────────────────────── Voice ─────────────────────────────────────────

class Voice:
    """
    Una voz FM monofónica de duración finita.

    Genera muestras en bloques con fase continua entre llamadas sucesivas.
    La voz muere sola cuando termina la envolvente de amplitud.

    Parámetros FM (los mismos 8 del dataset):
        freq       — frecuencia portadora en Hz (controlada por el teclado)
        ratio      — relación moduladora/portadora
        index      — índice de modulación (profundidad)
        amp_attack, amp_sustain, amp_decay  — envolvente de amplitud (segundos)
        mod_attack, mod_decay               — envolvente de modulación (segundos)
    """

    def __init__(self, freq: float, params: dict, gate_mode: bool = False):
        self.freq  = float(freq)
        self.ratio = float(params['ratio'])
        self.index = float(params['index'])
        self.a_att = float(params['amp_attack'])
        self.a_sus = float(params['amp_sustain'])
        self.a_dec = float(params['amp_decay'])
        self.m_att = float(params['mod_attack'])
        self.m_dec = float(params['mod_decay'])

        # Tiempo acumulado desde el inicio de la nota (segundos)
        self.t: float = 0.0

        # El engine elimina la voz cuando done=True
        self.done: bool = False

        # Modo gate: el sustain dura mientras se mantiene la tecla pulsada
        self.gate_mode: bool = gate_mode
        self._gate_open: bool = True        # False cuando se suelta la tecla
        self._decay_start_t: float = None   # Se fija en el hilo de audio al cerrar el gate

        # En modo normal la duración es fija; en gate se calcula dinámicamente
        self._end: float = self.a_att + self.a_sus + self.a_dec + 0.05 if not gate_mode else None

    def gate_off(self) -> None:
        """Llamado desde el engine cuando se suelta la tecla en modo gate."""
        self._gate_open = False
        # _decay_start_t se fija en generate() desde el hilo de audio para
        # evitar condiciones de carrera con self.t

    def _amp_envelope_gate(self, t: np.ndarray) -> np.ndarray:
        """
        Envolvente de amplitud controlada por gate:
          · Mientras el gate está abierto: attack → sustain indefinido a 1.0
          · Tras gate_off: decay a partir del momento de la liberación
        """
        # Registrar el tiempo de inicio del decay en el hilo de audio
        if not self._gate_open and self._decay_start_t is None:
            self._decay_start_t = self.t

        env = np.zeros_like(t, dtype=np.float32)

        if self._gate_open or self._decay_start_t is None:
            # Attack
            idx_a = t <= self.a_att
            env[idx_a] = (t[idx_a] / self.a_att) if self.a_att > 0 else 1.0
            # Sustain indefinido
            env[t > self.a_att] = 1.0
        else:
            dt = self._decay_start_t
            # Antes del gate_off: attack + sustain
            idx_pre = t <= dt
            idx_a   = idx_pre & (t <= self.a_att)
            env[idx_a] = (t[idx_a] / self.a_att) if self.a_att > 0 else 1.0
            env[idx_pre & (t > self.a_att)] = 1.0
            # Decay tras gate_off
            idx_d = (~idx_pre) & (t <= dt + self.a_dec)
            if self.a_dec > 0:
                env[idx_d] = 1.0 - (t[idx_d] - dt) / self.a_dec

        return np.clip(env, 0.0, 1.0)

    def generate(self, n_frames: int) -> np.ndarray:
        """
        Genera n_frames muestras FM float32 y avanza el tiempo interno.
        Después de llamar a este método, self.done puede haberse puesto a True.
        """
        # Vector de tiempo absoluto para este bloque
        t = np.arange(n_frames, dtype=np.float32) / SR + self.t

        # Envolvente de amplitud: modo gate o modo fijo
        if self.gate_mode:
            amp_env = self._amp_envelope_gate(t)
        else:
            amp_env = generar_envolvente(t, self.a_att, self.a_sus, self.a_dec)

        mod_env = generar_envolvente(t, self.m_att, 0.0, self.m_dec)

        # Síntesis FM:  out(t) = amp_env(t) · sin( 2π·fc·t + I·mod_env(t)·sin( 2π·fm·t ) )
        fm_freq   = self.freq * self.ratio
        modulator = np.sin(2.0 * np.pi * fm_freq * t)
        samples   = amp_env * np.sin(
            2.0 * np.pi * self.freq * t + self.index * mod_env * modulator
        )

        # Avance de tiempo
        self.t += n_frames / SR

        # Condición de fin
        if self.gate_mode:
            if not self._gate_open and self._decay_start_t is not None:
                if self.t >= self._decay_start_t + self.a_dec + 0.05:
                    self.done = True
        else:
            if self.t >= self._end:
                self.done = True

        return samples.astype(np.float32)


# ─────────────────────────── Engine ────────────────────────────────────────

class FMSynth8Engine:
    """
    Motor de síntesis FM en tiempo real.

    Abre un sounddevice.OutputStream y mezcla en cada callback todas las
    voces activas (polifonía simple). Las voces se crean con note_on() y
    mueren solas al terminar su envolvente.

    También soporta grabación interna de 2 segundos para la inferencia
    del modelo (start_recording / callback).

    Uso típico:
        engine = FMSynth8Engine()
        engine.start()
        engine.note_on('a', 261.63, params)   # do central
        ...
        engine.stop()
    """

    def __init__(self):
        self._voices: dict[str, Voice] = {}
        self._lock   = threading.Lock()
        self._stream = None

        # ── Grabación ──
        self._recording : bool  = False
        self._rec_buf   : list  = []
        self._rec_n     : int   = 0
        self._rec_target: int   = SR * 2      # 2 segundos de grabación
        self._rec_cb    = None                # callable(audio: np.ndarray)

    # ── Ciclo de vida ────────────────────────────────────────────────────

    def start(self):
        """Abre y arranca el stream de audio."""
        self._stream = sd.OutputStream(
            samplerate = SR,
            channels   = 1,
            dtype      = 'float32',
            blocksize  = BLOCK_SIZE,
            callback   = self._audio_callback,
        )
        self._stream.start()

    def stop(self):
        """Para y cierra el stream de audio."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # ── Control de notas ─────────────────────────────────────────────────

    def note_on(self, key: str, freq: float, params: dict, gate_mode: bool = False):
        """
        Activa una nota para la tecla indicada.
        Si ya había una voz con esa tecla, la reemplaza.
        """
        with self._lock:
            self._voices[key] = Voice(freq, params, gate_mode=gate_mode)

    def note_off(self, key: str):
        """
        Señal de tecla soltada.
        · Modo normal: la voz muere por su envolvente (no se hace nada).
        · Modo gate:   se cierra el gate y la voz entra en fase de decay.
        """
        with self._lock:
            voice = self._voices.get(key)
            if voice is not None and voice.gate_mode:
                voice.gate_off()

    # ── Grabación ────────────────────────────────────────────────────────

    def start_recording(self, callback):
        """
        Graba los próximos 2 segundos de audio sintetizado y llama a
        callback(audio) con un array float32 de longitud SR*2.

        El callback se ejecuta desde el hilo de audio: usa tkinter.after()
        para procesarlo en el hilo principal si actualizas la UI.
        """
        self._rec_buf = []
        self._rec_n   = 0
        self._rec_cb  = callback
        self._recording = True      # activar al final (evita carrera)

    # ── Callback de audio (hilo de tiempo real) ──────────────────────────

    def _audio_callback(
        self,
        outdata  : np.ndarray,
        frames   : int,
        time_info,
        status,
    ):
        """
        Llamado por sounddevice cada BLOCK_SIZE muestras.
        NO bloquear, NO asignar memoria grande, NO lanzar excepciones.
        """
        out = np.zeros(frames, dtype=np.float32)

        with self._lock:
            # 1. Eliminar voces que terminaron en la iteración anterior
            done_keys = [k for k, v in self._voices.items() if v.done]
            for k in done_keys:
                del self._voices[k]

            # 2. Mezclar todas las voces activas
            for voice in self._voices.values():
                out += voice.generate(frames)

        # Soft-clip con tanh: comprime la suma gradualmente en vez de recortarla
        # en duro. np.clip creaba distorsión audible cuando 2+ voces sumaban >1.0.
        #   · 1 voz al máximo  → tanh(0.7) ≈ 0.60  (volumen pleno)
        #   · 2 voces al máx.  → tanh(1.4) ≈ 0.89  (compresión suave)
        #   · 3 voces al máx.  → tanh(2.1) ≈ 0.97  (saturación musical, sin clic)
        out = np.tanh(out * 0.7).astype(np.float32)

        # 4. Grabación (si está activa)
        if self._recording:
            self._rec_buf.append(out.copy())
            self._rec_n += frames
            if self._rec_n >= self._rec_target:
                self._recording = False
                recorded = np.concatenate(self._rec_buf)[:self._rec_target]
                self._rec_buf = []
                cb, self._rec_cb = self._rec_cb, None
                if cb is not None:
                    cb(recorded)

        outdata[:, 0] = out


# ─────────────────────────── Constantes UI ─────────────────────────────────

# Configuración de cada slider: (clave_params, etiqueta, min, max, valor_defecto)
_SLIDER_PARAMS = [
    ('ratio',       'Ratio  (fm/fc)',   0.05,  2.0,  0.50),
    ('index',       'Index  (I)',       1.0,  10.0,  3.0 ),
    ('amp_attack',  'Amp Attack  (s)',  0.015, 1.9,  0.10),
    ('amp_sustain', 'Amp Sustain (s)',  0.015, 1.9,  0.30),
    ('amp_decay',   'Amp Decay   (s)',  0.015, 1.9,  0.50),
    ('mod_attack',  'Mod Attack  (s)',  0.010, 1.9,  0.10),
    ('mod_decay',   'Mod Decay   (s)',  0.010, 1.9,  0.80),
]

# Valores por defecto para el botón "Reset"
_DEFAULTS = {
    'ratio': 0.50, 'index': 3.0,
    'amp_attack': 0.10, 'amp_sustain': 0.30, 'amp_decay': 0.50,
    'mod_attack': 0.10, 'mod_decay': 0.80,
}

# Layout del piano visual
# Teclas blancas en orden de izquierda a derecha
_WHITE_KEYS = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l']

# Teclas negras: (índice de la blanca a su izquierda, tecla)
# La tecla negra se centra en el límite derecho de esa blanca.
_BLACK_LAYOUT = [
    (0, 'w'),   # C#4 entre C(0) y D(1)
    (1, 'e'),   # D#4 entre D(1) y E(2)
    (3, 't'),   # F#4 entre F(3) y G(4)
    (4, 'y'),   # G#4 entre G(4) y A(5)
    (5, 'u'),   # A#4 entre A(5) y B(6)
    (7, 'o'),   # C#5 entre C5(7) y D5(8)
    (8, 'p'),   # D#5 después de D5(8)
]

# Nombres de los 8 parámetros FM (orden del modelo)
PARAM_NAMES = [
    'carrier', 'ratio', 'index',
    'amp_attack', 'amp_sustain', 'amp_decay',
    'mod_attack', 'mod_decay',
]


# ─────────────────────────── Ventana Tkinter ───────────────────────────────

class FMSynth8Window(tk.Toplevel):
    """
    Sintetizador FM interactivo con teclado QWERTY.

    Abre una ventana Toplevel con:
      • 7 sliders de parámetros FM (ratio, index, amp_attack, amp_sustain, amp_decay, mod_attack, mod_decay)
      • Controles de octava y botones Aleatorio/Reset
      • Piano visual que resalta las teclas pulsadas
      • (Fase 3) Panel de inferencia con el modelo CNN

    Uso desde main.py:
        win = FMSynth8Window(root, model=model, means=means,
                             stds=stds, device=str(device))
    """

    # Dimensiones del piano visual
    _KW  = 44   # Anchura tecla blanca (px)
    _KH  = 80   # Altura tecla blanca (px)
    _BW  = 28   # Anchura tecla negra (px)
    _BH  = 50   # Altura tecla negra (px)
    _KM  = 8    # Margen lateral (px)

    def __init__(self, parent, initial_params=None, original_params=None):
        """
        parent          — ventana padre (tk.Tk o tk.Toplevel)
        initial_params  — lista de 8 valores FM en el orden PARAM_NAMES,
                          procedente de la predicción del modelo.
        original_params — lista de 8 valores FM originales del dataset,
                          si se conocen (el WAV estaba en el training set).
        """
        super().__init__(parent)
        self.title('FMSynth8 — Sintetizador FM')
        self.resizable(True, True)

        self._octave    = 0      # desplazamiento de octava actual
        self._held      = set()  # teclas actualmente pulsadas
        self._gate_mode = False  # True → sustain controlado por duración de pulsación

        self._prediction_params = initial_params
        self._original_params   = original_params

        # Motor de audio
        self._engine = FMSynth8Engine()
        self._engine.start()

        self._build_ui()

        # Cargar parámetros predichos en los sliders si se proporcionan
        if initial_params is not None:
            self._apply_params(initial_params, label='Parámetros cargados desde la predicción del modelo.')

        # Captura de teclado en la propia ventana
        self.bind('<KeyPress>',   self._on_key_press)
        self.bind('<KeyRelease>', self._on_key_release)
        self.focus_set()

        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ── Construcción de la UI ────────────────────────────────────────────

    def _build_ui(self):
        # ── Sección 1: Sliders de parámetros FM ──
        pf = tk.LabelFrame(self, text='Parámetros FM', padx=8, pady=6)
        pf.pack(fill='x', padx=10, pady=(10, 4))

        self._sliders: dict[str, tk.DoubleVar] = {}

        for row, (name, label, mn, mx, default) in enumerate(_SLIDER_PARAMS):
            tk.Label(pf, text=label, width=18, anchor='e').grid(
                row=row, column=0, padx=4, pady=3)

            var = tk.DoubleVar(value=default)
            ttk.Scale(pf, from_=mn, to=mx, variable=var,
                      orient='horizontal', length=300).grid(
                row=row, column=1, padx=6, pady=3)

            disp = tk.StringVar(value=f'{default:.3f}')
            entry = tk.Entry(pf, textvariable=disp, width=7,
                             font=('Courier', 9), justify='right')
            entry.grid(row=row, column=2, padx=4, pady=3)

            commit = self._make_entry_commit(disp, var, mn, mx)
            entry.bind('<Return>',   lambda e, c=commit: (c(e), self.focus_set()))
            entry.bind('<FocusOut>', commit)

            var.trace_add('write', self._make_disp_trace(var, disp))
            self._sliders[name] = var

        # ── Sección 2: Controles (Aleatorio / Reset / Octava) ──
        cf = tk.Frame(self)
        cf.pack(fill='x', padx=10, pady=4)

        tk.Button(cf, text='Aleatorio', width=10,
                  command=self._randomize).pack(side='left', padx=4)
        tk.Button(cf, text='Reset', width=8,
                  command=self._reset).pack(side='left', padx=4)

        self._btn_gate = tk.Button(cf, text='Sustain: slider', width=16,
                                   bg='#f0f0f0', command=self._toggle_gate_mode)
        self._btn_gate.pack(side='left', padx=(16, 4))

        # Botones para cargar predicción / original
        self._btn_pred = tk.Button(
            cf, text='Predicción', width=10, bg='#cce5ff',
            state='normal' if self._prediction_params is not None else 'disabled',
            command=self._load_prediction)
        self._btn_pred.pack(side='left', padx=4)

        self._btn_orig = tk.Button(
            cf, text='Original', width=10, bg='#d4edda',
            state='normal' if self._original_params is not None else 'disabled',
            command=self._load_original)
        self._btn_orig.pack(side='left', padx=4)

        tk.Label(cf, text='Octava:').pack(side='left', padx=(20, 4))
        tk.Button(cf, text='−', width=2,
                  command=self._octave_down).pack(side='left', padx=2)
        self._oct_label = tk.Label(cf, text='0', width=3,
                                   font=('Arial', 11, 'bold'))
        self._oct_label.pack(side='left')
        tk.Button(cf, text='+', width=2,
                  command=self._octave_up).pack(side='left', padx=2)

        tk.Label(cf, text='(Z / X para cambiar octava)',
                 fg='#888', font=('Arial', 8)).pack(side='left', padx=(10, 0))

        # ── Sección 3: Piano visual ──
        kf = tk.LabelFrame(self, text='Teclado  (QWERTY → notas)', padx=8, pady=6)
        kf.pack(fill='x', padx=10, pady=4)
        self._build_keyboard(kf)

        # ── Sección 4: Info de parámetros cargados ──
        self._info_var = tk.StringVar(value='Parámetros por defecto.')
        tk.Label(self, textvariable=self._info_var,
                 fg='#555', font=('Arial', 9)).pack(pady=(2, 8))

    def _build_keyboard(self, parent):
        """Dibuja el piano visual en un Canvas."""
        KW, KH, BW, BH, M = self._KW, self._KH, self._BW, self._BH, self._KM
        n_white    = len(_WHITE_KEYS)
        canvas_w   = M + n_white * KW + M
        canvas_h   = KH + M * 2

        canvas = tk.Canvas(parent, width=canvas_w, height=canvas_h, bg='#2b2b2b')
        canvas.pack()
        self._canvas = canvas
        self._key_rects: dict[str, tuple] = {}  # tecla → (rect_id, 'white'|'black')

        # Teclas blancas (primero, para que las negras queden encima)
        for i, key in enumerate(_WHITE_KEYS):
            x0 = M + i * KW
            x1 = x0 + KW - 2
            rid = canvas.create_rectangle(
                x0, M, x1, M + KH, fill='#f0f0f0', outline='#888', width=1)
            canvas.create_text(
                x0 + KW // 2, M + KH - 14,
                text=key.upper(), font=('Arial', 9, 'bold'), fill='#666')
            self._key_rects[key] = (rid, 'white')

        # Teclas negras (encima, usando la fórmula de centro de borde)
        for wi, key in _BLACK_LAYOUT:
            cx  = M + (wi + 1) * KW        # centro X = borde derecho de blanca[wi]
            x0  = cx - BW // 2
            x1  = x0 + BW
            rid = canvas.create_rectangle(
                x0, M, x1, M + BH, fill='#1a1a1a', outline='#000', width=1)
            canvas.create_text(
                (x0 + x1) // 2, M + BH - 10,
                text=key.upper(), font=('Arial', 8), fill='#bbb')
            self._key_rects[key] = (rid, 'black')

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_disp_trace(var: tk.DoubleVar, sv: tk.StringVar):
        """Devuelve un trace que actualiza sv con el valor formateado de var."""
        def _trace(*_):
            sv.set(f'{var.get():.3f}')
        return _trace

    @staticmethod
    def _make_entry_commit(sv: tk.StringVar, var: tk.DoubleVar, mn: float, mx: float):
        """Devuelve un handler que parsea el Entry y mueve el slider al confirmar."""
        def _commit(event=None):
            try:
                val = float(sv.get().replace(',', '.'))
                var.set(max(mn, min(mx, val)))
            except ValueError:
                sv.set(f'{var.get():.3f}')   # revertir si no es un número válido
        return _commit

    def _get_params(self) -> dict:
        """Devuelve un dict con los valores actuales de todos los sliders."""
        return {name: var.get() for name, var in self._sliders.items()}

    def _set_key_color(self, key: str, pressed: bool):
        """Ilumina o apaga una tecla en el piano visual."""
        if key not in self._key_rects:
            return
        rid, kind = self._key_rects[key]
        if kind == 'black':
            color = '#5566ff' if pressed else '#1a1a1a'
        else:
            color = '#aabbff' if pressed else '#f0f0f0'
        self._canvas.itemconfig(rid, fill=color)

    # ── Eventos de teclado ───────────────────────────────────────────────

    def _on_key_press(self, event):
        key = event.keysym.lower()

        # Atajos de octava
        if key == 'z':
            self._octave_down(); return
        if key == 'x':
            self._octave_up();   return

        if key in KEY_TO_SEMITONE and key not in self._held:
            self._held.add(key)
            freq = key_to_freq(key, self._octave)
            self._engine.note_on(key, freq, self._get_params(), gate_mode=self._gate_mode)
            self._set_key_color(key, True)

    def _on_key_release(self, event):
        key = event.keysym.lower()
        if key in self._held:
            self._held.discard(key)
            self._engine.note_off(key)
            self._set_key_color(key, False)

    # ── Controles de parámetros ──────────────────────────────────────────

    def _octave_up(self):
        if self._octave < 3:
            self._octave += 1
            self._oct_label.config(text=str(self._octave))

    def _octave_down(self):
        if self._octave > -3:
            self._octave -= 1
            self._oct_label.config(text=str(self._octave))

    def _toggle_gate_mode(self):
        self._gate_mode = not self._gate_mode
        if self._gate_mode:
            self._btn_gate.config(text='Sustain: pulsación', bg='#aee8a0')
        else:
            self._btn_gate.config(text='Sustain: slider', bg='#f0f0f0')

    def _randomize(self):
        import random
        for name, var in self._sliders.items():
            lo, hi = GEN_PARAMS.get(name, (0.0, 1.0))
            var.set(random.uniform(lo, hi))

    def _reset(self):
        for name, var in self._sliders.items():
            var.set(_DEFAULTS[name])
        self._info_var.set('Parámetros por defecto.')

    def _apply_params(self, params, label='Parámetros cargados.'):
        """
        Carga una lista de 8 valores FM (orden PARAM_NAMES) en los sliders.
        El parámetro 'carrier' se ignora porque lo controla el teclado.
        """
        pred_dict = dict(zip(PARAM_NAMES, params))
        for name, var in self._sliders.items():
            if name in pred_dict:
                lo, hi = GEN_PARAMS.get(name, (0.0, 1.0))
                var.set(max(lo, min(hi, pred_dict[name])))
        self._info_var.set(label)

    def _load_prediction(self):
        if self._prediction_params is not None:
            self._apply_params(self._prediction_params, label='Parámetros de la predicción del modelo.')

    def _load_original(self):
        if self._original_params is not None:
            self._apply_params(self._original_params, label='Parámetros originales del dataset.')

    # ── Cierre limpio ────────────────────────────────────────────────────

    def _on_close(self):
        self._engine.stop()
        self.destroy()

