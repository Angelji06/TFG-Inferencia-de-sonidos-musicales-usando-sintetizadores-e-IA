import math
import itertools
import time
from pyo import *
import os
import csv
import glob
import shutil

# ===============================================================================================================
#  Script que genera un dataset de sonidos FM (.wav) mediante un barrido, ademas guarda las etiquetas en un CSV.
# ===============================================================================================================

# Crea la carpeta de salida
DATASETWAV_DIR = "datasetFMwav"
script_dir = os.path.dirname(os.path.abspath(__file__))    # Guarda la ruta de este script
main_dir = os.path.dirname(script_dir)  # Sube al directorio principal del repositorio
datasets_dir = os.path.join(main_dir, "Datasets")
os.makedirs(datasets_dir, exist_ok=True)
out_path = os.path.join(datasets_dir, DATASETWAV_DIR)
if os.path.exists(out_path):    # Si la carpeta ya existe, la borramos completamente
    shutil.rmtree(out_path)
os.makedirs(out_path, exist_ok=True)
print("Carpeta 'out' creada en:", out_path)

# Ruta del CSV de etiquetas
csv_path = os.path.join(out_path, "labels.csv")
# Abrimos el CSV en modo escritura y escribimos la cabecera
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "carrier", "ratio", "index"])  # cabecera

# Inicializamos el servidor de audio de PYO en modo offline (no reproduce en tiempo real)
s = Server(audio="offline", midi="jack", nchnls=1, sr=44100)
s.boot()

# Definición de parámetros para el barrido (granulador)
params = {       # ini   fin   step
    'carrier'   : (100,  2000, 100),    # frecuencia portadora en Hz
    'ratio'     : (0.05, 2,    0.05),   # ratio portadora/moduladora
    'index'     : (1,    10,   0.5),    # índice de modulación
}

# Comprobación rápida del contexto local
vars = locals()
print(vars)  

# Para poder cambiar los parámetros en tiempo real (dentro del bucle de generación) los mapeamos a objetos Sig 
# de pyo. Sig actúa como una señal cuyo valor se puede setear dinámicamente con .setValue(). 
for p in params:
    vars[p] = Sig(params[p][2])     # Aquí inicializamos cada Sig con el valor del step como valor inicial 

# Hacemos referencias locales explícitas a los objetos Sig creados arriba.
carrier = vars['carrier']
ratio = vars['ratio']
index = vars['index']

# Creamos el sintetizador FM de pyo. Importante: 'carrier', 'ratio' y 'index' son objetos Sig (señales) y pyo 
# los aceptará como parámetros dinámicos. (mul y add se dejan fijos: multiplicador y offset de la señal)
synth = FM(
    carrier=carrier,
    ratio=ratio,
    index=index,
    mul=1,
    add=0
)

# synth.ctrl() crea una pequeña GUI de control (útil si se ejecuta con GUI)
synth.ctrl()
synth.out()  # enviamos la señal al servidor (salida)

# Duración en segundos de cada muestra generada (cada grabación)
# Nota: la primer parte de la señal puede necesitar un recorte para estabilizar
TIME = 0.5

# Generador que produce una secuencia aritmética sin construir la lista completa.
def cover(ini, end, step, steps):
    return (ini + k * step for k in range(steps))

# Preparativos previos
ranges = []
ext = {}
for p in params:
    ini, end, step = params[p][0], params[p][1], params[p][2]   # Se obtienen ini, end, step
    
    # Se calcula el número de pasos y lo mete en la lista ranges
    steps = int((end - ini) / step) + 1                     
    ranges.append(steps)                        

    # Se revisa si se acaba exactamente en end
    if (ini + steps * step < end):                  
        print(f"Warning: param {p} does not end")
 
    # Crea un generador que sabe cómo producir esos valores cuando se le itere.
    ext[p] = cover(ini, end, step, steps)
            
# total de combinaciones
combs = math.prod(ranges)
print(f"Total combinations: {combs}")

time.sleep(2)

# ======================================================================================================== #

keys = list(ext)
# itertools.product genera el producto cartesiano de los generadores anteriores.
# iter es un iterador que devolverá, en cada llamada a __next__(), una tupla con valores (carrier, ratio, index) 
iter = itertools.product(*map(ext.get, keys))

# Tomamos la primera combinación para inicializar los Sig y arrancar el proceso
newVals = dict(zip(keys, iter.__next__()))

# Establecemos el valor inicial en cada Sig para que el sintetizador comience
for p in params:
    vars[p].setValue(newVals[p])

out = "file"  # modo de salida: 'file' para grabar a wav, 'speaker' para salida realtime (no usado aquí)
s.recordOptions(dur=TIME, filename='kk', fileformat=0, sampletype=3, quality=0.4) # Configuración de la grabacion del servidor Pyo

g = 0           # Contador global de combinaciones generadas

# Función de comprobación
def check_dataset():
    # Listar todos los WAV generados
    wav_files = sorted(glob.glob(os.path.join(out_path, "*.wav")))
    n_wavs = len(wav_files)

    # Leer el CSV y contar filas (sin cabecera)
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
        n_csv = len(rows) - 1  # restamos cabecera

    print("\n=== COMPROBACIÓN DE DATASET ===")
    print(f"Total combinaciones esperadas: {combs}")
    print(f"Archivos WAV generados: {n_wavs}")
    print(f"Filas en CSV: {n_csv}")

    # Comparación
    if n_wavs == n_csv == combs:
        print("✅ Todo coincide correctamente.")
    else:
        print("⚠ Atención: hay discrepancia en el número de muestras o filas CSV!")

    print("\nArchivos generados (primeros 10 mostrados):")
    for fpath in wav_files[:10]:
        print(fpath)
    if n_wavs > 10:
        print("...")

def update():   # Función llamada periódicamente por Pattern cada TIME segundos.
    global g
    g += 1

    # Paramos el servidor por seguridad antes de cambiar opciones/archivos
    s.stop()

    # Obtenemos la siguiente combinación del producto cartesiano
    try:
        newVals = dict(zip(keys, iter.__next__()))
    except StopIteration:
        print("✅ Todas las combinaciones procesadas.")
        pat2.stop()         # Detener el Pattern
        check_dataset()     # Ejecutar la verificación final
        return

    print(newVals)
    print(f'g: {g}')
    for p in params:
        # Actualizamos cada señal Sig con el nuevo valor
        vars[p].setValue(newVals[p])
        print(p, vars[p].value)
    print("\n\n")

    # Si estamos en modo 'file', creamos un archivo con nombre secuencial y
    # configuramos las opciones de grabación para renderizar TIME segundos.
    if out == 'file':
        file = os.path.join(out_path, f"pru_{g}.wav")

        # Guardamos los labels en el CSV **antes** de renderizar para asegurar consistencia
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"pru_{g}.wav", newVals['carrier'], newVals['ratio'], newVals['index']])

        s.recordOptions(dur=TIME, filename=file, fileformat=0, sampletype=3, quality=0.4)
        # Arrancamos el servidor; como está en modo offline, pyo renderizará la salida
        s.start()

# Pattern ejecuta la función update cada TIME segundos en un hilo de pyo
pat2 = Pattern(function=update, time=TIME).play()

# Arrancamos el servidor para comenzar el proceso (el Pattern ya está en play)
s.start()
