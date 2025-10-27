import math
import itertools
import time
from pyo import *
import os

# SINTE FM  GENERACIÓN DATASET WAV
DATASETWAV_DIR = "datasetwav"
DATASETFM_DIR = "datasetFM"           
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(script_dir, DATASETWAV_DIR)
os.makedirs(out_path, exist_ok=True)
print("Carpeta 'out' creada en:", out_path)
print("✅ Carpeta creada")
s = Server(audio="offline",midi="jack",nchnls=1,sr=44100)
s.boot()

# FM(carrier=100, ratio=0.5, index=5, mul=1, add=0)

# cto de parámetros utilizados por el granulador con sus intervalos de recorrido y valor de variación 
params = {       # ini fin step
    'carrier'   : (100,2000,100),
    'ratio'     : (0.05,2,0.05),
    'index'     : (1,10,0.5),
    }

vars = locals()
print(vars)

# para hacer variación dinámica de los parámetros los definimos como signals (Sig)
# mapeamos los parámetros a Signal (Sig) para poder variar su valor en tiempo real
for p in params: 
    vars[p] = Sig(params[p][2])


# granulador con los parámetros establecidos asignados a las variables/señal anteriores
synth = FM(
    carrier=carrier,
    ratio=ratio,
    index=index,
    mul=1,  # estos dos los dejamos fijos
    add=0
)
 

synth.ctrl()
synth.out()


# duración de las muestras generadas 
# luego habrá que hacer un trim de la parte inicial para dar tiempo a que la señal
# se estabilice

TIME = 0.5

# lista intensional de valores para un intervalo con un paso
def cover(ini,end,step,steps): 
    return (ini+k*step for k in range(steps))

ranges = []
ext = {}
for p in params:
    ini, end, step = params[p][0], params[p][1], params[p][2]
    steps = int((end-ini)/step)+ 1   
    ranges.append(steps)
    if (ini+steps*step<end):
        print(f"Warning: param {p} does not end")

    # esta lista    
    # ext[p] = [ini+k*step for k in range(steps)]
    # la generamos con llamada a funcion (cover) para que
    # no comparta valores entre variables en las 
    # iteraciones y funcione la evaluacion perezosa        
    ext[p] = cover(ini,end,step,steps)
            
# total de combinaciones
combs = math.prod(ranges)
print(f"Total combinations: {combs}")    

time.sleep(2)

keys = list(ext)    
iter = itertools.product(*map(ext.get, keys))


newVals = dict(zip(keys,iter.__next__()))

for p in params: 
    vars[p].setValue(newVals[p])


out = "file"
#out = "speaker"

s.recordOptions(dur=TIME, filename='kk', fileformat=0, sampletype=3, quality=0.4)


g = 0
def update():    
    global g
    g += 1

    s.stop()
    newVals = dict(zip(keys,iter.__next__()))
    print(newVals)

    print(f'g: {g}')
    for p in params:         
        vars[p].setValue(newVals[p])
        print(p,vars[p].value)
        
    print("\n\n")

    #pitch.setValue(pitch.value + 0.01)
    #grn.pitch = pitch

    if out=='file':
        file = os.path.join(out_path, f"pru_{g}.wav")
        s.recordOptions(dur=TIME, filename=file, fileformat=0, sampletype=3, quality=0.4)
        #s.recordOptions(dur=TIME, filename=file, fileformat=0, sampletype=4)
        #s.recstart()
        s.start()   
    else:
        s.start()

pat2 = Pattern(function=update, time=TIME).play()

s.start()




#s.gui(locals())