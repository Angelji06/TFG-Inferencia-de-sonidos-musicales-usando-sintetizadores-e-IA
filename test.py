import numpy as np

def generar_envolvente(t, attack, decay):
    env = np.zeros_like(t, dtype=np.float32)
    
    # Fase de Ataque
    idx_a = t <= attack
    print("idxa    ", idx_a)
    if attack > 0:
        env[idx_a] = t[idx_a] / attack
    else:
        env[idx_a] = 1.0
        
    # Fase de Decaimiento
    idx_d = (t > attack) & (t <= attack + decay)
    print("idxd    ", idx_d)
    if decay > 0:
        env[idx_d] = 1.0 - (t[idx_d] - attack) / decay
        
    # (Lo que supera attack+decay se queda en 0.0)
    return env
# Definimos el tiempo de 0 a 1 segundo (solo 10 puntos para ver la salida clara)
t = np.linspace(0, 1, 11) 
attack = 0.2
decay = 0.5

resultado = generar_envolvente(t, attack, decay)

print("Tiempo (t):    ", t)
print("Envolvente:    ", np.round(resultado, 2))