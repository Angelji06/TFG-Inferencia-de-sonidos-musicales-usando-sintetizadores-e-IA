**Autores**: David Cendejas Rodríguez y Ángel Jiménez Izquierdo

**Tutores**: Miguel Gómez-Zamalloa Gil y Jaime Sánchez Hernández

Entrenamiento de un modelo de Red Neuronal para el procesamiento, identificación y reproducción de un timbre dado una muestra.

**Prototipo 1** (PrototipoIA1.ipynb)

Primera toma de contacto con el aprendizaje automático y el audio.

    Dataset: 
        Pequeño (50 elementos)
        Muestras .wav de 1seg
        Generado mediante una lista de frecuencias aleatorias y los tipos de onda clásicos (sine, square, sawtooth, triangle y noise), usando numpy.

    Entrenamiento: 
        Usando tensorflow.
        Mediante una extracción de carácterísticas simple usando MFCC, para después asociarlas a su respectiva forma de onda (etiqueta).

    Modelo: 
        Clasificación
        Poco preciso
        No convolucional
        Arquitectura casi arbitraria.

**Prototipo 2** (PrototipoIA2.ipynb)

Optimizaciones sobre el primer prototipo.

    Dataset: 
        Mismo que el prototipo 1, pero transformado en espectrogramas (.png) usando librosa
        Menos gasto en cómputo y almacenamiento, permite entrenar con modelos de imagen.

    Entrenamiento:
        Usando un modelo de imagen preentrenado de fastAI (resnet34) y especializado con nuestro dataset usando los DataBlocks.

    Modelo: 
        Clasificación
        Bastante preciso (si se introduzca una onda pura y en el pequeño margen de frecuencias entrenado)
        Convolucional
        Arquitectura ajena 

**Prototipo 3** (PrototipoIA4_Regresion.ipynb)

    Dataset:
        Mediano (15k elementos)
        Se generan los wavs mediante un barrido de un sintetizador FM de pyo.
        Se convierten en tensores de espectrogramas via torchaudio.

    Entrenamiento:
        Usando pytorch y una clase llamada SpectrogramTensorDataset que hereda de la clase Dataset de pytorch.
        Se usa una clase llamada SmallCNNRegressor para definir la arquitectura.
        Bucle de entrenamiento de 5 etapas, función de pérdida: MSELoss

    Modelo:
        Regresivo
        Poco preciso

