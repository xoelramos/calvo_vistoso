import os
from PIL import Image
import numpy as np

# Directorio donde se encuentran tus imágenes
directorios = ['./circulo', './cuadrado', './pentagonos', './hexagono']

# Arrays para las imágenes y las etiquetas
X = []
Y = []

# Tamaño al que se redimensionarán las imágenes (ajustar según tus necesidades)
tamaño_imagen = (64, 64)
for directorio in directorios:
    # Buscar en el directorio
    for archivo in os.listdir(directorio):
        if archivo.startswith('cir'):
            etiqueta = 0
        elif archivo.startswith('cua'):
            etiqueta = 1
        elif archivo.startswith('pen'):
            etiqueta = 2
        elif archivo.startswith('hex'):
            etiqueta = 3
        else:
            continue  # Si el archivo no coincide con los criterios, continúa con el siguiente

        # Cargar y procesar la imagen
        ruta_imagen = os.path.join(directorio, archivo)
        imagen = Image.open(ruta_imagen)
        imagen = imagen.resize(tamaño_imagen)
        imagen = np.array(imagen)

        # Añadir la imagen y la etiqueta a los arrays
        X.append(imagen)
        Y.append(etiqueta)

print(X,Y)