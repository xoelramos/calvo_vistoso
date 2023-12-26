import cv2
import numpy as np
import os

# Función para manejar eventos de clic en la ventana
def clic_en_ventana(event, x, y, flags, param):
    global puntos
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos.append((x, y))

# Ruta del directorio con las imágenes
directorio = 'parriba'  # Cambiar 'ruta_del_directorio' por la ubicación de tus imágenes
extensiones_validas = ['.jpg', '.jpeg', '.png']  # Extensiones de imagen válidas

# Obtener la lista de archivos en el directorio con las extensiones deseadas
archivos = [f for f in os.listdir(directorio) if f.lower().endswith(tuple(extensiones_validas))]

# Variable para almacenar los puntos
puntos = []

# Configurar la ventana y el manejador de eventos
cv2.namedWindow('Imagen')
cv2.setMouseCallback('Imagen', clic_en_ventana)

# Recorrer cada imagen en el directorio
for archivo in archivos:
    imagen = cv2.imread(os.path.join(directorio, archivo))
    imagen = cv2.resize(imagen, (1080, 860))  # Redimensionar la imagen según lo necesites
    cv2.imshow('Imagen', imagen)
    
    # Esperar a que el usuario haga clic en la ventana
    while True:
        key = cv2.waitKey(1)
        
        # Al presionar 'Enter', salir del bucle
        if key == 13:  # Código ASCII para la tecla 'Enter'
            break
    
    # Guardar los puntos en un archivo .npy
    np.save(archivo.split('.')[0] + '_puntos.npy', np.array(puntos))
    puntos = []

# Cerrar la ventana y finalizar
cv2.destroyAllWindows()

# Convertir las imágenes a archivos .npy
for archivo in archivos:
    imagen = cv2.imread(os.path.join(directorio, archivo))
    np.save(archivo.split('.')[0] + '_imagen.npy', imagen)
