import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Cargar el modelo previamente entrenado
model = load_model('modelo_entrenado.h5')

# Lista de las rutas de las imágenes que deseas probar
img_paths = ['xoel_test.jpg', 'xoel_test2.jpg', 'xoel_test3.jpg']

for img_path in img_paths:
    # Cargar la imagen
    img = image.load_img(img_path, target_size=(64, 64))  # Ajusta el tamaño según lo necesites

    # Convertir la imagen a un arreglo numpy
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión extra al principio (lote de una sola muestra)

    # Normalizar la imagen
    img_array = img_array / 255.0  # Ajusta la normalización según la que se usó durante el entrenamiento

    # Realizar la predicción
    prediction = model.predict(img_array)

    img_cv2 = cv2.imread(img_path)
    img_cv2 = cv2.resize(img_cv2, (1080, 860))  # Redimensionar la imagen según lo necesites
    
    
    # Mostrar la etiqueta predicha para la imagen actual
    if prediction[0][0] > 0.5:  # Suponiendo que es un problema de clasificación binaria
        cv2.putText(img_cv2, f'tas parriba', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img_cv2, f'tas pabajo', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Imagen con Etiqueta', img_cv2)
    cv2.waitKey(0)

cv2.destroyAllWindows()