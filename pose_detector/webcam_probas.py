import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Cargar el modelo previamente entrenado
model = load_model('modelo_entrenado.h5')

# Iniciar la captura de la webcam
cap = cv2.VideoCapture(0)  # El argumento 0 indica la cámara predeterminada

while True:
    ret, frame = cap.read()  # Capturar un frame de la webcam
    
    # Redimensionar el frame a 64x64 (ajusta según el modelo)
    img = cv2.resize(frame, (64, 64))
    
    # Convertir el frame a un arreglo numpy
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión extra al principio (lote de una sola muestra)

    # Normalizar la imagen
    img_array = img_array / 255.0  # Ajusta la normalización según la que se usó durante el entrenamiento

    # Realizar la predicción
    prediction = model.predict(img_array)

    # Decodificar la predicción y mostrarla en la ventana de la webcam
    if prediction[0][0] > 0.5:  # Suponiendo clasificación binaria
        cv2.putText(frame, 'parriba', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'pabajo', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Mostrar la ventana con la webcam
    cv2.imshow('Webcam', frame)
    
    # Si se presiona 'q', salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
