import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from sklearn.model_selection import train_test_split
'''
# Supongamos que tienes dos archivos que contienen tus imágenes y etiquetas
imagenes = np.load('arriba_npy.npy')  # Las imágenes deben ser un array de Numpy
puntos_clave = np.load('puntos_arriba.npy')  # Las coordenadas de los puntos clave como array de Numpy
'''

directorios= ['./arriba_npy','./puntos_Arriba']

i=1

for filename in os.listdir(directorios[0]):
    
    print(filename)
    image_path = os.path.join(directorios[0], filename)

    puntos_calve_path = os.path.join(directorios[1],'arriba'+str(i)+'_puntos.npy')

    print(puntos_calve_path)
    
    imagenes=np.load(image_path)

    puntos_clave=np.load(puntos_calve_path)
    # Cargar la imagen

    # Normalizar las imágenes
    imagenes = imagenes / 255.0

    # Crear generadores de datos (opcional)
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                                height_shift_range=0.1, zoom_range=0.1)

    
    # Crear conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(imagenes, puntos_clave, test_size=0.1)


    # Definir el modelo
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
    # ... agregar más capas convolucionales y de pooling según sea necesario
    model.add(Flatten())
    # ... agregar más capas densas según sea necesario
    # La capa de salida debe tener tantas neuronas como coordenadas necesites predecir
    # Por ejemplo, si tienes 14 puntos clave, necesitas 28 salidas (x e y para cada punto)
    model.add(Dense(28, activation='linear'))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Entrenar el modelo
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64)

    i+=1