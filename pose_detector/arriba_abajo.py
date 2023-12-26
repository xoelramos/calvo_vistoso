import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os
from PIL import Image

# Directorio donde se encuentran tus imágenes
directorios = ['./parriba', './pabajo']

# Arrays para las imágenes y las etiquetas
x = []
y = []

# Tamaño al que se redimensionarán las imágenes (ajustar según tus necesidades)
tamaño_imagen = (64, 64)
for directorio in directorios:
    # Buscar en el directorio
    for archivo in os.listdir(directorio):
        if archivo.startswith('arr'):
            etiqueta = 0
        elif archivo.startswith('aba'):
            etiqueta = 1
        else:
            continue  # Si el archivo no coincide con los criterios, continúa con el siguiente

        ruta_imagen = os.path.join(directorio, archivo)
        imagen = Image.open(ruta_imagen)
        imagen = imagen.convert('RGB')  # Asegúrate de que esté en formato RGB
        imagen = imagen.resize(tamaño_imagen)
        imagen = np.array(imagen)
        x.append(imagen)
        y.append(etiqueta)
print(x,y)
# Asumiendo que X e y ya están definidos
# X = np.array([...])  # Tus imágenes
# y = np.array([...])  # Tus etiquetas

# Normalizar las imágenes
X_normalized = np.array([(x1 / 255.0) for x1 in x])  # Convierte a array y normaliza

# Convertir las etiquetas a categorías one-hot
y_categorical = np.array([to_categorical(y1, num_classes=2) for y1 in y])  # Convierte a array y categoriza

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_categorical, test_size=0.2, random_state=42)

# Definir el modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))  # 3 canales para imágenes RGB
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Aplanar la salida y añadir capas densas
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))  # 4 clases de salida

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()
 
# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=200, batch_size=90, validation_split=0.2)

# Evaluar el modelo con los datos de prueba
score = model.evaluate(X_test, y_test, verbose=0)

# Imprimir el rendimiento del modelo
print(f'Test loss: {score[0]} / Test accuracy: {score[1]*100}')