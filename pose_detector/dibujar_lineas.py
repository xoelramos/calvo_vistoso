import cv2
import numpy as np

# Crear una imagen en blanco
img = np.zeros((500, 500, 3), dtype=np.uint8)

# Coordenadas del rectángulo
x, y, w, h = 100, 100, 300, 200

# Dibujar el rectángulo sin relleno (solo contorno)
Rectangulo=cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)

# Obtener el centro del rectángulo
centro_x = x + w // 2
centro_y = y + h // 2

# Puntos extremos de la línea
punto_inicio = (x, y + h // 2)
punto_fin = (x + w, y + h // 2)

# Dibujar la línea
cv2.line(img, punto_inicio, punto_fin, (255, 255, 255), 2)

# Dibujar el círculo en el extremo de la línea
cv2.circle(img, punto_fin, 5, (0, 0, 255), -1)

# Mostrar la imagen con la línea y el círculo
cv2.imshow('Rectangulo con linea y circulo', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
