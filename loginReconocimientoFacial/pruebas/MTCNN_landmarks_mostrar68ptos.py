"""import cv2
import dlib

# Cargar el predictor de puntos de referencia faciales de Dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()



# Cargar la imagen
image = cv2.imread("CeciliaLOG.jpg")


# Detectar puntos de referencia faciales y dibujar un rectángulo alrededor de cada rostro y puntos de referencia faciales en la imagen
image.detect_landmarks(image)

# Mostrar la imagen resultante
cv2.imshow("Image", image)
# Guardar la imagen resultante
cv2.imwrite("output_a.jpg", image)
cv2.waitKey(0)
"""

#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

import cv2
import dlib



# Cargar el predictor de puntos de referencia faciales de Dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

# Cargar la imagen
image = cv2.imread("image.jpg")
alto_original, ancho_original = image.shape[:2]
nuevo_ancho = 150
nuevo_alto = int(nuevo_ancho * (alto_original / ancho_original))
imagen_redimensionada = cv2.resize(image, (nuevo_ancho, nuevo_alto))

# Cambiar el tamaño y suavizar la imagen
gray = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2GRAY)

# Detectar rostros en la imagen
faces = detector(gray)

# Para cada rostro detectado
for face in faces:
    # Obtener las coordenadas del rectángulo del rostro
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    # Dibujar un rectángulo alrededor del rostro
    cv2.rectangle(gray, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Detectar puntos de referencia faciales en el rostro
    landmarks = predictor(gray, face)

    # Para cada punto de referencia facial
    for n in range(0, 68):
        # Obtener las coordenadas del punto de referencia facial
        x = landmarks.part(n).x
        y = landmarks.part(n).y

        # Dibujar un círculo en el punto de referencia facial
        cv2.circle(gray, (x, y), 3, (255, 0, 0), -1)

# Mostrar la imagen resultante
cv2.imshow("Image", gray)
# Guardar la imagen resultante
cv2.imwrite("output_f.jpg", gray)
cv2.waitKey(0)
