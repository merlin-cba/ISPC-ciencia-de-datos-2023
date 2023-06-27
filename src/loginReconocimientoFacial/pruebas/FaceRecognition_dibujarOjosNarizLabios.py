from PIL import Image, ImageDraw
from face_recognition import face_landmarks
import face_recognition
import numpy as np

# Carga la imagen y detecta los rostros
image = face_recognition.load_image_file("image.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)

# Crea un objeto ImageDraw para dibujar en la imagen
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

# Para cada rostro detectado
for i, face_landmarks in enumerate(face_landmarks_list):
    # Dibuja los puntos de referencia del labio superior
    top_lip_landmarks = face_landmarks['top_lip']
    d.line(top_lip_landmarks, fill=(255, 255, 0), width=1)
    #d.point(top_lip_landmarks, fill=(255, 255, 0))
    
    # Dibuja los puntos de referencia del labio inferior
    bottom_lip_landmarks = face_landmarks['bottom_lip']
    d.line(bottom_lip_landmarks, fill=(255, 255, 0), width=1)
    #d.point(bottom_lip_landmarks, fill=(255, 255, 0))
    
    # Calcula el ancho del labio superior
    top_lip_left = top_lip_landmarks[0]
    top_lip_right = top_lip_landmarks[6]
    top_lip_width = np.linalg.norm(np.array(top_lip_right) - np.array(top_lip_left))

    # Dibuja los puntos de referencia del ojo izquierdo
    left_eye_landmarks = face_landmarks['left_eye']
    d.line(left_eye_landmarks, fill=(255, 255, 0), width=1)
    #d.point(left_eye_landmarks, fill=(255, 255, 0))
    
    # Dibuja los puntos de referencia del ojo derecho
    right_eye_landmarks = face_landmarks['right_eye']
    d.point(right_eye_landmarks, fill=(255, 255, 0))
    
    # Dibuja los puntos de referencia del puente nasal
    nose_bridge_landmarks = face_landmarks['nose_bridge']
    d.point(nose_bridge_landmarks, fill=(255, 255, 0))
    
    # Dibuja los puntos de referencia de la punta de la nariz
    nose_tip_landmarks = face_landmarks['nose_tip']
    d.point(nose_tip_landmarks, fill=(255, 255, 0))
    
    # Calcula el ancho del labio inferior
    bottom_lip_left = bottom_lip_landmarks[0]
    bottom_lip_right = bottom_lip_landmarks[6]
    bottom_lip_width = np.linalg.norm(np.array(bottom_lip_right) - np.array(bottom_lip_left))
    
    # Calcula el alto del labio superior
    top_lip_top = top_lip_landmarks[3]
    top_lip_bottom = top_lip_landmarks[9]
    top_lip_height = np.linalg.norm(np.array(top_lip_bottom) - np.array(top_lip_top))
    
    # Calcula el alto del labio inferior
    bottom_lip_top = bottom_lip_landmarks[3]
    bottom_lip_bottom = bottom_lip_landmarks[9]
    bottom_lip_height = np.linalg.norm(np.array(bottom_lip_bottom) - np.array(bottom_lip_top))

    # Calcula la distancia entre los ojos
    left_eye_center = np.mean(left_eye_landmarks, axis=0)
    right_eye_center = np.mean(right_eye_landmarks, axis=0)
    eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
    
    # Calcula la longitud del puente nasal
    nose_bridge_length = np.linalg.norm(np.array(nose_bridge_landmarks[0]) - np.array(nose_bridge_landmarks[-1]))
    
    # Calcula el ancho de la punta de la nariz
    nose_tip_width = np.linalg.norm(np.array(nose_tip_landmarks[0]) - np.array(nose_tip_landmarks[-1]))
    
    # Imprime el resultado
    print(f"Ancho del labio superior para el rostro {i+1}: {top_lip_width}")
    print(f"Ancho del labio inferior para el rostro {i+1}: {bottom_lip_width}")
    print(f"Alto del labio superior para el rostro {i+1}: {top_lip_height}")
    print(f"Alto del labio inferior para el rostro {i+1}: {bottom_lip_height}")
    print(f"Distancia entre ojos {i+1}: {eye_distance}")
    print(f"Longitud del puente nasal {i+1}: {nose_bridge_length}")
    print(f"Anchura de la punta de la nariz {i+1}: {nose_tip_width}")

# Muestra la imagen con los puntos de referencia dibujados
pil_image.show()
pil_image.save('mi_imagen.png')

