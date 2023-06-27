"""
============================
Librerías
============================
pip install face-recognition
pip install opencv-python
pip install Pillow
pip install numpy
==============================
Descripción del modelo
==============================
Este modelo de login facial utiliza La librería Face Recognition de Python para reconocer y manipular caras en imágenes. 
Está construida utilizando el reconocimiento facial de última generación de dlib, 
una biblioteca de aprendizaje automático y visión por computadora, con aprendizaje profundo. 
El modelo tiene una precisión del 99.38% en el benchmark Labeled Faces in the Wild, 
lo que significa que es capaz de reconocer correctamente el 99.38% de las caras en un conjunto de datos de prueba.

Permite a los usuarios registrar su nombre de usuario e imágenes de su rostro para utilizarlas en un sistema de
inicio de sesión con reconocimiento facial.

Pasos: 
1. Registrar un nombre de usuario. 
2. Registrar 5 imágenes faciales, mirando a la cámara, en un ambiente con buena iluminación. 
   Detectar los puntos de referencia en las 5 imágenes cargadas y almacenarlos en una lista. 
   Las medidas que  son: distancia entre los ojos, ancho de la boca, ancho del labio superio e inferior, longitud del puente nasal y ancho de la punta e la nariz.  
3. Iniciar sesión con captura de la imagen facial
4. Verificar la similitud entre la magen de inicio  de sesión con las capturadas ene l registro del usuario/a. 



"""

import face_recognition
from face_recognition import face_landmarks
import cv2
import os
from PIL import Image
import numpy as np


class LoginFacial:
    def __init__(self):
        self.usuario_info = None
        self.usuario_img = None
        self.usuario_login = None

    #Registrar el nombre del usuario
    def registrar_usuario(self):
        while True:
            self.usuario_info = input("Ingrese usuario/a: ")
            if os.path.exists(self.usuario_info):
                print("El nombre de usuario ya existe. Por favor, ingrese otro nombre.")
            else:
                archivo = open(self.usuario_info, "w")
                archivo.write(self.usuario_info + "\n")
                archivo.close()
                break
        return self.usuario_info
    
    #Registrar 5 imágenes de tu rostro, mirando de frente a la cámara, en un ambiente con buena iluminación
    def registro_facial(self):
        print("Registramos 5 imagenes de nuestro rostro, mirando a cámara, en un ambiente con buena iluminación.\nPara capturar cada imagen presiona la tecla Escape")
        cap = cv2.VideoCapture(0) #inicia la cámara
        count = 0
        while count < 5:
            ret, frame = cap.read() #captura el último frame
            cv2.imshow('Registro Facial', frame) 
            key = cv2.waitKey(1) #para capturar cada imagen presionamos la tecla Escape
            if key == 27:
                cv2.imwrite(f"{self.usuario_info}_{count}.jpg", frame) #guarda las 5 imagenes tomadas con el nombre del usuario ingresado + número de captura + .jpg
                cv2.imshow(f"Imagen Capturada {count + 1}", frame) #muestra las 5 imagenes tomadas
                count += 1
        cap.release()
        cv2.destroyAllWindows() #limpia los campos para cuando ingresen nuevos usuarios
        self.usuario_img = self.usuario_info
        return self.usuario_img

    """
    Hacemos un recorte del rostro en las imágenes faciales capturadas. 
    Además, para mejorar la presición del programa, vamos a tomar 5 medidas de control: 
       * Distancia entre los ojos.
       * Ancho de la boca.
       * Ancho del labio superio e inferior.
       * Longitud del puente nasal. 
       * Ancho de la punta e la nariz.   """
    def reg_rostro(self):
        eye_distances = []
        mouse_widths = []
        top_lip_heights = []
        bottom_lip_heights = []
        nose_lengths = []
        nose_widths = []

        for i in range(5): #Detectamos puntos de referencia faciales en las 5 imágenes con face_landmarks
            img = f"{self.usuario_img}_{i}.jpg"
            image = face_recognition.load_image_file(img)
            face_landmarks_list = face_recognition.face_landmarks(image)
            for face_landmarks in face_landmarks_list:

                # Calcular la distancia entre los ojos
                left_eye = face_landmarks['left_eye']
                right_eye = face_landmarks['right_eye']
                left_eye_center = np.mean(left_eye, axis=0)
                right_eye_center = np.mean(right_eye, axis=0)
                eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
                eye_distances.append(eye_distance)

                # Calcular el ancho de la boca y alto del labio superior e inferior
                top_lip_landmarks = face_landmarks['top_lip']
                bottom_lip_landmarks = face_landmarks['bottom_lip']
               
                mouse_width = np.linalg.norm(np.array(top_lip_landmarks[0]) - np.array(top_lip_landmarks[6]))
                top_lip_height = np.linalg.norm(np.array(top_lip_landmarks[3]) - np.array(top_lip_landmarks[9]))
                bottom_lip_height = np.linalg.norm(np.array(bottom_lip_landmarks[3]) - np.array(bottom_lip_landmarks[9]))

                mouse_widths.append(mouse_width)
                top_lip_heights.append(top_lip_height)
                bottom_lip_heights.append(bottom_lip_height)

                # Calcular la longitud del puente nasal y el ancho de la punta de la nariz
                nose_bridge = face_landmarks['nose_bridge']
                nose_tip = face_landmarks['nose_tip']
                nose_length = np.linalg.norm(np.array(nose_bridge[0]) - np.array(nose_tip[2]))
                nose_width = np.linalg.norm(np.array(nose_tip[0]) - np.array(nose_tip[2]))

                nose_lengths.append(nose_length)
                nose_widths.append(nose_width)
        
        self.avg_eye_distance_reg = sum(eye_distances) / len(eye_distances)
        self.avg_nose_length_reg = sum(nose_lengths) / len(nose_lengths)
        self.avg_nose_width_reg = sum(nose_widths) / len(nose_widths)
        self.avg_mouse_width_reg = sum(mouse_widths) / len(mouse_widths)
        self.avg_top_lip_height_reg = sum(top_lip_heights) / len(top_lip_heights)
        self.avg_bottom_lip_height_reg = sum(bottom_lip_heights) / len(bottom_lip_heights)

        #guardamos el archivo con los promedios guardados para después usarlos en Login
        with open(f"{self.usuario_info}_promedios.txt", "w") as f:
            f.write(f"{self.avg_eye_distance_reg}\n")
            f.write(f"{self.avg_nose_length_reg}\n")
            f.write(f"{self.avg_nose_width_reg}\n")
            f.write(f"{self.avg_mouse_width_reg}\n")
            f.write(f"{self.avg_top_lip_height_reg}\n")
            f.write(f"{self.avg_bottom_lip_height_reg}\n")

    #Registramos el usuario con el nombre/rostros asociados 
    def registro(self):
        self.registrar_usuario()
        self.registro_facial()
        self.reg_rostro()

#===========================================

    #Inicio de sesión facial. 
    def login_facial(self):
        print("Para iniciar sesión con tu rostro vamos a tomar una foto de cara mirando a cámara, en un ambiente con buena iluminación.\nPara capturar la imagen presioná la tecla Escape")
        self.usuario_login = self.usuario_info
        cap = cv2.VideoCapture(0) #inicia la cámara
        while(True):
            ret,frame = cap.read() #captura el último frame
            cv2.imshow('Login Facial',frame) 
            if cv2.waitKey(1) == 27: #presionamos la tecla Escape
                break
        cv2.imwrite(self.usuario_login+"LOG.jpg",frame) #guardamos la imagen con el nombre del usuario ingresado + LOG.jpg 
        cap.release()
        cv2.destroyAllWindows() #limpia los campos para cuando ingresen nuevos usuarios
        return self.usuario_login

    #Reconocemos el rostro en la la imagen capturada para el login. 
    def log_rostro(self):
        img = self.usuario_login+"LOG.jpg"
        image = face_recognition.load_image_file(img)
        face_locations = face_recognition.face_locations(image)
        #desempaquetamos el valor de face_location en cuatro variables (top, right, bottom y left), que representan las coordenadas del rectángulo que encierra el rostro detectado.
        for face_location in face_locations: 
            top, right, bottom, left = face_location
            #utilizamos estas coordenadas para obtener solo el rostro y alamcenamos esta imagen recortadada en la variable face_image
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image.save(self.usuario_login+"LOG.jpg") #guardamos la imagen recortada del primer rostro detectado
    

    def login(self):
        self.usuario_info = input("Ingrese usuario/a: ")
        if not os.path.exists(self.usuario_info):
            print("El nombre de usuario no existe. Por favor, regístrese primero.")
            return
        self.login_facial()
        img_path = self.usuario_login + "LOG.jpg" #imagen usuarioLOG.jpg
        unknown_image = face_recognition.load_image_file(img_path)
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        im_archivos = os.listdir()
        
        similitudes = []
        for i in range(5):
            if f"{self.usuario_login}_{i}.jpg" in im_archivos: #verifica que haya imágenes usuario_i.jpg
                known_image = face_recognition.load_image_file(f"{self.usuario_login}_{i}.jpg")
                known_encoding = face_recognition.face_encodings(known_image)[0]
                #compara la imagen de login con las 5 tomadas en el registro con la función compare_faces con un valor de tolerancia de 0.6
                #esta función devuelve una lista de valores booleanos que indican si el rostro de login coincide o no con los rostros del registro.
                results = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.6) 
                similitudes.append(results[0])

        
        face_landmarks_list = face_recognition.face_landmarks(unknown_image)
        for face_landmarks in face_landmarks_list:

            # Calcular la distancia entre los ojos
            left_eye = face_landmarks['left_eye']
            right_eye = face_landmarks['right_eye']
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            self.eye_distance_log = np.linalg.norm(left_eye_center - right_eye_center)

            # Calcular el ancho de la boca y alto del labio superior e inferior
            top_lip_landmarks = face_landmarks['top_lip']
            bottom_lip_landmarks = face_landmarks['bottom_lip']
        
            self.mouse_width_log = np.linalg.norm(np.array(top_lip_landmarks[0]) - np.array(top_lip_landmarks[6]))
            self.top_lip_height_log = np.linalg.norm(np.array(top_lip_landmarks[3]) - np.array(top_lip_landmarks[9]))
            self.bottom_lip_height_log = np.linalg.norm(np.array(bottom_lip_landmarks[3]) - np.array(bottom_lip_landmarks[9]))

            # Calcular la longitud del puente nasal y el ancho de la punta de la nariz
            nose_bridge = face_landmarks['nose_bridge']
            nose_tip = face_landmarks['nose_tip']
            self.nose_length_log = np.linalg.norm(np.array(nose_bridge[0]) - np.array(nose_tip[2]))
            self.nose_width_log = np.linalg.norm(np.array(nose_tip[0]) - np.array(nose_tip[2]))

            #abrimos el archivo con los promedios guardados
            with open(f"{self.usuario_info}_promedios.txt", "r") as f:
                self.avg_eye_distance_reg = float(f.readline().strip())
                self.avg_nose_length_reg = float(f.readline().strip())
                self.avg_nose_width_reg = float(f.readline().strip())
                self.avg_mouse_width_reg = float(f.readline().strip())
                self.avg_top_lip_height_reg = float(f.readline().strip())
                self.avg_bottom_lip_height_reg = float(f.readline().strip())


            # calculamos el promedio de medidas tomadas en el registro de las imágenes y lo comparamos con las mismas medidas de inicio de sesión
            def is_similar(avg_value, login_value, threshold=5): #valor de umbral 5
                return abs(avg_value - login_value) < threshold

            is_eye_distance_similar = is_similar(self.avg_eye_distance_reg, self.eye_distance_log)
            is_nose_length_similar = is_similar(self.avg_nose_length_reg, self.nose_length_log)
            is_nose_width_similar = is_similar(self.avg_nose_width_reg, self.nose_width_log)
            is_mouse_width_similar = is_similar(self.avg_mouse_width_reg, self.mouse_width_log)
            is_top_lip_height_similar = is_similar(self.avg_top_lip_height_reg, self.top_lip_height_log)
            is_bottom_lip_height_similar = is_similar(self.avg_bottom_lip_height_reg, self.bottom_lip_height_log)

        num_similar = sum([is_eye_distance_similar, is_nose_length_similar, is_nose_width_similar, is_mouse_width_similar, is_top_lip_height_similar, is_bottom_lip_height_similar])
        if similitudes:
            max_similitud = max(similitudes)
            #para iniciar sesión la función compare_faces debe ser True y al menos 5 de las medidas tomadas
            if max_similitud == True and num_similar >= 5: 
                print("Inicio de sesión exitoso")
                print("Bienvenido/a al sistema usuario: ", self.usuario_login)
            else:
                print("Rostro incorrecto, verifique su usuario")
        else:
            print("Usuario/a no encontrado")
        
        print("\n======================")
        #médidas
        print("Distancia promedio entre los ojos en registro: ", self.avg_eye_distance_reg," y en inicio de sesión: ", self.eye_distance_log)        
        print("Longitud promedio de la nariz: ", self.avg_nose_length_reg," y en inicio de sesión: ", self.nose_length_log) 
        print("Ancho promedio de punta de la nariz: ", self.avg_nose_width_reg," y en inicio de sesión: ", self.nose_width_log) 
        print("Ancho promedio de la boca: ", self.avg_mouse_width_reg," y en inicio de sesión: ", self.mouse_width_log) 
        print("Alto promedio del labio superior para el rostro: ", self.avg_top_lip_height_reg," y en inicio de sesión: ", self.top_lip_height_log) 
        print("Alto promedio del labio inferior para el rostro: ", self.avg_bottom_lip_height_reg ," y en inicio de sesión: ", self.bottom_lip_height_log) 


    def pantalla_principal(self):
      print("Login Inteligente")
      print("1. Iniciar Sesion")
      print("2. Registro")
      print("======================")
      choice=input("Ingrese su opción: ")
      if choice=='1':
          self.login()
      elif choice=='2':
          self.registro()

login_facial = LoginFacial()
login_facial.pantalla_principal()
        
