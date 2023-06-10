"""
============================
Librerías
============================
pip install matplotlib
pip install mtcnn
pip install keras
pip install dlib
pip install opencv-python
==============================
Descripción del modelo
==============================
Este modelo de login facial utiliza la biblioteca MTCNN y significa Redes Convolucionales en Cascada Multitarea. 
Es un marco desarrollado para la detección y alineación de rostros. El proceso consta de tres etapas de redes 
convolucionales capaces de reconocer rostros y ubicaciones de puntos de referencia como ojos, nariz y boca (pasa de una 
CNN poco profunda a una más compleja).

Para mejorar la presición del modelo, se cargar el predictor de puntos de referencia faciales de Dlib que cpntine dos módulos. 
Por un lado, el archivo "shape_predictor_68_face_landmarks.dat" que contiene el modelo entrenado para detectar 68 puntos de referencia en una imagen de un rostro humano. 
Por otro, "get_frontal_face_detector", que carga carga un detector de rostros frontales que se utiliza para detectar rostros humanos en una imagen.

Además, se utiliza la función land_marks para detectar puntos de referencia faciales en una imagen, el detector ORB para calcular descriptores en las imágenes y el comparador BFMatcher 
para encontrar coincidencias entre los descriptores de las imágenes y calcula una medida de similitud.
  

Pasos: 
1. Registrar un nombre de usuario. 
2. Registrar 5 imágenes faciales, mirando a la cámara, en un ambiente con buena iluminación. 
3. Guardar las imágenes con uan medida específica y en escala de grises para detectar puntos de referencia faciales con landmarks. 
4. Iniciar sesión con captura de la imagen facial
5. Verificar la similitud entre la magen de inicio  de sesión con las capturadas en el registro del usuario/a. 

"""

import cv2
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import os
import dlib
import numpy as np

class LoginFacialMTCNN:
    def __init__(self):
        self.usuario_info = None
        self.usuario_img = None
        self.usuario_login = None

        #Cargar el predictor de puntos de referencia faciales de Dlib (dos módulos)
        #El archivo "shape_predictor_68_face_landmarks.dat" contiene el modelo entrenado para detectar 68 puntos de referencia en una imagen de un rostro humano
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        #El siguiente módulo, "get_frontal_face_detector", carga un detector de rostros frontales que se utiliza para detectar rostros humanos en una imagen.
        self.detector = dlib.get_frontal_face_detector()

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
    
    #Hacemos un recorte del rostro en las imágenes faciales capturadas. 
    def reg_rostro(self, lista_resultados):
        for i in range(5):
            img = f"{self.usuario_img}_{i}.jpg"
            data = pyplot.imread(img)
            def process_result(j):
                x1,y1,ancho, alto = lista_resultados[j]['box']
                x2,y2 = x1 + ancho, y1 + alto
                pyplot.subplot(1, len(lista_resultados), j+1)
                pyplot.axis('off')
                cara_reg = data[y1:y2, x1:x2]
                #pasamos la imagen a un tamaño específico para poder compararla y en escala de grises para que funcione mejor con lamdmarks
                cara_reg = cv2.cvtColor(cv2.resize(cara_reg,(150,200)), cv2.COLOR_BGR2GRAY)
                cv2.imwrite(f"{self.usuario_img}_processed_{i}.jpg",cara_reg)
            list(map(process_result, range(len(lista_resultados))))
           
    #Registramos el usuario con el nombre/rostros asociados 
    def registro(self):
        self.registrar_usuario()
        self.registro_facial()
        detector = MTCNN()
        for i in range(5):
            img = f"{self.usuario_img}_{i}.jpg"
            pixeles = pyplot.imread(img)
            caras = detector.detect_faces(pixeles)
            self.reg_rostro(caras)

    #Inicio de sesión facial.
    def login_facial(self):
        print("Para iniciar sesión con tu rostro vamos a tomar una foto de cara mirando a cámara, en un ambiente con buena iluminación.\nPara capturar la imagen presioná la tecla Escape")
        self.usuario_login = self.usuario_info
        cap = cv2.VideoCapture(0) #inicia la cámara
        while(True):
            ret,frame = cap.read()
            cv2.imshow('Login Facial',frame)
            if cv2.waitKey(1) == 27: #presionamos la tecla Escape
                break
        cv2.imwrite(self.usuario_login+"LOG.jpg",frame)
        cap.release()
        cv2.destroyAllWindows()
        return self.usuario_login
    
    #Reconocemos el rostro en la la imagen capturada para el login. 
    def log_rostro(self, img, lista_resultados):
        data = pyplot.imread(img)
        for i in range(len(lista_resultados)):
            x1,y1,ancho, alto = lista_resultados[i]['box']
            x2,y2 = x1 + ancho, y1 + alto
            pyplot.subplot(1, len(lista_resultados), i+1)
            pyplot.axis('off')
            cara_reg = data[y1:y2, x1:x2]
            #pasamos la imagen a un tamaño específico para poder compararla y en escala de grises para que funcione mejor con lamdmarks
            cara_reg = cv2.cvtColor(cv2.resize(cara_reg,(150,200)), cv2.COLOR_BGR2GRAY)  
            cv2.imwrite(self.usuario_login+"LOG.jpg",cara_reg)
            return pyplot.imshow(data[y1:y2, x1:x2])
        pyplot.show()
    

    """
    Función para detectar puntos de referencia faciales en una imagen
    
    Toma una imagen como entrada y utiliza el detector de rostros frontales y el predictor de puntos de referencia faciales 
    cargados previamente para detectar rostros y puntos de referencia faciales en la imagen. 
    Luego, dibuja un rectángulo alrededor de cada rostro detectado y dibuja puntos en cada uno de los puntos de referencia faciales.
    """
    def detect_landmarks(self, image):
        # Detectar rostros en la imagen
        faces = self.detector(image)

        # Para cada rostro detectado
        for face in faces:
            # Obtener las coordenadas del rostro
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            # Dibujar un rectángulo alrededor del rostro
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Detectar puntos de referencia faciales en el rostro
            landmarks = self.predictor(image,face)

            # Dibujar puntos de referencia faciales en la imagen
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(image, (x, y), 4, (255, 0, 0), -1)

    """
    Detectamos puntos de referencia faciales en las imágenes utilizando el método detect_landmarks y
    el detector ORB para calcular descriptores en las imágenes. Además, apelamos al comparador BFMatcher 
    para encontrar coincidencias entre los descriptores de las imágenes y calcula una medida de similitud.
    """
    def orb_sim(self, img1,img2):
        # Detectar puntos de referencia faciales en las imágenes
        self.detect_landmarks(img1)
        self.detect_landmarks(img2)

        orb = cv2.ORB_create()
        kpa, descr_a = orb.detectAndCompute(img1, None)
        kpb, descr_b = orb.detectAndCompute(img2, None)
        comp = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #NORM_HAMMING es un tipo de norma que se utiliza para medir la distancia entre dos descriptores
        matches = comp.match(descr_a, descr_b)
        regiones_similares = [i for i in matches if i.distance < 75] #el valor de distancia se puede ajustar
        if len(matches) == 0:
            return 0
        return len(regiones_similares) / len(matches)

#===========================================

    #Inicio de sesión facial. 
    def login(self):
        self.usuario_info = input("Ingrese usuario/a: ")
        if not os.path.exists(self.usuario_info):
            print("El nombre de usuario no existe. Por favor, regístrese primero.")
            return
        self.login_facial()
        img = self.usuario_login+"LOG.jpg"
        pixeles = pyplot.imread(img)
        detector = MTCNN()
        caras = detector.detect_faces(pixeles)
        self.log_rostro(img, caras)
        im_archivos = os.listdir()
        similitudes = []
        for i in range(5):
            if f"{self.usuario_login}_{i}.jpg" in im_archivos:
                rostro_reg = cv2.imread(f"{self.usuario_login}_{i}.jpg",0)
                rostro_log = cv2.imread(self.usuario_login+"LOG.jpg",0)
                similitud = self.orb_sim(rostro_reg, rostro_log)
                similitudes.append(similitud)
        if similitudes:
            max_similitud = max(similitudes)
            if max_similitud >= 0.9: #inicia la sesión si supera a 0.9
                print("Inicio de sesión exitoso")
                print("Bienvenido/a al sistema usuario: ",self.usuario_login)
                print("Compatibilidad con la foto del registro: ",max_similitud)
            else:
                print("Rostro incorrecto, verifique su usuario")
                print("Compatibilidad con la foto del registro: ",max_similitud)
                print("Incompatibilidad de rostros")
        else:
            print("Usuario/a no encontrado")

    def pantalla_principal(self):
        print("Login Inteligente")
        print("1. Iniciar Sesion")
        print("2. Registro")
        print("======================")
        choice = input("Ingrese su opción: ")
        if choice == '1':
            self.login()
        elif choice == '2':
            self.registro()

login_facial = LoginFacialMTCNN()
login_facial.pantalla_principal()

