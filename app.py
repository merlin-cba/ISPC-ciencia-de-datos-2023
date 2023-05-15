# Traemos las librerias
from flask import Flask, render_template, request, url_for, redirect, jsonify

# Inicializar la aplicacion
app = Flask(__name__)


@app.before_request
def before_request():
    print("Antes de la petición...")

@app.after_request
def after_request(response):
    print("Después de la petición")
    return response

@app.route('/')
def index():
    data = {
        'titulo': 'Proyecto ISPC - Grupo 3', # Titulo de la pestaña
        'bienvenida': 'Objetivos'
        }
    return render_template('index.html', data=data)

@app.route('/service')
def service():
    service = {
        'titulo': '¿Cómo funciona?', # Titulo de la pestaña
        'descripcion': 'Predicción del consumo eléctrico / entrenamiento...',
        'link_repo': 'https://github.com/merlin-cba/ISPC-ciencia-de-datos-2023'
        }
    return render_template('pages/service.html', data=service)


@app.route('/predict')
def predict():
    predict = {
        'titulo': 'Predicción del consumo eléctrico', # Titulo de la pestaña
        'descripcion': 'Pära conocer...',
        }
    return render_template('pages/predict.html', data=predict)

@app.route('/train')
def train():
    train = {
        'titulo': 'Entrenamiento del modelo', # Titulo de la pestaña
        'descripcion': 'En esta sección podés...',
        }
    return render_template('pages/train.html', data=train)

@app.route('/team')
def team():
    team_members = [
        {
            'name': 'Oscar Ferreira',
            'role': 'Desarrollador Full Stack',
            'image': 'oscar.png'
        },
        {
            'name': 'Emmanuel Reynoso',
            'role': 'Especialista en Análisis de Datos',
            'image': 'emmanuel.png'
        }, 
        {
            'name': 'Cecilia Heredia',
            'role': 'Diseñadora de Interfaces de Usuario',
            'image': 'cecilia.png'
        }
        
    ]
    return render_template('pages/team.html', team_members=team_members)


@app.errorhandler(404)
def pagina_no_encontrada(error):
    return render_template('./errors/404.html'), 404


# Si estamos en el archivo principal
if __name__ == '__main__':
    app.register_error_handler( 404, pagina_no_encontrada)

    # Executa la aplicacion
    app.run(debug=True, port=5000)