# Traemos las librerias
from flask import Flask, render_template, request, url_for, redirect, jsonify
from model.predictor import DataProcessor


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
        'titulo': 'SmartEnergy', # Titulo de la pestaña
        'bienvenida': 'Objetivos'
        }
    return render_template('index.html', data=data)

@app.route('/login')
def login():
    login = {
        
        }
    return render_template('pages/login.html', data=login)

@app.route('/service')
def service():
    service = {
        'titulo': '¿Cómo funciona?', # Titulo de la pestaña
        'descripcion': 'Predicción del consumo eléctrico / entrenamiento...',
        'link_repo': 'https://github.com/merlin-cba/ISPC-ciencia-de-datos-2023'
        }
    return render_template('pages/service.html', data=service)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    instancia = DataProcessor('https://docs.google.com/spreadsheets/d/e/2PACX-1vS2YjVwU3IQAo2ITTvtN6rYqZjHiRYAiX0nH2wcRtnKzkbusE6OHWzkpyq8l6R8pybap_x4MhsJKuAK/pubhtml?gid=0&single=true')
    if request.method == 'POST':
        input_number = int(request.form['input-number']) # Obtener el valor del formulario
        result = prediccion(input_number) # función de prediccion
        return str(result)
    
    return render_template('pages/predict.html', data={
        'titulo': 'Predicción del consumo eléctrico',
        'descripcion': 'Ingrese un valor para hacer la predicción.',
        'prediccion' : instancia.load_data()
    })

# Función de predicción de ejemplo
def prediccion(valor):
    return valor * 2


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        input_link = str(request.form['input-text']) # Obtener el valor del formulario
        result = carga_link(input_link) # función de prediccion
        return str(result)
    return render_template('pages/train.html', data={
        'titulo': 'Entrenamiento del modelo', # Titulo de la pestaña
        'descripcion': 'En esta sección podés...',
        })

def carga_link(valor):
    return valor



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