# Traemos las librerias
from flask import Flask, render_template, request, url_for, redirect, jsonify
import os
from model.predictor import DataProcessor
from model.trained.modelPredictFuture import Evaluador, coberturaIntervalo, Predictor, formatoFecha, GoogleSheetsReader

# Inicializar la aplicacion
app = Flask(__name__, 
            template_folder='src/templates', 
            static_folder='src/static')

metrica_prediccion = 0

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


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predicciones = None
    predicciones_str = []

    if request.method == 'POST':
        input_link = request.form['input-link']

        reader = GoogleSheetsReader(input_link)
        df = reader.read()

        processor = formatoFecha(df)
        df = processor.process_dates()

        model_file_path = os.path.join(os.path.dirname(__file__), 'model', 'trained', 'model.pkl')
        predictor = Predictor(model_file = model_file_path)

        df = df.dropna(subset=['Demanda'])

        df['Demanda'] = df['Demanda'].fillna(0)

        steps = 10
        predicciones = predictor.predict_future(data=df['Demanda'].astype(float), steps=steps)
        predicciones_str = [f'Día {i+1}: {str(p)}' for i,p in enumerate(predicciones)]
        return f'Predicciones para los próximos 10 días: {str(predicciones_str)}'

    return render_template('pages/predict.html', data={
        'titulo': 'Predicción del consumo eléctrico',
        'descripcion': 'Ingrese un valor para hacer la predicción.',
        'link': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTxBQZwGZEB5WI5-9LUovmkjoevEVyLD6zCt2G_1IISZO1IDkah6O9OCOgByA7dcy5wcUAwLx29D0KM/pubhtml',
        'prediccion' : predicciones_str ,
    })



@app.route('/train', methods=['GET', 'POST'])
def train():
    
    if request.method == 'POST':
        if 'train-btn' in request.form:
            result, data_processor = carga_link()
            train_data, val_data, test_data = data_processor.split_data('2021-07-01 00:00:00', '2022-05-01 23:00:00', '2022-03-31 00:00:00')
            metric, predictions = train_model(data_processor, train_data)  
            
            return str(metric)
        
    return render_template('pages/train.html', data={
        'titulo': 'Entrenamiento del modelo',
        'descripcion': 'En esta sección podés...',
    })

def carga_link():
    data_processor = DataProcessor('model/datos/completo_ok.csv')
    result = data_processor.load_data()
    return result, data_processor

def train_model(data_processor, train_data):
    metric, predictions = data_processor.train(train_data)
    return metric, predictions

# Función de predicción de ejemplo
def realizar_prediccion(valor):
    return valor * metrica_prediccion

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