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
        'bienvenida': '¡Saludos!'
        }
    return render_template('index.html', data=data)

@app.route('/EDA')
def eda():
    eda = {
        'titulo': 'Análisis Exploratorio de los Datos', # Titulo de la pestaña
        'descripcion': 'La conformacion de nuestros datos es...'
        }
    return render_template('pages/EDA.html', data=eda)

@app.route('/team')
def team():
    team_members = [
        {
            'name': 'Oscar Ferreira',
            'role': 'Desarrollador Full Stack',
            'image': 'oscar.png'
        },
        {
            'name': 'Cecilia Heredia',
            'role': 'Diseñadora de Interfaces de Usuario',
            'image': 'cecilia.png'
        },
        {
            'name': 'Viviana Farabolloni',
            'role': 'Desarrollador de Software',
            'image': 'viviana.png'
        },
        {
            'name': 'Emmanuel Reynoso',
            'role': 'Especialista en Análisis de Datos',
            'image': 'emmanuel.png'
        }
    ]
    return render_template('pages/team.html', team_members=team_members)

@app.route('/about')
def about():
    data = {
        'titulo': 'Breve explicacion del nuestro proyecto', # Titulo de la pestaña
        'descripcion': 'Aqui pasamos a comentarles un poco sobre el proyecto y sus objetivos',
        'link_repo': 'https://github.com/merlin-cba/ISPC-ciencia-de-datos-2023'
        }
    return render_template('pages/about.html', data=data)

@app.errorhandler(404)
def pagina_no_encontrada(error):
    return render_template('./errors/404.html'), 404


# Si estamos en el archivo principal
if __name__ == '__main__':
    app.register_error_handler( 404, pagina_no_encontrada)

    # Executa la aplicacion
    app.run(debug=True, port=5000)