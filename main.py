# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from joblib import load
from data_processor import DataProcessor
from demand_predictor import DemandPredictor
from data_plotter import DataPlotter

# Cargar el modelo entrenado
model = load('model.pkl')

# Datos de entrada para la predicción
data_file = 'datos.csv'  # Cambiar por el nombre y ubicación de tus datos
data_processor = DataProcessor(data_file)
data_processor.load_data()
data_processor.transform_data()

# Crear una instancia de DemandPredictor
demand_predictor = DemandPredictor(data_processor)

# Realizar la predicción
predictions = demand_predictor.predict_demand()

# Crear una instancia de DataPlotter
plotter = DataPlotter(data_processor.dates, data_processor.demand, predictions)

# Graficar los datos y las predicciones
plotter.plot_data()

# Calcular el error cuadrático medio (MSE)
mse = plotter.calculate_mse()

# Calcular la precisión (R2 score)
r2 = plotter.calculate_r2_score()

# Imprimir el error cuadrático medio y la precisión
print(f'Error cuadrático medio (MSE): {mse:.2f}')
print(f'Precisión (R2 score): {r2:.2f}')
