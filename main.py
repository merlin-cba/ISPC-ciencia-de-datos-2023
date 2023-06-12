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

def main():
    # Inicializar DataProcessor
    data_processor = DataProcessor()
    data_processor.load_data("datos.csv")
    data_processor.transform_data()

    # Dividir los datos en entrenamiento y prueba
    data_train, data_test = data_processor.split_data("2022-01-01", "2022-12-31")

    # Crear instancia de DemandPredictor
    demand_predictor = DemandPredictor()
    demand_predictor.load_model("model.pkl")

    # Predecir demanda utilizando regresión lineal
    X_train = data_train[['Demanda', 'Fecha']]
    y_train = data_train['Demanda']
    y_pred_linear = demand_predictor.predict_linear(X_train)

    # Predecir demanda utilizando red neuronal
    X_test = data_test[['Demanda', 'Fecha']]
    y_pred_neural = demand_predictor.predict_neural(X_test)

    # Calcular métricas de error
    mse_linear = demand_predictor.calculate_mse(y_train, y_pred_linear)
    mse_neural = demand_predictor.calculate_mse(data_test['Demanda'], y_pred_neural)

    # Calcular precisión de las predicciones
    accuracy_linear = demand_predictor.calculate_accuracy(y_train, y_pred_linear)
    accuracy_neural = demand_predictor.calculate_accuracy(data_test['Demanda'], y_pred_neural)

    # Graficar los datos y las predicciones
    data_plotter = DataPlotter()
    data_plotter.plot_data(data_test.index, data_test['Demanda'], y_pred_linear, y_pred_neural)

    # Mostrar métricas de error y precisión
    print("MSE (Regresión lineal):", mse_linear)
    print("MSE (Red neuronal):", mse_neural)
    print("Precisión (Regresión lineal):", accuracy_linear)
    print("Precisión (Red neuronal):", accuracy_neural)

if __name__ == '__main__':
    main()
