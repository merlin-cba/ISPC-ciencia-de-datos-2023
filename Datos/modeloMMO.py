# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple

class DataProcessor:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = None
    
    def load_data(self):
        self.data = pd.read_csv(self.data_file)
    
    def transform_data(self):
        self.data['Fecha'] = pd.to_datetime(self.data['Fecha'])
        self.data = self.data.set_index('Fecha')
        self.data = self.data.asfreq('60min')
        self.data = self.data.sort_index()
    
    def split_data(self, train_start_date, train_end_date, test_start_date, test_end_date):
        train_data = self.data[(self.data.index >= train_start_date) & (self.data.index <= train_end_date)]
        test_data = self.data[(self.data.index >= test_start_date) & (self.data.index <= test_end_date)]
        return train_data, test_data


class DemandPredictor:
    def __init__(self, model_file):
        self.model_file = model_file
        self.linear_model = None
        self.neural_model = None
    
    def load_models(self):
        with open(self.model_file, 'rb') as f:
            self.linear_model, self.neural_model = pickle.load(f)
    
   def train_linear_model(self, X_train, y_train):
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_train, y_train)
    
    def train_neural_network(self, X_train, y_train):
        self.neural_network = MLPRegressor(hidden_layer_sizes=(3,), random_state=1)
        self.neural_network.fit(X_train, y_train)
    
    def predict_linear(self, X):
        return self.linear_model.predict(X)
    
    def predict_neural_network(self, X):
        return self.neural_network.predict(X)
    
    def evaluate_model(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, r2

class DataPlotter:
    def __init__(self):
        self.data = None
    
    def load_data(self, data):
        self.data = data
    
    def plot_data(self):
        plt.plot(self.data.index, self.data['Demanda'], label='Datos reales')
        plt.plot(self.data.index, self.data['Prediccion_Lineal'], label='Predicción Lineal')
        plt.plot(self.data.index, self.data['Prediccion_Neuronal'], label='Predicción Neuronal')
        plt.xlabel('Fecha')
        plt.ylabel('Demanda')
        plt.title('Predicción de Demanda')
        plt.legend()
        plt.show()


