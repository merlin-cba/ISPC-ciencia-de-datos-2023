import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

class DemandPredictor:
    def __init__(self, model_file):
        self.model_file = model_file
        self.linear_model = None
        self.neural_model = None
    
    def load_models(self):
        with open(self.model_file, 'rb') as f:
            self.linear_model, self.neural_model = pickle.load(f)
    
    def predict(self, features):
        linear_predictions = self.linear_model.predict(features)
        neural_predictions = self.neural_model.predict(features)
        return linear_predictions, neural_predictions
    
    def calculate_accuracy(self, actual_values, predicted_values):
        mse = mean_squared_error(actual_values, predicted_values)
        r2 = r2_score(actual_values, predicted_values)
        return mse, r2
