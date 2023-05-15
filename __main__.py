import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

class DataProcessor:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = None
    
    def load_data(self):
        self.data = pd.read_csv(self.data_file)
    
    def transform_data(self):
        # Realizar la ingeniería de características aquí
        pass
    
    def split_data(self, test_size=0.2, random_state=42):
        X = self.data.drop("target_variable", axis=1)
        y = self.data["target_variable"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

class DemandPredictor:
    def __init__(self, model):
        self.model = model
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return mae, mse

# Implementa una clase específica para cada modelo que desees utilizar

class RandomForestPredictor(DemandPredictor):
    def __init__(self):
        model = RandomForestRegressor()
        super().__init__(model)

class XGBoostPredictor(DemandPredictor):
    def __init__(self):
        model = XGBRegressor()
        super().__init__(model)

class LightGBMPredictor(DemandPredictor):
    def __init__(self):
        model = LGBMRegressor()
        super().__init__(model)

# Ejemplo de uso

# Crear una instancia del DataProcessor y cargar los datos
data_processor = DataProcessor("datos.csv")
data_processor.load_data()

# Transformar los datos
data_processor.transform_data()

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = data_processor.split_data()

# Crear una instancia del predictor y entrenar el modelo
predictor = RandomForestPredictor()
predictor.train(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = predictor.predict(X_test)

# Evaluar el modelo
mae, mse = predictor.evaluate(y_test, y_pred)
print("MAE:", mae)
print("MSE:", mse)
