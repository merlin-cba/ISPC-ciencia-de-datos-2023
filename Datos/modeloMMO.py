# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from typing import Tuple


class DataProcessor:

    def __init__(self, data_file: str):
        self._data_file = data_file
        self._datos = None
    
    def load_data(self) -> None:
        self._datos = pd.read_csv(self._data_file)
    
    def transform_data(self) -> None:
        raise NotImplementedError("Subclasses must implement transform_data method.")
    
    def split_data(self, start_date_train: str, end_date_train: str, 
                   end_date_validation: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError("Subclasses must implement split_data method.")


class ElectricityDemand(DataProcessor):

    def __init__(self, data_file: str):
        super().__init__(data_file)
    
    def transform_data(self) -> None:
        self._datos['Fecha'] = pd.to_datetime(self._datos['Fecha'])
        self._datos = self._datos.set_index('Fecha')
        self._datos = self._datos.asfreq('60min')
        self._datos = self._datos.sort_index()
    
    def split_data(self, start_date_train: str, end_date_train: str, 
                   end_date_validation: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self._datos = self._datos.loc[start_date_train:end_date_validation]
        self._datos_train = self._datos.loc[:end_date_train]
        self._datos_val = self._datos.loc[end_date_train:end_date_validation]
        self._datos_test = self._datos.loc[end_date_validation:]
        return self._datos_train, self._datos_val, self._datos_test


class DemandPredictor:

    def __init__(self):
        self._model = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError("Subclasses must implement train method.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement predict method.")
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
        y_pred = self.predict(X)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return mae, rmse


class RidgeModelTrainer(DemandPredictor):

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self._alpha = alpha
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model = Ridge(alpha=self._alpha)
        self._model.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


# Ejemplo de uso
data_file = "data.csv"
start_date_train = "2022-01-01"
end_date_train = "2022-01-31"
end_date_validation = "2022-02-28"
model_file = "model.pkl"

# Crear instancia de ElectricityDemand
electricity_demand = ElectricityDemand(data_file)
electricity_demand.load_data()
electricity_demand.transform_data()

# Dividir los datos
datos_train, datos_val, datos_test = electricity_demand.split_data(start_date_train, end_date_train, end_date_validation)

# Cargar el modelo entrenado
with open(model_file, "rb") as f:
    trained_model = pickle.load(f)

# Usar el modelo entrenado para predecir
X = datos_test.drop("target", axis=1).values
y_true = datos_test["target"].values
y_pred = trained_model.predict(X)

# Evaluar el rendimiento
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)
