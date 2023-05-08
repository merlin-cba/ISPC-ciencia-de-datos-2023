import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster, backtesting_forecaster
from skforecast.utils import save_forecaster, load_forecaster
from typing import Tuple, Union, List

class DataProcessor:
    def __init__(self, data_file: str):
        self._data_file = data_file
        self._datos = None
        self._datos_train = None
        self._datos_val = None
        self._datos_test = None
    
    def load_data(self) -> None:
        self._datos = pd.read_csv(self._data_file)
    
    def transform_data(self) -> None:
        pass
    
    def split_data(self, start_date_train: str, end_date_train: str, end_date_validation: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self._datos = self._datos.loc[start_date_train:end_date_validation]
        self._datos_train = self._datos.loc[:end_date_train]
        self._datos_val = self._datos.loc[end_date_train:end_date_validation]
        self._datos_test = self._datos.loc[end_date_validation:]
        return self._datos_train, self._datos_val, self._datos_test

class DemandPredictor:
    def __init__(self, data_file: str):
        self._data_file = data_file
        self._model = None
        self._predictions = None
        self._error = None
        self._train_data = None
        self._val_data = None
        self._test_data = None
    
    def load_data(self) -> None:
        self._train_data, self._val_data, self._test_data = DataProcessor(self._data_file).split_data('2010-01-01', '2017-12-31', '2018-12-31')
    
    def train_model(self) -> None:
        pass
    
    def predict(self, start_date: str, end_date: str) -> pd.DataFrame:
        pass
    
    def plot_data(self) -> None:
        self.plot_data_helper(self._train_data, self._val_data, self._test_data, 'Electricity Demand')
    
    def plot_predictions(self, start_date: str, end_date: str) -> None:
        pass
    
    def plot_error(self) -> None:
        pass

class LinearRegressionPredictor(DemandPredictor):
    def __init__(self, data_file: str):
        super().__init__(data_file)
        self._model = make_pipeline(StandardScaler(), LinearRegression())
    
    def train_model(self) -> None:
        self._model.fit(self._train_data.drop(columns=['Demanda']), self._train_data['Demanda'])
    
    def predict(self, start_date: str, end_date: str) -> pd.DataFrame:
        self._
