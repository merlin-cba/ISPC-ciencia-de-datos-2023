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
from typing import List, Tuple


class DataProcessor:
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
    
    def load_data(self) -> None:
        self.data = pd.read_csv(self.data_file)
    
    def transform_data(self) -> None:
        pass
    
    def split_data(self, start_date_train: str, end_date_train: str, end_date_val: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.data = self.data.loc[start_date_train:end_date_val]
        self.train_data = self.data.loc[start_date_train:end_date_train]
        self.val_data = self.data.loc[end_date_train:end_date_val]
        self.test_data = self.data.loc[end_date_val:]
        return self.train_data, self.val_data, self.test_data


class ElectricityDemand(DataProcessor):
    """
    Clase que implementa un procesador de datos específico para la demanda eléctrica.
    """

    def __init__(self, data_file: str):
        super().__init__(data_file)

    def transform_data(self, freq: str = '60min') -> None:
        """
        Transforma los datos de la demanda eléctrica.

        Args:
        - freq: frecuencia a la cual se va a remuestrear los datos. Por defecto '60min'.

        Returns:
        None
        """
        self._datos['Fecha'] = pd.to_datetime(self._datos['Fecha'])
        self._datos = self._datos.set_index('Fecha')
        self._datos = self._datos.asfreq(freq)
        self._datos = self._datos.sort_index()

    def plot_data(self) -> None:
        """
        Grafica los datos de la demanda eléctrica.

        Returns:
        None
        """
        self.plot_data_helper(self._datos_train, self._datos_val, self._datos_test, 'Demanda eléctrica')

    def plot_data_zoom(self, start_date: str, end_date: str) -> None:
        """
        Grafica los datos de la demanda eléctrica en un rango de fechas especificado.

        Args:
        - start_date: fecha de inicio del rango en formato 'YYYY-MM-DD'.
        - end_date: fecha de fin del rango en formato 'YYYY-MM-DD'.

        Returns:
        None
        """
        zoom = (start_date, end_date)
        fig = plt.figure(figsize=(12, 6))
        grid = plt.GridSpec(nrows=8, ncols=1, hspace=0.6, wspace=0)

        main_ax = fig.add_subplot(grid[1:3, :])
        zoom_ax = fig.add_subplot(grid[5:, :])

        self.plot_data_zoom_helper(self._datos_train, self._datos_val, self._datos_test, zoom, main_ax, zoom_ax)

    def calculate_errors(self) -> pd.DataFrame:
        """
        Calcula los errores del modelo de predicción.

        Returns:
        Un objeto de tipo DataFrame de pandas con los errores calculados.
        """
        predictions = self.predict(self._datos_val.index)
        errors = pd.DataFrame()
        errors['MAE'] = [mean_absolute_error(self._datos_val['Demanda'], predictions)]
        errors['RMSE'] = [np.sqrt(mean_squared_error(self._datos_val['Demanda'], predictions))]
        return errors

    def predict(self, dates: Union[str, pd.DatetimeIndex]) -> Union[float, np.ndarray]:
        """
        Realiza la predicción de la demanda eléctrica.

        Args:
        - dates: fecha o fechas a predecir. Puede ser un objeto de tipo str o pd.DatetimeIndex.

        Returns:
        Si se ingresa una sola fecha, devuelve un objeto de tipo float con la predicción para esa fecha.
        Si se ingresa un rango de fechas, devuelve un objeto de tipo np.ndarray con las predicciones para cada fecha.
        """
        if isinstance(dates, str):
            dates = pd.date_range(dates, periods=1)
        predictions = self._model.predict(self.create_features(dates))
        return predictions

   def create_features(self, lag_demand: int, window_size: int) -> pd.DataFrame:
    # Crear la columna 'y' con la demanda eléctrica
    df = pd.DataFrame({'y': self._datos['Demanda eléctrica']})
    
    # Crear las características usando las funciones rolling y shift de pandas
    for lag in range(1, lag_demand + 1):
        df[f'y_lag{lag}'] = df['y'].shift(lag)
    
    for window in range(1, window_size + 1):
        df[f'y_roll{window}_mean'] = df['y'].rolling(window=window).mean()
        df[f'y_roll{window}_max'] = df['y'].rolling(window=window).max()
        df[f'y_roll{window}_min'] = df['y'].rolling(window=window).min()
    
    # Eliminar las filas con valores faltantes (NaN)
    df = df.dropna()
    
    return df


class DemandPredictor:
    def __init__(self, model: LinearRegression, scaler: StandardScaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return y_pred

    def predict_next_day(self, X: np.ndarray) -> float:
        next_day = X[-24:].reshape(1, -1)
        next_day_scaled = self.scaler.transform(next_day)
        next_day_pred = self.model.predict(next_day_scaled)
        return next_day_pred[0]
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        return mae, rmse




class LinearRegressionPredictor:
    """
    Clase que entrena y predice utilizando un modelo de regresión lineal de scikit-learn.
    """
    def __init__(self):
        self.model = None
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]):
        """
        Entrena el modelo con los datos de entrenamiento.
        """
        self.X_train = X_train
        self.y_train = y_train
        
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predice los valores de salida utilizando el modelo entrenado.
        """
        y_pred = self.model.predict(X_test)
        return y_pred


class RidgeModelTrainer:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = None
        
    def train(self, X_train, y_train):
        pipeline = make_pipeline(
            StandardScaler(),
            Ridge(alpha=self.alpha)
        )
        pipeline.fit(X_train, y_train)
        self.model = pipeline
        
    def predict(self, X_test):
        return self.model.predict(X_test)
