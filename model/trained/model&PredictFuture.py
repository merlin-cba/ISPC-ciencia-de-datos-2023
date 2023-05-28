# -*- coding: utf-8 -*-

# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd


# Modelado y Forecasting
# ==============================================================================

from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
import pickle
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error




# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')


# Carga de datos
# ==============================================================================

class GoogleSheetsReader:
    def __init__(self, url):
        self.url = url

    def read(self):
        df_list = pd.read_html(self.url)
        df = pd.DataFrame(df_list[0])
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        return df


# Conversión del formato fecha
# ==============================================================================
class formatoFecha:
    def __init__(self, df):
        self.df = df

    def process_dates(self):
        self.df['Fecha'] = pd.to_datetime(self.df['Fecha'])
        self.df = self.df.set_index('Fecha')
        self.df = self.df.asfreq('60min')
        self.df = self.df.sort_index()
        return self.df


# Backtesting
# ==============================================================================

class Predictor:
    def __init__(self, model_file):
        with open(model_file, 'rb') as file:
            self.forecaster = pickle.load(file)

    def make_predictions(self, data, initial_train_size, steps, interval, n_boot):
        metric, predictions = backtesting_forecaster(
            forecaster=self.forecaster,
            y=data,
            initial_train_size=initial_train_size,
            fixed_train_size=False,
            steps=steps,
            refit=True,
            interval=interval,
            n_boot=n_boot,
            metric='mean_squared_error',
            verbose=False
        )
        return predictions
    
    def predict_future(self, data, steps):
        self.forecaster.fit(data)
        forecast = self.forecaster.predict(steps)
        return forecast
    


class coberturaIntervalo:
    def __init__(self, data):
        self.data = data

    def calculate_coverage(self, predictions):
        inside_interval = np.where(
            (self.data.loc[predictions.index, 'Demanda'] >= predictions['lower_bound']) &
            (self.data.loc[predictions.index, 'Demanda'] <= predictions['upper_bound']),
            True,
            False
        )
        coverage = inside_interval.mean()
        return coverage

class Evaluador:
    def __init__(self, datos):
        self.datos = datos
        
    def calcular_mse(self, datos, predicciones):
        error_mse = mean_squared_error(
            y_true = datos,
            y_pred = predicciones
        )
        return error_mse
        
    

url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTGFdnehaioXiuJbzV5zVtyu3jnt5z5-wtNgqJ_WrcfOdq90Qg_j-esIsxRlBq_NDEvr3JKfNkdBRFw/pubhtml'
reader = GoogleSheetsReader(url)
df = reader.read()

processor = formatoFecha(df)
df = processor.process_dates()
df.info()

predictor = Predictor(model_file='model.pkl')
predictions = predictor.make_predictions(df.Demanda['2021-10-01 00:00':'2021-11-02 23:00'].astype(float), initial_train_size=47, steps=7, interval=[10, 90], n_boot=1000)
print(predictions.shape)

cobIntervalo = coberturaIntervalo(df.astype(float))
coverage = cobIntervalo.calculate_coverage(predictions)
print("=============================================")
print(f"Cobertura del intervalo predicho con los datos de test: {100 * coverage}")
print("=============================================")
print()


evaluador = Evaluador(df)
error_mse = evaluador.calcular_mse(df.Demanda['2021-10-01 00:00':'2021-11-01 00:00'], predictions.iloc[:, 0])
#datos=df.Demanda establecida en predictions - initial_train_size (en este caso 47 registros)
print("=============================================")
print(f"Test error (mse): {error_mse}")
print("=============================================")
print()

# Eliminar filas con valores faltantes en la columna Demanda
df = df.dropna(subset=['Demanda'])

# Rellenar valores faltantes en la columna Demanda con ceros
df['Demanda'] = df['Demanda'].fillna(0)

#demanda de los próximos 10 dias

#demanda de los próximos 10 dias
steps = 10
predicciones = predictor.predict_future(data=df['Demanda'].astype(float), steps=steps)
print("=============================================")
print(f"La demanda de los próximos {steps} días va a ser:")
print("=============================================")
for i, prediccion in enumerate(predicciones):
    print(f'Día {i+1}: {prediccion}')
