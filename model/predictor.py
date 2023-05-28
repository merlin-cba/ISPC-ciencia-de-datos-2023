import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster


class DataProcessor:
    def __init__(self, data_file):
        self.data_file = data_file
        #self.data = None
    
    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_file)
            self.data['Fecha'] = pd.to_datetime(self.data['Fecha'])
            self.data = self.data.set_index('Fecha')
            self.data = self.data.asfreq('60min')
            self.data = self.data.sort_index()
            print('Cargo correctamente')
            return self.data
        except:
            print('Error: Data file not found.')
    
    #def transform_data(self):
    #    # Realizar la ingeniería de características aquí
    #    pass

    def split_data(self, start_date_train, end_date_train, end_date_val):
        try:
            self.data       = self.data.loc[start_date_train:end_date_val]
            self.train_data = self.data.loc[start_date_train:end_date_train]  # Datos de entrenamiento
            self.val_data   = self.data.loc[end_date_train:end_date_val]      # Datos de validacion
            self.test_data  = self.data.loc[end_date_val:]                    # Datos de test

            print("Dividio los datos")
            # self.train_data.to_excel('datos_entrenamiento.xlsx')
            return self.train_data, self.val_data, self.test_data
        except:
            print("Error al dividir datos")
            return None, None, None, None

# class DemandPredictor:
#     def __init__(self, model):
#         self.model = model
    
    def train(self, datos):
        try:
            self.forecaster = ForecasterAutoreg(
                regressor = LGBMRegressor(),
                lags = 7
            )
            
            self.metric, self.predictions = backtesting_forecaster(
                            forecaster         = self.forecaster,
                            #y                  = datos.Demanda['2021-10-01 00:00':'2021-11-02 23:00'],#datos['Demanda'],
                            y                  = datos['Demanda']['2021-10-01 00:00':'2021-11-02 23:00'],
                            #X                  = self.create_features(7, 1),
                            initial_train_size = 45,
                            fixed_train_size   = False,
                            steps              = 7,
                            refit              = False,# True,
                            interval           = [10, 90],
                            n_boot             = 1000,
                            metric             = 'mean_squared_error',
                            verbose            = False
                    )
            
            self.forecaster.fit(datos['Demanda']['2021-10-01 00:00':'2021-11-02 23:00'])  # Ajustar el modelo
            
            print('Se entreno correctamente')
            print(self.metric)
            print(self.predictions)
            return self.metric, self.predictions
        except Exception as e:
            print("Error al entrenar modelo", e)
    
    def predict(self, fecha):
        try:
            if isinstance(fecha, str):
                self.fecha = pd.date_range(fecha, periods=1)
            # self.predictions = self.model.predict(self.create_features(self.fecha, 1))
            self.predictions = self.forecaster.predict(len(self.fecha))
            print("La prediccuon es:", self.predictions)
            return self.predictions
            
            # return self.model.predict(X_test)
        except Exception as e:
            print("Error al predecir demanda - ", e)

    def create_features(self, lag_demand: int, window_size: int):# -> pd.DataFrame:
        try:
            # Crear la columna 'y' con la demanda eléctrica
            # self.df = pd.DataFrame({'y': self.data['Demanda']})
            # self.df = pd.DataFrame({'y': self.data})
            # self.df = pd.DataFrame({'y': self.data.values}, index=self.data.index)
            self.df = pd.DataFrame({'y': self.data['Demanda'].values}, index=self.data.index)



            
            # Crear las características usando las funciones rolling y shift de pandas
            for lag in range(1, lag_demand + 1):
                # self.df[f'y_lag{lag}'] = self.df['y'].shift(lag)
                self.df[f'y_lag{lag}'] = self.df['y'].shift(lag * self.df.index.freq)
                

            
            for window in range(1, window_size + 1):
                self.df[f'y_roll{window}_mean'] = self.df['y'].rolling(window=window).mean()
                self.df[f'y_roll{window}_max']  = self.df['y'].rolling(window=window).max()
                self.df[f'y_roll{window}_min']  = self.df['y'].rolling(window=window).min()
            
            # Eliminar las filas con valores faltantes (NaN)
            # self.df = self.df.dropna()
            self.df = self.df.dropna().set_index(self.data.index[lag_demand + window_size - 1:])

            return self.df
        except Exception as e:
            print("Error en funcion: create_features - ", e)
    
    def evaluate(self, y_true, y_pred):
        try:
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            return mae, mse
        except:
            print("Error al evaluar modelo")
            return None, None

# Implementa una clase específica para cada modelo que desees utilizar

#class RandomForestPredictor(DemandPredictor):
#    def __init__(self, dataframe):
#        self.data = dataframe
#        model = RandomForestRegressor()
#        super().__init__(model)

# Ejemplo de uso

# Crear una instancia del DataProcessor y cargar los datos
data_processor = DataProcessor('model\completo_ok.csv')
data_processor.load_data()

# Transformar los datos
# data_processor.transform_data()

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = None,None,None,None
train_data, val_data, test_data = data_processor.split_data( '2021-07-01 00:00:00', '2022-05-01 23:00:00' , '2022-03-31 00:00:00' )

# Crear una instancia del predictor y entrenar el modelo
# predictor = RandomForestPredictor(train_data)
metric, predictions = data_processor.train(train_data)

# Realizar predicciones en el conjunto de prueba
y_pred = data_processor.predict('2023-02-02 14:00')

# Evaluar el modelo
mae, mse = data_processor.evaluate(y_test, y_pred)
print("MAE:", mae)
print("MSE:", mse)