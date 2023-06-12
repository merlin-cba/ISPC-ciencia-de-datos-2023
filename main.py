# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
from ElectricityDemand import ElectricityDemand

# Ruta del archivo de datos
data_file = "data.csv"

# Ruta del archivo del modelo entrenado
model_file = "model.pkl"

# Fechas para dividir los datos
start_date_train = "2022-01-01"
end_date_train = "2022-01-31"
end_date_validation = "2022-02-28"

# Crear instancia de ElectricityDemand
electricity_demand = ElectricityDemand(data_file)
electricity_demand.load_data()
electricity_demand.transform_data()

# Dividir los datos
datos_train, datos_val, datos_test = electricity_demand.split_data(start_date_train, end_date_train, end_date_validation)

# Cargar el modelo entrenado
with open(model_file, "rb") as file:
    model = pickle.load(file)

# Predecir con el modelo
X_test = datos_test.drop("target", axis=1).values
y_pred = model.predict(X_test)

# Evaluar el rendimiento (puedes adaptar esta parte según tus necesidades)
mae = mean_absolute_error(datos_test["target"].values, y_pred)
rmse = mean_squared_error(datos_test["target"].values, y_pred, squared=False)

print("MAE:", mae)
print("RMSE:", rmse)
data_plotter = DataPlotter(datos_test, y_pred)
data_plotter.plot_predictions()
