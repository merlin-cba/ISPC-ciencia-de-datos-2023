import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

class DataPlotter:
    def __init__(self, dates, demand, predictions):
        self.dates = dates
        self.demand = demand
        self.predictions = predictions
    
    def plot_data(self):
        # Graficar los datos reales y las predicciones
        plt.figure(figsize=(10, 6))
        plt.plot(self.dates, self.demand, label='Datos reales')
        plt.plot(self.dates, self.predictions, label='Predicciones')
        plt.xlabel('Fecha')
        plt.ylabel('Demanda eléctrica')
        plt.title('Predicciones de demanda eléctrica')
        plt.legend()
        plt.show()
    
    def calculate_mse(self):
        # Calcular el error cuadrático medio (MSE)
        mse = mean_squared_error(self.demand, self.predictions)
        return mse
    
    def calculate_r2_score(self):
        # Calcular la precisión (R2 score)
        r2 = r2_score(self.demand, self.predictions)
        return r2
