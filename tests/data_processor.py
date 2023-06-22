import pandas as pd

class DataProcessor:
    def __init__(self, data_file):
        self.data_file = data_file
        self.dates = None
        self.demand = None
    
    def load_data(self):
        # Cargar los datos desde el archivo CSV
        data = pd.read_csv(self.data_file)
        
        # Extraer las fechas y la demanda eléctrica
        self.dates = pd.to_datetime(data['Fecha'])
        self.demand = data['Demanda']
    
    def transform_data(self):
        self.dates = pd.to_datetime(self.dates)
        self.demand = self.demand.set_index(self.dates)
        self.demand = self.demand.asfreq('60min')
        self.demand = self.demand.sort_index()
