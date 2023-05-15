import unittest

# Importar las clases y funciones que se van a probar
from modelo import DataProcessor, DemandPredictor, ElectricityDemand

class TestElectricityDemand(unittest.TestCase):
    
    def setUp(self):
        # Configuración inicial para las pruebas
        self.data_processor = DataProcessor('datos.csv')
        self.data_processor.load_data()
        self.data_processor.transform_data()
        self.electricity_demand = ElectricityDemand(self.data_processor.get_datos())
        self.demand_predictor = DemandPredictor(self.electricity_demand.get_datos())
    
    def tearDown(self):
        # Limpiar configuración después de cada prueba
        self.data_processor = None
        self.electricity_demand = None
        self.demand_predictor = None
    
    def test_create_features(self):
        # Caso de prueba para la función create_features
        features = self.demand_predictor.create_features(lag_demand=3, window_size=4)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(features.shape, (len(self.data_processor.get_datos()) - 3 - 4 + 1, 13))
        self.assertTrue(all([col in features.columns for col in ['y', 'y_lag1', 'y_lag2', 'y_lag3',
                                                                   'y_roll1_mean', 'y_roll1_max', 'y_roll1_min',
                                                                   'y_roll2_mean', 'y_roll2_max', 'y_roll2_min',
                                                                   'y_roll3_mean', 'y_roll3_max', 'y_roll3_min']]))
    
    def test_train_model(self):
        # Caso de prueba para la función train_model
        self.demand_predictor.create_features(lag_demand=3, window_size=4)
        self.demand_predictor.train_model(model_type='linear_regression')
        
        self.assertIsNotNone(self.demand_predictor.get_model())
        self.assertEqual(self.demand_predictor.get_model_type(), 'linear_regression')
    
    def test_predict_demand(self):
        # Caso de prueba para la función predict_demand
        self.demand_predictor.create_features(lag_demand=3, window_size=4)
        self.demand_predictor.train_model(model_type='linear_regression')
        predictions = self.demand_predictor.predict_demand()
        
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(predictions.shape[0], len(self.electricity_demand.get_datos()) - 3 - 4 + 1)
        self.assertEqual(predictions.shape[1], 2)
        self.assertTrue(all([col in predictions.columns for col in ['Fecha', 'Prediccion']]))
    
if __name__ == '__main__':
    unittest.main()
