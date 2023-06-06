import unittest
from modelo import DataProcessor, ElectricityDemand, RidgeModelTrainer, LinearRegressionPredictor

class ModelTestCase(unittest.TestCase):
    def setUp(self):
        # Configurar el estado inicial para las pruebas
        data_file = 'data.csv'
        self.data_processor = DataProcessor(data_file)
        self.demand = ElectricityDemand(data_file)
        self.trainer = RidgeModelTrainer()
        self.predictor = LinearRegressionPredictor()
        
    def test_data_loading(self):
        # Verificar si se cargan los datos correctamente
        self.data_processor.load_data()
        self.assertIsNotNone(self.data_processor.data)
    
    def test_data_transform(self):
        # Verificar si los datos se transforman correctamente
        self.data_processor.load_data()
        self.data_processor.transform_data()
        self.assertIsNotNone(self.data_processor.data_transformed)
    
    def test_data_split(self):
        # Verificar si la división de datos se realiza correctamente
        self.data_processor.load_data()
        self.data_processor.transform_data()
        self.data_processor.split_data()
        self.assertIsNotNone(self.data_processor.data_train)
        self.assertIsNotNone(self.data_processor.data_val)
        self.assertIsNotNone(self.data_processor.data_test)
    
    def test_demand_prediction(self):
        # Verificar si la predicción de demanda se realiza correctamente
        self.demand.load_data()
        self.demand.transform_data()
        self.demand.split_data()
        self.demand.create_features()
        self.demand.train_model()
        predictions = self.demand.predict()
        self.assertIsNotNone(predictions)
    
    def test_model_training(self):
        # Verificar si el entrenamiento del modelo se realiza correctamente
        self.demand.load_data()
        self.demand.transform_data()
        self.demand.split_data()
        self.demand.create_features()
        self.trainer.train_model(self.demand.features_train, self.demand.target_train)
        self.assertIsNotNone(self.trainer.model)
    
    def test_model_prediction(self):
        # Verificar si la predicción del modelo se realiza correctamente
        self.demand.load_data()
        self.demand.transform_data()
        self.demand.split_data()
        self.demand.create_features()
        self.trainer.train_model(self.demand.features_train, self.demand.target_train)
        predictions = self.predictor.predict(self.trainer.model, self.demand.features_test)
        self.assertIsNotNone(predictions)

if __name__ == '__main__':
    unittest.main()
