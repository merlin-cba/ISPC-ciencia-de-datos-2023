import unittest
from my_module import MyClass

class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Configurar el estado inicial para las pruebas
        self.data_file = 'data.csv'
        self.predictor = MyClass(self.data_file)
    
    def test_data_loading(self):
        # Verificar si los datos se cargan correctamente
        self.predictor.load_data()
        self.assertIsNotNone(self.predictor.data)
        self.assertIsInstance(self.predictor.data, pd.DataFrame)
    
    def test_data_transform(self):
        # Verificar si los datos se transforman correctamente
        self.predictor.transform_data()
        self.assertIsNotNone(self.predictor.transformed_data)
        self.assertIsInstance(self.predictor.transformed_data, pd.DataFrame)
    
    def test_data_split(self):
        # Verificar si los datos se dividen correctamente
        self.predictor.split_data()
        self.assertIsNotNone(self.predictor.train_data)
        self.assertIsNotNone(self.predictor.test_data)
        self.assertIsNotNone(self.predictor.validation_data)
        self.assertIsInstance(self.predictor.train_data, pd.DataFrame)
        self.assertIsInstance(self.predictor.test_data, pd.DataFrame)
        self.assertIsInstance(self.predictor.validation_data, pd.DataFrame)
    
    def test_prediction(self):
        # Verificar si las predicciones se realizan correctamente
        self.predictor.load_data()
        self.predictor.transform_data()
        self.predictor.split_data()
        self.predictor.train_model()
        predictions = self.predictor.predict()
        self.assertIsNotNone(predictions)
        self.assertIsInstance(predictions, np.ndarray)

if __name__ == '__main__':
    unittest.main()
