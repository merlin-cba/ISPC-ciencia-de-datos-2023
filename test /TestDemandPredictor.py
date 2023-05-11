import pandas as pd
from modelo import DemandPredictor

class TestDemandPredictor:
    def test_predict_demand(self):
        # crear datos de prueba
        X_train = pd.DataFrame({'Hour': [0, 1, 2, 3, 4], 'Temperature': [20, 22, 21, 23, 25]})
        y_train = pd.Series([100, 110, 105, 115, 120])
        X_test = pd.DataFrame({'Hour': [5, 6, 7, 8, 9], 'Temperature': [27, 26, 25, 24, 23]})
        # crear modelo y entrenar
        predictor = DemandPredictor()
        predictor.train(X_train, y_train)
        # hacer predicción
        y_pred = predictor.predict(X_test)
        # verificar que la longitud de las predicciones sea igual al número de filas en X_test
        assert len(y_pred) == X_test.shape[0]
