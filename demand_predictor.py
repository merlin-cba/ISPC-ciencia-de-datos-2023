import pickle
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

class DemandPredictor:
    def __init__(self):
        self.linear_model = LinearRegression()
        self.neural_network = MLPRegressor(hidden_layer_sizes=(3,), random_state=42, max_iter=500)
        self.loaded_model = False

    def load_model(self, model_file):
        with open(model_file, 'rb') as file:
            model_data = pickle.load(file)
        self.linear_model = model_data['linear_model']
        self.neural_network = model_data['neural_network']
        self.loaded_model = True

    def predict_linear(self, X):
        return self.linear_model.predict(X)

    def predict_neural(self, X):
        return self.neural_network.predict(X)
