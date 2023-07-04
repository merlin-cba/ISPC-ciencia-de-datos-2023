import pickle

class ModelTrainer:
    def __init__(self, model_file):
        self.model_file = model_file
        self.model = None
    
    def train_linear_model(self, X_train, y_train):
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_train, y_train)
    
    def train_neural_network(self, X_train, y_train):
        self.neural_network = MLPRegressor(hidden_layer_sizes=(3,), random_state=1)
        self.neural_network.fit(X_train, y_train)

    
    def save_model(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
