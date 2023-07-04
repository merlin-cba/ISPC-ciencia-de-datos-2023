from flask import Flask, request
from modelo import DataProcessor
import pandas as pd

app = Flask(__name__)

@app.route("/load_data", methods=["POST"])


class DataProcessor:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = None
    
    def load_data(self):
        self.data = pd.read_csv(self.data_file)
        
    def split_data(self, start_date_train, end_date_train, end_date_validation):
        train_data = self.data[(self.data['date'] >= start_date_train) & (self.data['date'] <= end_date_train)]
        validation_data = self.data[(self.data['date'] > end_date_train) & (self.data['date'] <= end_date_validation)]
        test_data = self.data[self.data['date'] > end_date_validation]
        return train_data, validation_data, test_data
