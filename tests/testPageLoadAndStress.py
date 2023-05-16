"""
Testear el tiempo de carga de la web
Rendimiento bajo stress, con 100 usuarios (con un retraso de 0.5 segundos entre cada solicitud)

Guarda los resultados de los tiempos de carga de cada solicitud en results.csv y luego toma es archivo para generar las métricas. 
"""

from concurrent.futures import ThreadPoolExecutor
import requests
from time import sleep, time
import csv
import os
import pandas as pd

class LoadTester:
    def __init__(self, urls, num_requests):
        self.urls = urls
        self.num_requests = num_requests
        self.response_times = []
        self.status_codes = []

    def load_page(self, url):
        start_time = time()
        response = requests.get(url)
        end_time = time()
        self.response_times.append(end_time - start_time)
        self.status_codes.append(response.status_code)
        return response.status_code

    def run_test(self):
        with ThreadPoolExecutor() as executor:
            for url in self.urls:
                for _ in range(self.num_requests):
                    executor.submit(self.load_page, url)
                    sleep(0.5) #retraso de 0.5" entre cada petición
            executor.shutdown(wait=True)
        
        avg_response_time = sum(self.response_times) / len(self.response_times)
        success_rate = self.status_codes.count(200) / len(self.status_codes)

        print("==============================================================")
        print("Prueba de stress: tiempo de respuesta promedio y tasa de éxito")
        print("==============================================================")
        print(f"Average response time: {avg_response_time:.2f} seconds")
        print(f"Success rate: {success_rate:.2%}")

    def save_results(self, file_path):
        dir_path = 'results'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = os.path.join(dir_path, 'results.csv')
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a') as f:
            writer = csv.writer(f, delimiter=';')
            if not file_exists:
                writer.writerow(['URL', 'Tiempo de carga', 'Marca de tiempo'])
            for url, response_time in zip(self.urls * self.num_requests, self.response_times):
                writer.writerow([url, response_time])

class LoadTimeMetrics:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path, delimiter=';')
    
    def calculate_metrics(self):
        metrics = self.df.groupby('URL')['Tiempo de carga'].agg(['mean', 'median', 'std'])
        return metrics


if __name__ == '__main__':
    urls = ["http://127.0.0.1:5000/", "http://127.0.0.1:5000/predict", "http://127.0.0.1:5000/train"] #cambiar las urls una seleccionado el servidor
    num_requests = 100 #100 usuarios
    tester = LoadTester(urls, num_requests)
    tester.run_test()
    
    file_path = os.path.join('results', 'results.csv')
    tester.save_results(file_path)

    metrics_calculator = LoadTimeMetrics(file_path)
    metrics = metrics_calculator.calculate_metrics()
    print()
    print("==============================================================")
    print("Métricas de tiempo de carga")
    print("==============================================================")
    print(metrics) #un tiempo promedio inferior a 2" se considera de buen rendimiento
