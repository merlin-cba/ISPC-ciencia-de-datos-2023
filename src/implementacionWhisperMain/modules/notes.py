import datetime
import glob

class Notes:
    def __init__(self):
        self.ruta_notas = 'Archivos/Texto/'
    def file_name(self):
        self.fecha_actual = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        self.nombre_base = f'nota-{ self.fecha_actual}-'
        self.archivos_encontrados = glob.glob(f'{self.ruta_notas}{ self.nombre_base}*')
        self.numero_siguiente = len( self.archivos_encontrados) + 1
        self.nombre_archivo = f'{ self.nombre_base}{self.numero_siguiente}.txt'
        return f'{ self.ruta_notas}{ self.nombre_archivo}'




