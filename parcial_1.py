import os
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Limpiar pantalla según el sistema operativo
if os.name == "nt":  # Windows
    os.system("cls")
else:  # macOS/Linux
    os.system("clear")

# import dataset
heart_disease = fetch_ucirepo(id=45)

# access data
X = heart_disease.data.features
y = heart_disease.data.targets

# Combinar features y targets en un DataFrame
df = pd.concat([X, y], axis=1)

# Filtrar solo columnas relevantes para exploración del miocardio
# Estas son las variables más importantes para análisis cardíaco
miocardio_columns = [
    'age',           # edad
    'sex',           # sexo
    'cp',            # tipo de dolor en el pecho
    'trestbps',      # presión arterial en reposo
    'chol',          # colesterol sérico
    'fbs',           # azúcar en sangre en ayunas
    'restecg',       # electrocardiograma en reposo
    'thalach',       # frecuencia cardíaca máxima alcanzada
    'exang',         # angina inducida por ejercicio
    'oldpeak',       # depresión del ST inducida por ejercicio
    'slope',         # pendiente del segmento ST de ejercicio pico
    'ca',            # número de vasos principales coloreados por fluoroscopia
    'thal',          # talasemia
    'num'            # diagnóstico (target)
]

# Crear DataFrame solo con datos de miocardio
df_miocardio = df[miocardio_columns]

# Mostrar información del dataset de miocardio
print("Información del dataset - Exploración del Miocardio:")
print(f"- Dimensiones: {df_miocardio.shape[0]} filas x {df_miocardio.shape[1]} columnas")
print(f"- Columnas: {list(df_miocardio.columns)}")

# Mostrar los primeros 5 datos en tabla
print(f"\nPrimeros 5 registros - Datos de Miocardio:")
print("=" * 100)
print(df_miocardio.head().to_string(index=False))