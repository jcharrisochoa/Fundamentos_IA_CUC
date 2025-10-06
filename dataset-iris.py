# Importar librería donde está el dataset Iris
# Como alternativa se puede descargar como .xlsx, .xls o .csv y luego cargar el archivo
from sklearn import datasets

# Importar librería para el manejo de dataframes
import pandas as pd

# Cargar al dataset Iris a una variable
iris = datasets.load_iris()

# Crear un dataframe del dataset
datos = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Agregar la columna de etiquetas (clases)
# Para ver los nombres de las clases se usa iris.target_names
datos['target'] = iris.target

print("\n", "-"*30)
print("Primeras cinco filas")
print(datos)

print("\n", "-"*30)
print("Columns",datos.columns)

print("\n", "-"*30)
print("shape:",datos.shape)

print("\n", "-"*30)
print("Size:", datos.size)

print("\n", "-"*30)
print("Número de muestras por cada tipo:")
print(datos["target"].value_counts())

print("\n", "-"*30)
print(datos.describe())
