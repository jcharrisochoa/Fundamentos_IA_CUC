# Importar librerías
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

# Cargar el dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Gráfico de sépalos y pétalos
fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 fila, 2 columnas

# Primera gráfica: sepal length vs sepal width
scatter1 = axs[0].scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
axs[0].set_xlabel(iris.feature_names[0])
axs[0].set_ylabel(iris.feature_names[1])
axs[0].set_title('Sépalos')
axs[0].legend(scatter1.legend_elements()[0], iris.target_names, title="Classes", loc="lower right")

# Segunda gráfica: petal length vs petal width
scatter2 = axs[1].scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target)
axs[1].set_xlabel(iris.feature_names[2])
axs[1].set_ylabel(iris.feature_names[3])
axs[1].set_title('Pétalos')
axs[1].legend(scatter2.legend_elements()[0], iris.target_names, title="Classes", loc="lower right")

# Mostrar gráfica
plt.show()