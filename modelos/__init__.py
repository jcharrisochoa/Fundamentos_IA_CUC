"""
Paquete de modelos de machine learning para clasificación de enfermedades cardíacas
"""

from .random_forest_model import (
    entrenar_random_forest,
    graficar_importancia_caracteristicas,
    graficar_matriz_confusion as rf_matriz_confusion
)

from .logistic_regression_model import (
    entrenar_logistic_regression,
    graficar_coeficientes,
    graficar_matriz_confusion as lr_matriz_confusion,
    comparar_regularizacion
)

__all__ = [
    'entrenar_random_forest',
    'graficar_importancia_caracteristicas',
    'rf_matriz_confusion',
    'entrenar_logistic_regression',
    'graficar_coeficientes',
    'lr_matriz_confusion',
    'comparar_regularizacion'
]
