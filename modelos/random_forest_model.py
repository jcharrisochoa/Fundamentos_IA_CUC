"""
Modelo de Random Forest para clasificación de enfermedades cardíacas
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def entrenar_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, random_state=42):
    """
    Entrena un modelo de Random Forest para clasificación
    
    Parámetros:
    - X_train: Datos de entrenamiento (features)
    - X_test: Datos de prueba (features)
    - y_train: Etiquetas de entrenamiento
    - y_test: Etiquetas de prueba
    - n_estimators: Número de árboles en el bosque
    - random_state: Semilla para reproducibilidad
    
    Retorna:
    - modelo: Modelo entrenado
    - metricas: Diccionario con métricas de evaluación
    """
    
    print("=" * 60)
    print("ENTRENANDO MODELO RANDOM FOREST")
    print("=" * 60)
    
    # Crear y entrenar el modelo
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    print(f"Entrenando Random Forest con {n_estimators} árboles...")
    rf_model.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Calcular métricas
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    print(f"\nPrecisión en entrenamiento: {accuracy_train:.4f}")
    print(f"Precisión en prueba: {accuracy_test:.4f}")
    
    # Crear reporte de clasificación
    print("\nReporte de Clasificación (Conjunto de Prueba):")
    print("-" * 50)
    print(classification_report(y_test, y_pred_test))
    
    # Mostrar importancia de características
    feature_importance = pd.DataFrame({
        'caracteristica': X_train.columns,
        'importancia': rf_model.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    print("\nImportancia de Características:")
    print("-" * 30)
    print(feature_importance.to_string(index=False))
    
    # Guardar métricas
    metricas = {
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test,
        'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
        'feature_importance': feature_importance,
        'y_pred_test': y_pred_test
    }
    
    return rf_model, metricas


def graficar_importancia_caracteristicas(metricas, top_n=10):
    """
    Grafica la importancia de las características más importantes
    """
    plt.figure(figsize=(10, 6))
    top_features = metricas['feature_importance'].head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importancia'])
    plt.yticks(range(len(top_features)), top_features['caracteristica'])
    plt.xlabel('Importancia')
    plt.title(f'Top {top_n} Características Más Importantes - Bosque Aleatorio')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def graficar_matriz_confusion(y_test, y_pred, titulo="Matriz de Confusión - Bosque Aleatorio"):
    """
    Grafica la matriz de confusión
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Vive', 'Muere'],
                yticklabels=['Vive', 'Muere'])
    plt.title(titulo)
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.tight_layout()
    plt.show()
