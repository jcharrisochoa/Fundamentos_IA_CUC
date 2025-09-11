"""
Modelo de Regresión Logística Regularizada para clasificación de enfermedades cardíacas
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def entrenar_logistic_regression(X_train, X_test, y_train, y_test, C=1.0, random_state=42):
    """
    Entrena un modelo de Regresión Logística regularizada para clasificación
    
    Parámetros:
    - X_train: Datos de entrenamiento (features)
    - X_test: Datos de prueba (features)
    - y_train: Etiquetas de entrenamiento
    - y_test: Etiquetas de prueba
    - C: Parámetro de regularización (inverso de la fuerza de regularización)
    - random_state: Semilla para reproducibilidad
    
    Retorna:
    - modelo: Modelo entrenado
    - scaler: Escalador usado para normalizar los datos
    - metricas: Diccionario con métricas de evaluación
    """
    
    print("=" * 60)
    print("ENTRENANDO MODELO REGRESIÓN LOGÍSTICA REGULARIZADA")
    print("=" * 60)
    
    # Normalizar los datos (importante para regresión logística)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Normalizando datos con StandardScaler...")
    print(f"Parámetro de regularización C = {C}")
    
    # Crear y entrenar el modelo
    lr_model = LogisticRegression(
        C=C,
        random_state=random_state,
        max_iter=1000,
        solver='liblinear'  # Bueno para datasets pequeños
    )
    
    print("Entrenando Regresión Logística...")
    lr_model.fit(X_train_scaled, y_train)
    
    # Realizar predicciones
    y_pred_train = lr_model.predict(X_train_scaled)
    y_pred_test = lr_model.predict(X_test_scaled)
    
    # Calcular métricas
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    
    print(f"\nPrecisión en entrenamiento: {accuracy_train:.4f}")
    print(f"Precisión en prueba: {accuracy_test:.4f}")
    
    # Crear reporte de clasificación
    print("\nReporte de Clasificación (Conjunto de Prueba):")
    print("-" * 50)
    print(classification_report(y_test, y_pred_test))
    
    # Mostrar coeficientes de las características
    feature_coef = pd.DataFrame({
        'caracteristica': X_train.columns,
        'coeficiente': lr_model.coef_[0]
    }).sort_values('coeficiente', key=abs, ascending=False)
    
    print("\nCoeficientes de Características (ordenados por valor absoluto):")
    print("-" * 60)
    print(feature_coef.to_string(index=False))
    
    # Guardar métricas
    metricas = {
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test,
        'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
        'feature_coefficients': feature_coef,
        'y_pred_test': y_pred_test,
        'intercept': lr_model.intercept_[0]
    }
    
    return lr_model, scaler, metricas


def graficar_coeficientes(metricas, top_n=10):
    """
    Grafica los coeficientes de las características más importantes
    """
    plt.figure(figsize=(12, 6))
    top_features = metricas['feature_coefficients'].head(top_n)
    
    colors = ['red' if x < 0 else 'blue' for x in top_features['coeficiente']]
    plt.barh(range(len(top_features)), top_features['coeficiente'], color=colors)
    plt.yticks(range(len(top_features)), top_features['caracteristica'])
    plt.xlabel('Valor del Coeficiente')
    plt.title(f'Top {top_n} Coeficientes de Características - Regresión Logística')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def graficar_matriz_confusion(y_test, y_pred, titulo="Matriz de Confusión - Regresión Logística"):
    """
    Grafica la matriz de confusión
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Vive', 'Muere'],
                yticklabels=['Vive', 'Muere'])
    plt.title(titulo)
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.tight_layout()
    plt.show()


def comparar_regularizacion(X_train, X_test, y_train, y_test, C_values=[0.01, 0.1, 1.0, 10.0, 100.0]):
    """
    Compara diferentes valores de regularización
    """
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE DIFERENTES VALORES DE REGULARIZACIÓN")
    print("=" * 60)
    
    results = []
    
    for C in C_values:
        # Normalizar datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entrenar modelo
        lr_model = LogisticRegression(C=C, random_state=42, max_iter=1000, solver='liblinear')
        lr_model.fit(X_train_scaled, y_train)
        
        # Evaluar
        y_pred_test = lr_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred_test)
        
        results.append({
            'C': C,
            'Accuracy': accuracy,
            'Coef_Sum': np.sum(np.abs(lr_model.coef_[0]))
        })
        
        print(f"C = {C:6.2f} | Accuracy = {accuracy:.4f} | Suma |coef| = {np.sum(np.abs(lr_model.coef_[0])):.4f}")
    
    # Graficar resultados
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results_df['C'], results_df['Accuracy'], 'bo-')
    plt.xscale('log')
    plt.xlabel('Valor de C (Regularización)')
    plt.ylabel('Precisión')
    plt.title('Precisión vs Regularización')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(results_df['C'], results_df['Coef_Sum'], 'ro-')
    plt.xscale('log')
    plt.xlabel('Valor de C (Regularización)')
    plt.ylabel('Suma de |Coeficientes|')
    plt.title('Suma de Coeficientes vs Regularización')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df
