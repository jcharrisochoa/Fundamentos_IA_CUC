import os
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Importar los modelos personalizados
from modelos.random_forest_model import entrenar_random_forest, graficar_importancia_caracteristicas, graficar_matriz_confusion as rf_matriz_confusion
from modelos.logistic_regression_model import entrenar_logistic_regression, graficar_coeficientes, graficar_matriz_confusion as lr_matriz_confusion, comparar_regularizacion

# Importar métricas de clasificación
from sklearn.metrics import confusion_matrix
from metricas import compute_metrics, print_report

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

# Análisis exploratorio básico
print(f"\nAnálisis Exploratorio del Dataset:")
print("=" * 50)
print(f"\nEstadísticas descriptivas:")
print(df_miocardio.describe())

# Preparar datos para modelado
print(f"\nPreparando datos para modelado...")
print("=" * 50)

# Manejar valores faltantes
# Rellenar valores faltantes con la mediana para variables numéricas
df_miocardio_clean = df_miocardio.copy()
df_miocardio_clean['ca'] = df_miocardio_clean['ca'].fillna(df_miocardio_clean['ca'].median())
df_miocardio_clean['thal'] = df_miocardio_clean['thal'].fillna(df_miocardio_clean['thal'].median())

# Separar características (X) y variable objetivo (y)
X = df_miocardio_clean.drop('num', axis=1)  # Todas las columnas excepto 'num'
y = df_miocardio_clean['num']  # Variable objetivo

# Convertir variable objetivo a binaria para predicción de supervivencia
# El dataset original tiene valores 0, 1, 2, 3, 4 para diferentes niveles de enfermedad
# Para predicción de supervivencia: 0 = Vive (sin enfermedad o enfermedad leve), 1 = Muere (enfermedad severa)
# Mapeo: 0,1 = Vive (0), 2,3,4 = Muere (1)
y_binary = (y >= 2).astype(int)

print(f"Distribución de la variable objetivo (supervivencia):")
print(f"- Vive (0): {sum(y_binary == 0)} casos ({sum(y_binary == 0)/len(y_binary)*100:.1f}%)")
print(f"- Muere (1): {sum(y_binary == 1)} casos ({sum(y_binary == 1)/len(y_binary)*100:.1f}%)")

# División train/test (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, 
    test_size=0.3, 
    random_state=42, 
    stratify=y_binary  # Mantener proporción de clases
)

print(f"\nDivisión de datos:")
print(f"- Conjunto de entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"- Conjunto de prueba: {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)")

# Verificar distribución de clases en cada conjunto
print(f"\nDistribución de clases en entrenamiento:")
print(f"- Vive: {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")
print(f"- Muere: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")

print(f"\nDistribución de clases en prueba:")
print(f"- Vive: {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
print(f"- Muere: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")


# FUNCIÓN PARA MOSTRAR EL MENÚ
def mostrar_menu():
    print(f"\n" + "="*50)
    print("SELECCIÓN DE MODELO DE PREDICCIÓN DE SUPERVIVENCIA")
    print("="*50)
    print(f"\nSeleccione el modelo a entrenar:")
    print(f"1. Random Forest")
    print(f"2. Regresión Logística")
    print(f"3. Salir")

# FUNCIÓN PARA OBTENER LA OPCIÓN DEL USUARIO
def obtener_opcion():
    while True:
        try:
            opcion = int(input(f"\nIngrese su opción (1, 2 o 3): "))
            if opcion in [1, 2, 3]:
                return opcion
            else:
                print("Por favor, ingrese una opción válida (1, 2 o 3)")
        except ValueError:
            print("Por favor, ingrese un número válido")

# FUNCIÓN PARA EJECUTAR RANDOM FOREST
def ejecutar_random_forest():
    print(f"\n" + "="*50)
    print("INICIANDO ENTRENAMIENTO DE RANDOM FOREST")
    print("="*50)
    
    print(f"\n1. ENTRENANDO RANDOM FOREST")
    print("-" * 40)
    rf_model, rf_metricas = entrenar_random_forest(X_train, X_test, y_train, y_test)
    
    # Graficar importancia de características
    print(f"\nGenerando gráficos para Random Forest...")
    graficar_importancia_caracteristicas(rf_metricas, top_n=10)
    rf_matriz_confusion(y_test, rf_metricas['y_pred_test'], "Matriz de Confusión - Bosque Aleatorio (Predicción de Supervivencia)")
    
    # Calcular métricas detalladas de la matriz de confusión
    # Calcular matriz de confusión
    cm = confusion_matrix(y_test, rf_metricas['y_pred_test'])
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n" + "="*60)
    print("MÉTRICAS DETALLADAS - RANDOM FOREST")
    print("="*60)
    
    # Calcular métricas usando las funciones existentes
    results = compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
    print_report(results, True)
    
    print(f"\n" + "="*50)
    print("ANÁLISIS DE RANDOM FOREST COMPLETADO")
    print("="*50)

    # Pausa para que el usuario pueda leer las métricas
    input("\nPresione una tecla para continuar...")
    
  

# FUNCIÓN PARA EJECUTAR REGRESIÓN LOGÍSTICA
def ejecutar_regresion_logistica():
    print(f"\n" + "="*50)
    print("INICIANDO ENTRENAMIENTO DE REGRESIÓN LOGÍSTICA")
    print("="*50)
    
    print(f"\n2. ENTRENANDO REGRESIÓN LOGÍSTICA REGULARIZADA")
    print("-" * 50)
    lr_model, scaler, lr_metricas = entrenar_logistic_regression(X_train, X_test, y_train, y_test, C=1.0)
    
    # Graficar coeficientes
    print(f"\nGenerando gráficos para Regresión Logística...")
    graficar_coeficientes(lr_metricas, top_n=10)
    lr_matriz_confusion(y_test, lr_metricas['y_pred_test'], "Matriz de Confusión - Regresión Logística (Predicción de Supervivencia)")
    
    # Comparar diferentes valores de regularización
    print(f"\nComparando diferentes valores de regularización...")
    comparar_regularizacion(X_train, X_test, y_train, y_test)
    
    # Calcular métricas detalladas de la matriz de confusión
    # Calcular matriz de confusión
    cm = confusion_matrix(y_test, lr_metricas['y_pred_test'])
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n" + "="*60)
    print("MÉTRICAS DETALLADAS - REGRESIÓN LOGÍSTICA")
    print("="*60)
    
    # Calcular métricas usando las funciones existentes
    results = compute_metrics(tp=tp, fp=fp, tn=tn, fn=fn)
    print_report(results, True)
    
    print(f"\n" + "="*50)
    print("ANÁLISIS DE REGRESIÓN LOGÍSTICA COMPLETADO")
    print("="*50)

    # Pausa para que el usuario pueda leer las métricas
    input("\nPresione una tecla para continuar...")
    
   

# BUCLE PRINCIPAL DEL PROGRAMA
while True:
    mostrar_menu()
    opcion = obtener_opcion()
    
    if opcion == 1:
        ejecutar_random_forest()
    elif opcion == 2:
        ejecutar_regresion_logistica()
    elif opcion == 3:
        print(f"\n" + "="*50)
        print("SALIENDO DEL SISTEMA")
        print("="*50)
        print("¡Gracias por usar el sistema de predicción de supervivencia!")
        break