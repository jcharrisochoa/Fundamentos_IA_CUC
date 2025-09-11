## Fundamentos IA – Universidad de la Costa (CUC)

Repositorio creado para el desarrollo de ejercicios pedagógicos de la carrera de Ingeniería Electrónica de la Universidad de la Costa.

- Autor: Johan Charris
- Ubicación: Colombia
- Periodo: 2025

### Objetivo
Apoyar el aprendizaje práctico de conceptos de Inteligencia Artificial, métricas de evaluación y manejo básico de herramientas en Python, usando ejemplos simples y reproducibles.

### Librerías utilizadas
- Python 3.x (biblioteca estándar).
- `math` para operaciones numéricas elementales.
- NumPy (cálculo numérico)
  - Instalación: `pip install numpy`
  - Documentación: `https://numpy.org/doc/`
- pandas (manipulación de datos)
  - Instalación: `pip install pandas`
  - Documentación: `https://pandas.pydata.org/docs/`
- Matplotlib (visualización)
  - Instalación: `pip install matplotlib`
  - Documentación: `https://matplotlib.org/stable/`
- scikit-learn (aprendizaje automático)
  - Instalación: `pip install scikit-learn`
  - Documentación: `https://scikit-learn.org/stable/`
- ucimlrepo (carga de datasets UCI)
  - Instalación: `pip install ucimlrepo`
  - Documentación: `https://pypi.org/project/ucimlrepo/`
- seaborn (visualización avanzada)
  - Instalación: `pip install seaborn`
  - Documentación: `https://seaborn.pydata.org/`
- Opcional para actividades en cuadernos:
  - Jupyter
    - Instalación: `pip install jupyter`
    - Documentación: `https://jupyter.org/`
  - IPykernel
    - Instalación: `pip install ipykernel`
    - Documentación: `https://ipykernel.readthedocs.io/en/latest/`

---

### Entorno virtual recomendado
Para aislar dependencias y facilitar la reproducción, se recomienda usar un entorno virtual con `venv`:

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows (PowerShell):
```powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
```

Instalar las librerías requeridas:
```bash
pip install numpy pandas matplotlib scikit-learn ucimlrepo seaborn
# Opcional para cuadernos
pip install jupyter ipykernel
```

O usar el archivo de dependencias del proyecto:
```bash
pip install -r requirements.txt
```

Para salir del entorno: `deactivate`.

---

## Proyecto de Predicción de Supervivencia por Infarto al Miocardio

### ¿Qué hace este proyecto?
Este proyecto ayuda a predecir si una persona vivirá o morirá después de un infarto al corazón. Usa dos métodos diferentes de inteligencia artificial: Random Forest y Regresión Logística.

### Estructura del Proyecto

```
Fundamentos_IA_CUC/
├── parcial_1.py                    # Script principal con análisis completo
├── modelos/                        # Directorio de modelos de ML
│   ├── __init__.py                # Inicializador del paquete
│   ├── random_forest_model.py     # Modelo Random Forest
│   └── logistic_regression_model.py # Modelo Regresión Logística
├── metricas/                       # Directorio de métricas de evaluación
│   ├── __init__.py                # Inicializador del paquete
│   └── metricas_clasificacion.py  # Funciones para calcular métricas
├── requirements.txt               # Dependencias del proyecto
└── README.md                     # Este archivo
```

### ¿Qué incluye este proyecto?

#### 1. Análisis de Datos
- Carga información de pacientes con problemas del corazón
- Muestra estadísticas básicas de los datos
- Ve cuántos pacientes viven y cuántos mueren

#### 2. Preparación de Datos
- Arregla datos que faltan
- Convierte la información a formato binario (vive/muere)
- Divide los datos: 70% para entrenar, 30% para probar

#### 3. Modelos de Predicción

##### Random Forest (Bosque Aleatorio)
- Usa 100 árboles de decisión
- Muestra qué características son más importantes
- Crea gráficos fáciles de entender

##### Regresión Logística
- Método matemático para predecir
- Normaliza los datos para mejor precisión
- Muestra qué factores influyen más en la supervivencia

#### 4. Resultados y Gráficos
- **Primero**: Muestra gráficos de colores para entender mejor los resultados
- **Después**: Calcula métricas detalladas de precisión usando funciones especializadas
- Compara diferentes configuraciones
- Usa el archivo `metricas_clasificacion.py` para cálculos precisos

### ¿Qué tan bien funciona?

| Modelo | Precisión en Entrenamiento | Precisión en Prueba |
|--------|----------------------------|---------------------|
| Random Forest | 96.70% | 83.52% |
| Regresión Logística | 88.21% | 84.62% |

**El mejor modelo**: Regresión Logística con 84.62% de precisión.

### Distribución de Pacientes
- **Vive**: 219 pacientes (72.3%)
- **Muere**: 84 pacientes (27.7%)

### ¿Cómo usar este proyecto?

1. **Instalar las herramientas necesarias**:
```bash
pip install -r requirements.txt
```

2. **Ejecutar el programa**:
```bash
python parcial_1.py
```

3. **Elegir qué hacer**:
   - **Opción 1**: Usar Random Forest (Bosque Aleatorio)
   - **Opción 2**: Usar Regresión Logística
   - **Opción 3**: Salir del programa

### Características del Sistema

- **Fácil de usar**: Menú simple para elegir qué hacer
- **Todo en español**: Gráficos y textos fáciles de entender
- **Análisis completo**: Muestra qué tan bien funciona cada modelo
- **Menú interactivo**: Después de cada análisis, regresa al menú principal
- **Flexible**: Puedes usar diferentes modelos o salir cuando quieras
- **Orden lógico**: Primero muestra gráficos, después métricas detalladas
- **Cálculos precisos**: Usa funciones especializadas para métricas de clasificación

### Información de los Datos

- **Total de pacientes**: 303 personas
- **Información que se usa**: 14 características médicas
- **Objetivo**: Predecir si el paciente vive o muere
- **Datos importantes**: edad, sexo, dolor en el pecho, presión arterial, colesterol, etc.

### Detalles Técnicos

- Se arreglan datos que faltan usando el valor promedio
- Se convierte la información a dos opciones: vive (0) o muere (1)
- Se divide la información: 70% para aprender, 30% para probar
- Los datos se normalizan para que funcionen mejor

### Mejoras Implementadas

#### Uso de Funciones Especializadas
- **Eliminación de código duplicado**: Se removió la función intermedia `calcular_metricas_desde_matriz_confusion`
- **Uso directo**: Ahora se usan directamente `compute_metrics()` y `print_report()` del archivo `metricas_clasificacion.py`
- **Código más limpio**: Mejor organización y reutilización de funciones existentes

#### Orden de Presentación Optimizado
- **Gráficos primero**: Se muestran las visualizaciones inmediatamente después del entrenamiento
- **Métricas después**: Las métricas detalladas aparecen al final como resumen completo
- **Flujo natural**: Análisis visual precede al análisis numérico detallado

#### Estructura Modular
- **Separación de responsabilidades**: Cada archivo tiene una función específica
- **Fácil mantenimiento**: Cambios en métricas se hacen en un solo lugar
- **Reutilización**: Las funciones de métricas pueden usarse en otros proyectos

### ¿Cómo funciona el programa?

#### Pasos que sigue:

1. **Carga los datos**: Toma información de pacientes con problemas del corazón
2. **Analiza los datos**: Muestra estadísticas básicas
3. **Prepara los datos**: Arregla información faltante y la organiza
4. **Divide los datos**: 70% para aprender, 30% para probar
5. **Muestra el menú**: Te deja elegir qué hacer
6. **Ejecuta tu opción**: Entrena el modelo que elegiste
7. **Muestra resultados**: Gráficos y números de qué tan bien funciona
8. **Regresa al menú**: Puedes elegir otra opción o salir

#### Opciones disponibles:

**Opción 1 - Random Forest (Bosque Aleatorio)**:
- Usa 100 árboles de decisión
- Muestra qué características son más importantes
- Crea gráficos fáciles de entender

**Opción 2 - Regresión Logística**:
- Método matemático para predecir
- Muestra qué factores influyen más
- Compara diferentes configuraciones

**Opción 3 - Salir**:
- Termina el programa
- Muestra mensaje de despedida

#### Lo que obtienes:

- **Gráficos de colores**: Para entender mejor los resultados (se muestran primero)
- **Números de precisión**: Qué tan bien funciona cada modelo (se muestran después)
- **Análisis detallado**: Qué características son más importantes
- **Comparaciones**: Diferentes formas de configurar el modelo
- **Métricas completas**: Exactitud, sensibilidad, especificidad, F1-Score, etc.
- **Cálculos precisos**: Usando funciones especializadas del archivo `metricas_clasificacion.py`

#### Orden de presentación:

1. **Entrenamiento del modelo** con reporte básico
2. **Gráficos visuales** (importancia de características, matriz de confusión)
3. **Comparaciones** (solo para Regresión Logística)
4. **Métricas detalladas** con análisis completo de precisión


