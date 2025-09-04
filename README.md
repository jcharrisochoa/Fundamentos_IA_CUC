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

Instalar las librerías requeridas (ejemplos):
```bash
pip install numpy pandas matplotlib scikit-learn
# Opcional para cuadernos
pip install jupyter ipykernel
```

Para salir del entorno: `deactivate`.


