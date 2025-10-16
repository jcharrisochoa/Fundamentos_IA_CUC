# 🎭 Detector de Tapabocas con IA

Sistema inteligente de detección automática de uso de tapabocas utilizando **YOLOv8** y **OpenCV** con análisis de características faciales en tiempo real.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Detection-green.svg" alt="YOLO">
  <img src="https://img.shields.io/badge/OpenCV-4.x-red.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características Principales](#-características-principales)
- [Funcionamiento General](#-funcionamiento-general)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Instalación](#-instalación)
- [Uso Rápido](#-uso-rápido)
- [Capturas de Pantalla](#-capturas-de-pantalla)
- [Documentación Adicional](#-documentación-adicional)
- [Autor](#-autor)

---

## 🎯 Descripción

El **Detector de Tapabocas** es una aplicación de escritorio con interfaz gráfica que permite detectar automáticamente si las personas en una imagen están usando tapabocas o no. Utiliza técnicas avanzadas de **visión por computador** e **inteligencia artificial** para:

- ✅ Detectar personas en tiempo real a través de la cámara web
- ✅ Identificar rostros humanos (frontales y de perfil)
- ✅ Analizar la región facial (nariz/boca) para determinar presencia de tapabocas
- ✅ Clasificar múltiples personas simultáneamente
- ✅ Generar métricas detalladas del análisis
- ✅ Registrar historial de detecciones con timestamp

---

## ⭐ Características Principales

### 🎥 Detección en Tiempo Real
- Captura de video continuo desde cámara web
- Actualización a **30 FPS** para fluidez
- Efecto espejo para facilitar interacción

### 🤖 IA de Última Generación
- **YOLOv8n (nano)**: Detección rápida y precisa de personas
- **Haar Cascade**: Localización de rostros frontales y de perfil
- **Análisis multicriterio**: 6 métricas para clasificación

### 🎨 Detección Universal de Colores
El sistema detecta tapabocas de cualquier color:
- 🤍 Blancos y grises claros
- ⚫ Negros y grises oscuros
- 🔵 Azules (quirúrgicos)
- 🟢 Verdes (clínicos)
- 🟣 Rosas y morados
- 🔴 Rojos
- 🟡 Amarillos y naranjas

### 📊 Métricas Detalladas
Para cada persona detectada se calculan:
1. **Ratio de piel visible** - Porcentaje de piel en región facial
2. **Ratio de área no-piel** - Porcentaje de cobertura
3. **Ratio de colores de tapabocas** - Presencia de colores característicos
4. **Densidad de bordes** - Análisis de contornos (algoritmo Canny)
5. **Varianza de color** - Uniformidad en la región
6. **Desviación estándar de textura** - Análisis de patrones

### 📝 Sistema de Log Completo
- Historial cronológico con timestamps
- Resumen de detecciones por análisis
- Tabla de métricas detalladas por persona
- Scroll automático al último análisis

### 🖥️ Interfaz Intuitiva
- Panel de video en tiempo real
- Panel de imagen capturada con análisis visual
- Controles simples (Capturar, Limpiar, Salir)
- Indicador de estado en tiempo real
- Bounding boxes con colores semánticos:
  - 🟢 Verde = CON TAPABOCAS
  - 🔴 Rojo = SIN TAPABOCAS
  - 🟠 Naranja = NO DETECTADO

---

## 🔄 Funcionamiento General

### Pipeline de Procesamiento

```
┌─────────────────┐
│  Cámara Web     │
│  (Video Live)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Captura Frame  │
│  (Usuario click)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   YOLO v8n      │
│ Detecta Personas│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Haar Cascade   │
│ Localiza Rostros│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Análisis Región  │
│  Nariz/Boca     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  6 Métricas     │
│  + Score Final  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Clasificación  │
│ CON/SIN/NO DET  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualización  │
│  + Log Entry    │
└─────────────────┘
```

### Flujo de Uso

1. **Inicio**: La aplicación abre la cámara y muestra video en tiempo real
2. **Captura**: El usuario hace clic en "Capturar Imagen"
3. **Detección**: YOLO detecta todas las personas en la imagen
4. **Localización**: Haar Cascade encuentra rostros en cada persona
5. **Análisis**: Se extraen 6 métricas de la región nariz/boca
6. **Clasificación**: Sistema de scoring determina presencia de tapabocas
7. **Resultado**: Se dibujan bounding boxes y se actualiza el log
8. **Repetición**: El usuario puede capturar nuevas imágenes indefinidamente

---

## 🛠️ Tecnologías Utilizadas

### Lenguajes y Frameworks
- **Python 3.8+** - Lenguaje principal
- **Tkinter** - Interfaz gráfica (GUI)

### Bibliotecas de IA/ML
- **Ultralytics YOLOv8** - Detección de objetos (personas)
- **OpenCV 4.x** - Procesamiento de imágenes y video
- **NumPy** - Operaciones numéricas y matrices

### Bibliotecas Auxiliares
- **Pillow (PIL)** - Manipulación de imágenes para Tkinter
- **datetime** - Manejo de timestamps

### Modelos Preentrenados
- **yolov8n.pt** - YOLO v8 Nano para detección de personas (COCO dataset)
- **haarcascade_frontalface_default.xml** - Detección de rostros frontales
- **haarcascade_profileface.xml** - Detección de rostros de perfil

---

## 📦 Instalación

### Requisitos Previos
- Python 3.8 o superior
- Cámara web funcional
- Sistema operativo: Windows, Linux o macOS

### Paso 1: Navegar a la carpeta
```bash
cd tapaboca
```

### Paso 2: Instalar Dependencias
```bash
pip install ultralytics opencv-python numpy pillow
```

### Paso 3: Verificar Modelo YOLO
Asegúrate de que el archivo `yolov8n.pt` esté en la carpeta del proyecto.

---

## 🚀 Uso Rápido

### Ejecutar la Aplicación
```bash
python main.py
```

### Controles de la Interfaz

| Botón | Función |
|-------|---------|
| **Capturar Imagen** | Congela el frame actual y lo analiza |
| **Limpiar Imagen** | Borra la imagen capturada y resetea |
| **Limpiar Log** | Limpia el historial de análisis |
| **Salir** | Cierra la aplicación |

### Interpretación de Resultados

#### Bounding Boxes
- **🟢 Verde** → CON TAPABOCAS (Score ≥ 5)
- **🔴 Rojo** → SIN TAPABOCAS (Score ≤ -3)
- **🟠 Naranja** → NO DETECTADO (Score intermedio o error)

#### Log de Análisis
```
======================================================================
📅 2025-10-16 15:30:45
✅ Análisis completado exitosamente
👥 Personas detectadas: 2
   ✅ Con tapabocas: 1
   ❌ Sin tapabocas: 1

📊 MÉTRICAS DETALLADAS POR PERSONA:
#    Piel  No-Piel  Color  Bordes    Var   Text  Score  Resultado      
1   0.025    0.975  0.450   0.180    245   18.5      8  CON TAPABOCAS  
2   0.620    0.380  0.120   0.095    680   28.3     -5  SIN TAPABOCAS  
======================================================================
```

---

## 📸 Interfaz de Usuario

```
┌────────────────────────────────────────────────────────────────┐
│           Detector de Tapabocas con IA                         │
├────────────────────┬───────────────────────────────────────────┤
│ Cámara en Tiempo   │  Imagen Capturada - Análisis             │
│      Real          │                                           │
│                    │  [Imagen con bounding boxes]              │
│ [Video live]       │                                           │
│                    │                                           │
├────────────────────┴───────────────────────────────────────────┤
│                  Historial de Análisis                         │
│ ┌──────────────────────────────────────────────────────────┐  │
│ │ [Scroll de log con métricas y resultados]                │  │
│ └──────────────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────┤
│  Estado: 2 persona(s): 1 con tapabocas, 1 sin tapabocas       │
│                                                                │
│ [Capturar] [Limpiar Imagen] [Limpiar Log] [Salir]             │
└────────────────────────────────────────────────────────────────┘
```

---

## 📚 Documentación Adicional

Para información más detallada, consulta:

- **[DOCUMENTACION_TECNICA.md](./DOCUMENTACION_TECNICA.md)** - Arquitectura y algoritmos
- **[MANUAL_USUARIO.md](./MANUAL_USUARIO.md)** - Guía paso a paso de uso
- **[ANALISIS_DETECCION.md](./ANALISIS_DETECCION.md)** - Proceso de detección y análisis

---

## 👨‍💻 Autor

**Johan Charris Ochoa**  
Universidad de la Costa (CUC)  
2025

---

## 📄 Licencia

Este proyecto está desarrollado con fines educativos y de investigación.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

---

<p align="center">
  Hecho con ❤️ para la seguridad y salud pública
</p>

