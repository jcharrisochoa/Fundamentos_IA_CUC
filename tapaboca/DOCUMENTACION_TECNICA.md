# 🔧 Documentación Técnica - Detector de Tapabocas

Documentación técnica detallada del sistema de detección de tapabocas con IA.

---

## 📋 Tabla de Contenidos

- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Componentes Principales](#-componentes-principales)
- [Algoritmos y Modelos](#-algoritmos-y-modelos)
- [Pipeline de Procesamiento](#-pipeline-de-procesamiento)
- [Sistema de Métricas](#-sistema-de-métricas)
- [Configuración y Parámetros](#-configuración-y-parámetros)
- [Estructura de Datos](#-estructura-de-datos)
- [Optimizaciones](#-optimizaciones)

---

## 🏗️ Arquitectura del Sistema

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                   (Punto de Entrada)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  DetectorTapabocas.py                       │
│                  (Clase Principal)                          │
│  ┌──────────────┬──────────────┬──────────────────────┐    │
│  │   GUI        │   Cámara     │   Procesamiento IA   │    │
│  │   (Tkinter)  │   (OpenCV)   │   (YOLO + Haar)      │    │
│  └──────────────┴──────────────┴──────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      config.py                              │
│            (Configuración Centralizada)                     │
│  • Parámetros YOLO    • Umbrales HSV    • UI Config        │
└─────────────────────────────────────────────────────────────┘
```

### Módulos del Sistema

#### 1. **main.py**
- Punto de entrada de la aplicación
- Instancia la clase principal
- Inicia el loop de Tkinter

#### 2. **DetectorTapabocas.py**
- Clase principal del sistema
- Gestión de GUI, cámara y procesamiento
- ~665 líneas de código
- Métodos organizados por funcionalidad

#### 3. **config.py**
- Configuración centralizada
- 217 líneas de constantes
- Parámetros de YOLO, HSV, UI, mensajes

#### 4. **yolov8n.pt**
- Modelo preentrenado YOLOv8 Nano
- 6.2 MB de tamaño
- Entrenado en COCO dataset (80 clases)

---

## 🔩 Componentes Principales

### 1. Sistema de Captura de Video

```python
def init_camera(self):
    """Inicializa cámara web."""
    self.cap = cv2.VideoCapture(cfg.CAMERA_INDEX)  # Index 0 = cámara predeterminada
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    self.update_video_feed()  # Loop de actualización

def update_video_feed(self):
    """Actualiza video a 30 FPS."""
    ret, frame = self.cap.read()
    self.current_frame = cv2.flip(frame, 1)  # Efecto espejo
    self.root.after(30, self.update_video_feed)  # 30ms ≈ 30 FPS
```

**Características:**
- Resolución: 640x480 píxeles
- Frame rate: ~30 FPS
- Efecto espejo para UX intuitiva
- Actualización asíncrona con Tkinter

### 2. Interfaz Gráfica (GUI)

**Estructura:**
```python
┌─ root (Tk) - 803x660px
│  ├─ main_frame (Frame)
│  │  ├─ titulo (Label)
│  │  ├─ video_frame (Frame)
│  │  │  ├─ left_panel (Frame 350x250)
│  │  │  │  └─ video_label (Label) → Video en vivo
│  │  │  └─ right_panel (Frame 350x250)
│  │  │     └─ analysis_label (Label) → Imagen capturada
│  │  ├─ log_frame (Frame)
│  │  │  ├─ log_text (Text + Scrollbar)
│  │  │  └─ scrollbar (Scrollbar)
│  │  └─ control_frame (Frame)
│  │     ├─ estado_label (Label)
│  │     └─ buttons (4 Button widgets)
```

**Componentes:**
- **Paneles**: 350x250px cada uno
- **Log**: Text widget con scroll automático
- **Botones**: 15x2 caracteres, colores semánticos
- **Estado**: Label dinámico con contador en vivo

### 3. Motor de Detección YOLO

```python
def _init_yolo_model(self):
    """Carga YOLOv8n."""
    model = YOLO('yolov8n.pt')
    return model

def process_captured_image(self):
    """Procesa con YOLO."""
    results = self.model(
        self.imagen_capturada, 
        conf=0.6,      # Umbral de confianza
        verbose=False  # Sin output en consola
    )
    
    # Extraer personas (clase 0 en COCO)
    for box in result.boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
```

**Parámetros YOLO:**
- Modelo: YOLOv8n (nano - más rápido)
- Confianza mínima: 0.6 (60%)
- Clase detectada: 0 (persona en COCO)
- Input: Imagen BGR de OpenCV

### 4. Detección de Rostros (Haar Cascade)

```python
# Dos cascadas para mayor cobertura
self.face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
self.profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_profileface.xml'
)

def detect_faces_in_person(self, x1, y1, x2, y2):
    """Detecta rostros en región de persona."""
    person_roi = self.imagen_capturada[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
    
    for cascade in [self.face_cascade, self.profile_cascade]:
        faces = cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,      # Escala de reducción
            minNeighbors=3,       # Mínimo de vecinos
            minSize=(40, 40)      # Tamaño mínimo de rostro
        )
```

**Parámetros Haar:**
- `scaleFactor=1.1`: Reducción del 10% por nivel
- `minNeighbors=3`: Mínimo de detecciones cercanas
- `minSize=(40, 40)`: Rostro mínimo de 40x40px
- Dos cascadas: frontal + perfil

---

## 🧮 Algoritmos y Modelos

### 1. YOLOv8 (You Only Look Once v8)

**Arquitectura:**
- **Backbone**: CSPDarknet (Cross Stage Partial)
- **Neck**: PANet (Path Aggregation Network)
- **Head**: Detección de objetos

**Características:**
- One-stage detector (una sola pasada)
- Anchor-free (sin anclas predefinidas)
- Multi-escala (detecta objetos de varios tamaños)
- 80 clases del dataset COCO

**Output:**
```python
box.xyxy[0]  # [x1, y1, x2, y2] coordenadas
box.conf[0]  # Confianza [0-1]
box.cls[0]   # Clase detectada [0-79]
```

### 2. Haar Cascade Classifier

**Principio:**
- Características de Haar (bordes, líneas, rectángulos)
- Cascada de clasificadores débiles
- AdaBoost para selección de características

**Proceso:**
1. Múltiples escalas de la imagen
2. Sliding window en cada escala
3. Cascada de etapas (early rejection)
4. Retorna bounding boxes de rostros

### 3. Análisis HSV para Detección de Tapabocas

**Espacio de Color HSV:**
- **H (Hue)**: Matiz [0-180]
- **S (Saturation)**: Saturación [0-255]
- **V (Value)**: Valor/Brillo [0-255]

**Ventajas sobre RGB:**
- Independiente de iluminación
- Mejor para segmentación de color
- Rangos más intuitivos

**Rangos Definidos:**
```python
# Blancos/grises claros
MASK_WHITE_LOWER = [0, 0, 180]
MASK_WHITE_UPPER = [180, 40, 255]

# Azules (quirúrgicos)
MASK_BLUE_LOWER = [100, 40, 40]
MASK_BLUE_UPPER = [130, 255, 255]

# ... (8 rangos totales)
```

### 4. Algoritmo de Canny (Detección de Bordes)

```python
edge_density = np.sum(cv2.Canny(gray, 15, 60) > 0) / gray.size
```

**Parámetros:**
- Umbral inferior: 15
- Umbral superior: 60
- Ratio 1:4 (recomendado por Canny)

**Proceso:**
1. Suavizado gaussiano
2. Gradiente de intensidad (Sobel)
3. Supresión no-máxima
4. Umbralización por histéresis

---

## 🔄 Pipeline de Procesamiento

### Flujo Completo de Detección

```
1. CAPTURA
   └─> Frame actual (640x480 BGR)

2. YOLO DETECTION
   └─> Personas detectadas con bbox y confianza
       └─> Filtrado por confianza ≥ 0.6
           └─> Filtrado de duplicados (IoU < 0.3)

3. FACE DETECTION
   └─> Para cada persona:
       ├─> ROI de persona extraída
       ├─> Conversión a escala de grises
       ├─> Haar Cascade frontal
       ├─> Haar Cascade perfil
       └─> Merge y filtrado de rostros

4. FACE REGION ESTIMATION
   └─> Si rostro detectado: usar bbox real
   └─> Si no: estimar región (25% superior, 80% ancho)

5. MASK ANALYSIS
   └─> Región nariz/boca (50% inferior de rostro)
       ├─> Conversión HSV
       ├─> Detección de piel (2 rangos)
       ├─> Detección de colores tapabocas (8 rangos)
       ├─> Detección de bordes (Canny)
       ├─> Análisis de textura (std, variance)
       └─> Cálculo de 6 métricas

6. SCORING SYSTEM
   └─> Acumulador de puntos
       ├─> Criterio 1: Skin ratio (-5 a +5)
       ├─> Criterio 2: Non-skin ratio (+1 a +4)
       ├─> Criterio 3: Mask colors (+1 a +2)
       ├─> Criterio 4: Edge density (-1 a +2)
       ├─> Criterio 5: Color variance (-1 a +2)
       └─> Criterio 6: Texture std (-1 a +2)

7. CLASSIFICATION
   └─> Score ≥ 5  → CON TAPABOCAS
   └─> Score ≤ -3 → SIN TAPABOCAS
   └─> Intermedio  → Análisis secundario

8. VISUALIZATION
   └─> Bounding box con color semántico
   └─> Etiquetas (Persona #X, Resultado)
   └─> Log entry con timestamp y métricas
```

---

## 📊 Sistema de Métricas

### Métricas Calculadas

#### 1. **Skin Ratio** (Ratio de Piel)
```python
skin_mask = cv2.bitwise_or(
    cv2.inRange(hsv, [0, 40, 70], [20, 255, 255]),
    cv2.inRange(hsv, [20, 50, 70], [30, 255, 255])
)
skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
```
- **Rango**: [0.0 - 1.0]
- **Interpretación**: 
  - < 0.05 → Muy cubierto (tapabocas probable)
  - > 0.50 → Muy visible (sin tapabocas)

#### 2. **Non-Skin Ratio** (Ratio No-Piel)
```python
non_skin_ratio = 1 - skin_ratio
```
- **Rango**: [0.0 - 1.0]
- **Interpretación**: Complemento de skin_ratio

#### 3. **Mask Color Ratio** (Ratio de Colores)
```python
mask_colors = cv2.bitwise_or(white_mask, black_mask)
mask_colors = cv2.bitwise_or(mask_colors, blue_mask)
# ... (combinar 8 máscaras)
mask_color_ratio = np.sum(mask_colors > 0) / mask_colors.size
```
- **Rango**: [0.0 - 1.0]
- **Interpretación**: Presencia de colores típicos de tapabocas

#### 4. **Edge Density** (Densidad de Bordes)
```python
edges = cv2.Canny(gray, 15, 60)
edge_density = np.sum(edges > 0) / edges.size
```
- **Rango**: [0.0 - 1.0]
- **Interpretación**: Tapabocas tienen bordes definidos

#### 5. **Color Variance** (Varianza de Color)
```python
color_variance = np.var(gray)
```
- **Rango**: [0 - ∞]
- **Interpretación**: 
  - Bajo → Uniforme (tapabocas)
  - Alto → Variado (piel con sombras)

#### 6. **Texture Std** (Desviación Estándar)
```python
texture_std = np.std(gray)
```
- **Rango**: [0 - ∞]
- **Interpretación**: Similitud de textura

### Sistema de Scoring

```python
def _calculate_mask_score(self, skin_ratio, non_skin_ratio, 
                          mask_color_ratio, edge_density, 
                          color_variance, texture_std):
    score = 0
    
    # Criterio 1: Piel visible (peso: -5 a +5)
    if skin_ratio < 0.03: score += 5
    elif skin_ratio < 0.08: score += 4
    elif skin_ratio < 0.15: score += 3
    # ... más umbrales
    
    # Criterios 2-6: similares
    # ...
    
    return score  # Rango típico: [-8, +15]
```

**Umbrales de Decisión:**
- `score ≥ 5` → **CON TAPABOCAS**
- `score ≤ -3` → **SIN TAPABOCAS**
- `-3 < score < 5` → Análisis secundario por skin_ratio

---

## ⚙️ Configuración y Parámetros

### Parámetros Críticos (config.py)

#### YOLO
```python
YOLO_MODEL_PATH = 'yolov8n.pt'
YOLO_CONFIDENCE = 0.6     # 60% confianza mínima
PERSON_CLASS_ID = 0       # Clase persona en COCO
```

#### Detección de Rostros
```python
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 3
FACE_MIN_SIZE = (40, 40)
FACE_REGION_HEIGHT_RATIO = 0.25  # 25% superior
FACE_REGION_WIDTH_RATIO = 0.8    # 80% ancho
```

#### Filtrado de Duplicados
```python
IOU_THRESHOLD = 0.3         # Intersection over Union
MIN_DETECTION_AREA = 2000   # Píxeles mínimos
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 2.0
```

#### Análisis de Imagen
```python
MOUTH_REGION_RATIO = 0.5    # 50% inferior del rostro
CANNY_THRESHOLD_1 = 15
CANNY_THRESHOLD_2 = 60
```

#### Scoring
```python
MASK_PRESENT_THRESHOLD = 5
MASK_ABSENT_THRESHOLD = -3
SKIN_RATIO_HIGH = 0.50
SKIN_RATIO_LOW = 0.05
```

---

## 📦 Estructura de Datos

### Objetos Principales

#### Detection Object
```python
{
    'bbox': (x1, y1, x2, y2),        # Bounding box
    'confidence': 0.85,               # Confianza YOLO
    'tiene_tapabocas': 'CON TAPABOCAS',  # Clasificación
    'metrics': {                      # Métricas detalladas
        'skin_ratio': 0.025,
        'non_skin_ratio': 0.975,
        'mask_color_ratio': 0.450,
        'edge_density': 0.180,
        'color_variance': 245.3,
        'texture_std': 18.5,
        'score': 8
    }
}
```

#### Analysis Data
```python
{
    'num_personas': 2,
    'con_tapabocas': 1,
    'sin_tapabocas': 1,
    'no_detectado': 0,
    'detecciones': [detection1, detection2, ...]
}
```

---

## ⚡ Optimizaciones

### 1. Filtrado de Duplicados (IoU)
```python
def calculate_iou(self, box1, box2):
    """Intersection over Union."""
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0
```
- Evita detecciones duplicadas
- Umbral IoU > 0.3 → duplicado

### 2. Reducción de Frames
```python
def _resize_frame(self, frame, max_size=(350, 250)):
    scale = min(max_width/width, max_height/height)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size)
```
- Reduce tamaño para GUI
- Mantiene aspect ratio

### 3. Procesamiento Asíncrono
```python
self.root.after(100, self.process_captured_image)
```
- No bloquea UI durante procesamiento
- Procesamiento en siguiente ciclo de eventos

### 4. Caching de Modelos
```python
# Cargar una sola vez al inicio
self.model = YOLO('yolov8n.pt')
self.face_cascade = cv2.CascadeClassifier(...)
```
- Modelos cargados una vez
- Reutilizados en cada detección

---

## 🐛 Manejo de Errores

### Try-Except Blocks
```python
try:
    # Código de procesamiento
except Exception as e:
    self.estado_label.config(text=cfg.MSG_ERROR_PROCESS)
    return 'NO DETECTADO', {}
```

### Validaciones
- Verificar frame no vacío
- Verificar modelo cargado
- Verificar ROI válido
- Verificar cámara abierta

---

## 📈 Métricas de Rendimiento

### Tiempo de Procesamiento (típico)

| Etapa | Tiempo |
|-------|--------|
| Captura de frame | < 1ms |
| Detección YOLO | 50-150ms |
| Detección rostros | 20-50ms |
| Análisis HSV | 5-15ms |
| Scoring | < 1ms |
| Dibujado | 5-10ms |
| **Total** | **80-230ms** |

### Recursos

| Recurso | Uso |
|---------|-----|
| RAM | ~300-500 MB |
| GPU | Opcional (CUDA) |
| CPU | 15-30% (1 core) |
| Disco | 10 MB (código + modelo) |

---

## 🔐 Consideraciones de Seguridad

- ✅ No almacena imágenes en disco
- ✅ No envía datos a internet
- ✅ Procesamiento local 100%
- ✅ Log solo con métricas numéricas
- ✅ No guarda información personal

---

## 📚 Referencias Técnicas

1. **YOLOv8**: [Ultralytics Documentation](https://docs.ultralytics.com/)
2. **OpenCV**: [OpenCV Documentation](https://docs.opencv.org/)
3. **Haar Cascades**: Viola-Jones Object Detection Framework
4. **HSV Color Space**: Computer Vision: Algorithms and Applications
5. **Canny Edge Detection**: J. Canny, 1986

---

<p align="center">
  Documentación Técnica v1.0 - 2025
</p>

