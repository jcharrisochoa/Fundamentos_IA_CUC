"""
Configuración del Detector de Tapabocas
Todas las constantes del sistema centralizadas para fácil mantenimiento
"""

# ==================== CONFIGURACIÓN DE VENTANA ====================
WINDOW_TITLE = "Detector de Tapabocas - YOLO + OpenCV"
WINDOW_GEOMETRY = "803x660" # width x height
WINDOW_RESIZABLE = False

# ==================== CONFIGURACIÓN DE PANELES DE VIDEO Y ANÁLISIS ====================
PANEL_WIDTH = 350
PANEL_HEIGHT = 250

# ==================== CONFIGURACIÓN DE VIDEO Y CÁMARA ====================
VIDEO_FPS = 30
VIDEO_UPDATE_INTERVAL = 30  # milisegundos
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_INDEX = 0

# ==================== CONFIGURACIÓN DEL MODELO YOLO ====================
YOLO_MODEL_PATH = 'yolov8n.pt'
YOLO_CONFIDENCE = 0.6
YOLO_VERBOSE = False
PERSON_CLASS_ID = 0  # Clase persona en COCO dataset

# ==================== CONFIGURACIÓN DE DETECCIÓN ====================
IOU_THRESHOLD = 0.3
MIN_DETECTION_AREA = 2000
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 2.0

# ==================== CONFIGURACIÓN DE DETECCIÓN FACIAL ====================
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 3
FACE_MIN_SIZE = (40, 40)
FACE_REGION_HEIGHT_RATIO = 0.25  # 25% superior para rostro
FACE_REGION_WIDTH_RATIO = 0.8    # 80% ancho centrado
MOUTH_REGION_RATIO = 0.5          # 50% inferior del rostro
FACE_CONFIDENCE_PENALTY = 0.7     # Penalización si no se detecta rostro

# ==================== CONFIGURACIÓN DE DETECCIÓN DE BORDES ====================
CANNY_THRESHOLD_1 = 15
CANNY_THRESHOLD_2 = 60

# ==================== RANGOS HSV PARA DETECCIÓN DE PIEL ====================
SKIN_RANGE_1_LOWER = [0, 40, 70]
SKIN_RANGE_1_UPPER = [20, 255, 255]
SKIN_RANGE_2_LOWER = [20, 50, 70]
SKIN_RANGE_2_UPPER = [30, 255, 255]

# ==================== RANGOS HSV PARA DETECCIÓN DE TAPABOCAS ====================
# Blancos/grises claros
MASK_WHITE_LOWER = [0, 0, 180]
MASK_WHITE_UPPER = [180, 40, 255]

# Negros/grises oscuros
MASK_BLACK_LOWER = [0, 0, 0]
MASK_BLACK_UPPER = [180, 255, 80]

# Azules (quirúrgicos)
MASK_BLUE_LOWER = [100, 40, 40]
MASK_BLUE_UPPER = [130, 255, 255]

# Verdes (clínicos)
MASK_GREEN_LOWER = [40, 40, 40]
MASK_GREEN_UPPER = [80, 255, 255]

# Rosas/Morados
MASK_PINK_LOWER = [140, 40, 40]
MASK_PINK_UPPER = [170, 255, 255]

# Rojos
MASK_RED_1_LOWER = [0, 40, 100]
MASK_RED_1_UPPER = [10, 255, 255]
MASK_RED_2_LOWER = [170, 40, 100]  # Rojos (wrap around)
MASK_RED_2_UPPER = [180, 255, 255]

# Amarillos/naranjas
MASK_YELLOW_LOWER = [20, 40, 100]
MASK_YELLOW_UPPER = [40, 255, 255]

# ==================== UMBRALES DE CLASIFICACIÓN ====================
# Score para decisión final
MASK_PRESENT_THRESHOLD = 5
MASK_ABSENT_THRESHOLD = -3

# Umbrales de skin ratio para decisión alternativa
SKIN_RATIO_HIGH = 0.50  # Sin tapabocas si > 0.50
SKIN_RATIO_LOW = 0.05   # Con tapabocas si < 0.05

# ==================== CRITERIOS DE SCORING ====================
# Criterio 1: Piel visible
SKIN_RATIO_THRESHOLDS = {
    'very_low': (0.03, 5),   # < 0.03 -> +5
    'low': (0.08, 4),        # < 0.08 -> +4
    'medium_low': (0.15, 3), # < 0.15 -> +3
    'medium': (0.25, 2),     # < 0.25 -> +2
    'high': (0.35, 1),       # < 0.35 -> +1
    'very_high': (0.60, -5), # > 0.60 -> -5
    'high_neg': (0.45, -4),  # > 0.45 -> -4
}

# Criterio 2: Color no-piel
NON_SKIN_RATIO_THRESHOLDS = {
    'very_high': (0.80, 4),
    'high': (0.60, 3),
    'medium': (0.40, 2),
    'low': (0.20, 1),
}

# Criterio 3: Colores de tapabocas
MASK_COLOR_RATIO_THRESHOLDS = {
    'high': (0.30, 2),
    'medium': (0.15, 1),
}

# Criterio 4: Bordes
EDGE_DENSITY_THRESHOLDS = {
    'high': (0.20, 2),
    'medium': (0.12, 1),
    'low': (0.05, -1),
}

# Criterio 5: Uniformidad (varianza de color)
COLOR_VARIANCE_THRESHOLDS = {
    'low': (150, 2),
    'medium': (300, 1),
    'high': (800, -1),
}

# Criterio 6: Textura
TEXTURE_STD_THRESHOLDS = {
    'low': (12, 2),
    'medium': (20, 1),
    'high': (35, -1),
}

# ==================== CONFIGURACIÓN DE UI ====================
# Colores
COLOR_BACKGROUND = "SystemButtonFace"
COLOR_PANEL_BG = "lightgray"
COLOR_VIDEO_BG = "black"
COLOR_ANALYSIS_BG = "gray"
COLOR_LOG_BG = "#f0f0f0"
COLOR_TEXT_BLACK = "black"
COLOR_TEXT_WHITE = "white"

# Botones
BUTTON_COLOR_CAPTURE = "green"
BUTTON_COLOR_CLEAR = "orange"
BUTTON_COLOR_CLEAR_LOG = "cyan"
BUTTON_COLOR_EXIT = "red"
BUTTON_WIDTH = 15
BUTTON_HEIGHT = 2

# Fuentes
FONT_TITLE = ("Arial", 18, "bold")
FONT_SUBTITLE = ("Arial", 12, "bold")
FONT_PANEL_TITLE = ("Arial", 10, "bold")
FONT_VIDEO_LABEL = ("Arial", 14)
FONT_ANALYSIS_LABEL = ("Arial", 12)
FONT_STATUS = ("Arial", 11)
FONT_BUTTON = ("Arial", 12)
FONT_LOG = ("Courier", 9)
FONT_DETECTION = 0.6  # OpenCV font scale

# Dimensiones
LOG_HEIGHT = 8
PADDING_X = 10
PADDING_Y = 10

# Detección - Colores de bounding boxes
BBOX_COLOR_WITH_MASK = (0, 255, 0)      # Verde
BBOX_COLOR_WITHOUT_MASK = (0, 0, 255)   # Rojo
BBOX_COLOR_UNKNOWN = (0, 165, 255)      # Naranja
BBOX_THICKNESS = 3

# ==================== ARCHIVOS DE CASCADAS HAAR ====================

CASCADE_FRONTAL_FACE = 'haarcascade_frontalface_default.xml'
CASCADE_PROFILE_FACE = 'haarcascade_profileface.xml'

# ==================== MENSAJES DEL SISTEMA ====================
MSG_INIT_CAMERA = "Inicializando cámara..."
MSG_NO_IMAGE = "Sin imagen capturada"
MSG_ERROR_NO_FRAME = "Error: No hay frame disponible"
MSG_ERROR_NO_IMAGE_MODEL = "Error: No hay imagen o modelo"
MSG_ERROR_CAPTURE = "Error al capturar imagen"
MSG_ERROR_PROCESS = "Error al procesar imagen"
MSG_ERROR_CAMERA = "Error: No se pudo acceder a la cámara"
MSG_ERROR_INIT_CAMERA = "Error al inicializar cámara"
MSG_ERROR_READ_FRAME = "Error leyendo frame"
MSG_ERROR_VIDEO_FEED = "Error en video feed"
MSG_CAPTURING = "Estado: Foto capturada - Procesando..."
MSG_NO_DETECTION = "Estado: No se detectaron personas"
MSG_NO_PHOTO = "Estado: Sin foto - Esperando captura"
MSG_SYSTEM_INIT = "Sistema iniciado. Esperando primera captura..."
MSG_IMAGE_CLEARED = "🧹 Imagen limpiada. Sistema listo para nuevo análisis."
MSG_LOG_CLEARED = "🔄 Log limpiado. Sistema reiniciado."
MSG_ANALYSIS_SUCCESS = "✅ Análisis completado exitosamente"
MSG_NO_PERSONS = "⚠️ No se detectaron personas en la imagen"
MSG_CLOSING = "Cerrando aplicación..."
MSG_STARTING = "Iniciando Detector de Tapabocas..."

# ==================== TÍTULOS Y ETIQUETAS ====================
LABEL_MAIN_TITLE = "Detector de Tapabocas con IA"
LABEL_CAMERA_PANEL = "Cámara en Tiempo Real"
LABEL_ANALYSIS_PANEL = "Imagen Capturada - Análisis"
LABEL_LOG_TITLE = "Historial de Análisis"
LABEL_BUTTON_CAPTURE = "Capturar Imagen"
LABEL_BUTTON_CLEAR = "Limpiar Imagen"
LABEL_BUTTON_CLEAR_LOG = "Limpiar Log"
LABEL_BUTTON_EXIT = "Salir"

