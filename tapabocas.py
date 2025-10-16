"""
Aplicación de Detección de Tapabocas usando YOLO y OpenCV

Esta aplicación utiliza YOLOv8 para detectar personas y algoritmos de
visión artificial para clasificar si llevan tapabocas o no.

Autor: Sistema de IA
Fecha: 2025
Descripción: 
    - Interfaz gráfica con tkinter
    - Video en tiempo real desde webcam
    - Detección de personas con YOLOv8
    - Clasificación de tapabocas mediante análisis de características
    - Visualización con bounding boxes de colores según resultado
"""

import tkinter as tk
from tkinter import ttk, Label, Button, Frame
import cv2
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO
import os

# =============================================================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================================================

class Config:
    """Configuración centralizada de la aplicación"""
    
    # Configuración YOLO
    YOLO_CONFIDENCE = 0.6
    YOLO_MODEL_PATH = 'yolov8n.pt'
    PERSON_CLASS_ID = 0
    
    # Configuración de filtrado
    IOU_THRESHOLD = 0.3
    MIN_AREA = 2000
    MIN_ASPECT_RATIO = 0.3
    MAX_ASPECT_RATIO = 2.0
    
    # Configuración de detección de rostros
    FACE_SCALE_FACTOR = 1.1
    FACE_MIN_NEIGHBORS = 3
    FACE_MIN_SIZE = (40, 40)
    FACE_ROTATION_ANGLES = [15, 30, 45, -15, -30, -45]
    
    # Configuración de clasificación
    FACE_REGION_RATIO = 0.5  # 50% inferior del rostro
    MASK_SCORE_THRESHOLD_CON = 5
    MASK_SCORE_THRESHOLD_SIN = -3
    
    # Configuración GUI
    WINDOW_TITLE = "Detector de Tapabocas con IA"
    PANEL_WIDTH = 580
    PANEL_HEIGHT = 480
    VIDEO_UPDATE_MS = 30
    
    # Rangos HSV para detección de colores
    SKIN_RANGE_1 = ([0, 40, 70], [20, 255, 255])
    SKIN_RANGE_2 = ([20, 50, 70], [30, 255, 255])
    WHITE_GRAY_RANGE = ([0, 0, 200], [180, 30, 255])
    BLUE_RANGE = ([100, 50, 50], [130, 255, 255])
    BLACK_RANGE = ([0, 0, 0], [180, 255, 50])


class DetectorTapabocas:
    """
    Clase principal para la detección de tapabocas.
    
    Componentes:
        - GUI con tkinter (2 paneles: video en vivo y análisis)
        - Modelo YOLOv8 para detección de personas
        - Clasificador de tapabocas basado en características
        - Visualización con bounding boxes codificados por color
    
    Atributos:
        root: Ventana principal de tkinter
        model: Modelo YOLOv8 para detección
        cap: Objeto de captura de video OpenCV
        imagen_capturada: Frame capturado para análisis
        detecciones: Lista de detecciones con clasificación
    """
    
    def __init__(self):
        """Inicializa la aplicación con configuración optimizada."""
        self.root = tk.Tk()
        self.root.title(Config.WINDOW_TITLE)
        self.root.geometry("1200x700")
        self.root.resizable(False, False)
        
        # Variables de estado
        self.hay_foto = False
        self.imagen_capturada = None
        self.imagen_procesada = None
        self.cap = None
        self.video_running = False
        self.current_frame = None
        self.detecciones = []
        
        # Inicializar modelo YOLO
        print("Cargando modelo YOLOv8...")
        self.model = None
        self.init_yolo_model()
        
        # Configurar la GUI
        self.setup_gui()
        # Inicializar cámara
        self.init_camera()
        
    def init_yolo_model(self):
        """Carga el modelo YOLOv8 pre-entrenado."""
        try:
            self.model = YOLO(Config.YOLO_MODEL_PATH)
            print("Modelo YOLOv8 cargado correctamente")
        except Exception as e:
            print(f"Error al cargar modelo YOLO: {e}")
            self.model = None
    
    def setup_gui(self):
        """
        Crea todos los componentes de la interfaz gráfica.
        
        Estructura:
            - Panel izquierdo: Video en tiempo real (580x480)
            - Panel derecho: Imagen capturada con análisis (580x480)
            - Botones: Capturar, Limpiar, Salir
            - Label de estado: Muestra resultados del análisis
        """
        # Frame principal
        main_frame = Frame(self.root, bg="SystemButtonFace")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        titulo = Label(main_frame, text="Detector de Tapabocas con IA", 
                      font=("Arial", 18, "bold"), bg="SystemButtonFace", fg="black")
        titulo.pack(pady=10)
        
        # Frame para los paneles de video
        video_frame = Frame(main_frame, bg="SystemButtonFace")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Panel izquierdo - Video en tiempo real
        left_panel = Frame(video_frame, bg="lightgray", relief="solid", bd=1, 
                          width=Config.PANEL_WIDTH, height=Config.PANEL_HEIGHT)
        left_panel.pack(side=tk.LEFT, fill=None, expand=True, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        left_title = Label(left_panel, text="Cámara en Tiempo Real", 
                          font=("Arial", 12, "bold"), bg="lightgray", fg="black")
        left_title.pack(pady=5)
        
        self.video_label = Label(left_panel, bg="black", text="Inicializando cámara...", 
                                fg="white", font=("Arial", 14))
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)


        
        # Panel derecho - Imagen capturada con análisis
        right_panel = Frame(video_frame, bg="lightgray", relief="solid", bd=1, width=580, height=480)
        right_panel.pack(side=tk.RIGHT, fill=None, expand=True, padx=(5, 0))
        right_panel.pack_propagate(False)  # Fijar tamaño exacto del Frame
        
        right_title = Label(right_panel, text="Imagen Capturada - Análisis", 
                           font=("Arial", 12, "bold"), bg="lightgray", fg="black")
        right_title.pack(pady=5)
        
        self.analysis_label = Label(right_panel, bg="gray", text="Sin imagen capturada", 
                                   fg="black", font=("Arial", 14))
        self.analysis_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)


        
        # Frame para controles
        control_frame = Frame(main_frame, bg="SystemButtonFace")
        control_frame.pack(pady=20)
        
        # Label de estado
        self.estado_label = Label(control_frame, text="Estado: Sin foto - Esperando captura", 
                                 font=("Arial", 12), bg="SystemButtonFace", fg="black")
        self.estado_label.pack(pady=10)
        
        # Frame para botones
        button_frame = Frame(control_frame, bg="SystemButtonFace")
        button_frame.pack()
        
        # Botón Capturar
        self.btn_capturar = Button(
            button_frame, 
            text="Capturar Imagen", 
            command=self.capture_image,
            font=("Arial", 12), 
            bg="green", 
            fg="black",
            width=15, 
            height=2,
            highlightbackground="lightgray",
            highlightthickness=1,
            bd=1,
            relief="solid"
        )
        self.btn_capturar.pack(side=tk.LEFT, padx=10)
        
        # Botón Limpiar
        self.btn_limpiar = Button(
            button_frame, 
            text="Limpiar Imagen", 
            command=self.clear_image,
            font=("Arial", 12), 
            bg="orange", 
            fg="black",
            width=15, 
            height=2,
            highlightbackground="lightgray",
            highlightthickness=1,
            bd=1,
            relief="solid"
            )

        self.btn_limpiar.pack(side=tk.LEFT, padx=10)
        
        # Botón Salir
        self.btn_salir = Button(
            button_frame, 
            text="Salir", 
            command=self.salir_aplicacion,
            font=("Arial", 12), 
            bg="red", 
            fg="black",
            width=15, 
            height=2,
            highlightbackground="lightgray",
            highlightthickness=1,
            bd=1,
            relief="solid"
            )
        self.btn_salir.pack(side=tk.LEFT, padx=10)
        
    def capture_image(self):
        """
        Captura el frame actual de la webcam.
        
        Proceso:
            1. Guarda el frame actual
            2. Muestra imagen en panel derecho
            3. Llama a process_captured_image para análisis
        """
        if self.current_frame is None:
            self.estado_label.config(text="Error: No hay frame disponible para capturar", fg="red")
            print("Error: No hay frame disponible")
            return
        
        try:
            # Guardar el frame capturado
            self.imagen_capturada = self.current_frame.copy()
            self.hay_foto = True
            
            # Convertir de BGR a RGB para mostrar
            frame_rgb = cv2.cvtColor(self.imagen_capturada, cv2.COLOR_BGR2RGB)
            
            # Redimensionar para que quepa en el panel de análisis
            height, width = frame_rgb.shape[:2]
            max_width, max_height = 580, 480
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convertir a formato PIL y luego a PhotoImage
            image_pil = Image.fromarray(frame_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)
            
            # Actualizar el label del análisis
            self.analysis_label.config(image=image_tk, text="", bg="lightgray")
            self.analysis_label.image = image_tk  # Mantener referencia
            
            # Actualizar estado
            self.estado_label.config(text="Estado: Foto capturada - Procesando...", fg="orange")
            
            # Procesar la imagen (detectar rostros y tapabocas)
            # Por ahora solo mostramos que se capturó
            self.root.after(100, self.process_captured_image)
            
            print("Imagen capturada correctamente")
            
        except Exception as e:
            print(f"Error al capturar imagen: {e}")
            self.estado_label.config(text="Error al capturar imagen", fg="red")
    
    def detect_faces_in_person(self, x1, y1, x2, y2):
        """
        Detecta rostros específicos dentro de una región de persona detectada.
        Incluye detección multi-orientación para manejar rotaciones.
        
        Args:
            x1, y1, x2, y2: Coordenadas del bounding box de la persona
        
        Returns:
            list: Lista de coordenadas de rostros detectados
        """
        try:
            # Extraer región de la persona
            person_roi = self.imagen_capturada[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                return []
            
            # Usar múltiples cascadas para diferentes orientaciones
            face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            face_cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            
            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            
            all_faces = []
            
            # Detección frontal
            faces_frontal = face_cascade_frontal.detectMultiScale(
                gray_roi, 
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (fx, fy, fw, fh) in faces_frontal:
                abs_x1 = x1 + fx
                abs_y1 = y1 + fy
                abs_x2 = x1 + fx + fw
                abs_y2 = y1 + fy + fh
                all_faces.append((abs_x1, abs_y1, abs_x2, abs_y2, 'frontal'))
            
            # Detección de perfil (rostros girados)
            faces_profile = face_cascade_profile.detectMultiScale(
                gray_roi, 
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (fx, fy, fw, fh) in faces_profile:
                abs_x1 = x1 + fx
                abs_y1 = y1 + fy
                abs_x2 = x1 + fx + fw
                abs_y2 = y1 + fy + fh
                all_faces.append((abs_x1, abs_y1, abs_x2, abs_y2, 'profile'))
            
            # Detección con imagen rotada (para rostros girados)
            for angle in [15, 30, 45, -15, -30, -45]:
                rotated = self.rotate_image(gray_roi, angle)
                faces_rotated = face_cascade_frontal.detectMultiScale(
                    rotated, 
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(40, 40),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                for (fx, fy, fw, fh) in faces_rotated:
                    # Convertir coordenadas de vuelta a la imagen original
                    orig_coords = self.rotate_coordinates_back((fx, fy, fw, fh), angle, gray_roi.shape)
                    if orig_coords:
                        fx, fy, fw, fh = orig_coords
                        abs_x1 = x1 + fx
                        abs_y1 = y1 + fy
                        abs_x2 = x1 + fx + fw
                        abs_y2 = y1 + fy + fh
                        all_faces.append((abs_x1, abs_y1, abs_x2, abs_y2, f'rotated_{angle}'))
            
            # Filtrar duplicados en detecciones de rostros
            filtered_faces = self.filter_face_duplicates(all_faces)
            
            # Detección completada sin debug
            return [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in filtered_faces]
            
        except Exception as e:
            print(f"Error detectando rostros: {e}")
            return []
    
    def rotate_image(self, image, angle):
        """
        Rota una imagen por el ángulo especificado.
        
        Args:
            image: Imagen en escala de grises
            angle: Ángulo en grados
        
        Returns:
            Imagen rotada
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated
    
    def rotate_coordinates_back(self, coords, angle, original_shape):
        """
        Convierte coordenadas de imagen rotada de vuelta a la imagen original.
        
        Args:
            coords: (x, y, w, h) en imagen rotada
            angle: Ángulo de rotación aplicado
            original_shape: (height, width) de imagen original
        
        Returns:
            Coordenadas en imagen original o None si fuera de límites
        """
        try:
            fx, fy, fw, fh = coords
            height, width = original_shape
            center = (width // 2, height // 2)
            
            # Puntos del rectángulo en imagen rotada
            points = np.array([
                [fx, fy],
                [fx + fw, fy],
                [fx + fw, fy + fh],
                [fx, fy + fh]
            ], dtype=np.float32)
            
            # Matriz de rotación inversa
            rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
            
            # Transformar puntos de vuelta
            transformed_points = cv2.transform(points.reshape(1, -1, 2), rotation_matrix).reshape(-1, 2)
            
            # Calcular bounding box
            x_coords = transformed_points[:, 0]
            y_coords = transformed_points[:, 1]
            
            new_x1 = int(np.min(x_coords))
            new_y1 = int(np.min(y_coords))
            new_x2 = int(np.max(x_coords))
            new_y2 = int(np.max(y_coords))
            
            # Verificar que esté dentro de límites
            if (new_x1 >= 0 and new_y1 >= 0 and 
                new_x2 < width and new_y2 < height and
                new_x2 > new_x1 and new_y2 > new_y1):
                return (new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1)
            
            return None
            
        except Exception as e:
            print(f"Error convirtiendo coordenadas: {e}")
            return None
    
    def filter_face_duplicates(self, faces):
        """
        Filtra detecciones duplicadas de rostros usando IoU.
        
        Args:
            faces: Lista de (x1, y1, x2, y2, tipo)
        
        Returns:
            Lista filtrada sin duplicados
        """
        if len(faces) <= 1:
            return faces
        
        filtered = []
        for face in faces:
            x1, y1, x2, y2, face_type = face
            is_duplicate = False
            
            for accepted in filtered:
                ax1, ay1, ax2, ay2, _ = accepted
                iou = self.calculate_iou((x1, y1, x2, y2), (ax1, ay1, ax2, ay2))
                
                if iou > Config.IOU_THRESHOLD:  # Umbral más bajo para rostros
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(face)
        
        return filtered
    
    def estimate_face_region(self, x1, y1, x2, y2):
        """
        Estima la región del rostro cuando la detección específica falla.
        
        Args:
            x1, y1, x2, y2: Coordenadas del bounding box de la persona
        
        Returns:
            tuple: Coordenadas estimadas del rostro
        """
        height = y2 - y1
        width = x2 - x1
        
        # Calcular región del rostro más conservadora
        # Rostro típicamente ocupa 20-25% de la altura de una persona
        face_height = int(height * 0.25)  # Más conservador: 25% superior
        
        # Centrar horizontalmente
        face_width = int(width * 0.8)  # 80% del ancho
        face_x_offset = int((width - face_width) / 2)
        
        # Coordenadas del rostro estimado
        face_x1 = x1 + face_x_offset
        face_y1 = y1
        face_x2 = face_x1 + face_width
        face_y2 = y1 + face_height
        
        # Región estimada calculada
        return (face_x1, face_y1, face_x2, face_y2)
    
    def process_captured_image(self):
        """
        Procesa la imagen capturada con YOLO y clasificación.
        
        Pasos:
            1. Detecta personas con YOLOv8 (conf >= 0.5) - umbral más alto
            2. Filtra detecciones duplicadas usando NMS
            3. Para cada persona, clasifica si tiene tapabocas
            4. Dibuja bounding boxes con colores según resultado
            5. Actualiza label de estado con estadísticas
        
        Colores de bounding box:
            - Verde: CON TAPABOCAS
            - Rojo: SIN TAPABOCAS
            - Naranja: NO DETECTADO
        """
        if self.imagen_capturada is None or self.model is None:
            self.estado_label.config(text="Error: No hay imagen o modelo no disponible", fg="red")
            return
        
        try:
            # Realizar detección con YOLO con umbral más alto para evitar duplicados
            results = self.model(self.imagen_capturada, conf=Config.YOLO_CONFIDENCE, verbose=False)
            
            # Copiar imagen para dibujar sobre ella
            self.imagen_procesada = self.imagen_capturada.copy()
            self.detecciones = []
            
            # Procesar resultados y aplicar filtrado de duplicados
            detections_raw = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Obtener clase detectada
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Solo procesar personas (clase 0 en COCO)
                    if cls == 0:  # Persona
                        # Obtener coordenadas del bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections_raw.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf,
                            'class': 'Persona'
                        })
            
            # Aplicar filtrado de duplicados usando IoU
            filtered_detections = self.filter_duplicate_detections(detections_raw)
            
            # Procesar cada detección filtrada
            for det in filtered_detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                
                # Detectar rostros dentro de la persona
                faces = self.detect_faces_in_person(x1, y1, x2, y2)
                
                if faces:
                    # Procesar cada rostro detectado
                    for face_x1, face_y1, face_x2, face_y2 in faces:
                        # Clasificar si tiene tapabocas
                        tiene_tapabocas, metrics = self.classify_mask_in_bbox(face_x1, face_y1, face_x2, face_y2)
                        
                        # Guardar detección del rostro
                        self.detecciones.append({
                            'bbox': (face_x1, face_y1, face_x2, face_y2),
                            'confidence': conf,
                            'class': 'Rostro',
                            'tiene_tapabocas': tiene_tapabocas,
                            'metrics': metrics
                        })
                else:
                    # Si no se detecta rostro específico, estimar región del rostro
                    face_x1, face_y1, face_x2, face_y2 = self.estimate_face_region(x1, y1, x2, y2)
                    
                    # Clasificar si tiene tapabocas
                    tiene_tapabocas, metrics = self.classify_mask_in_bbox(face_x1, face_y1, face_x2, face_y2)
                    
                    # Guardar detección estimada del rostro
                    self.detecciones.append({
                        'bbox': (face_x1, face_y1, face_x2, face_y2),
                        'confidence': conf * 0.7,  # Reducir confianza por estimación
                        'class': 'Rostro Estimado',
                        'tiene_tapabocas': tiene_tapabocas,
                        'metrics': metrics
                    })
            
            # Dibujar resultados en la imagen
            self.draw_detections()
            
            # Actualizar estado con información de tapabocas
            num_personas = len(self.detecciones)
            if num_personas > 0:
                con_tapabocas = sum(1 for d in self.detecciones if d['tiene_tapabocas'] == 'CON TAPABOCAS')
                sin_tapabocas = sum(1 for d in self.detecciones if d['tiene_tapabocas'] == 'SIN TAPABOCAS')
                no_detectado = sum(1 for d in self.detecciones if d['tiene_tapabocas'] == 'NO DETECTADO')
                
                estado_texto = f"Estado: Hay foto - {num_personas} persona(s): "
                estado_texto += f"{con_tapabocas} con tapabocas, {sin_tapabocas} sin tapabocas"
                if no_detectado > 0:
                    estado_texto += f", {no_detectado} no detectado(s)"
                
                self.estado_label.config(text=estado_texto, fg="green")
            else:
                self.estado_label.config(
                    text="Estado: Hay foto - No se detectaron personas", 
                    fg="orange"
                )
            
            # Mostrar resumen de análisis
            self.print_analysis_summary(num_personas, con_tapabocas, sin_tapabocas, no_detectado)
            
        except Exception as e:
            print(f"Error en procesamiento: {e}")
            self.estado_label.config(text="Error al procesar imagen", fg="red")
    
    def print_analysis_summary(self, num_personas, con_tapabocas, sin_tapabocas, no_detectado):
        """
        Imprime un resumen limpio del análisis realizado.
        Limpia la consola antes de mostrar el resumen.
        
        Args:
            num_personas: Número total de personas detectadas
            con_tapabocas: Número de personas con tapabocas
            sin_tapabocas: Número de personas sin tapabocas
            no_detectado: Número de personas no detectadas
        """
        # Limpiar consola antes del resumen
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
        

        print("RESUMEN DE ANÁLISIS - DETECTOR DE TAPABOCAS")
        
        # Estadísticas generales
        print(f"👥 Personas detectadas: {num_personas}")
        print(f"✅ Con tapabocas: {con_tapabocas}")
        print(f"❌ Sin tapabocas: {sin_tapabocas}")
        print(f"❓ No detectado: {no_detectado}")
        
        # Configuración del sistema
        print("\n🔧 CONFIGURACIÓN DEL SISTEMA:")
        print(f"   • Modelo YOLO: {self.model.model_name if hasattr(self.model, 'model_name') else 'YOLOv8n'}")
        print(f"   • Umbral confianza: 0.6")
        print(f"   • Filtrado IoU: 0.3")
        print(f"   • Área mínima: 2000 píxeles")
        print(f"   • Validación proporción: Activada")
        print(f"   • Detección multi-orientación: Activada")
        
        # Métricas de clasificación si hay detecciones
        if num_personas > 0 and hasattr(self, 'detecciones'):
            print("\n📈 MÉTRICAS DE CLASIFICACIÓN:")
            for i, det in enumerate(self.detecciones, 1):
                if 'metrics' in det and det['metrics']:
                    metrics = det['metrics']
                    print(f"   Persona #{i}:")
                    print(f"     • Piel visible: {metrics.get('skin_ratio', 0):.3f}")
                    print(f"     • Color no-piel: {metrics.get('non_skin_ratio', 0):.3f}")
                    print(f"     • Color tapabocas: {metrics.get('mask_color_ratio', 0):.3f}")
                    print(f"     • Densidad bordes: {metrics.get('edge_density', 0):.3f}")
                    print(f"     • Varianza color: {metrics.get('color_variance', 0):.1f}")
                    print(f"     • Desv. estándar: {metrics.get('texture_std', 0):.1f}")
                    print(f"     • Score final: {metrics.get('score', 0)}")
                    print(f"     • Resultado: {det['tiene_tapabocas']}")
    
    def filter_duplicate_detections(self, detections):
        """
        Filtra detecciones duplicadas usando IoU y validación de tamaño.
        
        Args:
            detections: Lista de detecciones con bbox y confidence
        
        Returns:
            Lista filtrada sin duplicados
        """
        if len(detections) <= 1:
            return detections
        
        # Ordenar por confianza (mayor a menor)
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for det in sorted_detections:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Validación de tamaño mínimo (eliminar detecciones muy pequeñas)
            if area < Config.MIN_AREA:
                continue
            
            # Validación de proporción (personas no son muy anchas ni muy altas)
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < Config.MIN_ASPECT_RATIO or aspect_ratio > Config.MAX_ASPECT_RATIO:
                continue
            
            is_duplicate = False
            
            # Comparar con detecciones ya aceptadas
            for accepted in filtered:
                ax1, ay1, ax2, ay2 = accepted['bbox']
                
                # Calcular IoU
                iou = self.calculate_iou((x1, y1, x2, y2), (ax1, ay1, ax2, ay2))
                
                # Si IoU > 0.3, considerar duplicado
                if iou > Config.IOU_THRESHOLD:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(det)
        
        return filtered
    
    def calculate_iou(self, box1, box2):
        """
        Calcula Intersection over Union (IoU) entre dos bounding boxes.
        
        Args:
            box1, box2: Tuplas (x1, y1, x2, y2)
        
        Returns:
            float: Valor IoU entre 0 y 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calcular intersección
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calcular unión
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _create_color_masks(self, hsv_region):
        """Crea máscaras de color para diferentes tipos de tapabocas."""
        masks = {}
        
        # Máscara de piel
        skin_mask1 = cv2.inRange(hsv_region, 
                                np.array(Config.SKIN_RANGE_1[0], dtype=np.uint8),
                                np.array(Config.SKIN_RANGE_1[1], dtype=np.uint8))
        skin_mask2 = cv2.inRange(hsv_region, 
                                np.array(Config.SKIN_RANGE_2[0], dtype=np.uint8),
                                np.array(Config.SKIN_RANGE_2[1], dtype=np.uint8))
        masks['skin'] = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        # Máscaras de colores de tapabocas
        masks['white_gray'] = cv2.inRange(hsv_region, 
                                         np.array(Config.WHITE_GRAY_RANGE[0], dtype=np.uint8),
                                         np.array(Config.WHITE_GRAY_RANGE[1], dtype=np.uint8))
        masks['blue'] = cv2.inRange(hsv_region, 
                                   np.array(Config.BLUE_RANGE[0], dtype=np.uint8),
                                   np.array(Config.BLUE_RANGE[1], dtype=np.uint8))
        masks['black'] = cv2.inRange(hsv_region, 
                                    np.array(Config.BLACK_RANGE[0], dtype=np.uint8),
                                    np.array(Config.BLACK_RANGE[1], dtype=np.uint8))
        
        return masks
        """
        Clasifica presencia de tapabocas en un rostro detectado.
        
        Método mejorado universal para tapabocas de cualquier color:
            Analiza la región de nariz/boca usando múltiples criterios
            optimizados para detectar tapabocas de cualquier color (azul, blanco, negro, etc.).
        
        Args:
            x1, y1, x2, y2: Coordenadas del bounding box del rostro
        
        Returns:
            tuple: (resultado, metrics) donde resultado es 'CON TAPABOCAS', 'SIN TAPABOCAS', o 'NO DETECTADO'
                   y metrics es un diccionario con las métricas calculadas
        """
        try:
            # Obtener región del rostro
            roi = self.imagen_capturada[y1:y2, x1:x2]
            
            if roi.size == 0:
                return 'NO DETECTADO', {}
            
            height, width = roi.shape[:2]
            
            # Analizar región específica de nariz/boca (50% inferior del rostro)
            mouth_nose_start = int(height * 0.5)
            mouth_nose_region = roi[mouth_nose_start:, :]
            
            if mouth_nose_region.size == 0:
                return 'NO DETECTADO', {}
            
            # Convertir a escala de grises
            gray_region = cv2.cvtColor(mouth_nose_region, cv2.COLOR_BGR2GRAY)
            
            # 1. Análisis de bordes (detección de Canny) - más sensible
            edges = cv2.Canny(gray_region, 15, 60)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 2. Análisis de color piel en HSV (rangos más específicos)
            hsv_region = cv2.cvtColor(mouth_nose_region, cv2.COLOR_BGR2HSV)
            
            # Detectar colores típicos de piel (sin tapabocas)
            # Rango 1: tonos naranjas/rojizos (piel clara)
            lower_skin1 = np.array([0, 40, 70], dtype=np.uint8)
            upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask1 = cv2.inRange(hsv_region, lower_skin1, upper_skin1)
            
            # Rango 2: tonos amarillentos (piel más oscura)
            lower_skin2 = np.array([20, 50, 70], dtype=np.uint8)
            upper_skin2 = np.array([30, 255, 255], dtype=np.uint8)
            skin_mask2 = cv2.inRange(hsv_region, lower_skin2, upper_skin2)
            
            # Combinar máscaras de piel
            skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
            
            # 3. Análisis de textura (desviación estándar)
            texture_std = np.std(gray_region)
            
            # 4. Análisis de uniformidad de color (varianza)
            color_variance = np.var(gray_region)
            
            # 5. Detección de colores no-piel (tapabocas de cualquier color)
            # Detectar colores que NO son piel (tapabocas blancos, azules, negros, etc.)
            non_skin_mask = cv2.bitwise_not(skin_mask)
            non_skin_ratio = np.sum(non_skin_mask > 0) / non_skin_mask.size
            
            # Detección específica de colores comunes de tapabocas
            # Blancos/grises claros
            white_gray_mask = cv2.inRange(hsv_region, 
                                        np.array([0, 0, 200], dtype=np.uint8),   # Blanco/gris claro
                                        np.array([180, 30, 255], dtype=np.uint8))
            
            # Azules (quirúrgicos)
            blue_mask = cv2.inRange(hsv_region, 
                                   np.array([100, 50, 50], dtype=np.uint8),   # Azul
                                   np.array([130, 255, 255], dtype=np.uint8))
            
            # Negros/grises oscuros
            black_mask = cv2.inRange(hsv_region, 
                                    np.array([0, 0, 0], dtype=np.uint8),       # Negro
                                    np.array([180, 255, 50], dtype=np.uint8))   # Gris oscuro
            
            # Combinar máscaras de colores de tapabocas
            mask_colors = cv2.bitwise_or(white_gray_mask, blue_mask)
            mask_colors = cv2.bitwise_or(mask_colors, black_mask)
            mask_color_ratio = np.sum(mask_colors > 0) / mask_colors.size
            
            # 6. Análisis de brillo promedio
            brightness_mean = np.mean(gray_region)
            
            # SISTEMA DE PUNTUACIÓN OPTIMIZADO PARA TAPABOCAS UNIVERSALES
            tiene_tapabocas_score = 0
            
            # Criterio 1: Piel visible (CRITERIO PRINCIPAL)
            # Con tapabocas: muy poca piel visible en región nariz/boca
            if skin_ratio < 0.03:  # Muy poca piel (tapabocas cubre casi todo)
                tiene_tapabocas_score += 5
            elif skin_ratio < 0.08:  # Poca piel (tapabocas cubre la mayoría)
                tiene_tapabocas_score += 4
            elif skin_ratio < 0.15:  # Algo de piel (tapabocas parcial)
                tiene_tapabocas_score += 3
            elif skin_ratio < 0.25:  # Bastante piel (tapabocas mal colocado)
                tiene_tapabocas_score += 2
            elif skin_ratio < 0.35:  # Mucha piel (probablemente sin tapabocas)
                tiene_tapabocas_score += 1
            elif skin_ratio > 0.60:  # Mucha piel visible (sin tapabocas)
                tiene_tapabocas_score -= 5
            elif skin_ratio > 0.45:  # Bastante piel (probablemente sin tapabocas)
                tiene_tapabocas_score -= 4
            
            # Criterio 2: Detección de colores no-piel (tapabocas de cualquier color)
            if non_skin_ratio > 0.80:  # Mucho color no-piel (tapabocas cubre casi todo)
                tiene_tapabocas_score += 4
            elif non_skin_ratio > 0.60:  # Bastante color no-piel
                tiene_tapabocas_score += 3
            elif non_skin_ratio > 0.40:  # Algo de color no-piel
                tiene_tapabocas_score += 2
            elif non_skin_ratio > 0.20:  # Poco color no-piel
                tiene_tapabocas_score += 1
            
            # Criterio 2b: Detección específica de colores comunes de tapabocas
            if mask_color_ratio > 0.30:  # Mucho color de tapabocas detectado
                tiene_tapabocas_score += 2
            elif mask_color_ratio > 0.15:  # Algo de color de tapabocas
                tiene_tapabocas_score += 1
            
            # Criterio 3: Densidad de bordes
            # Tapabocas tiene bordes definidos en los contornos
            if edge_density > 0.20:  # Muchos bordes (tapabocas bien definido)
                tiene_tapabocas_score += 2
            elif edge_density > 0.12:  # Algunos bordes (tapabocas visible)
                tiene_tapabocas_score += 1
            elif edge_density < 0.05:  # Muy pocos bordes (sin tapabocas)
                tiene_tapabocas_score -= 1
            
            # Criterio 4: Uniformidad de color
            # Tapabocas tiende a ser más uniforme que piel
            if color_variance < 150:  # Muy uniforme (tapabocas)
                tiene_tapabocas_score += 2
            elif color_variance < 300:  # Bastante uniforme
                tiene_tapabocas_score += 1
            elif color_variance > 800:  # Muy variado (piel)
                tiene_tapabocas_score -= 1
            
            # Criterio 5: Textura
            # Tapabocas tiene textura más homogénea
            if texture_std < 12:  # Muy homogéneo (tapabocas)
                tiene_tapabocas_score += 2
            elif texture_std < 20:  # Bastante homogéneo
                tiene_tapabocas_score += 1
            elif texture_std > 35:  # Muy variado (piel)
                tiene_tapabocas_score -= 1
            
            # Guardar métricas para resumen
            metrics = {
                'skin_ratio': skin_ratio,
                'non_skin_ratio': non_skin_ratio,
                'mask_color_ratio': mask_color_ratio,
                'edge_density': edge_density,
                'color_variance': color_variance,
                'texture_std': texture_std,
                'score': tiene_tapabocas_score
            }
            
            # Decisión final (umbrales ajustados para tapabocas universales)
            if tiene_tapabocas_score >= 5:
                return 'CON TAPABOCAS', metrics
            elif tiene_tapabocas_score <= -3:
                return 'SIN TAPABOCAS', metrics
            else:
                # Si no estamos seguros, usar criterio principal (piel visible)
                if skin_ratio > 0.50:
                    return 'SIN TAPABOCAS', metrics
                elif skin_ratio < 0.05:
                    return 'CON TAPABOCAS', metrics
                else:
                    return 'NO DETECTADO', metrics
                
        except Exception as e:
            print(f"Error en clasificación: {e}")
            return 'NO DETECTADO', {}
    
    def draw_detections(self):
        """
        Dibuja bounding boxes y labels en la imagen.
        
        Características:
            - Rectángulos con grosor de 3px
            - Colores según clasificación (verde/rojo/naranja)
            - Labels en 2 líneas: "Persona #N" y resultado
            - Fondo coloreado para mejor legibilidad
        """
        if self.imagen_procesada is None:
            return
        
        # Dibujar cada detección
        for i, det in enumerate(self.detecciones):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            tiene_tapabocas = det.get('tiene_tapabocas', 'NO DETECTADO')
            
            # Color según resultado: Verde=CON, Rojo=SIN, Naranja=NO DETECTADO
            if tiene_tapabocas == 'CON TAPABOCAS':
                color = (0, 255, 0)  # Verde
                color_texto = 'Verde'
            elif tiene_tapabocas == 'SIN TAPABOCAS':
                color = (0, 0, 255)  # Rojo
                color_texto = 'Rojo'
            else:
                color = (0, 165, 255)  # Naranja
                color_texto = 'Naranja'
            
            thickness = 3
            
            # Dibujar rectángulo
            cv2.rectangle(self.imagen_procesada, (x1, y1), (x2, y2), color, thickness)
            
            # Preparar texto del label principal
            label = f"Persona #{i+1}"
            label2 = f"{tiene_tapabocas}"
            
            # Configuración de fuente
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Calcular tamaño del texto
            (text_width1, text_height1), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            (text_width2, text_height2), _ = cv2.getTextSize(label2, font, font_scale, font_thickness)
            
            max_width = max(text_width1, text_width2)
            total_height = text_height1 + text_height2 + 15
            
            # Dibujar fondo del label
            cv2.rectangle(self.imagen_procesada, 
                         (x1, y1 - total_height - 5), 
                         (x1 + max_width + 10, y1), 
                         color, -1)
            
            # Dibujar texto del label (2 líneas)
            cv2.putText(self.imagen_procesada, label, 
                       (x1 + 5, y1 - text_height2 - 10), 
                       font, font_scale, (255, 255, 255), font_thickness)
            
            cv2.putText(self.imagen_procesada, label2, 
                       (x1 + 5, y1 - 5), 
                       font, font_scale, (255, 255, 255), font_thickness)
        
        # Mostrar imagen procesada en el panel
        self.display_processed_image()
        
    def display_processed_image(self):
        """
        Muestra la imagen procesada en el panel derecho.
        
        Convierte de BGR a RGB y redimensiona para ajustar al panel.
        """
        if self.imagen_procesada is None:
            return
        
        try:
            # Convertir de BGR a RGB
            frame_rgb = cv2.cvtColor(self.imagen_procesada, cv2.COLOR_BGR2RGB)
            
            # Redimensionar para que quepa en el panel
            height, width = frame_rgb.shape[:2]
            max_width, max_height = 580, 480
            
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convertir a formato PIL y luego a PhotoImage
            image_pil = Image.fromarray(frame_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)
            
            # Actualizar el label del análisis
            self.analysis_label.config(image=image_tk, text="", bg="lightgray")
            self.analysis_label.image = image_tk  # Mantener referencia
            
        except Exception as e:
            print(f"Error al mostrar imagen procesada: {e}")
    
    def clear_image(self):
        """
        Limpia el panel de análisis y resetea el estado.
        
        Resetea todas las variables de imagen y detección.
        """
        self.hay_foto = False
        self.imagen_capturada = None
        self.imagen_procesada = None
        self.detecciones = []
        # Limpiar el panel de análisis completamente
        self.analysis_label.config(image='', text="Sin imagen capturada", bg="gray")
        self.analysis_label.image = None  # Eliminar referencia a la imagen mostrada
        self.estado_label.config(text="Estado: Sin foto - Esperando captura", fg="black")
        print("Imagen limpiada")
        
    def init_camera(self):
        """
        Inicializa la cámara web (dispositivo 0).
        
        Configura resolución 640x480 e inicia el loop de video.
        """
        try:
            self.cap = cv2.VideoCapture(0)  # Cámara predeterminada
            if not self.cap.isOpened():
                self.video_label.config(text="Error: No se pudo acceder a la cámara", fg="red")
                return
            
            # Configurar resolución
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.video_running = True
            self.update_video_feed()
            print("Cámara inicializada correctamente")
            
        except Exception as e:
            print(f"Error al inicializar cámara: {e}")
            self.video_label.config(text="Error al inicializar cámara", fg="red")
    
    def update_video_feed(self):
        """
        Actualiza el video en tiempo real (~30 FPS).
        
        Lee frames de la cámara, aplica efecto espejo y muestra en panel izquierdo.
        Se ejecuta recursivamente cada 30ms.
        """
        if not self.video_running or not self.cap:
            return
            
        try:
            ret, frame = self.cap.read()
            if ret:
                # Voltear la imagen horizontalmente (efecto espejo)
                frame = cv2.flip(frame, 1)
                
                # Guardar frame actual para captura
                self.current_frame = frame.copy()
                
                # Convertir de BGR a RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Redimensionar para que quepa en el panel
                height, width = frame_rgb.shape[:2]
                max_width, max_height = Config.PANEL_WIDTH, Config.PANEL_HEIGHT
                
                if width > max_width or height > max_height:
                    scale = min(max_width/width, max_height/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                
                # Convertir a formato PIL y luego a PhotoImage
                image_pil = Image.fromarray(frame_rgb)
                image_tk = ImageTk.PhotoImage(image_pil)
                
                # Actualizar el label del video
                self.video_label.config(image=image_tk, text="")
                self.video_label.image = image_tk  # Mantener referencia
                
            else:
                self.video_label.config(text="Error leyendo frame de cámara", fg="red")
                
        except Exception as e:
            print(f"Error actualizando video: {e}")
            self.video_label.config(text="Error en video feed", fg="red")
        
        # Programar siguiente actualización
        if self.video_running:
            self.root.after(30, self.update_video_feed)  # ~30 FPS
    
    def salir_aplicacion(self):
        """
        Cierra la aplicación de forma segura.
        
        Libera la cámara y destruye la ventana.
        """
        print("Cerrando aplicación...")
        self.video_running = False
        
        if self.cap:
            self.cap.release()
            
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """
        Inicia el loop principal de la aplicación tkinter.
        """
        print("Iniciando Detector de Tapabocas...")
        self.root.mainloop()


def main():
    """
    Punto de entrada de la aplicación.
    
    Crea una instancia de DetectorTapabocas y la ejecuta.
    """
    app = DetectorTapabocas()
    app.run()


if __name__ == "__main__":
    main()
