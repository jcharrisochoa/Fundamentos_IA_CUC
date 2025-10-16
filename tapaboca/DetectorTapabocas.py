"""
Detector de Tapabocas con YOLO + OpenCV
Autor: Johan Charris Ochoa - Universidad de la Costa (CUC) - 2025
"""

import tkinter as tk
from tkinter import Label, Button, Frame, Text, Scrollbar, END
import cv2
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import config as cfg
import os
import platform

class DetectorTapabocas:
    """Detector de tapabocas con YOLOv8 y análisis de características faciales."""
    
    def __init__(self):
        """Inicializa ventana, modelo YOLO, GUI y cámara."""
        self.root = tk.Tk()
        self.root.title(cfg.WINDOW_TITLE)
        self.root.geometry(cfg.WINDOW_GEOMETRY)
        self.root.resizable(cfg.WINDOW_RESIZABLE, cfg.WINDOW_RESIZABLE)
        
        # Estado de la aplicación
        self.hay_foto = False
        self.imagen_capturada = None
        self.imagen_procesada = None
        self.cap = None
        self.video_running = False
        self.current_frame = None
        self.detecciones = []
        
        # Cargar modelo YOLO
        self.model = self._init_yolo_model()
        
        # Inicializar cascadas de Haar para detección de rostros
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cfg.CASCADE_FRONTAL_FACE)
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cfg.CASCADE_PROFILE_FACE)
        
        self.setup_gui()
        self.init_camera()
        
    def _init_yolo_model(self):
        """Carga el modelo YOLOv8 para detección de personas."""
        try:
            model = YOLO(cfg.YOLO_MODEL_PATH)
            return model
        except Exception as e:
            print(f"Error al cargar modelo YOLO: {e}")
            return None
    
    # ==================== CÁMARA Y VIDEO ====================
    def init_camera(self):
        """Inicializa cámara web y comienza captura de video."""
        try:
            self.cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
            if not self.cap.isOpened():
                self.video_label.config(text=cfg.MSG_ERROR_CAMERA, fg=cfg.COLOR_TEXT_BLACK)
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
            self.video_running = True
            self.update_video_feed()
        except:
            self.video_label.config(text=cfg.MSG_ERROR_INIT_CAMERA, fg=cfg.COLOR_TEXT_BLACK)
    
    def update_video_feed(self):
        """Actualiza video en tiempo real usando FPS configurado."""
        if not self.video_running or not self.cap:
            return
            
        try:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = cv2.flip(frame, 1)  # Efecto espejo
                self._update_label_image(self.video_label, self.current_frame)
            else:
                self.video_label.config(text=cfg.MSG_ERROR_READ_FRAME, fg=cfg.COLOR_TEXT_BLACK)
        except:
            self.video_label.config(text=cfg.MSG_ERROR_VIDEO_FEED, fg=cfg.COLOR_TEXT_BLACK)
                
        if self.video_running:
            self.root.after(cfg.VIDEO_UPDATE_INTERVAL, self.update_video_feed)
 
    # ==================== MÉTODOS AUXILIARES DE CAMARA VIDEO Y CAPTURA DE IMAGENES INTERFAZ ====================
    
    def _resize_frame(self, frame, max_size=None): # Redimensiona frame para ajustar al panel si es necesario
        """Redimensiona frame para ajustar al panel si es necesario."""
        if max_size is None:
            max_size = (cfg.PANEL_WIDTH, cfg.PANEL_HEIGHT)
        height, width = frame.shape[:2]
        max_width, max_height = max_size
        
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_size = (int(width * scale), int(height * scale))
            return cv2.resize(frame, new_size)
        return frame
    
    def _frame_to_photoimage(self, frame): # Convierte frame BGR a PhotoImage para tkinter
        """Convierte frame BGR a PhotoImage para tkinter."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = self._resize_frame(frame_rgb)
        image_pil = Image.fromarray(frame_resized)
        return ImageTk.PhotoImage(image_pil)
    
    def _update_label_image(self, label, frame):
        """Actualiza un label con una nueva imagen."""
        try:
            photo = self._frame_to_photoimage(frame) # Convierte frame BGR a PhotoImage para tkinter
            label.config(image=photo, text="", bg=cfg.COLOR_PANEL_BG)
            label.image = photo
        except:
            pass
    
    # ==================== CONFIGURACIÓN DE GUI ====================
    
    def setup_gui(self):
        """Crea la interfaz gráfica con paneles de video y controles."""
        main_frame = Frame(self.root, bg=cfg.COLOR_BACKGROUND)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=cfg.PADDING_X, pady=cfg.PADDING_Y)
        
        Label(main_frame, text=cfg.LABEL_MAIN_TITLE, 
              font=cfg.FONT_TITLE, bg=cfg.COLOR_BACKGROUND, fg=cfg.COLOR_TEXT_BLACK).pack(pady=10)
        
        video_frame = Frame(main_frame, bg=cfg.COLOR_BACKGROUND)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Panel izquierdo - Video en tiempo real
        left_panel = self._create_panel(video_frame, cfg.LABEL_CAMERA_PANEL, tk.LEFT)
        self.video_label = Label(left_panel, bg=cfg.COLOR_VIDEO_BG, text=cfg.MSG_INIT_CAMERA, 
                                fg=cfg.COLOR_TEXT_WHITE, font=cfg.FONT_VIDEO_LABEL)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=cfg.PADDING_X, pady=cfg.PADDING_Y)

        # Panel derecho - Imagen capturada
        right_panel = self._create_panel(video_frame, cfg.LABEL_ANALYSIS_PANEL, tk.RIGHT)
        self.analysis_label = Label(right_panel, bg=cfg.COLOR_ANALYSIS_BG, text=cfg.MSG_NO_IMAGE, 
                                    fg=cfg.COLOR_TEXT_BLACK, font=cfg.FONT_ANALYSIS_LABEL)
        self.analysis_label.pack(fill=tk.BOTH, expand=True, padx=cfg.PADDING_X, pady=cfg.PADDING_Y)
        
        # Log de análisis con scroll
        log_frame = Frame(main_frame, bg=cfg.COLOR_BACKGROUND)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        Label(log_frame, text=cfg.LABEL_LOG_TITLE, font=cfg.FONT_SUBTITLE, 
              bg=cfg.COLOR_BACKGROUND, fg=cfg.COLOR_TEXT_BLACK).pack(pady=5)
        
        log_container = Frame(log_frame, relief="solid", bd=1)
        log_container.pack(fill=tk.BOTH, expand=True, padx=cfg.PADDING_X)
        
        scrollbar = Scrollbar(log_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = Text(log_container, height=cfg.LOG_HEIGHT, wrap=tk.WORD, 
                            yscrollcommand=scrollbar.set, font=cfg.FONT_LOG,
                            bg=cfg.COLOR_LOG_BG, fg=cfg.COLOR_TEXT_BLACK)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)
        
        # Mensaje inicial en el log
        self._add_log_entry(cfg.MSG_SYSTEM_INIT)
        
        # Controles
        control_frame = Frame(main_frame, bg=cfg.COLOR_BACKGROUND)
        control_frame.pack(pady=15)
        
        self.estado_label = Label(control_frame, text=cfg.MSG_NO_PHOTO, 
                                 font=cfg.FONT_STATUS, bg=cfg.COLOR_BACKGROUND, fg=cfg.COLOR_TEXT_BLACK)
        self.estado_label.pack(pady=8)
        
        button_frame = Frame(control_frame, bg=cfg.COLOR_BACKGROUND)
        button_frame.pack()
        
        # Botones de control
        self._create_button(button_frame, cfg.LABEL_BUTTON_CAPTURE, self.capture_image, cfg.BUTTON_COLOR_CAPTURE)
        self._create_button(button_frame, cfg.LABEL_BUTTON_CLEAR, self.clear_image, cfg.BUTTON_COLOR_CLEAR)
        self._create_button(button_frame, cfg.LABEL_BUTTON_CLEAR_LOG, self.clear_log, cfg.BUTTON_COLOR_CLEAR_LOG)
        self._create_button(button_frame, cfg.LABEL_BUTTON_EXIT, self.salir_aplicacion, cfg.BUTTON_COLOR_EXIT)
    
    def _create_panel(self, parent, title, side):
        """Crea un panel con título para video o análisis."""
        panel = Frame(parent, bg=cfg.COLOR_PANEL_BG, relief="solid", bd=1, 
                     width=cfg.PANEL_WIDTH, height=cfg.PANEL_HEIGHT)
        panel.pack(side=side, fill=None, expand=True, padx=(0, 5) if side == tk.LEFT else (5, 0))
        panel.pack_propagate(False)
        Label(panel, text=title, font=cfg.FONT_PANEL_TITLE, bg=cfg.COLOR_PANEL_BG, fg=cfg.COLOR_TEXT_BLACK).pack(pady=5)
        return panel
    
    def _create_button(self, parent, text, command, bg_color):
        """Crea un botón estilizado."""
        Button(parent, text=text, command=command, font=cfg.FONT_BUTTON, bg=bg_color, 
               fg=cfg.COLOR_TEXT_BLACK, width=cfg.BUTTON_WIDTH, height=cfg.BUTTON_HEIGHT, 
               highlightbackground=cfg.COLOR_PANEL_BG, highlightthickness=1, bd=1, 
               relief="solid").pack(side=tk.LEFT, padx=cfg.PADDING_X)
    
    def _add_log_entry(self, message, analysis_data=None):
        """Agrega una entrada al log con timestamp y métricas detalladas."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        separator = "─" * 70
        
        self.log_text.insert(END, f"\n{separator}\n")
        self.log_text.insert(END, f"📅 {timestamp}\n")
        self.log_text.insert(END, f"{message}\n")
        
        if analysis_data:
            num_personas = analysis_data.get('num_personas', 0)
            con = analysis_data.get('con_tapabocas', 0)
            sin = analysis_data.get('sin_tapabocas', 0)
            no_det = analysis_data.get('no_detectado', 0)
            
            self.log_text.insert(END, f"👥 Personas detectadas: {num_personas}\n")
            if num_personas > 0:
                self.log_text.insert(END, f"   ✅ Con tapabocas: {con}\n")
                self.log_text.insert(END, f"   ❌ Sin tapabocas: {sin}\n")
                if no_det > 0:
                    self.log_text.insert(END, f"   ❓ No detectado: {no_det}\n")
                
                # Agregar métricas detalladas por persona
                detecciones = analysis_data.get('detecciones', [])
                if detecciones:
                    self.log_text.insert(END, f"\n MÉTRICAS DETALLADAS POR PERSONA:\n")
                    self.log_text.insert(END, f"{'#':<3} {'Piel':>6} {'No-Piel':>8} {'Color':>6} {'Bordes':>7} {'Var':>6} {'Text':>6} {'Score':>6} {'Resultado':<15}\n")
                    
                    for i, det in enumerate(detecciones, 1):
                        if 'metrics' in det and det['metrics']:
                            m = det['metrics']
                            resultado = det.get('tiene_tapabocas', 'NO DETECTADO')
                            self.log_text.insert(END, 
                                f"{i:<3} "
                                f"{m.get('skin_ratio', 0):>6.3f} "
                                f"{m.get('non_skin_ratio', 0):>8.3f} "
                                f"{m.get('mask_color_ratio', 0):>6.3f} "
                                f"{m.get('edge_density', 0):>7.3f} "
                                f"{m.get('color_variance', 0):>6.0f} "
                                f"{m.get('texture_std', 0):>6.1f} "
                                f"{m.get('score', 0):>6} "
                                f"{resultado:<15}\n"
                            )
        
        self.log_text.insert(END, "\n")
        self.log_text.see(END)  # Auto-scroll al final
        
    # ==================== CAPTURA Y PROCESAMIENTO ====================
    
    def capture_image(self):
        """Captura el frame actual y lo procesa."""
        if self.current_frame is None:
            self.estado_label.config(text=cfg.MSG_ERROR_NO_FRAME, fg=cfg.COLOR_TEXT_BLACK)
            return
        
        try:
            self.imagen_capturada = self.current_frame.copy()
            self.hay_foto = True
            self._update_label_image(self.analysis_label, self.imagen_capturada)
            self.estado_label.config(text=cfg.MSG_CAPTURING, fg=cfg.COLOR_TEXT_BLACK)
            self.root.after(100, self.process_captured_image)
        except:
            self.estado_label.config(text=cfg.MSG_ERROR_CAPTURE, fg=cfg.COLOR_TEXT_BLACK)
    
    def detect_faces_in_person(self, x1, y1, x2, y2):
        """Detecta rostros dentro de una región de persona."""
        try:
            person_roi = self.imagen_capturada[y1:y2, x1:x2]
            if person_roi.size == 0:
                return []
            
            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            all_faces = []
            
            # Detección frontal y de perfil
            for cascade in [self.face_cascade, self.profile_cascade]:
                faces = cascade.detectMultiScale(gray_roi, scaleFactor=cfg.FACE_SCALE_FACTOR, 
                                                 minNeighbors=cfg.FACE_MIN_NEIGHBORS, minSize=cfg.FACE_MIN_SIZE)
                for (fx, fy, fw, fh) in faces:
                    all_faces.append((x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh))
            
            return self._filter_duplicates_simple(all_faces)
        except:
            return []
    
    def _filter_duplicates_simple(self, boxes):
        """Filtra bounding boxes duplicados usando IoU."""
        if len(boxes) <= 1:
            return boxes
        
        filtered = []
        for box in boxes:
            if not any(self.calculate_iou(box, accepted) > cfg.IOU_THRESHOLD for accepted in filtered):
                filtered.append(box)
        return filtered
    
    def estimate_face_region(self, x1, y1, x2, y2):
        """Estima región del rostro usando proporciones configurables."""
        height, width = y2 - y1, x2 - x1
        face_width = int(width * cfg.FACE_REGION_WIDTH_RATIO)
        face_x_offset = int((width - face_width) / 2)
        return (x1 + face_x_offset, y1, x1 + face_x_offset + face_width, 
                y1 + int(height * cfg.FACE_REGION_HEIGHT_RATIO))
    
    def process_captured_image(self):
        """Procesa imagen con YOLO y clasifica tapabocas."""
        if self.imagen_capturada is None or self.model is None:
            self.estado_label.config(text=cfg.MSG_ERROR_NO_IMAGE_MODEL, fg=cfg.COLOR_TEXT_BLACK)
            return
        
        try:
            results = self.model(self.imagen_capturada, conf=cfg.YOLO_CONFIDENCE, verbose=cfg.YOLO_VERBOSE)
            
            self.imagen_procesada = self.imagen_capturada.copy()
            self.detecciones = []
            
            # Extraer detecciones de personas
            detections_raw = []
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == cfg.PERSON_CLASS_ID:  # Clase persona en COCO
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections_raw.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(box.conf[0])
                        })
            
            # Filtrar duplicados y procesar cada persona
            for det in self.filter_duplicate_detections(detections_raw):
                x1, y1, x2, y2 = det['bbox']
                faces = self.detect_faces_in_person(x1, y1, x2, y2)
                
                # Usar rostros detectados o estimación
                face_regions = faces if faces else [self.estimate_face_region(x1, y1, x2, y2)]
                
                for face_bbox in face_regions:
                    tiene_tapabocas, metrics = self.classify_mask_in_bbox(*face_bbox)
                    self.detecciones.append({
                        'bbox': face_bbox,
                        'confidence': det['confidence'] * (1.0 if faces else cfg.FACE_CONFIDENCE_PENALTY),
                        'tiene_tapabocas': tiene_tapabocas,
                        'metrics': metrics
                    })
            
            self.draw_detections()
            self._update_estado_label()
            self._print_analysis_summary()
            
        except:
            self.estado_label.config(text=cfg.MSG_ERROR_PROCESS, fg=cfg.COLOR_TEXT_BLACK)

    def filter_duplicate_detections(self, detections):
        """Filtra detecciones duplicadas por IoU, tamaño y proporción."""
        if len(detections) <= 1:
            return detections
        
        sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        filtered = []
        
        for det in sorted_dets:
            x1, y1, x2, y2 = det['bbox']
            width, height = x2 - x1, y2 - y1
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            # Validar tamaño y proporción
            if area < cfg.MIN_DETECTION_AREA or aspect_ratio < cfg.MIN_ASPECT_RATIO or aspect_ratio > cfg.MAX_ASPECT_RATIO:
                continue
            
            # Verificar duplicados
            if not any(self.calculate_iou(det['bbox'], a['bbox']) > cfg.IOU_THRESHOLD 
                      for a in filtered):
                filtered.append(det)
        
        return filtered
    
    def calculate_iou(self, box1, box2):
        """Calcula Intersection over Union entre dos bounding boxes."""
        x1_i = max(box1[0], box2[0])
        y1_i = max(box1[1], box2[1])
        x2_i = min(box1[2], box2[2])
        y2_i = min(box1[3], box2[3])
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # ==================== CLASIFICACIÓN DE TAPABOCAS ====================
    
    def classify_mask_in_bbox(self, x1, y1, x2, y2):
        """Clasifica si hay tapabocas en la región facial (analiza nariz/boca)."""
        try:
            roi = self.imagen_capturada[y1:y2, x1:x2]
            if roi.size == 0:
                return 'NO DETECTADO', {}
            
            # Extraer región nariz/boca (usando ratio de configuración)
            mouth_region = roi[int(roi.shape[0] * cfg.MOUTH_REGION_RATIO):, :]
            if mouth_region.size == 0:
                return 'NO DETECTADO', {}
            
            # Calcular métricas
            gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
            
            # Detección de bordes
            edge_density = np.sum(cv2.Canny(gray, cfg.CANNY_THRESHOLD_1, cfg.CANNY_THRESHOLD_2) > 0) / gray.size
            
            # Detección de piel (rangos HSV para diversos tonos)
            skin_mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array(cfg.SKIN_RANGE_1_LOWER), np.array(cfg.SKIN_RANGE_1_UPPER)),
                cv2.inRange(hsv, np.array(cfg.SKIN_RANGE_2_LOWER), np.array(cfg.SKIN_RANGE_2_UPPER))
            )
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
            non_skin_ratio = 1 - skin_ratio
            
            # Detección UNIVERSAL de tapabocas (todos los colores posibles del mercado)
            # Incluye: blancos, azules, negros, grises, verdes, rosas, rojos, amarillos, etc.
            mask_colors_list = [
                cv2.inRange(hsv, np.array(cfg.MASK_WHITE_LOWER), np.array(cfg.MASK_WHITE_UPPER)),   # Blancos/grises claros
                cv2.inRange(hsv, np.array(cfg.MASK_BLACK_LOWER), np.array(cfg.MASK_BLACK_UPPER)),   # Negros/grises oscuros
                cv2.inRange(hsv, np.array(cfg.MASK_BLUE_LOWER), np.array(cfg.MASK_BLUE_UPPER)),     # Azules
                cv2.inRange(hsv, np.array(cfg.MASK_GREEN_LOWER), np.array(cfg.MASK_GREEN_UPPER)),   # Verdes
                cv2.inRange(hsv, np.array(cfg.MASK_PINK_LOWER), np.array(cfg.MASK_PINK_UPPER)),     # Rosas/Morados
                cv2.inRange(hsv, np.array(cfg.MASK_RED_1_LOWER), np.array(cfg.MASK_RED_1_UPPER)),   # Rojos
                cv2.inRange(hsv, np.array(cfg.MASK_RED_2_LOWER), np.array(cfg.MASK_RED_2_UPPER)),   # Rojos (wrap)
                cv2.inRange(hsv, np.array(cfg.MASK_YELLOW_LOWER), np.array(cfg.MASK_YELLOW_UPPER)), # Amarillos/naranjas
            ]
            
            # Combinar todas las máscaras de colores
            mask_colors = mask_colors_list[0]
            for mask in mask_colors_list[1:]:
                mask_colors = cv2.bitwise_or(mask_colors, mask)
            
            mask_color_ratio = np.sum(mask_colors > 0) / mask_colors.size
            
            # Análisis de textura y color
            texture_std = np.std(gray)
            color_variance = np.var(gray)
            
            # Sistema de puntuación
            score = self._calculate_mask_score(skin_ratio, non_skin_ratio, mask_color_ratio, 
                                               edge_density, color_variance, texture_std)
            
            metrics = {
                'skin_ratio': skin_ratio,
                'non_skin_ratio': non_skin_ratio,
                'mask_color_ratio': mask_color_ratio,
                'edge_density': edge_density,
                'color_variance': color_variance,
                'texture_std': texture_std,
                'score': score
            }
            
            # Decisión final
            if score >= cfg.MASK_PRESENT_THRESHOLD:
                return 'CON TAPABOCAS', metrics
            elif score <= cfg.MASK_ABSENT_THRESHOLD:
                return 'SIN TAPABOCAS', metrics
            else:
                return ('SIN TAPABOCAS' if skin_ratio > cfg.SKIN_RATIO_HIGH else 
                       'CON TAPABOCAS' if skin_ratio < cfg.SKIN_RATIO_LOW else 'NO DETECTADO'), metrics
                
        except Exception as e:
            return 'NO DETECTADO', {}
    
    def _calculate_mask_score(self, skin_ratio, non_skin_ratio, mask_color_ratio, 
                             edge_density, color_variance, texture_std):
        """Calcula score de tapabocas basado en múltiples métricas."""
        score = 0
        
        # Criterio 1: Piel visible (principal)
        t = cfg.SKIN_RATIO_THRESHOLDS
        if skin_ratio < t['very_low'][0]: score += t['very_low'][1]
        elif skin_ratio < t['low'][0]: score += t['low'][1]
        elif skin_ratio < t['medium_low'][0]: score += t['medium_low'][1]
        elif skin_ratio < t['medium'][0]: score += t['medium'][1]
        elif skin_ratio < t['high'][0]: score += t['high'][1]
        elif skin_ratio > t['very_high'][0]: score += t['very_high'][1]
        elif skin_ratio > t['high_neg'][0]: score += t['high_neg'][1]
        
        # Criterio 2: Color no-piel
        t = cfg.NON_SKIN_RATIO_THRESHOLDS
        if non_skin_ratio > t['very_high'][0]: score += t['very_high'][1]
        elif non_skin_ratio > t['high'][0]: score += t['high'][1]
        elif non_skin_ratio > t['medium'][0]: score += t['medium'][1]
        elif non_skin_ratio > t['low'][0]: score += t['low'][1]
        
        # Criterio 3: Colores de tapabocas
        t = cfg.MASK_COLOR_RATIO_THRESHOLDS
        if mask_color_ratio > t['high'][0]: score += t['high'][1]
        elif mask_color_ratio > t['medium'][0]: score += t['medium'][1]
        
        # Criterio 4: Bordes
        t = cfg.EDGE_DENSITY_THRESHOLDS
        if edge_density > t['high'][0]: score += t['high'][1]
        elif edge_density > t['medium'][0]: score += t['medium'][1]
        elif edge_density < t['low'][0]: score += t['low'][1]
        
        # Criterio 5: Uniformidad
        t = cfg.COLOR_VARIANCE_THRESHOLDS
        if color_variance < t['low'][0]: score += t['low'][1]
        elif color_variance < t['medium'][0]: score += t['medium'][1]
        elif color_variance > t['high'][0]: score += t['high'][1]
        
        # Criterio 6: Textura
        t = cfg.TEXTURE_STD_THRESHOLDS
        if texture_std < t['low'][0]: score += t['low'][1]
        elif texture_std < t['medium'][0]: score += t['medium'][1]
        elif texture_std > t['high'][0]: score += t['high'][1]
        
        return score
    
    # ==================== VISUALIZACIÓN ====================
    
    def _update_estado_label(self):
        """Actualiza el label de estado con estadísticas y registra en el log."""
        num_personas = len(self.detecciones)
        if num_personas > 0:
            con = sum(1 for d in self.detecciones if d['tiene_tapabocas'] == 'CON TAPABOCAS')
            sin = sum(1 for d in self.detecciones if d['tiene_tapabocas'] == 'SIN TAPABOCAS')
            no_det = sum(1 for d in self.detecciones if d['tiene_tapabocas'] == 'NO DETECTADO')
            
            texto = f"Estado: {num_personas} persona(s): {con} con tapabocas, {sin} sin tapabocas"
            if no_det > 0:
                texto += f", {no_det} no detectado(s)"
            self.estado_label.config(text=texto, fg=cfg.COLOR_TEXT_BLACK)
            
            # Agregar al log de análisis con métricas detalladas
            analysis_data = {
                'num_personas': num_personas,
                'con_tapabocas': con,
                'sin_tapabocas': sin,
                'no_detectado': no_det,
                'detecciones': self.detecciones  # Incluir detecciones con métricas
            }
            self._add_log_entry(cfg.MSG_ANALYSIS_SUCCESS, analysis_data)
        else:
            self.estado_label.config(text=cfg.MSG_NO_DETECTION, fg=cfg.COLOR_TEXT_BLACK)
            self._add_log_entry(cfg.MSG_NO_PERSONS)
    
    def _clear_console(self):
        """Limpia la consola de manera multiplataforma."""
        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')
    
    def _print_system_config(self):
        """Imprime configuración del sistema y parámetros."""
        print(f"\n PARÁMETROS DEL MODELO:")
        print(f"   • Modelo YOLO:          YOLOv8n (nano)")
        print(f"   • Umbral confianza:     {cfg.YOLO_CONFIDENCE}")
        print(f"   • Umbral IoU:           {cfg.IOU_THRESHOLD}")
        print(f"   • Área mínima:          {cfg.MIN_DETECTION_AREA} píxeles")
        print(f"   • FPS video:            ~{cfg.VIDEO_FPS}")
        print(f"   • Resolución panel:     {cfg.PANEL_WIDTH}x{cfg.PANEL_HEIGHT}")
        print()
    
    def _print_analysis_summary(self):
        """Imprime resumen completo del análisis con variables y métricas."""
        self._clear_console()
        self._print_system_config()
        
        num_personas = len(self.detecciones)
        con = sum(1 for d in self.detecciones if d['tiene_tapabocas'] == 'CON TAPABOCAS')
        sin = sum(1 for d in self.detecciones if d['tiene_tapabocas'] == 'SIN TAPABOCAS')
        no_det = num_personas - con - sin
        
        print(f"\nRESUMEN GENERAL:")
        print(f"   • Total personas detectadas:  {num_personas}")
        print(f"   • ✅ Con tapabocas:            {con}")
        print(f"   • ❌ Sin tapabocas:            {sin}")
        print(f"   • ❓ No detectado:             {no_det}")
        
        if num_personas > 0:
            print(f"\n MÉTRICAS DETALLADAS POR PERSONA:")
            print(f"   {'#':<3} {'Piel':>6} {'No-Piel':>8} {'Color':>6} {'Bordes':>7} {'Varianza':>9} {'Textura':>8} {'Score':>6} {'Resultado':<15}")
            print(f"   {'-'*3} {'-'*6} {'-'*8} {'-'*6} {'-'*7} {'-'*9} {'-'*8} {'-'*6} {'-'*15}")
            
            for i, det in enumerate(self.detecciones, 1):
                if 'metrics' in det and det['metrics']:
                    m = det['metrics']
                    print(f"   {i:<3} {m.get('skin_ratio', 0):>6.3f} "
                          f"{m.get('non_skin_ratio', 0):>8.3f} "
                          f"{m.get('mask_color_ratio', 0):>6.3f} "
                          f"{m.get('edge_density', 0):>7.3f} "
                          f"{m.get('color_variance', 0):>9.1f} "
                          f"{m.get('texture_std', 0):>8.1f} "
                          f"{m.get('score', 0):>6} "
                          f"{det['tiene_tapabocas']:<15}")
        
        print("\n" + "="*66)
    
    def draw_detections(self):
        """Dibuja bounding boxes con labels en la imagen procesada."""
        if self.imagen_procesada is None:
            return
        
        colors = {
            'CON TAPABOCAS': cfg.BBOX_COLOR_WITH_MASK, 
            'SIN TAPABOCAS': cfg.BBOX_COLOR_WITHOUT_MASK, 
            'NO DETECTADO': cfg.BBOX_COLOR_UNKNOWN
        }
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for i, det in enumerate(self.detecciones):
            x1, y1, x2, y2 = det['bbox']
            resultado = det.get('tiene_tapabocas', 'NO DETECTADO')
            color = colors.get(resultado, cfg.BBOX_COLOR_UNKNOWN)
            
            # Dibujar rectángulo
            cv2.rectangle(self.imagen_procesada, (x1, y1), (x2, y2), color, cfg.BBOX_THICKNESS)
            
            # Preparar labels
            labels = [f"Persona #{i+1}", resultado]
            sizes = [cv2.getTextSize(l, font, cfg.FONT_DETECTION, 2)[0] for l in labels]
            max_w = max(s[0] for s in sizes)
            total_h = sum(s[1] for s in sizes) + 15
            
            # Fondo y texto
            cv2.rectangle(self.imagen_procesada, (x1, y1 - total_h - 5), 
                         (x1 + max_w + 10, y1), color, -1)
            cv2.putText(self.imagen_procesada, labels[0], (x1 + 5, y1 - sizes[1][1] - 10), 
                       font, cfg.FONT_DETECTION, (255, 255, 255), 2)
            cv2.putText(self.imagen_procesada, labels[1], (x1 + 5, y1 - 5), 
                       font, cfg.FONT_DETECTION, (255, 255, 255), 2)
        
        self._update_label_image(self.analysis_label, self.imagen_procesada)
    
    def clear_image(self):
        """Limpia imagen capturada y resetea estado."""
        self.hay_foto = False
        self.imagen_capturada = None
        self.imagen_procesada = None
        self.detecciones = []
        self.analysis_label.config(image='', text=cfg.MSG_NO_IMAGE, bg=cfg.COLOR_ANALYSIS_BG, fg=cfg.COLOR_TEXT_BLACK)
        self.analysis_label.image = None
        self.estado_label.config(text=cfg.MSG_NO_PHOTO, fg=cfg.COLOR_TEXT_BLACK)
        self._add_log_entry(cfg.MSG_IMAGE_CLEARED)
    
    def clear_log(self):
        """Limpia todo el historial del log de análisis."""
        self.log_text.delete(1.0, END)
        self._add_log_entry(cfg.MSG_LOG_CLEARED)
        
    # ==================== SALIR DE LA APLICACIÓN ====================
    def salir_aplicacion(self):
        """Cierra la aplicación y libera recursos."""
        self.video_running = False
        if self.cap:
            self.cap.release()
        self.root.quit()
        self.root.destroy()

    # ==================== BUCLE PRINCIPAL DE LA APLICACIÓN ====================
    def run(self):
        """Inicia el loop principal de tkinter."""
        print(cfg.MSG_STARTING)
        self.root.mainloop()