"""
Detector de Tapabocas con YOLO + OpenCV
Autor: Sistema de IA - 2025
"""

import tkinter as tk
from tkinter import Label, Button, Frame, Text, Scrollbar, END
import cv2
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO
from datetime import datetime


class DetectorTapabocas:
    """Detector de tapabocas con YOLOv8 y análisis de características faciales."""
    
    # Constantes de configuración
    PANEL_SIZE = (406, 336)  # Reducido 30% (antes 580x480)
    VIDEO_FPS = 30
    YOLO_CONF = 0.6
    IOU_THRESHOLD = 0.3
    MIN_AREA = 2000
    
    def __init__(self):
        """Inicializa ventana, modelo YOLO, GUI y cámara."""
        self.root = tk.Tk()
        self.root.title("Detector de Tapabocas - YOLO + OpenCV")
        self.root.geometry("950x800")
        self.root.resizable(False, False)
        
        # Estado de la aplicación
        self.hay_foto = False
        self.imagen_capturada = None
        self.imagen_procesada = None
        self.cap = None
        self.video_running = False
        self.current_frame = None
        self.detecciones = []
        
        # Cargar modelo YOLO
        print("Cargando modelo YOLOv8...")
        self.model = self._init_yolo_model()
        
        # Inicializar cascadas de Haar para detección de rostros
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        self.setup_gui()
        self.init_camera()
        
    def _init_yolo_model(self):
        """Carga el modelo YOLOv8 para detección de personas."""
        try:
            model = YOLO('yolov8n.pt')
            print("Modelo YOLOv8 cargado correctamente")
            return model
        except Exception as e:
            print(f"Error al cargar modelo YOLO: {e}")
            return None
    
    # ==================== MÉTODOS AUXILIARES DE IMÁGENES ====================
    
    def _resize_frame(self, frame, max_size=None):
        """Redimensiona frame para ajustar al panel si es necesario."""
        if max_size is None:
            max_size = self.PANEL_SIZE
        height, width = frame.shape[:2]
        max_width, max_height = max_size
        
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_size = (int(width * scale), int(height * scale))
            return cv2.resize(frame, new_size)
        return frame
    
    def _frame_to_photoimage(self, frame):
        """Convierte frame BGR a PhotoImage para tkinter."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = self._resize_frame(frame_rgb)
        image_pil = Image.fromarray(frame_resized)
        return ImageTk.PhotoImage(image_pil)
    
    def _update_label_image(self, label, frame):
        """Actualiza un label con una nueva imagen."""
        try:
            photo = self._frame_to_photoimage(frame)
            label.config(image=photo, text="", bg="lightgray")
            label.image = photo
        except:
            pass
    
    # ==================== CONFIGURACIÓN DE GUI ====================
    
    def setup_gui(self):
        """Crea la interfaz gráfica con paneles de video y controles."""
        main_frame = Frame(self.root, bg="SystemButtonFace")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        Label(main_frame, text="Detector de Tapabocas con IA", 
              font=("Arial", 18, "bold"), bg="SystemButtonFace", fg="black").pack(pady=10)
        
        video_frame = Frame(main_frame, bg="SystemButtonFace")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Panel izquierdo - Video en tiempo real
        left_panel = self._create_panel(video_frame, "Cámara en Tiempo Real", tk.LEFT)
        self.video_label = Label(left_panel, bg="black", text="Inicializando cámara...", 
                                fg="white", font=("Arial", 14))
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel derecho - Imagen capturada
        right_panel = self._create_panel(video_frame, "Imagen Capturada - Análisis", tk.RIGHT)
        self.analysis_label = Label(right_panel, bg="gray", text="Sin imagen capturada", 
                                    fg="black", font=("Arial", 12))
        self.analysis_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Log de análisis con scroll
        log_frame = Frame(main_frame, bg="SystemButtonFace")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        Label(log_frame, text="Historial de Análisis", font=("Arial", 12, "bold"), 
              bg="SystemButtonFace", fg="black").pack(pady=5)
        
        log_container = Frame(log_frame, relief="solid", bd=1)
        log_container.pack(fill=tk.BOTH, expand=True, padx=10)
        
        scrollbar = Scrollbar(log_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = Text(log_container, height=8, wrap=tk.WORD, 
                            yscrollcommand=scrollbar.set, font=("Courier", 9),
                            bg="#f0f0f0", fg="black")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)
        
        # Mensaje inicial en el log
        self._add_log_entry("Sistema iniciado. Esperando primera captura...")
        
        # Controles
        control_frame = Frame(main_frame, bg="SystemButtonFace")
        control_frame.pack(pady=15)
        
        self.estado_label = Label(control_frame, text="Estado: Sin foto - Esperando captura", 
                                 font=("Arial", 11), bg="SystemButtonFace", fg="black")
        self.estado_label.pack(pady=8)
        
        button_frame = Frame(control_frame, bg="SystemButtonFace")
        button_frame.pack()
        
        # Botones de control
        self._create_button(button_frame, "Capturar Imagen", self.capture_image, "green")
        self._create_button(button_frame, "Limpiar Imagen", self.clear_image, "orange")
        self._create_button(button_frame, "Limpiar Log", self.clear_log, "cyan")
        self._create_button(button_frame, "Salir", self.salir_aplicacion, "red")
    
    def _create_panel(self, parent, title, side):
        """Crea un panel con título para video o análisis."""
        panel = Frame(parent, bg="lightgray", relief="solid", bd=1, 
                     width=self.PANEL_SIZE[0], height=self.PANEL_SIZE[1])
        panel.pack(side=side, fill=None, expand=True, padx=(0, 5) if side == tk.LEFT else (5, 0))
        panel.pack_propagate(False)
        Label(panel, text=title, font=("Arial", 10, "bold"), bg="lightgray", fg="black").pack(pady=5)
        return panel
    
    def _create_button(self, parent, text, command, bg_color):
        """Crea un botón estilizado."""
        Button(parent, text=text, command=command, font=("Arial", 12), bg=bg_color, 
               fg="black", width=15, height=2, highlightbackground="lightgray",
               highlightthickness=1, bd=1, relief="solid").pack(side=tk.LEFT, padx=10)
    
    def _add_log_entry(self, message, analysis_data=None):
        """Agrega una entrada al log con timestamp."""
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
        
        self.log_text.insert(END, "\n")
        self.log_text.see(END)  # Auto-scroll al final
        
    # ==================== CAPTURA Y PROCESAMIENTO ====================
    
    def capture_image(self):
        """Captura el frame actual y lo procesa."""
        if self.current_frame is None:
            self.estado_label.config(text="Error: No hay frame disponible", fg="black")
            return
        
        try:
            self.imagen_capturada = self.current_frame.copy()
            self.hay_foto = True
            self._update_label_image(self.analysis_label, self.imagen_capturada)
            self.estado_label.config(text="Estado: Foto capturada - Procesando...", fg="black")
            self.root.after(100, self.process_captured_image)
        except:
            self.estado_label.config(text="Error al capturar imagen", fg="black")
    
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
                faces = cascade.detectMultiScale(gray_roi, scaleFactor=1.1, 
                                                 minNeighbors=3, minSize=(40, 40))
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
            if not any(self.calculate_iou(box, accepted) > 0.3 for accepted in filtered):
                filtered.append(box)
        return filtered
    
    def estimate_face_region(self, x1, y1, x2, y2):
        """Estima región del rostro (25% superior, 80% ancho centrado)."""
        height, width = y2 - y1, x2 - x1
        face_width = int(width * 0.8)
        face_x_offset = int((width - face_width) / 2)
        return (x1 + face_x_offset, y1, x1 + face_x_offset + face_width, y1 + int(height * 0.25))
    
    def process_captured_image(self):
        """Procesa imagen con YOLO y clasifica tapabocas."""
        if self.imagen_capturada is None or self.model is None:
            self.estado_label.config(text="Error: No hay imagen o modelo", fg="black")
            return
        
        try:
            results = self.model(self.imagen_capturada, conf=self.YOLO_CONF, verbose=False)
            self.imagen_procesada = self.imagen_capturada.copy()
            self.detecciones = []
            
            # Extraer detecciones de personas
            detections_raw = []
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:  # Clase persona en COCO
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
                        'confidence': det['confidence'] * (1.0 if faces else 0.7),
                        'tiene_tapabocas': tiene_tapabocas,
                        'metrics': metrics
                    })
            
            self.draw_detections()
            self._update_estado_label()
            self._print_analysis_summary()
            
        except:
            self.estado_label.config(text="Error al procesar imagen", fg="black")
    
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
            self.estado_label.config(text=texto, fg="black")
            
            # Agregar al log de análisis
            analysis_data = {
                'num_personas': num_personas,
                'con_tapabocas': con,
                'sin_tapabocas': sin,
                'no_detectado': no_det
            }
            self._add_log_entry("✅ Análisis completado exitosamente", analysis_data)
        else:
            self.estado_label.config(text="Estado: No se detectaron personas", fg="black")
            self._add_log_entry("⚠️ No se detectaron personas en la imagen")
    
    def _clear_console(self):
        """Limpia la consola de manera multiplataforma."""
        import os
        import platform
        if platform.system() == 'Windows':
            os.system('cls')
        else:
            os.system('clear')
    
    def _print_system_config(self):
        """Imprime configuración del sistema y parámetros."""
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║           CONFIGURACIÓN DEL SISTEMA - DETECTOR IA             ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        print(f"\n📋 PARÁMETROS DEL MODELO:")
        print(f"   • Modelo YOLO:          YOLOv8n (nano)")
        print(f"   • Umbral confianza:     {self.YOLO_CONF}")
        print(f"   • Umbral IoU:           {self.IOU_THRESHOLD}")
        print(f"   • Área mínima:          {self.MIN_AREA} píxeles")
        print(f"   • FPS video:            ~{self.VIDEO_FPS}")
        print(f"   • Resolución panel:     {self.PANEL_SIZE[0]}x{self.PANEL_SIZE[1]}")
        
        print(f"\n🎨 DETECCIÓN DE COLORES DE TAPABOCAS:")
        print(f"   • Blancos/Grises claros")
        print(f"   • Negros/Grises oscuros")
        print(f"   • Azules (quirúrgicos)")
        print(f"   • Verdes (clínicos)")
        print(f"   • Rosas/Morados")
        print(f"   • Rojos")
        print(f"   • Amarillos/Naranjas")
        
        print(f"\n🔍 CRITERIOS DE CLASIFICACIÓN:")
        print(f"   • Ratio de piel visible")
        print(f"   • Ratio de colores no-piel")
        print(f"   • Ratio de colores de tapabocas")
        print(f"   • Densidad de bordes")
        print(f"   • Uniformidad de color")
        print(f"   • Análisis de textura")
        print()
    
    def _print_analysis_summary(self):
        """Imprime resumen completo del análisis con variables y métricas."""
        self._clear_console()
        self._print_system_config()
        
        num_personas = len(self.detecciones)
        con = sum(1 for d in self.detecciones if d['tiene_tapabocas'] == 'CON TAPABOCAS')
        sin = sum(1 for d in self.detecciones if d['tiene_tapabocas'] == 'SIN TAPABOCAS')
        no_det = num_personas - con - sin
        
        print("╔════════════════════════════════════════════════════════════════╗")
        print("║              RESULTADOS DEL ANÁLISIS ACTUAL                    ║")
        print("╚════════════════════════════════════════════════════════════════╝")
        print(f"\n📊 RESUMEN GENERAL:")
        print(f"   • Total personas detectadas:  {num_personas}")
        print(f"   • ✅ Con tapabocas:            {con}")
        print(f"   • ❌ Sin tapabocas:            {sin}")
        print(f"   • ❓ No detectado:             {no_det}")
        
        if num_personas > 0:
            print(f"\n📈 MÉTRICAS DETALLADAS POR PERSONA:")
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
            if area < self.MIN_AREA or aspect_ratio < 0.3 or aspect_ratio > 2.0:
                continue
            
            # Verificar duplicados
            if not any(self.calculate_iou(det['bbox'], a['bbox']) > self.IOU_THRESHOLD 
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
            
            # Extraer región nariz/boca (50% inferior del rostro)
            mouth_region = roi[int(roi.shape[0] * 0.5):, :]
            if mouth_region.size == 0:
                return 'NO DETECTADO', {}
            
            # Calcular métricas
            gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
            
            # Detección de bordes
            edge_density = np.sum(cv2.Canny(gray, 15, 60) > 0) / gray.size
            
            # Detección de piel (rangos HSV para diversos tonos)
            skin_mask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 40, 70]), np.array([20, 255, 255])),
                cv2.inRange(hsv, np.array([20, 50, 70]), np.array([30, 255, 255]))
            )
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
            non_skin_ratio = 1 - skin_ratio
            
            # Detección UNIVERSAL de tapabocas (todos los colores posibles del mercado)
            # Incluye: blancos, azules, negros, grises, verdes, rosas, rojos, amarillos, etc.
            mask_colors_list = [
                cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255])),     # Blancos/grises claros
                cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80])),       # Negros/grises oscuros
                cv2.inRange(hsv, np.array([100, 40, 40]), np.array([130, 255, 255])),  # Azules
                cv2.inRange(hsv, np.array([40, 40, 40]), np.array([80, 255, 255])),    # Verdes
                cv2.inRange(hsv, np.array([140, 40, 40]), np.array([170, 255, 255])),  # Rosas/Morados
                cv2.inRange(hsv, np.array([0, 40, 100]), np.array([10, 255, 255])),    # Rojos
                cv2.inRange(hsv, np.array([170, 40, 100]), np.array([180, 255, 255])), # Rojos (wrap)
                cv2.inRange(hsv, np.array([20, 40, 100]), np.array([40, 255, 255])),   # Amarillos/naranjas
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
            if score >= 5:
                return 'CON TAPABOCAS', metrics
            elif score <= -3:
                return 'SIN TAPABOCAS', metrics
            else:
                return ('SIN TAPABOCAS' if skin_ratio > 0.50 else 
                       'CON TAPABOCAS' if skin_ratio < 0.05 else 'NO DETECTADO'), metrics
                
        except Exception as e:
            return 'NO DETECTADO', {}
    
    def _calculate_mask_score(self, skin_ratio, non_skin_ratio, mask_color_ratio, 
                             edge_density, color_variance, texture_std):
        """Calcula score de tapabocas basado en múltiples métricas."""
        score = 0
        
        # Criterio 1: Piel visible (principal)
        if skin_ratio < 0.03: score += 5
        elif skin_ratio < 0.08: score += 4
        elif skin_ratio < 0.15: score += 3
        elif skin_ratio < 0.25: score += 2
        elif skin_ratio < 0.35: score += 1
        elif skin_ratio > 0.60: score -= 5
        elif skin_ratio > 0.45: score -= 4
        
        # Criterio 2: Color no-piel
        if non_skin_ratio > 0.80: score += 4
        elif non_skin_ratio > 0.60: score += 3
        elif non_skin_ratio > 0.40: score += 2
        elif non_skin_ratio > 0.20: score += 1
        
        # Criterio 3: Colores de tapabocas
        if mask_color_ratio > 0.30: score += 2
        elif mask_color_ratio > 0.15: score += 1
        
        # Criterio 4: Bordes
        if edge_density > 0.20: score += 2
        elif edge_density > 0.12: score += 1
        elif edge_density < 0.05: score -= 1
        
        # Criterio 5: Uniformidad
        if color_variance < 150: score += 2
        elif color_variance < 300: score += 1
        elif color_variance > 800: score -= 1
        
        # Criterio 6: Textura
        if texture_std < 12: score += 2
        elif texture_std < 20: score += 1
        elif texture_std > 35: score -= 1
        
        return score
    
    # ==================== VISUALIZACIÓN ====================
    
    def draw_detections(self):
        """Dibuja bounding boxes con labels en la imagen procesada."""
        if self.imagen_procesada is None:
            return
        
        colors = {'CON TAPABOCAS': (0, 255, 0), 'SIN TAPABOCAS': (0, 0, 255), 
                 'NO DETECTADO': (0, 165, 255)}
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for i, det in enumerate(self.detecciones):
            x1, y1, x2, y2 = det['bbox']
            resultado = det.get('tiene_tapabocas', 'NO DETECTADO')
            color = colors.get(resultado, (0, 165, 255))
            
            # Dibujar rectángulo
            cv2.rectangle(self.imagen_procesada, (x1, y1), (x2, y2), color, 3)
            
            # Preparar labels
            labels = [f"Persona #{i+1}", resultado]
            sizes = [cv2.getTextSize(l, font, 0.6, 2)[0] for l in labels]
            max_w = max(s[0] for s in sizes)
            total_h = sum(s[1] for s in sizes) + 15
            
            # Fondo y texto
            cv2.rectangle(self.imagen_procesada, (x1, y1 - total_h - 5), 
                         (x1 + max_w + 10, y1), color, -1)
            cv2.putText(self.imagen_procesada, labels[0], (x1 + 5, y1 - sizes[1][1] - 10), 
                       font, 0.6, (255, 255, 255), 2)
            cv2.putText(self.imagen_procesada, labels[1], (x1 + 5, y1 - 5), 
                       font, 0.6, (255, 255, 255), 2)
        
        self._update_label_image(self.analysis_label, self.imagen_procesada)
    
    def clear_image(self):
        """Limpia imagen capturada y resetea estado."""
        self.hay_foto = False
        self.imagen_capturada = None
        self.imagen_procesada = None
        self.detecciones = []
        self.analysis_label.config(image='', text="Sin imagen capturada", bg="gray", fg="black")
        self.analysis_label.image = None
        self.estado_label.config(text="Estado: Sin foto - Esperando captura", fg="black")
        self._add_log_entry("🧹 Imagen limpiada. Sistema listo para nuevo análisis.")
    
    def clear_log(self):
        """Limpia todo el historial del log de análisis."""
        self.log_text.delete(1.0, END)
        self._add_log_entry("🔄 Log limpiado. Sistema reiniciado.")
        
    # ==================== CÁMARA Y VIDEO ====================
    
    def init_camera(self):
        """Inicializa cámara web y comienza captura de video."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.video_label.config(text="Error: No se pudo acceder a la cámara", fg="black")
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video_running = True
            self.update_video_feed()
        except:
            self.video_label.config(text="Error al inicializar cámara", fg="black")
    
    def update_video_feed(self):
        """Actualiza video en tiempo real (~30 FPS)."""
        if not self.video_running or not self.cap:
            return
            
        try:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = cv2.flip(frame, 1)  # Efecto espejo
                self._update_label_image(self.video_label, self.current_frame)
            else:
                self.video_label.config(text="Error leyendo frame", fg="black")
        except:
            self.video_label.config(text="Error en video feed", fg="black")
                
        if self.video_running:
            self.root.after(30, self.update_video_feed)
    
    def salir_aplicacion(self):
        """Cierra la aplicación y libera recursos."""
        print("Cerrando aplicación...")
        self.video_running = False
        if self.cap:
            self.cap.release()
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """Inicia el loop principal de tkinter."""
        print("Iniciando Detector de Tapabocas...")
        self.root.mainloop()


def main():
    """Punto de entrada de la aplicación."""
    app = DetectorTapabocas()
    app.run()


if __name__ == "__main__":
    main()

