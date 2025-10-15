"""
Aplicación de Detección de Tapabocas usando YOLO y OpenCV
Autor: Sistema de IA
Descripción: Aplicación GUI que detecta tapabocas en tiempo real desde webcam
"""

import tkinter as tk
from tkinter import ttk, Label, Button, Frame
import cv2
from PIL import Image, ImageTk
import numpy as np


class DetectorTapabocas:
    """
    Clase principal para la detección de tapabocas usando YOLO y OpenCV
    """
    
    def __init__(self):
        """
        Inicializa la aplicación con GUI y configuraciones básicas
        """
        self.root = tk.Tk()
        self.root.title("Detector de Tapabocas - YOLO + OpenCV")
        self.root.geometry("1200x700")
        self.root.resizable(False, False)
        
        # Variables de estado
        self.hay_foto = False
        self.imagen_capturada = None
        self.imagen_procesada = None
        self.cap = None  # Captura de video
        self.video_running = False
        self.current_frame = None  # Frame actual para captura
        
        # Configurar la GUI
        self.setup_gui()
        # Inicializar cámara
        self.init_camera()
        
    def setup_gui(self):
        """
        Configura y crea todos los elementos de la interfaz gráfica
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
        left_panel = Frame(video_frame, bg="lightgray", relief="solid", bd=1, width=580, height=480)
        left_panel.pack(side=tk.LEFT, fill=None, expand=True, padx=(0, 5))
        left_panel.pack_propagate(False)  # Fijar tamaño exacto del Frame
        
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
        Captura una imagen desde la webcam para análisis
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
    
    def process_captured_image(self):
        """
        Procesa la imagen capturada (placeholder para detección)
        """
        # Por ahora, solo actualizamos el estado
        # La detección se implementará en las siguientes fases
        self.estado_label.config(
            text="Estado: Hay foto - Análisis pendiente (detectar rostros y tapabocas)", 
            fg="blue"
        )
        print("Imagen lista para análisis")
        
    def clear_image(self):
        """
        Limpia la imagen capturada y resetea el análisis,
        asegurando que la imagen desaparezca del panel.
        """
        self.hay_foto = False
        self.imagen_capturada = None
        self.imagen_procesada = None
        # Limpiar el panel de análisis completamente
        self.analysis_label.config(image='', text="Sin imagen capturada", bg="gray")
        self.analysis_label.image = None  # Eliminar referencia a la imagen mostrada
        self.estado_label.config(text="Estado: Sin foto - Esperando captura", fg="black")
        print("Imagen limpiada")
        
    def init_camera(self):
        """
        Inicializa la cámara web para captura de video
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
        Actualiza el feed de video en tiempo real
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
                max_width, max_height = 580, 480
                
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
        Cierra la aplicación de forma segura
        """
        print("Cerrando aplicación...")
        self.video_running = False
        
        if self.cap:
            self.cap.release()
            
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """
        Ejecuta la aplicación principal
        """
        print("Iniciando Detector de Tapabocas...")
        self.root.mainloop()


def main():
    """
    Función principal que inicia la aplicación
    """
    app = DetectorTapabocas()
    app.run()


if __name__ == "__main__":
    main()
