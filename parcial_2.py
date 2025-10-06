import cv2 as cv
import numpy as np
import os

# --- Limpieza de pantalla ---
if os.name == "nt":  # Windows
    os.system("cls")
else:  # macOS/Linux
    os.system("clear")

# --- Configuración del modelo YOLO ---
# Asegúrate de que estos archivos estén en el mismo directorio que el script
weights_path = "yolov3-wider_16000.weights"
config_path = "yolov3-face.cfg"

# --- Comprobación de archivos ---
if not os.path.exists(weights_path) or not os.path.exists(config_path):
    print("Error: No se encontraron los archivos del modelo YOLO.")
    print(f"Asegúrate de tener '{weights_path}' y '{config_path}' en tu directorio.")
    print("Puedes descargarlos desde la web (busca 'YOLOv3 face model weights and cfg').")
    exit()

# Cargar la red neuronal YOLO
net = cv.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Obtener los nombres de las capas de salida
output_layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]

# --- Configuración de la cámara ---
cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Error: No se pudo abrir la cámara web.")
    exit()

print("Iniciando detección de rostros... Presiona 'ESC' para salir.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: No se pudo leer el fotograma.")
        break

    height, width, _ = frame.shape

    # --- Detección con YOLO ---
    # Crear un blob a partir de la imagen para la red neuronal
    blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Realizar la detección
    detections = net.forward(output_layers)

    # --- Procesar detecciones ---
    boxes = []
    confidences = []
    
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # En los modelos de rostros, solo hay una clase, pero filtramos por confianza
            if confidence > 0.5:
                # Escalar las coordenadas del cuadro delimitador al tamaño de la imagen
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Coordenadas de la esquina superior izquierda
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Aplicar Non-Max Suppression para eliminar cuadros superpuestos
    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            # Dibujar el rectángulo y el texto de confianza
            color = (0, 255, 0) # Verde
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"Rostro: {confidences[i]:.2f}"
            cv.putText(frame, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- Mostrar el resultado ---
    cv.imshow('Detección de Rostros con YOLO', frame)

    if cv.waitKey(1) & 0xFF == 27:  # Tecla 'ESC'
        break

# --- Limpieza final ---
cam.release()
cv.destroyAllWindows()
print("Cámara y ventanas cerradas.")