# 📖 Manual de Usuario - Detector de Tapabocas

Guía completa paso a paso para usar el Detector de Tapabocas con IA.

---

## 📋 Tabla de Contenidos

- [Introducción](#-introducción)
- [Requisitos del Sistema](#-requisitos-del-sistema)
- [Instalación](#-instalación)
- [Inicio Rápido](#-inicio-rápido)
- [Interfaz de Usuario](#-interfaz-de-usuario)
- [Guía de Uso Detallada](#-guía-de-uso-detallada)
- [Interpretación de Resultados](#-interpretación-de-resultados)
- [Casos de Uso](#-casos-de-uso)
- [Solución de Problemas](#-solución-de-problemas)
- [Preguntas Frecuentes](#-preguntas-frecuentes)

---

## 👋 Introducción

El **Detector de Tapabocas** es una aplicación de escritorio que utiliza inteligencia artificial para detectar automáticamente si las personas están usando tapabocas. Es ideal para:

- 🏢 Control de acceso en empresas
- 🏫 Monitoreo en instituciones educativas
- 🏥 Verificación en centros de salud
- 🔬 Investigación y análisis de datos
- 📊 Generación de estadísticas de cumplimiento

---

## 💻 Requisitos del Sistema

### Hardware Mínimo
- **Procesador**: Intel i3 o equivalente
- **RAM**: 4 GB
- **Cámara web**: Cualquier cámara USB o integrada
- **Espacio en disco**: 50 MB libres

### Hardware Recomendado
- **Procesador**: Intel i5 o superior
- **RAM**: 8 GB o más
- **Cámara web**: 720p o superior
- **GPU** (opcional): Para procesamiento más rápido

### Software
- **Sistema Operativo**: 
  - Windows 10/11
  - Linux (Ubuntu 20.04+)
  - macOS 10.15+
- **Python**: 3.8 o superior
- **Drivers de cámara**: Actualizados

---

## 📦 Instalación

### Paso 1: Verificar Python

Abre una terminal o cmd y verifica la versión de Python:

```bash
python --version
```

Debe mostrar Python 3.8 o superior.

### Paso 2: Navegar a la Carpeta

```bash
cd tapaboca
```

### Paso 3: Instalar Dependencias

```bash
pip install ultralytics opencv-python numpy pillow
```

**Nota**: La instalación puede tardar 2-5 minutos dependiendo de tu conexión.

### Paso 4: Verificar Instalación

```bash
python -c "import cv2; import ultralytics; print('OK')"
```

Si muestra "OK", la instalación fue exitosa.

---

## 🚀 Inicio Rápido

### Ejecutar la Aplicación

1. Abre una terminal
2. Navega a la carpeta:
   ```bash
   cd tapaboca
   ```
3. Ejecuta:
   ```bash
   python main.py
   ```
4. La ventana de la aplicación se abrirá automáticamente

### Primera Ejecución

Al ejecutar por primera vez:
1. El modelo YOLO se cargará (puede tardar 5-10 segundos)
2. Se solicitará permiso para usar la cámara (acepta)
3. Aparecerá el video en tiempo real

---

## 🖥️ Interfaz de Usuario

### Diseño de la Ventana

```
┌─────────────────────────────────────────────────────┐
│       Detector de Tapabocas con IA                  │
├──────────────────────┬──────────────────────────────┤
│                      │                              │
│  CÁMARA EN TIEMPO    │  IMAGEN CAPTURADA           │
│      REAL            │     - ANÁLISIS              │
│                      │                              │
│  [Video continuo]    │  [Imagen con resultados]    │
│                      │                              │
├──────────────────────┴──────────────────────────────┤
│            HISTORIAL DE ANÁLISIS                    │
│  ┌──────────────────────────────────────────────┐  │
│  │ [Log con scroll]                             │  │
│  │ - Timestamp                                  │  │
│  │ - Resultados                                 │  │
│  │ - Métricas                                   │  │
│  └──────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────┤
│  Estado: Sin foto - Esperando captura              │
│                                                     │
│  [Capturar] [Limpiar] [Limpiar Log] [Salir]        │
└─────────────────────────────────────────────────────┘
```

### Componentes

#### 1. Panel de Cámara (Izquierda)
- Muestra video en **tiempo real**
- Efecto espejo para facilitar posicionamiento
- Actualización continua a 30 FPS

#### 2. Panel de Análisis (Derecha)
- Muestra la **imagen capturada**
- Dibuja bounding boxes de colores:
  - 🟢 **Verde**: Con tapabocas
  - 🔴 **Rojo**: Sin tapabocas
  - 🟠 **Naranja**: No detectado
- Etiquetas con número de persona y resultado

#### 3. Log de Análisis (Centro-Inferior)
- Historial cronológico con **timestamps**
- Resumen de cada análisis
- Tabla de métricas detalladas
- Scroll automático al último registro

#### 4. Barra de Estado
- Muestra resultado del último análisis
- Cuenta personas detectadas
- Resume clasificaciones

#### 5. Botones de Control
- **Capturar Imagen**: Congela y analiza el frame actual
- **Limpiar Imagen**: Borra imagen capturada
- **Limpiar Log**: Limpia historial
- **Salir**: Cierra la aplicación

---

## 📝 Guía de Uso Detallada

### Paso 1: Posicionamiento

1. **Abre la aplicación** (`python main.py`)
2. **Posiciónate** frente a la cámara
3. Asegúrate de tener **buena iluminación**
4. El rostro debe estar **visible** en el panel izquierdo

**Consejos:**
- ✅ Luz natural o artificial frontal
- ✅ Fondo simple y uniforme
- ✅ Distancia de 0.5 a 2 metros
- ❌ Evita contraluz
- ❌ Evita sombras fuertes en el rostro

### Paso 2: Captura de Imagen

1. Cuando estés listo, haz clic en **"Capturar Imagen"**
2. El sistema congela el frame actual
3. La imagen aparece en el panel derecho
4. El estado cambia a "Procesando..."

**Nota**: El procesamiento toma entre 0.1 y 0.5 segundos.

### Paso 3: Visualización de Resultados

Inmediatamente verás:

1. **Bounding boxes** de colores en la imagen
2. **Etiquetas** con:
   - Número de persona (#1, #2, ...)
   - Resultado (CON/SIN TAPABOCAS)
3. **Estado actualizado** con resumen
4. **Entrada en el log** con métricas

### Paso 4: Análisis de Métricas (Opcional)

En el log puedes ver:

```
======================================================================
📅 2025-10-16 15:30:45
✅ Análisis completado exitosamente
👥 Personas detectadas: 1
   ✅ Con tapabocas: 1

📊 MÉTRICAS DETALLADAS POR PERSONA:
#    Piel  No-Piel  Color  Bordes    Var   Text  Score  Resultado      
1   0.025    0.975  0.450   0.180    245   18.5      8  CON TAPABOCAS  
======================================================================
```

**Interpretación:**
- **Piel**: 0.025 = 2.5% de piel visible (muy cubierto ✅)
- **No-Piel**: 0.975 = 97.5% no es piel (tapabocas detectado ✅)
- **Color**: 0.450 = 45% colores de tapabocas (azul, blanco, etc. ✅)
- **Bordes**: 0.180 = Bordes definidos (tapabocas tiene bordes ✅)
- **Var**: 245 = Varianza baja (color uniforme ✅)
- **Text**: 18.5 = Textura uniforme ✅
- **Score**: 8 = Puntuación alta (≥5 = CON TAPABOCAS ✅)

### Paso 5: Nueva Captura

1. Para analizar otra imagen, simplemente haz clic en **"Capturar Imagen"** nuevamente
2. Puedes capturar **ilimitadas** veces
3. Cada análisis se agrega al log con timestamp

### Paso 6: Limpiar

- **Limpiar Imagen**: Borra solo la imagen capturada actual
- **Limpiar Log**: Borra todo el historial (útil para nueva sesión)

### Paso 7: Salir

- Clic en **"Salir"** o cierra la ventana
- La cámara se liberará automáticamente

---

## 🔍 Interpretación de Resultados

### Códigos de Color

| Color | Significado | Score | Interpretación |
|-------|-------------|-------|----------------|
| 🟢 **Verde** | **CON TAPABOCAS** | ≥ 5 | Tapabocas detectado correctamente |
| 🔴 **Rojo** | **SIN TAPABOCAS** | ≤ -3 | No hay tapabocas o mal colocado |
| 🟠 **Naranja** | **NO DETECTADO** | -3 a 4 | Caso ambiguo o error de detección |

### Métricas Explicadas

#### 1. **Skin Ratio (Piel)**
- **Qué es**: Porcentaje de piel visible en región nariz/boca
- **Con tapabocas**: < 0.10 (menos del 10%)
- **Sin tapabocas**: > 0.50 (más del 50%)
- **Ejemplo**: 0.025 = Solo 2.5% de piel → Muy cubierto ✅

#### 2. **Non-Skin Ratio (No-Piel)**
- **Qué es**: Porcentaje de área NO identificada como piel
- **Con tapabocas**: > 0.80 (más del 80%)
- **Sin tapabocas**: < 0.50 (menos del 50%)
- **Ejemplo**: 0.975 = 97.5% cubierto ✅

#### 3. **Mask Color Ratio (Color)**
- **Qué es**: Presencia de colores típicos de tapabocas
- **Colores detectados**: Blanco, azul, negro, verde, rosa, rojo, amarillo
- **Con tapabocas**: > 0.30
- **Sin tapabocas**: < 0.15
- **Ejemplo**: 0.450 = 45% coincide con colores de tapabocas ✅

#### 4. **Edge Density (Bordes)**
- **Qué es**: Densidad de bordes detectados (algoritmo Canny)
- **Con tapabocas**: > 0.12 (bordes definidos)
- **Sin tapabocas**: < 0.05 (piel tiene pocos bordes)
- **Ejemplo**: 0.180 = Bordes claros ✅

#### 5. **Color Variance (Varianza)**
- **Qué es**: Variación de intensidad de color
- **Con tapabocas**: < 300 (uniforme)
- **Sin tapabocas**: > 800 (piel tiene sombras y variaciones)
- **Ejemplo**: 245 = Color muy uniforme ✅

#### 6. **Texture Std (Textura)**
- **Qué es**: Desviación estándar de la textura
- **Con tapabocas**: < 20 (textura suave y uniforme)
- **Sin tapabocas**: > 35 (piel tiene poros y variaciones)
- **Ejemplo**: 18.5 = Textura lisa ✅

### Sistema de Puntuación (Score)

El score se calcula sumando puntos de los 6 criterios:

**Criterio 1: Piel visible** (-5 a +5 puntos)
- < 3% piel → +5 puntos
- < 8% piel → +4 puntos
- < 15% piel → +3 puntos
- > 60% piel → -5 puntos

**Criterio 2: No-Piel** (+1 a +4 puntos)
- > 80% no-piel → +4 puntos
- > 60% no-piel → +3 puntos

**Criterios 3-6**: Similares

**Resultado:**
- **Score ≥ 5** → CON TAPABOCAS ✅
- **Score ≤ -3** → SIN TAPABOCAS ❌
- **Score -3 a 4** → Análisis secundario

---

## 💼 Casos de Uso

### Caso 1: Control de Acceso en Oficina

**Escenario**: Verificar que empleados usen tapabocas al ingresar

**Procedimiento**:
1. Instalar laptop con cámara en entrada
2. Ejecutar aplicación
3. Empleado se posiciona frente a cámara
4. Seguridad hace clic en "Capturar"
5. Sistema muestra resultado instantáneo
6. Permitir acceso solo si hay ✅ verde

### Caso 2: Monitoreo en Aula

**Escenario**: Verificar cumplimiento de estudiantes

**Procedimiento**:
1. Profesor ejecuta aplicación
2. Captura imagen del aula
3. Sistema detecta múltiples personas
4. Genera estadística automática
5. Profesor revisa log para registro

### Caso 3: Análisis de Video Grabado

**Escenario**: Analizar cumplimiento en video existente

**Procedimiento**:
1. Reproducir video en otra ventana
2. Posicionar cámara frente a pantalla
3. Pausar en frames clave
4. Capturar y analizar
5. Generar reporte del log

### Caso 4: Investigación Académica

**Escenario**: Recopilar datos para estudio

**Procedimiento**:
1. Ejecutar aplicación
2. Capturar imágenes de participantes
3. Copiar métricas del log
4. Analizar datos en Excel/Python
5. Generar gráficas y conclusiones

---

## 🔧 Solución de Problemas

### Problema 1: No Abre la Cámara

**Síntomas**: 
- Error "No se pudo acceder a la cámara"
- Pantalla negra en panel izquierdo

**Soluciones**:
1. ✅ Verificar que cámara esté conectada
2. ✅ Cerrar otras apps que usen cámara (Zoom, Teams, etc.)
3. ✅ Reiniciar la aplicación
4. ✅ En Windows: Verificar permisos en Configuración → Privacidad → Cámara

### Problema 2: No Detecta Personas

**Síntomas**:
- Mensaje "No se detectaron personas"
- Panel derecho sin resultados

**Soluciones**:
1. ✅ Acércate más a la cámara (0.5-2 metros)
2. ✅ Mejora la iluminación
3. ✅ Asegúrate de estar en el encuadre
4. ✅ Evita fondos muy complejos

### Problema 3: Detección Incorrecta

**Síntomas**:
- Dice "CON TAPABOCAS" pero no lo lleva
- Dice "SIN TAPABOCAS" pero sí lo lleva

**Posibles Causas**:
- 🔸 Iluminación deficiente
- 🔸 Tapabocas transparente o de red
- 🔸 Tapabocas mal colocado (solo cubre boca)
- 🔸 Sombras fuertes en el rostro

**Soluciones**:
1. ✅ Mejorar iluminación frontal
2. ✅ Evitar contraluz
3. ✅ Usar tapabocas opaco estándar
4. ✅ Colocar correctamente (nariz + boca)

### Problema 4: Aplicación Lenta

**Síntomas**:
- Tarda más de 1 segundo en procesar
- Video entrecortado

**Soluciones**:
1. ✅ Cerrar otras aplicaciones
2. ✅ Reducir resolución de cámara
3. ✅ Actualizar drivers de cámara
4. ✅ Usar PC con mejores especificaciones

### Problema 5: Error al Iniciar

**Síntomas**:
- Error "No module named 'ultralytics'"
- Crash al abrir

**Soluciones**:
1. ✅ Reinstalar dependencias:
   ```bash
   pip install --upgrade ultralytics opencv-python numpy pillow
   ```
2. ✅ Verificar Python 3.8+:
   ```bash
   python --version
   ```
3. ✅ Verificar archivo `yolov8n.pt` en carpeta

---

## ❓ Preguntas Frecuentes

### ¿Funciona con múltiples personas?

✅ **Sí**. El sistema detecta y analiza hasta 10 personas simultáneamente en una sola imagen.

### ¿Funciona con tapabocas de tela?

✅ **Sí**. Detecta tapabocas de cualquier material: quirúrgicos, N95, tela, etc.

### ¿Qué colores de tapabocas detecta?

✅ **Todos**. El sistema detecta 8 rangos de colores:
- Blancos, negros, azules, verdes, rosas, rojos, amarillos y grises.

### ¿Necesita internet?

❌ **No**. El procesamiento es 100% local. No envía datos a internet.

### ¿Guarda las fotos?

❌ **No**. Las imágenes solo se procesan en memoria RAM y no se guardan en disco.

### ¿Puedo usar una foto en lugar de la cámara?

⚠️ **No directamente**. La aplicación solo captura de cámara. Pero puedes:
1. Mostrar la foto en pantalla
2. Apuntar cámara a la pantalla
3. Capturar

### ¿Funciona en la oscuridad?

❌ **No**. Requiere iluminación mínima para detectar rostros y analizar colores.

### ¿Detecta tapabocas transparentes?

⚠️ **Difícilmente**. Los tapabocas transparentes son complejos de detectar porque:
- No ocultan la piel
- No tienen color opaco
- Confunden al algoritmo

### ¿Cuál es la precisión del sistema?

📊 **Aproximadamente 85-92%** en condiciones óptimas:
- Buena iluminación
- Rostro frontal
- Tapabocas opaco estándar
- Distancia adecuada

### ¿Puedo modificar los parámetros?

✅ **Sí**. Puedes editar `config.py` para ajustar:
- Umbrales de detección
- Rangos de colores HSV
- Parámetros de scoring
- Configuración de UI

---

## 📞 Soporte y Ayuda

### Recursos

- 📖 **Documentación Técnica**: Ver `DOCUMENTACION_TECNICA.md`
- 🔍 **Análisis Detallado**: Ver `ANALISIS_DETECCION.md`
- 💻 **Código Fuente**: Revisar archivos `.py` (están comentados)

### Contacto

Para reportar problemas o sugerencias:
- Crear issue en repositorio
- Contactar al desarrollador
- Revisar la documentación

---

## ✅ Checklist de Uso

Antes de cada sesión, verifica:

- [ ] Python 3.8+ instalado
- [ ] Dependencias instaladas
- [ ] Cámara conectada y funcionando
- [ ] Buena iluminación en el lugar
- [ ] Archivo `yolov8n.pt` presente
- [ ] Espacio suficiente en RAM (>500MB libre)

---

## 🎯 Mejores Prácticas

### Para Mejores Resultados:

1. **Iluminación** ✨
   - Luz frontal o lateral suave
   - Evitar contraluz
   - Luz natural o LED blanca

2. **Distancia** 📏
   - 0.5 a 2 metros de la cámara
   - Rostro ocupando 30-70% del encuadre

3. **Posición** 👤
   - Rostro frontal o máximo 45° de ángulo
   - Tapabocas cubriendo nariz y boca
   - Sin obstrucciones (manos, objetos)

4. **Fondo** 🖼️
   - Fondo simple y uniforme
   - Evitar patrones complejos
   - Contraste con el tapabocas

5. **Captura** 📸
   - Persona quieta (no en movimiento)
   - Esperar 1 segundo antes de capturar
   - Capturar múltiples veces si hay duda

---

<p align="center">
  Manual de Usuario v1.0 - 2025<br>
  ¡Gracias por usar el Detector de Tapabocas! 🎭
</p>

