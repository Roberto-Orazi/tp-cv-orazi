# TP Computer Vision - Sistema de Detección y Clasificación de Razas de Perros

Trabajo Práctico Final - TUIA 2025

## Requisitos

- **Python 3.9, 3.10 o 3.11** (recomendado: 3.10)
- Mac M1/M2/M3, Windows o Linux

## Instalación

### 0. Instalar Python (si no lo tenés)

**Mac:**
```bash
brew install python@3.10
```

**Windows:**
Descargar desde [python.org](https://www.python.org/downloads/) (seleccionar versión 3.10.x)

**Linux:**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv
```

**Verificar instalación:**
```bash
python3.10 --version
```

### 1. Crear entorno virtual

```bash
python3.10 -m venv cv
source cv/bin/activate  # En Windows: cv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Descargar dataset

Descargar de Kaggle: [70 Dog Breeds Image Dataset](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set)

Colocar en carpeta `dataset/` con la siguiente estructura:
```
dataset/
├── train/
├── valid/
├── test/
└── dogs.csv
```

---

## Estructura del Proyecto

```
tp-cv-orazi/
├── utils/                              # Componentes reutilizables
│   ├── dataset.py                      # Dataset personalizado
│   ├── transforms.py                   # Transformaciones de imagen
│   ├── models.py                       # Funciones para cargar modelos
│   └── metrics.py                      # Métricas de evaluación
│
├── dataset/                            # Dataset de perros
│
├── etapa-1/                            # Buscador por similitud
│   ├── extraer_embeddings.py          # Genera embeddings con ResNet50
│   ├── evaluar_ndcg.py                # Calcula NDCG@10
│   ├── app_etapa1.py                  # App Gradio (búsqueda por similitud)
│   └── ETAPA_1_EXPLICACION.md
│
├── etapa-2/                            # Entrenamiento y comparación
│   ├── entrenar_resnet18.py           # Fine-tuning de ResNet18
│   ├── extraer_embeddings_resnet18.py # Genera embeddings con ResNet18
│   ├── evaluar_metricas.py            # Calcula métricas detalladas
│   ├── app_etapa2.py                  # App con selector de modelos
│   └── ETAPA_2_EXPLICACION.md
│
├── etapa-3/                            # Pipeline de detección
│   ├── app_deteccion.py                # YOLO + Clasificación
│   ├── evaluar_pipeline.py             # Evalúa pipeline completo
│   └── ETAPA_3_EXPLICACION.md
│
├── etapa-4/                            # Evaluación y optimización
│   ├── evaluar_pipeline.py             # Evalúa con imágenes anotadas
│   ├── optimizar_modelos.py            # Cuantización de modelos
│   ├── etiquetador_manual.py           # Herramienta de anotación manual
│   ├── anotar_automatico.py            # Genera anotaciones YOLOv5/COCO
│   ├── anotaciones_manuales_ejemplo.json
│   └── ETAPA_4_EXPLICACION.md
│
├── requirements.txt
└── README.md
```

---

## Uso

### Ejecución Rápida (Scripts)

Se han incluido scripts para automatizar la ejecución de cada etapa. Estos scripts verifican si los archivos necesarios existen y ejecutan los comandos en orden.

**Etapa 1:**
```bash
./correr_etapa_1.sh
```

**Etapa 2:**
```bash
./correr_etapa_2.sh
```

**Etapa 3:**
```bash
./correr_etapa_3.sh
```

**Etapa 4:**
```bash
./correr_etapa_4.sh
```

### Limpieza del Proyecto

Para borrar todos los archivos generados (modelos, embeddings, métricas) y comenzar de cero:

```bash
./borrar_archivos_generados.sh
```

---

### Ejecución Manual

### Etapa 1: Buscador de Imágenes por Similitud

Objetivo: Crear un sistema de búsqueda por similitud usando ResNet50 pre-entrenado.

**Pasos:**

1. **Extraer embeddings** (solo primera vez, ~10-30 min):
```bash
cd etapa-1
python extraer_embeddings.py
```
Esto genera `embeddings_data.pkl` con los vectores de características de todas las imágenes.

2. **Lanzar aplicación Gradio**:
```bash
python app_etapa1.py
```
Abre http://127.0.0.1:7860 en tu navegador.

3. **Evaluar con NDCG@10**:
```bash
python evaluar_ndcg.py
```

**Entregables Etapa 1:**
- Base de datos vectorial (FAISS)
- Aplicación Gradio funcional
- Clasificación por voto mayoritario
- Métrica NDCG@10

---

### Etapa 2: Entrenamiento y Comparación de Modelos

Objetivo: Entrenar ResNet18 con fine-tuning y comparar con el modelo pre-entrenado.

**Pasos:**

1. **Entrenar ResNet18** (~1-2 horas en CPU):
```bash
cd etapa-2
python entrenar_resnet18.py
```
Genera:
- `resnet18_best.pth` - Mejor modelo guardado
- `training_curves.png` - Gráficas de entrenamiento

2. **Extraer embeddings del modelo entrenado**:
```bash
python extraer_embeddings_resnet18.py
```
Genera `embeddings_resnet18.pkl`.

3. **Evaluar métricas**:
```bash
python evaluar_metricas.py
```
Genera:
- `metricas_detalladas.csv` - Métricas por clase
- `confusion_matrix.png` - Matriz de confusión

4. **Lanzar app con selector de modelos**:
```bash
python app_etapa2.py
```
Permite elegir entre ResNet50 (pre-entrenado) y ResNet18 (fine-tuned).

**Entregables Etapa 2:**
- Modelo ResNet18 entrenado
- Métricas: Sensibilidad, Especificidad, Precisión, Exactitud, F1-Score
- App Gradio con selector de modelos

---

### Etapa 3: Pipeline de Detección y Clasificación

Objetivo: Detectar perros con YOLO preentrenado y clasificar la raza de cada perro detectado.

**Lanzar aplicación:**

```bash
cd etapa-3
python app_deteccion.py
```
Abre http://127.0.0.1:7861

Usa YOLOv8n preentrenado (yolov8n.pt) - NO requiere entrenamiento.
El modelo detecta perros automáticamente (clase 16 en COCO).

**Evaluar pipeline completo:**
```bash
python evaluar_pipeline.py
```
Genera `metricas_pipeline.csv` con métricas del sistema completo.

**Entregables Etapa 3:**
- Detección de perros con YOLO
- Clasificación de razas en cada detección
- App Gradio con pipeline completo
- Evaluación de detección + clasificación
- Soporte para múltiples perros en una imagen

---

### Etapa 4: Evaluación, Optimización y Herramientas de Anotación

Objetivo: Evaluar pipeline con imágenes anotadas, optimizar modelos y generar anotaciones automáticas.

**Paso 1: Preparar anotaciones manuales**

1. Recopilar 10 imágenes complejas (con perros y otros objetos) en `etapa-4/imagenes/convertidas`.
2. Ejecutar la herramienta de anotación propia:
   ```bash
   python etapa-4/etiquetador_manual.py
   ```
   (Seguir las instrucciones en pantalla: 2 clicks por perro + 'n' + ID de raza).
3. Esto generará el archivo `anotaciones_manuales.json` automáticamente.

Formato:
```json
{
  "images": [
    {
      "file": "ruta/imagen.jpg",
      "annotations": [
        {"bbox": [x1, y1, x2, y2], "breed": "Beagle"}
      ]
    }
  ]
}
```

**Paso 2: Evaluar pipeline completo**

```bash
python evaluar_pipeline.py
```

Calcula:
- mAP@0.5 (Mean Average Precision)
- IoU (Intersection over Union)
- Precision, Recall, F1-Score
- Classification Accuracy

Genera `resultados_evaluacion.json`

**Opción A: Cuantización Estándar (x86/Linux/Windows)**
```bash
python optimizar_modelos.py
```
- Aplica cuantización completa (Linear + Conv2d).
- Ideal para servidores o PCs con procesadores Intel/AMD.
- Logra reducción de tamaño (~75%) y mejora de velocidad.

**Opción B: Cuantización Dinámica (Mac M1/M2)**
```bash
python optimizar_dinamico.py
```
- Script adaptado específicamente para Apple Silicon.
- Cuantiza solo capas lineales para evitar errores de backend `qnnpack`.
- Mejora velocidad (~1.17x) priorizando latencia sobre tamaño.

**Opción C: Exportación y Optimización ONNX**
```bash
python optimizar_onnx.py
```
- Exporta el modelo a formato ONNX estándar.
- Permite uso con ONNX Runtime (más rápido, portable).
- Genera reporte detallado de compatibilidad.

**Paso 4: Generar anotaciones automáticas**

```bash
python anotar_automatico.py
```

Procesa una carpeta de imágenes y genera:
- **YOLOv5 (.txt)**: `anotaciones_yolo/*.txt` - Formato YOLO
- **COCO (.json)**: `anotaciones_coco.json` - Formato COCO

**Entregables Etapa 4:**
- Evaluación con 10 imágenes anotadas manualmente
- Métricas: mAP, IoU, Precision, Recall, F1-Score
- Optimización por cuantización (o ONNX)
- Comparación velocidad/precisión
- Script de anotación automática
- Exportación a formatos YOLOv5 y COCO

---

## Solución de Problemas

### Error: "KMP_DUPLICATE_LIB_OK"
Ya está resuelto en el código. Si persiste:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Error: "Module not found"
Asegúrate de estar en el entorno virtual:
```bash
source cv/bin/activate
```

### Error: "embeddings_data.pkl not found"
Ejecuta primero `python extraer_embeddings.py` en etapa-1.

### Error: "resnet18_best.pth not found"
Ejecuta primero `python entrenar_resnet18.py` en etapa-2.

---

## Fechas de Entrega

- **Etapa 1 y 2**: Jueves 27 de Noviembre
- **Etapa 3 y 4**: Jueves 11 de Diciembre

---

## Autor

Roberto Orazi - TUIA 2025

