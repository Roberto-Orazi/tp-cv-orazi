#!/bin/bash

echo "=== Iniciando Etapa 2 ==="
cd etapa-2 || exit

if [ ! -f "resnet18_best.pth" ]; then
    echo "Modelo no encontrado. Iniciando entrenamiento (esto puede tardar)..."
    python entrenar_resnet18.py
else
    echo "Modelo encontrado. Saltando entrenamiento."
fi

if [ ! -f "embeddings_resnet18.pkl" ]; then
    echo "Embeddings de ResNet18 no encontrados. Generando..."
    python extraer_embeddings_resnet18.py
else
    echo "Embeddings de ResNet18 encontrados. Saltando extracción."
fi

echo "Evaluando métricas..."
python evaluar_metricas.py

echo "Lanzando aplicación Gradio con selector de modelos..."
python app_etapa2.py
