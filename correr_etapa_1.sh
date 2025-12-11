#!/bin/bash

echo "=== Iniciando Etapa 1 ==="
cd etapa-1 || exit

if [ ! -f "embeddings_data.pkl" ]; then
    echo "Embeddings no encontrados. Generando..."
    python extraer_embeddings.py
else
    echo "Embeddings encontrados. Saltando extracción."
fi

echo "Evaluando NDCG..."
python evaluar_ndcg.py

echo "Lanzando aplicación Gradio..."
python app_etapa1.py
