#!/bin/bash

echo "=== Iniciando Etapa 3 ==="
cd etapa-3 || exit

echo "Evaluando pipeline completo..."
python evaluar_pipeline.py

echo "Lanzando aplicación de detección..."
python app_deteccion.py
