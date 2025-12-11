#!/bin/bash

echo "=== Iniciando Etapa 4 ==="
cd etapa-4 || exit

echo "Evaluando pipeline con anotaciones..."
python evaluar_pipeline.py

echo "Ejecutando optimización de modelos..."
python optimizar_modelos.py

echo "Generando anotaciones automáticas..."
python anotar_automatico.py

echo "Etapa 4 completada."
