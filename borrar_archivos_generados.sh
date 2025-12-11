#!/bin/bash

# Script para limpiar archivos generados y reiniciar el proyecto

echo "=== LIMPIANDO PROYECTO ==="
echo "Advertencia: Esto borrara modelos entrenados y embeddings."
echo "Tendras que re-entrenar (aprox 1-2hs en CPU) y re-generar todo."
echo ""
read -p "¿Estas seguro? (y/n): " confirm
if [[ $confirm != "y" ]]; then
    echo "Cancelado."
    exit 1
fi

echo ""
echo "Borrando archivos..."

# Etapa 1
rm -v etapa-1/embeddings_data.pkl 2>/dev/null

# Etapa 2
rm -v etapa-2/resnet18_best.pth 2>/dev/null
rm -v etapa-2/embeddings_resnet18.pkl 2>/dev/null
rm -v etapa-2/metricas_detalladas.csv 2>/dev/null
rm -v etapa-2/*.png 2>/dev/null

# Etapa 3
rm -v etapa-3/yolov8n.pt 2>/dev/null

# Etapa 4
rm -v etapa-4/resnet18_quantized.pth 2>/dev/null
rm -v etapa-4/resultados_*.json 2>/dev/null
rm -v etapa-4/anotaciones_coco.json 2>/dev/null
rm -rv etapa-4/anotaciones_yolo 2>/dev/null
rm -v etapa-4/yolov8n.pt 2>/dev/null

echo ""
echo "=== LIMPIEZA COMPLETADA ==="
echo "Ahora puedes correr ./correr_etapa_1.sh para empezar de cero."
