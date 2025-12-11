#!/bin/bash

echo "=================================="
echo "PROBANDO MÉTODOS DE CUANTIZACIÓN"
echo "=================================="

cd "$(dirname "$0")"

echo ""
echo "=== OPCIÓN 1: CUANTIZACIÓN DINÁMICA ==="
echo "(Funciona en Mac M1/M2, más simple)"
echo ""
read -p "¿Ejecutar cuantización dinámica? (s/n): " run_dynamic

if [ "$run_dynamic" = "s" ]; then
    echo ""
    echo "Ejecutando cuantización dinámica..."
    python optimizar_dinamico.py
    echo ""
    echo "✓ Resultados en: resultados_cuantizacion_dinamica.json"
    echo "✓ Modelo guardado en: resnet18_dynamic_quantized.pth"
fi

echo ""
echo "=== OPCIÓN 2: ONNX ==="
echo "(Más portable, funciona en todos lados)"
echo ""
read -p "¿Ejecutar exportación a ONNX? (s/n): " run_onnx

if [ "$run_onnx" = "s" ]; then
    # Verificar si onnxruntime está instalado
    if ! python -c "import onnxruntime" 2>/dev/null; then
        echo ""
        echo "⚠️  onnxruntime no está instalado"
        read -p "¿Instalar onnxruntime? (s/n): " install_onnx

        if [ "$install_onnx" = "s" ]; then
            echo ""
            echo "Instalando onnxruntime..."
            pip install onnxruntime
        fi
    fi

    echo ""
    echo "Ejecutando exportación y cuantización ONNX..."
    python optimizar_onnx.py
    echo ""
    echo "✓ Resultados en: resultados_onnx.json"
    echo "✓ Modelos guardados:"
    echo "  - resnet18.onnx (original)"
    echo "  - resnet18_quantized.onnx (cuantizado)"
fi

echo ""
echo "=================================="
echo "✓ Proceso completado!"
echo "=================================="
echo ""
echo "COMPARACIÓN:"
echo "- Cuantización Dinámica: Más simple, funciona en Mac M1/M2"
echo "- ONNX: Más portable, mejor para producción"
echo ""
