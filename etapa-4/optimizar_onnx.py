import os
import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import onnxruntime as ort
import time
import numpy as np

from utils.dataset import DogDataset
from utils.models import get_resnet18
from utils.transforms import get_val_transform


print("="*80)
print("EXPORTACIÓN Y CUANTIZACIÓN CON ONNX")
print("="*80)

device = torch.device("cpu")
print(f"Usando: {device}\n")

# ==================== 1. CARGAR MODELO ====================
print("1. Cargando modelo ResNet18...")
csv_file = '../dataset/dogs.csv'
test_dataset = DogDataset(csv_file, '../dataset', 'test', transform=get_val_transform())
num_classes = test_dataset.num_classes
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = get_resnet18(num_classes, pretrained=False)
model.load_state_dict(torch.load('../etapa-2/resnet18_best.pth', map_location=device))
model = model.to(device)
model.eval()
print("   Modelo cargado\n")

# ==================== 2. EXPORTAR A ONNX ====================
print("2. Exportando modelo a ONNX...")
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# FORZAR exportador legacy (el nuevo dynamo falla)
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        'resnet18.onnx',
        export_params=True,
        opset_version=11,  # Versión más antigua para evitar conversión
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        verbose=False,
        dynamo=False  # DESACTIVAR dynamo exporter (nuevo y buggy)
    )

size_onnx = os.path.getsize('resnet18.onnx') / (1024 * 1024)
print(f"   Modelo exportado: resnet18.onnx ({size_onnx:.2f} MB)\n")

# ==================== 3. CUANTIZAR ONNX ====================
print("3. Cuantizando modelo ONNX...")

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    # Cuantizar con parámetros mínimos (compatibilidad con versiones viejas)
    quantize_dynamic(
        model_input='resnet18.onnx',
        model_output='resnet18_quantized.onnx',
        weight_type=QuantType.QInt8
    )

    size_onnx_quant = os.path.getsize('resnet18_quantized.onnx') / (1024 * 1024)
    print(f"   Modelo cuantizado: resnet18_quantized.onnx ({size_onnx_quant:.2f} MB)")
    print(f"   Reducción: {size_onnx/size_onnx_quant:.2f}x\n")

except Exception as e:
    print(f"   ERROR cuantizando: {e}")
    print("   Continuando con evaluación del modelo ONNX original...\n")
    size_onnx_quant = None

print("4. Evaluando modelos con ONNX Runtime...")

try:
    import onnxruntime as ort


    def evaluate_onnx(onnx_path, loader):
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name

        correct = 0
        total = 0
        inference_times = []

        for images, labels, _ in loader:
            images_np = images.cpu().numpy()

            start_time = time.time()
            outputs = session.run(None, {input_name: images_np})
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            predicted = np.argmax(outputs[0], axis=1)
            total += labels.size(0)
            correct += np.sum(predicted == labels.numpy())

        accuracy = 100. * correct / total
        avg_time = np.mean(inference_times) * 1000
        return accuracy, avg_time

    print("\n   Evaluando modelo ONNX original...")
    acc_onnx, time_onnx = evaluate_onnx('resnet18.onnx', test_loader)
    print(f"   Accuracy: {acc_onnx:.2f}%")
    print(f"   Tiempo: {time_onnx:.2f}ms por batch")

    acc_onnx_quant = None
    time_onnx_quant = None

    if size_onnx_quant:
        print("\n   Evaluando modelo ONNX cuantizado...")
        try:
            acc_onnx_quant, time_onnx_quant = evaluate_onnx('resnet18_quantized.onnx', test_loader)
            print(f"   Accuracy: {acc_onnx_quant:.2f}%")
            print(f"   Tiempo: {time_onnx_quant:.2f}ms por batch")
        except Exception as e:
            print(f"   ERROR ejecutando modelo cuantizado: {str(e)[:80]}...")
            print(f"   El modelo cuantizado existe ({size_onnx_quant:.2f} MB) pero no se puede")
            print(f"   ejecutar en Mac M1/M2 (falta soporte para ConvInteger en onnxruntime)")

    print("\n" + "="*80)
    print("COMPARACION DE MODELOS ONNX")
    print("="*80)

    if size_onnx_quant and acc_onnx_quant is not None:
        print(f"{'Metrica':<30} {'ONNX Original':<20} {'ONNX Cuantizado':<20} {'Diferencia':<15}")
        print("-"*80)
        print(f"{'Accuracy (%)':<30} {acc_onnx:<20.2f} {acc_onnx_quant:<20.2f} {acc_onnx_quant - acc_onnx:<15.2f}")
        print(f"{'Tiempo (ms/batch)':<30} {time_onnx:<20.2f} {time_onnx_quant:<20.2f} {time_onnx_quant - time_onnx:<15.2f}")
        print(f"{'Speedup':<30} {'-':<20} {time_onnx/time_onnx_quant:<20.2f}x {'-':<15}")
        print(f"{'Tamano (MB)':<30} {size_onnx:<20.2f} {size_onnx_quant:<20.2f} {size_onnx_quant - size_onnx:<15.2f}")
        print(f"{'Reduccion':<30} {'-':<20} {size_onnx/size_onnx_quant:<20.2f}x {'-':<15}")

        import json
        results = {
            'onnx_original': {
                'accuracy': float(acc_onnx),
                'inference_time_ms': float(time_onnx),
                'size_mb': float(size_onnx)
            },
            'onnx_quantized': {
                'accuracy': float(acc_onnx_quant),
                'inference_time_ms': float(time_onnx_quant),
                'size_mb': float(size_onnx_quant),
                'speedup': float(time_onnx / time_onnx_quant)
            }
        }

        with open('resultados_onnx.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("="*80)
        print("\nResultados guardados en resultados_onnx.json")
    elif size_onnx_quant:
        # El modelo cuantizado existe pero no se pudo evaluar
        print(f"Modelo ONNX original evaluado correctamente:")
        print(f"  Accuracy: {acc_onnx:.2f}%")
        print(f"  Tiempo: {time_onnx:.2f}ms")
        print(f"  Tamaño: {size_onnx:.2f} MB")
        print(f"\nModelo cuantizado creado ({size_onnx_quant:.2f} MB, reducción {size_onnx/size_onnx_quant:.2f}x)")
        print(f"pero no se pudo evaluar en este sistema (Mac M1/M2)")
        print("="*80)
    else:
        print(f"Solo se evaluó el modelo ONNX original")
        print(f"Accuracy: {acc_onnx:.2f}%")
        print(f"Tiempo: {time_onnx:.2f}ms")
        print("="*80)



except ImportError:
    print("\n   ERROR: onnxruntime no está instalado")
    print("   Instala con: pip install onnxruntime")
    print("\n   El modelo ONNX fue exportado pero no se pudo evaluar.")

print("Proceso completado!")
