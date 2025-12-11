import os
import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
import platform

from utils.dataset import DogDataset
from utils.models import get_resnet18
from utils.transforms import get_val_transform

# ==================== CONFIGURAR BACKEND PARA MAC M1/M2 ====================
arch = platform.machine().lower()
print(f"Detectada arquitectura: {arch}")

if 'arm' in arch or 'aarch64' in arch:
    # Mac M1/M2 o ARM
    torch.backends.quantized.engine = 'qnnpack'
    print(f"Usando motor de cuantizacion: qnnpack (para ARM)")
else:
    # Intel/AMD
    torch.backends.quantized.engine = 'fbgemm'
    print(f"Usando motor de cuantizacion: fbgemm (para x86)")

device = torch.device("cpu")
print(f"Usando CPU para cuantizacion dinamica\n")

print("Cargando modelo ResNet18 original...")
csv_file = '../dataset/dogs.csv'
test_dataset = DogDataset(csv_file, '../dataset', 'test', transform=get_val_transform())
num_classes = test_dataset.num_classes

model_original = get_resnet18(num_classes, pretrained=False)
model_original.load_state_dict(torch.load('../etapa-2/resnet18_best.pth', map_location=device))
model_original = model_original.to(device)
model_original.eval()

print("Modelo original cargado correctamente\n")

# ==================== CUANTIZACIÓN DINÁMICA ====================
print("="*80)
print("CUANTIZACION DINAMICA")
print("="*80)
print("Aplicando cuantización dinámica...")
print("(Esto cuantiza pesos offline, activaciones en runtime)")

# En Mac M1/M2 (ARM), solo cuantizamos Linear layers
# Conv2d tiene problemas con qnnpack
if 'arm' in arch or 'aarch64' in arch:
    print("Nota: En Mac M1/M2 solo se cuantizan capas Linear (no Conv2d)")
    qconfig_spec = {nn.Linear}
else:
    qconfig_spec = {nn.Linear, nn.Conv2d}

# Cuantización dinámica
model_dynamic = torch.quantization.quantize_dynamic(
    model_original,                      # modelo a cuantizar
    qconfig_spec,                        # capas a cuantizar
    dtype=torch.qint8                    # tipo de dato
)

print("Cuantización dinámica completada\n")

# Guardar modelo
torch.save(model_dynamic.state_dict(), 'resnet18_dynamic_quantized.pth')
print("Modelo guardado en: resnet18_dynamic_quantized.pth\n")

print("="*80)
print("EVALUANDO MODELOS")
print("="*80)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

def evaluate_model(model, loader, device):
    correct = 0
    total = 0
    inference_times = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    avg_inference_time = np.mean(inference_times) * 1000
    return accuracy, avg_inference_time

print("\nEvaluando modelo ORIGINAL...")
acc_original, time_original = evaluate_model(model_original, test_loader, device)
print(f"  Accuracy: {acc_original:.2f}%")
print(f"  Tiempo promedio: {time_original:.2f}ms por batch")

print("\nEvaluando modelo CUANTIZADO DINÁMICO...")
acc_dynamic, time_dynamic = evaluate_model(model_dynamic, test_loader, device)
print(f"  Accuracy: {acc_dynamic:.2f}%")
print(f"  Tiempo promedio: {time_dynamic:.2f}ms por batch")

# ==================== COMPARACIÓN ====================
print("\n" + "="*80)
print("COMPARACION DE MODELOS")
print("="*80)
print(f"{'Metrica':<30} {'Original':<20} {'Dinamico':<20} {'Diferencia':<15}")
print("-"*80)
print(f"{'Accuracy (%)':<30} {acc_original:<20.2f} {acc_dynamic:<20.2f} {acc_dynamic - acc_original:<15.2f}")
print(f"{'Tiempo (ms/batch)':<30} {time_original:<20.2f} {time_dynamic:<20.2f} {time_dynamic - time_original:<15.2f}")
print(f"{'Speedup':<30} {'-':<20} {time_original/time_dynamic:<20.2f}x {'-':<15}")

# Tamaños de archivo
torch.save(model_original.state_dict(), '/tmp/resnet18_original.pth')
torch.save(model_dynamic.state_dict(), '/tmp/resnet18_dynamic.pth')

size_original = os.path.getsize('/tmp/resnet18_original.pth') / (1024 * 1024)
size_dynamic = os.path.getsize('/tmp/resnet18_dynamic.pth') / (1024 * 1024)

print(f"{'Tamano (MB)':<30} {size_original:<20.2f} {size_dynamic:<20.2f} {size_dynamic - size_original:<15.2f}")
print(f"{'Reduccion':<30} {'-':<20} {size_original/size_dynamic:<20.2f}x {'-':<15}")
print("="*80)

# Guardar resultados
import json
results = {
    'original': {
        'accuracy': float(acc_original),
        'inference_time_ms': float(time_original),
        'size_mb': float(size_original)
    },
    'dynamic_quantized': {
        'accuracy': float(acc_dynamic),
        'inference_time_ms': float(time_dynamic),
        'size_mb': float(size_dynamic),
        'speedup': float(time_original / time_dynamic)
    }
}

with open('resultados_cuantizacion_dinamica.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResultados guardados en resultados_cuantizacion_dinamica.json")


