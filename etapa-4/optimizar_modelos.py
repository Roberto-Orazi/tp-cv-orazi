import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.quantization
from torch.utils.data import DataLoader
import time
import numpy as np

from utils.dataset import DogDataset
from utils.transforms import get_val_transform
from utils.models import get_resnet18
import platform

machine = platform.machine().lower()
if "arm" in machine or "aarch64" in machine:
    engine = "qnnpack"
else:
    engine = "fbgemm"

print(f"Detectada arquitectura: {machine}")
print(f"Usando motor de cuantizacion: {engine}")
torch.backends.quantized.engine = engine

device = torch.device("cpu")
print(f"Usando CPU para cuantizacion")

print("\nCargando modelo ResNet18 original...")
csv_file = "../dataset/dogs.csv"
test_dataset = DogDataset(csv_file, "../dataset", "test", transform=get_val_transform())
num_classes = test_dataset.num_classes

model_original = get_resnet18(num_classes, pretrained=False)
model_original.load_state_dict(
    torch.load("../etapa-2/resnet18_best.pth", map_location=device)
)
model_original = model_original.to(device)
model_original.eval()

print("Creando modelo cuantizado...")
model_quantized = get_resnet18(num_classes, pretrained=False)
model_quantized.load_state_dict(
    torch.load("../etapa-2/resnet18_best.pth", map_location=device)
)
model_quantized = model_quantized.to(device)
model_quantized.eval()

model_quantized.qconfig = torch.quantization.get_default_qconfig("x86")
model_quantized_prepared = torch.quantization.prepare(model_quantized, inplace=False)

print("Calibrando modelo con datos de validacion...")
val_dataset = DogDataset(csv_file, "../dataset", "valid", transform=get_val_transform())
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

with torch.no_grad():
    for i, (images, labels, _) in enumerate(val_loader):
        if i >= 10:
            break
        images = images.to(device)
        model_quantized_prepared(images)

print("Convirtiendo a modelo cuantizado...")
model_quantized_final = torch.quantization.convert(
    model_quantized_prepared, inplace=False
)

print("Guardando modelo cuantizado...")
torch.save(model_quantized_final.state_dict(), "resnet18_quantized.pth")
print("Modelo cuantizado guardado en resnet18_quantized.pth")

print("\nEvaluando modelo original...")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


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

    accuracy = 100.0 * correct / total
    avg_inference_time = np.mean(inference_times) * 1000
    return accuracy, avg_inference_time


acc_original, time_original = evaluate_model(model_original, test_loader, device)
print(f"Accuracy: {acc_original:.2f}%")
print(f"Tiempo promedio de inferencia: {time_original:.2f}ms por batch")

print("\nEvaluando modelo cuantizado...")
try:
    acc_quantized, time_quantized = evaluate_model(
        model_quantized_final, test_loader, device
    )
    print(f"Accuracy cuantizado: {acc_quantized:.2f}%")
    print(
        f"Tiempo promedio de inferencia cuantizado: {time_quantized:.2f}ms por batch"
    )

    if time_quantized > 0:
        speedup = time_original / time_quantized
        print(f"\nSpeedup: {speedup:.2f}x")
    else:
        print("\nSpeedup: N/A")

except NotImplementedError as e:
    print(f"\nADVERTENCIA: No se pudo evaluar el modelo cuantizado en este dispositivo.")
    print(f"Error de backend (comun en Mac M1/M2): {e}")
    print("Se omite el calculo de Speedup.")
    acc_quantized = 0.0
    time_quantized = 0.0
    speedup = 0.0
except Exception as e:
    print(f"\nError inesperado evaluando modelo cuantizado: {e}")
    acc_quantized = 0.0
    time_quantized = 0.0
    speedup = 0.0

print("\n" + "=" * 80)
print("COMPARACION DE MODELOS")
print("=" * 80)
print(f"{'Metrica':<30} {'Original':<20} {'Cuantizado':<20} {'Diferencia':<15}")
print("-" * 80)
print(
    f"{'Accuracy (%)':<30} {acc_original:<20.2f} {acc_quantized:<20.2f} {acc_quantized - acc_original:<15.2f}"
)

if time_quantized > 0:
    print(
        f"{'Tiempo (ms/batch)':<30} {time_original:<20.2f} {time_quantized:<20.2f} {time_quantized - time_original:<15.2f}"
    )
    print(
        f"{'Speedup':<30} {'-':<20} {time_original/time_quantized:<20.2f}x {'-':<15}"
    )
    speedup_val = float(time_original / time_quantized)
else:
    print(
        f"{'Tiempo (ms/batch)':<30} {time_original:<20.2f} {'N/A (Error)':<20} {'-':<15}"
    )
    print(f"{'Speedup':<30} {'-':<20} {'N/A':<20} {'-':<15}")
    speedup_val = -1.0

size_original = os.path.getsize("../etapa-2/resnet18_best.pth") / (1024 * 1024)
size_quantized = os.path.getsize("resnet18_quantized.pth") / (1024 * 1024)

print(
    f"{'Tamano (MB)':<30} {size_original:<20.2f} {size_quantized:<20.2f} {size_quantized - size_original:<15.2f}"
)
print("=" * 80)

results = {
    "original": {
        "accuracy": float(acc_original),
        "inference_time_ms": float(time_original),
        "size_mb": float(size_original),
    },
    "quantized": {
        "accuracy": float(acc_quantized),
        "inference_time_ms": float(time_quantized),
        "size_mb": float(size_quantized),
        "speedup": speedup_val,
    },
}

import json

with open("resultados_optimizacion.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResultados guardados en resultados_optimizacion.json")

