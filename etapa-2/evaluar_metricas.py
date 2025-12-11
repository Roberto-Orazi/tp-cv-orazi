import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from utils.dataset import DogDataset
from utils.transforms import get_val_transform
from utils.models import get_resnet18
from utils.metrics import calculate_metrics_per_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

print("Cargando dataset de test...")
test_dataset = DogDataset(
    "../dataset/dogs.csv", "../dataset", "test", transform=get_val_transform()
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

num_classes = test_dataset.num_classes
print(f"Numero de clases: {num_classes}")

print("\nCargando modelo ResNet18...")
model = get_resnet18(num_classes, pretrained=False)
model.load_state_dict(torch.load("resnet18_best.pth"))
model = model.to(device)
model.eval()

print("\nRealizando predicciones en conjunto de test...")
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = 100.0 * np.sum(all_preds == all_labels) / len(all_labels)
print(f"\nAccuracy en test: {accuracy:.2f}%")

print("\nCalculando metricas detalladas...")

unique_labels = np.unique(np.concatenate([all_labels, all_preds]))
target_names = [test_dataset.idx_to_label[i] for i in unique_labels]

report = classification_report(
    all_labels,
    all_preds,
    labels=unique_labels,
    target_names=target_names,
    output_dict=True,
    zero_division=0,
)

print("\n" + "=" * 80)
print("METRICAS POR CLASE (primeras 10 razas)")
print("=" * 80)

metrics_df = pd.DataFrame(report).transpose()
print(metrics_df.head(10).to_string())

print("\n" + "=" * 80)
print("METRICAS GLOBALES")
print("=" * 80)
print(f"Accuracy:        {report['accuracy']:.4f}")
print(f"Macro Avg:")
print(f"  Precision:     {report['macro avg']['precision']:.4f}")
print(f"  Recall:        {report['macro avg']['recall']:.4f}")
print(f"  F1-Score:      {report['macro avg']['f1-score']:.4f}")
print(f"Weighted Avg:")
print(f"  Precision:     {report['weighted avg']['precision']:.4f}")
print(f"  Recall:        {report['weighted avg']['recall']:.4f}")
print(f"  F1-Score:      {report['weighted avg']['f1-score']:.4f}")
print("=" * 80)

metrics_df.to_csv("metricas_detalladas.csv")
print("\nMetricas guardadas en metricas_detalladas.csv")

print("\nGenerando matriz de confusion...")
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(20, 20))
sns.heatmap(
    cm,
    annot=False,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
)
plt.xlabel("Prediccion")
plt.ylabel("Real")
plt.title("Matriz de Confusion - ResNet18")
plt.xticks(rotation=90, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Matriz de confusion guardada en confusion_matrix.png")

print("\nCalculando metricas detalladas por clase...")

for i, class_idx in enumerate(unique_labels[:5]):
    class_name = test_dataset.idx_to_label[class_idx]
    metrics = calculate_metrics_per_class(all_labels, all_preds, class_idx)

    print(f"\n{class_name}:")
    print(
        f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}"
    )
    print(f"  Sensibilidad (Recall):    {metrics['sensibilidad']:.4f}")
    print(f"  Especificidad:            {metrics['especificidad']:.4f}")
    print(f"  Precision:                {metrics['precision']:.4f}")
    print(f"  Exactitud (Accuracy):     {metrics['exactitud']:.4f}")
    print(f"  F1-Score:                 {metrics['f1']:.4f}")

print("\nEvaluacion completada!")
