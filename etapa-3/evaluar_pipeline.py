import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import pandas as pd

from utils.transforms import get_val_transform
from utils.models import get_resnet18
from utils.dataset import DogDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

transform = get_val_transform()

print("Cargando modelos...")

yolo_model = YOLO("yolov8n.pt")
print("YOLO preentrenado cargado (clase 'dog' = 16 en COCO)")

csv_file = "../dataset/dogs.csv"
test_dataset = DogDataset(csv_file, "../dataset", "test", transform=transform)
num_classes = test_dataset.num_classes

classifier = get_resnet18(num_classes, pretrained=False)
classifier.load_state_dict(
    torch.load("../etapa-2/resnet18_best.pth", map_location=device)
)
classifier = classifier.to(device)
classifier.eval()

print(f"\nEvaluando pipeline en {len(test_dataset)} imagenes de test...")

total_images = 0
detected_images = 0
correct_classifications = 0
detection_confidences = []
classification_confidences = []

all_true_labels = []
all_pred_labels = []

for idx in tqdm(range(len(test_dataset))):
    img_tensor, true_label, img_path = test_dataset[idx]

    pil_image = Image.open(img_path).convert("RGB")

    results = yolo_model(pil_image, conf=0.25, verbose=False)

    dog_detected = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 16:
                dog_detected = True
                detection_confidences.append(conf)

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                dog_crop = pil_image.crop((x1, y1, x2, y2))

                if dog_crop.width < 10 or dog_crop.height < 10:
                    continue

                img_tensor_crop = transform(dog_crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = classifier(img_tensor_crop)
                    _, predicted = outputs.max(1)
                    predicted_idx = predicted.item()

                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    conf_class = probabilities[0][predicted_idx].item()
                    classification_confidences.append(conf_class)

                all_true_labels.append(true_label)
                all_pred_labels.append(predicted_idx)

                if predicted_idx == true_label:
                    correct_classifications += 1

                break

        if dog_detected:
            break

    total_images += 1
    if dog_detected:
        detected_images += 1

detection_rate = 100.0 * detected_images / total_images
classification_acc = (
    100.0 * correct_classifications / detected_images if detected_images > 0 else 0
)
avg_detection_conf = np.mean(detection_confidences) if detection_confidences else 0
avg_classification_conf = (
    np.mean(classification_confidences) if classification_confidences else 0
)

print("\n" + "=" * 80)
print("RESULTADOS DEL PIPELINE COMPLETO")
print("=" * 80)
print(f"Imagenes procesadas:           {total_images}")
print(f"Perros detectados:             {detected_images} ({detection_rate:.2f}%)")
print(f"Clasificaciones correctas:     {correct_classifications}")
print(f"Accuracy de clasificacion:     {classification_acc:.2f}%")
print(f"Confianza promedio deteccion:  {avg_detection_conf:.4f}")
print(f"Confianza promedio clasificacion: {avg_classification_conf:.4f}")
print("=" * 80)

from sklearn.metrics import classification_report, accuracy_score

if len(all_true_labels) > 0:
    unique_labels = np.unique(np.concatenate([all_true_labels, all_pred_labels]))
    target_names = [test_dataset.idx_to_label[i] for i in unique_labels]

    report = classification_report(
        all_true_labels,
        all_pred_labels,
        labels=unique_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    print("\nMETRICAS GLOBALES:")
    print(f"Accuracy:        {report['accuracy']:.4f}")
    print(f"Macro Avg F1:    {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")

    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv("metricas_pipeline.csv")
    print("\nMetricas guardadas en metricas_pipeline.csv")

print("\nEvaluacion completada!")
