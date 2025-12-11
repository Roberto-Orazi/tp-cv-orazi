import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import json

from utils.transforms import get_val_transform
from utils.models import get_resnet18
from utils.dataset import DogDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

transform = get_val_transform()

print("Cargando modelos...")
yolo_model = YOLO("yolov8n.pt")

csv_file = "../dataset/dogs.csv"
temp_dataset = DogDataset(csv_file, "../dataset", "train", transform=transform)
num_classes = temp_dataset.num_classes

classifier = get_resnet18(num_classes, pretrained=False)
classifier.load_state_dict(
    torch.load("../etapa-2/resnet18_best.pth", map_location=device)
)
classifier = classifier.to(device)
classifier.eval()

print("\nCargando anotaciones manuales...")
annotations_file = "anotaciones_manuales.json"

if not os.path.exists(annotations_file):
    print(f"\nERROR: No se encontro {annotations_file}")
    print(
        "Por favor crea el archivo con las anotaciones manuales de 10 imagenes complejas."
    )
    print("Formato esperado:")
    print(
        """
{
  "images": [
    {
      "file": "ruta/imagen1.jpg", "annotations": [
        {
          "bbox": [x1, y1, x2, y2], "breed": "nombre_raza"
        }
      ]
    }
  ]
}
    """
    )
    sys.exit(1)

with open(annotations_file, "r") as f:
    ground_truth = json.load(f)


def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


print("\nEvaluando pipeline en imagenes anotadas manualmente...")

total_gt_boxes = 0
total_pred_boxes = 0
true_positives = 0
false_positives = 0
false_negatives = 0

iou_threshold = 0.5
ious = []
classification_correct = 0
classification_total = 0

for image_data in ground_truth["images"]:
    img_path = image_data["file"]
    gt_annotations = image_data["annotations"]

    total_gt_boxes += len(gt_annotations)

    pil_image = Image.open(img_path).convert("RGB")
    results = yolo_model(pil_image, conf=0.25, verbose=False)

    pred_boxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 16:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                dog_crop = pil_image.crop((int(x1), int(y1), int(x2), int(y2)))
                if dog_crop.width < 10 or dog_crop.height < 10:
                    continue

                img_tensor = transform(dog_crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = classifier(img_tensor)
                    _, predicted = outputs.max(1)
                    predicted_idx = predicted.item()
                    breed = temp_dataset.idx_to_label[predicted_idx]

                pred_boxes.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "breed": breed,
                        "conf": conf,
                        "matched": False,
                    }
                )

    total_pred_boxes += len(pred_boxes)

    for gt_ann in gt_annotations:
        gt_box = gt_ann["bbox"]
        gt_breed = gt_ann["breed"]
        best_iou = 0.0
        best_match = None

        for pred in pred_boxes:
            if pred["matched"]:
                continue

            iou = calculate_iou(gt_box, pred["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_match = pred

        if best_iou >= iou_threshold and best_match is not None:
            true_positives += 1
            best_match["matched"] = True
            ious.append(best_iou)

            if best_match["breed"] == gt_breed:
                classification_correct += 1
            classification_total += 1
        else:
            false_negatives += 1

    for pred in pred_boxes:
        if not pred["matched"]:
            false_positives += 1

precision = (
    true_positives / (true_positives + false_positives)
    if (true_positives + false_positives) > 0
    else 0
)
recall = (
    true_positives / (true_positives + false_negatives)
    if (true_positives + false_negatives) > 0
    else 0
)
f1_score = (
    2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
)
avg_iou = np.mean(ious) if ious else 0
classification_acc = (
    classification_correct / classification_total if classification_total > 0 else 0
)


def calculate_ap(precisions, recalls):
    precisions = np.array([0] + precisions + [0])
    recalls = np.array([0] + recalls + [1])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return ap


all_predictions = []
for image_data in ground_truth["images"]:
    img_path = image_data["file"]
    pil_image = Image.open(img_path).convert("RGB")
    results = yolo_model(pil_image, conf=0.25, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 16:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                all_predictions.append(
                    (conf, img_path, [int(x1), int(y1), int(x2), int(y2)])
                )

all_predictions.sort(key=lambda x: x[0], reverse=True)

tp_list = []
fp_list = []
for conf, img_path, pred_box in all_predictions:
    img_data = next(
        (img for img in ground_truth["images"] if img["file"] == img_path), None
    )
    if img_data is None:
        fp_list.append(1)
        tp_list.append(0)
        continue

    matched = False
    for gt_ann in img_data["annotations"]:
        iou = calculate_iou(gt_ann["bbox"], pred_box)
        if iou >= iou_threshold:
            matched = True
            break

    if matched:
        tp_list.append(1)
        fp_list.append(0)
    else:
        tp_list.append(0)
        fp_list.append(1)

tp_cumsum = np.cumsum(tp_list)
fp_cumsum = np.cumsum(fp_list)

precisions_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
recalls_curve = (
    tp_cumsum / total_gt_boxes if total_gt_boxes > 0 else np.zeros_like(tp_cumsum)
)

ap = calculate_ap(precisions_curve.tolist(), recalls_curve.tolist())

print("\n" + "=" * 80)
print("RESULTADOS DE EVALUACION DEL PIPELINE")
print("=" * 80)
print(f"Imagenes evaluadas:              {len(ground_truth['images'])}")
print(f"Total GT boxes:                  {total_gt_boxes}")
print(f"Total predicciones:              {total_pred_boxes}")
print(f"True Positives:                  {true_positives}")
print(f"False Positives:                 {false_positives}")
print(f"False Negatives:                 {false_negatives}")
print(f"\nPrecision:                       {precision:.4f}")
print(f"Recall:                          {recall:.4f}")
print(f"F1-Score:                        {f1_score:.4f}")
print(f"Average IoU:                     {avg_iou:.4f}")
print(f"mAP@0.5:                         {ap:.4f}")
print(
    f"\nClassification Accuracy:         {classification_acc:.4f} ({classification_correct}/{classification_total})"
)
print("=" * 80)

results_dict = {
    "num_images": len(ground_truth["images"]),
    "total_gt_boxes": total_gt_boxes,
    "total_predictions": total_pred_boxes,
    "true_positives": true_positives,
    "false_positives": false_positives,
    "false_negatives": false_negatives,
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1_score),
    "average_iou": float(avg_iou),
    "mAP@0.5": float(ap),
    "classification_accuracy": float(classification_acc),
}

with open("resultados_evaluacion.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("\nResultados guardados en resultados_evaluacion.json")
