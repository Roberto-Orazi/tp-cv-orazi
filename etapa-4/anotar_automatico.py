import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime

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

input_folder = input("Ingresa la ruta de la carpeta con imagenes: ").strip()
input_folder = Path(input_folder)

if not input_folder.exists():
    print(f"ERROR: La carpeta {input_folder} no existe")
    sys.exit(1)

image_files = (
    list(input_folder.glob("*.jpg"))
    + list(input_folder.glob("*.jpeg"))
    + list(input_folder.glob("*.png"))
)

if len(image_files) == 0:
    print("ERROR: No se encontraron imagenes en la carpeta")
    sys.exit(1)

print(f"\nEncontradas {len(image_files)} imagenes")

output_yolo = Path("anotaciones_yolo")
output_yolo.mkdir(exist_ok=True)

coco_output = {
    "info": {
        "description": "Anotaciones automaticas de perros",
        "date_created": datetime.now().isoformat(),
    },
    "images": [],
    "annotations": [],
    "categories": [],
}

breed_to_category_id = {}
category_id_counter = 1

for breed in temp_dataset.label_to_idx.keys():
    coco_output["categories"].append(
        {"id": category_id_counter, "name": breed, "supercategory": "dog"}
    )
    breed_to_category_id[breed] = category_id_counter
    category_id_counter += 1

annotation_id = 1

print("\nProcesando imagenes...")

for img_idx, img_path in enumerate(image_files):
    print(f"Procesando {img_idx+1}/{len(image_files)}: {img_path.name}")

    pil_image = Image.open(img_path).convert("RGB")
    img_width, img_height = pil_image.size

    coco_output["images"].append(
        {
            "id": img_idx + 1,
            "file_name": img_path.name,
            "width": img_width,
            "height": img_height,
        }
    )

    results = yolo_model(pil_image, conf=0.25, verbose=False)

    yolo_annotations = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 16:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                dog_crop = pil_image.crop((x1, y1, x2, y2))
                if dog_crop.width < 10 or dog_crop.height < 10:
                    continue

                img_tensor = transform(dog_crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = classifier(img_tensor)
                    _, predicted = outputs.max(1)
                    predicted_idx = predicted.item()
                    breed = temp_dataset.idx_to_label[predicted_idx]

                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                class_id = predicted_idx
                yolo_annotations.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

                coco_bbox = [x1, y1, x2 - x1, y2 - y1]
                coco_output["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": img_idx + 1,
                        "category_id": breed_to_category_id[breed],
                        "bbox": coco_bbox,
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

    yolo_file = output_yolo / f"{img_path.stem}.txt"
    with open(yolo_file, "w") as f:
        f.write("\n".join(yolo_annotations))

print(f"\nAnotaciones YOLO guardadas en: {output_yolo}/")

coco_file = "anotaciones_coco.json"
with open(coco_file, "w") as f:
    json.dump(coco_output, f, indent=2)

print(f"Anotaciones COCO guardadas en: {coco_file}")

print("\n" + "=" * 80)
print("RESUMEN")
print("=" * 80)
print(f"Imagenes procesadas:       {len(image_files)}")
print(f"Total de anotaciones:      {annotation_id - 1}")
print(
    f"Clases detectadas:         {len(set([ann['category_id'] for ann in coco_output['annotations']]))}"
)
print("=" * 80)
print("\nFormatos generados:")
print(f"  - YOLOv5 (.txt):         {output_yolo}/")
print(f"  - COCO (.json):          {coco_file}")
