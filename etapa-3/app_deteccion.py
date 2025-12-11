import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import cv2
from ultralytics import YOLO

from utils.transforms import get_val_transform
from utils.models import get_resnet18
from utils.dataset import DogDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

transform = get_val_transform()

print("Cargando YOLO preentrenado...")
yolo_model = YOLO("yolov8n.pt")
print("YOLO cargado (clase 'dog' = 16 en COCO)")

print("Cargando clasificador ResNet18...")
csv_file = "../dataset/dogs.csv"
temp_dataset = DogDataset(csv_file, "../dataset", "train", transform=transform)
num_classes = temp_dataset.num_classes

classifier = get_resnet18(num_classes, pretrained=False)
classifier_path = "../etapa-2/resnet18_best.pth"
if os.path.exists(classifier_path):
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()
    print("Clasificador ResNet18 cargado")
else:
    classifier = None
    print(
        "ERROR: ResNet18 no encontrado. Ejecuta 'python entrenar_resnet18.py' en etapa-2"
    )


def detect_and_classify(image):
    if classifier is None:
        return None, "Error: Clasificador no encontrado"

    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image

    results = yolo_model(pil_image, conf=0.25)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 16:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append(
                    {"box": (int(x1), int(y1), int(x2), int(y2)), "confidence": conf}
                )

    fallback_msg = ""
    if len(detections) == 0:
        # Fallback: Asumir que toda la imagen es el perro
        width, height = pil_image.size
        detections.append(
            {"box": (0, 0, width, height), "confidence": 0.0}
        )  # Confianza 0.0 para indicar fallback
        fallback_msg = "(FALLBACK: No se detecto perro, usando imagen completa)\n\n"

    output_image = pil_image.copy()
    draw = ImageDraw.Draw(output_image)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()

    results_text = f"{fallback_msg}Detectados {len(detections)} perro(s):\n\n"

    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["box"]
        conf = detection["confidence"]

        dog_crop = pil_image.crop((x1, y1, x2, y2))

        if dog_crop.width < 10 or dog_crop.height < 10:
            continue

        img_tensor = transform(dog_crop).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = classifier(img_tensor)
            _, predicted = outputs.max(1)
            predicted_idx = predicted.item()

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted_idx].item()

            breed = temp_dataset.idx_to_label[predicted_idx]

        # Solo dibujar caja si no es fallback (confianza > 0)
        if conf > 0:
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

        label = f"{breed} ({confidence*100:.1f}%)"

        bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle(bbox, fill="green")
        draw.text((x1, y1 - 25), label, fill="white", font=font)

        results_text += f"Perro {i+1}:\n"
        results_text += f"  Raza: {breed}\n"
        results_text += f"  Confianza: {confidence*100:.1f}%\n"
        results_text += f"  Deteccion: {conf*100:.1f}%\n\n"

    return output_image, results_text


with gr.Blocks() as demo:
    gr.Markdown("# Etapa 3: Deteccion y Clasificacion de Razas de Perros")
    gr.Markdown("Pipeline completo: YOLO detecta perros -> ResNet18 clasifica raza")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Subir imagen")
            detect_btn = gr.Button("Detectar y Clasificar")

        with gr.Column():
            output_image = gr.Image(type="pil", label="Detecciones")
            result_text = gr.Textbox(label="Resultados", lines=10)

    detect_btn.click(
        detect_and_classify, inputs=[input_image], outputs=[output_image, result_text]
    )

if __name__ == "__main__":
    print("\nLanzando aplicacion Gradio - Etapa 3...")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7861)
