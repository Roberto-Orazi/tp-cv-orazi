import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import faiss
from collections import Counter
import gradio as gr
import pickle
from PIL import Image

from utils.transforms import get_val_transform
from utils.models import get_feature_extractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

transform = get_val_transform()
model = get_feature_extractor("resnet50")
model = model.to(device)
model.eval()

print("Cargando embeddings...")
with open("embeddings_data.pkl", "rb") as f:
    data = pickle.load(f)
    index = data["index"]
    labels = data["labels"]
    image_paths = data["image_paths"]

print(f"Indice cargado con {index.ntotal} vectores")


def search_similar_images(query_image, k=10):
    img = Image.open(query_image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_embedding = model(img_tensor)
        query_embedding = query_embedding.squeeze().cpu().numpy().reshape(1, -1)

    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)

    similar_images = []
    similar_labels = []

    for idx in indices[0]:
        similar_images.append(image_paths[idx])
        similar_labels.append(labels[idx])

    breed_counts = Counter(similar_labels)
    predicted_breed = breed_counts.most_common(1)[0][0]

    return similar_images, similar_labels, predicted_breed


def gradio_search(image):
    if image is None:
        return "Por favor sube una imagen", []

    try:
        similar_images, similar_labels, predicted_breed = search_similar_images(
            image, k=10
        )
        result_text = f"Raza predicha: {predicted_breed}"

        gallery_images = []
        for img_path, label in zip(similar_images, similar_labels):
            gallery_images.append((img_path, label))

        return result_text, gallery_images
    except Exception as e:
        return f"Error: {str(e)}", []


with gr.Blocks() as demo:
    gr.Markdown("# Etapa 1: Buscador de Razas por Similitud")
    gr.Markdown("Modelo: **ResNet50** pre-entrenado (ImageNet)")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Subir imagen")
            search_btn = gr.Button("Buscar")

        with gr.Column():
            result_text = gr.Textbox(label="Resultado")

    gallery = gr.Gallery(label="Imagenes similares", columns=5)

    search_btn.click(gradio_search, inputs=input_image, outputs=[result_text, gallery])

if __name__ == "__main__":
    print("\nLanzando aplicacion Gradio - Etapa 1...")
    dataset_path = os.path.abspath("../dataset")
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        allowed_paths=[dataset_path],
    )
