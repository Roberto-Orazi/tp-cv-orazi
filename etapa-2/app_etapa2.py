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

from utils.dataset import DogDataset
from utils.transforms import get_val_transform
from utils.models import get_feature_extractor, get_resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

transform = get_val_transform()

print("Cargando modelos...")

model_resnet50 = get_feature_extractor("resnet50")
model_resnet50 = model_resnet50.to(device)
model_resnet50.eval()

csv_file = "../dataset/dogs.csv"
temp_dataset = DogDataset(csv_file, "../dataset", "train", transform=transform)
num_classes = temp_dataset.num_classes

model_resnet18 = get_resnet18(num_classes, pretrained=False)
if os.path.exists("resnet18_best.pth"):
    model_resnet18.load_state_dict(torch.load("resnet18_best.pth", map_location=device))
    model_resnet18 = model_resnet18.to(device)
    model_resnet18.eval()
    print("ResNet18 entrenado cargado")
else:
    model_resnet18 = None
    print("ResNet18 no encontrado. Ejecuta 'python entrenar_resnet18.py' primero")

print("Cargando embeddings ResNet50...")
embeddings_path_50 = "../etapa-1/embeddings_data.pkl"
if os.path.exists(embeddings_path_50):
    with open(embeddings_path_50, "rb") as f:
        data_50 = pickle.load(f)
        index_50 = data_50["index"]
        labels_50 = data_50["labels"]
        image_paths_50 = data_50["image_paths"]
    print(f"Embeddings ResNet50 cargados: {index_50.ntotal} vectores")
else:
    index_50 = None
    print("Embeddings ResNet50 no encontrados")

embeddings_path_18 = "embeddings_resnet18.pkl"
if os.path.exists(embeddings_path_18):
    with open(embeddings_path_18, "rb") as f:
        data_18 = pickle.load(f)
        index_18 = data_18["index"]
        labels_18 = data_18["labels"]
        image_paths_18 = data_18["image_paths"]
    print(f"Embeddings ResNet18 cargados: {index_18.ntotal} vectores")
else:
    index_18 = None
    print("Embeddings ResNet18 no encontrados. Se generaran al buscar")


def extract_features_resnet18(image_path):
    if model_resnet18 is None:
        return None

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model_resnet18.conv1(img_tensor)
        features = model_resnet18.bn1(features)
        features = model_resnet18.relu(features)
        features = model_resnet18.maxpool(features)
        features = model_resnet18.layer1(features)
        features = model_resnet18.layer2(features)
        features = model_resnet18.layer3(features)
        features = model_resnet18.layer4(features)
        features = model_resnet18.avgpool(features)
        features = features.squeeze().cpu().numpy().reshape(1, -1)

    return features


def search_similar_images(query_image, model_name, k=10):
    if model_name == "ResNet50 (Pre-entrenado)":
        if index_50 is None:
            return [], [], "Error: Embeddings ResNet50 no encontrados"

        img = Image.open(query_image).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            query_embedding = model_resnet50(img_tensor)
            query_embedding = query_embedding.squeeze().cpu().numpy().reshape(1, -1)

        faiss.normalize_L2(query_embedding)
        distances, indices = index_50.search(query_embedding, k)

        similar_images = [image_paths_50[idx] for idx in indices[0]]
        similar_labels = [labels_50[idx] for idx in indices[0]]

    elif model_name == "ResNet18 (Fine-tuned)":
        if model_resnet18 is None:
            return [], [], "Error: ResNet18 no encontrado"

        if index_18 is not None:
            query_features = extract_features_resnet18(query_image)
            faiss.normalize_L2(query_features)
            distances, indices = index_18.search(query_features, k)

            similar_images = [image_paths_18[idx] for idx in indices[0]]
            similar_labels = [labels_18[idx] for idx in indices[0]]
        else:
            return [], [], "Error: Embeddings ResNet18 no encontrados"

    else:
        return [], [], "Modelo no disponible"

    breed_counts = Counter(similar_labels)
    predicted_breed = breed_counts.most_common(1)[0][0]

    return similar_images, similar_labels, predicted_breed


def gradio_search(image, model_name):
    if image is None:
        return "Por favor sube una imagen", []

    try:
        similar_images, similar_labels, predicted_breed = search_similar_images(
            image, model_name, k=10
        )

        if isinstance(predicted_breed, str) and predicted_breed.startswith("Error"):
            return predicted_breed, []

        result_text = f"Modelo: {model_name}\nRaza predicha: {predicted_breed}"

        gallery_images = []
        for img_path, label in zip(similar_images, similar_labels):
            gallery_images.append((img_path, label))

        return result_text, gallery_images
    except Exception as e:
        return f"Error: {str(e)}", []


with gr.Blocks() as demo:
    gr.Markdown("# Etapa 2: Buscador con Selector de Modelos")
    gr.Markdown(
        "Compara el rendimiento de diferentes modelos de extraccion de caracteristicas"
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Subir imagen")

            model_selector = gr.Radio(
                choices=["ResNet50 (Pre-entrenado)", "ResNet18 (Fine-tuned)"],
                value="ResNet50 (Pre-entrenado)",
                label="Seleccionar Modelo",
            )

            search_btn = gr.Button("Buscar")

        with gr.Column():
            result_text = gr.Textbox(label="Resultado", lines=3)

    gallery = gr.Gallery(label="Imagenes similares", columns=5)

    search_btn.click(
        gradio_search,
        inputs=[input_image, model_selector],
        outputs=[result_text, gallery],
    )

if __name__ == "__main__":
    print("\nLanzando aplicacion Gradio - Etapa 2...")
    dataset_path = os.path.abspath("../dataset")
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        allowed_paths=[dataset_path],
    )
