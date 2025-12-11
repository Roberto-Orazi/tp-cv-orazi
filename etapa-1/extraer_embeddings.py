import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import numpy as np
import faiss
import pickle

from utils.dataset import DogDataset
from utils.transforms import get_val_transform
from utils.models import get_feature_extractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

transform = get_val_transform()
model = get_feature_extractor("resnet50")
model = model.to(device)
model.eval()

if not os.path.exists("embeddings_data.pkl"):
    print("Extrayendo embeddings del dataset...")
    dataset = DogDataset("../dataset/dogs.csv", "../dataset", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    embeddings = []
    labels = []
    image_paths = []

    with torch.no_grad():
        for i, (images, lbls, paths) in enumerate(dataloader):
            if i % 50 == 0:
                print(f"Procesando batch {i}/{len(dataloader)}")

            images = images.to(device)
            features = model(images)
            features = features.squeeze().cpu().numpy()

            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            embeddings.append(features)
            labels.extend(lbls.cpu().numpy())
            image_paths.extend(paths)

    embeddings = np.vstack(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"Indice creado con {index.ntotal} vectores")

    dataset_labels = [dataset.idx_to_label[idx] for idx in labels]

    print("Guardando embeddings...")
    with open("embeddings_data.pkl", "wb") as f:
        pickle.dump(
            {"index": index, "labels": dataset_labels, "image_paths": image_paths}, f
        )
    print("Embeddings guardados en embeddings_data.pkl")
else:
    print("Cargando embeddings previamente guardados...")
    with open("embeddings_data.pkl", "rb") as f:
        data = pickle.load(f)
        index = data["index"]
        labels = data["labels"]
        image_paths = data["image_paths"]
    print(f"Indice cargado con {index.ntotal} vectores")

print("\nExtraccion de embeddings completada!")
print("Usa 'python app_etapa1.py' para lanzar la aplicacion Gradio")
