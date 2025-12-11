import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import faiss
from collections import Counter
import pickle

from utils.transforms import get_val_transform
from utils.models import get_feature_extractor
from utils.metrics import ndcg_at_k

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

transform = get_val_transform()
model = get_feature_extractor("resnet50")
model = model.to(device)
model.eval()

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


print("\nPreparando conjunto de prueba...")
df = pd.read_csv("../dataset/dogs.csv")
test_df = df[df["data set"] == "test"]

breed_groups = test_df.groupby("labels")
test_samples = []

for breed, group in breed_groups:
    samples = group.sample(min(5, len(group)))
    test_samples.extend(samples["filepaths"].tolist())

print(f"Conjunto de prueba: {len(test_samples)} imagenes")

print("\nCalculando NDCG@10...")
ndcg_scores = []
accuracy = 0

for i, test_path in enumerate(test_samples):
    if i % 50 == 0:
        print(f"Evaluando imagen {i}/{len(test_samples)}")

    full_path = Path("../dataset") / test_path
    true_label = df[df["filepaths"] == test_path]["labels"].values[0]

    similar_images, similar_labels, predicted_breed = search_similar_images(
        str(full_path), k=10
    )

    if predicted_breed == true_label:
        accuracy += 1

    relevances = [1 if label == true_label else 0 for label in similar_labels]
    ndcg = ndcg_at_k(relevances, 10)
    ndcg_scores.append(ndcg)

avg_ndcg = np.mean(ndcg_scores)
accuracy_pct = (accuracy / len(test_samples)) * 100

print(f"\n{'='*50}")
print(f"RESULTADOS DE EVALUACION")
print(f"{'='*50}")
print(f"NDCG@10 promedio: {avg_ndcg:.4f}")
print(f"Accuracy: {accuracy_pct:.2f}% ({accuracy}/{len(test_samples)})")
print(f"{'='*50}")
