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
from utils.models import get_resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

transform = get_val_transform()

print("Cargando modelo ResNet18 entrenado...")
dataset = DogDataset("../dataset/dogs.csv", "../dataset", transform=transform)
num_classes = dataset.num_classes

model = get_resnet18(num_classes, pretrained=False)
model.load_state_dict(torch.load("resnet18_best.pth", map_location=device))
model = model.to(device)
model.eval()

print("Extrayendo embeddings con ResNet18...")
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

embeddings = []
labels = []
image_paths = []

with torch.no_grad():
    for i, (images, lbls, paths) in enumerate(dataloader):
        if i % 50 == 0:
            print(f"Procesando batch {i}/{len(dataloader)}")

        images = images.to(device)

        features = model.conv1(images)
        features = model.bn1(features)
        features = model.relu(features)
        features = model.maxpool(features)
        features = model.layer1(features)
        features = model.layer2(features)
        features = model.layer3(features)
        features = model.layer4(features)
        features = model.avgpool(features)
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
with open("embeddings_resnet18.pkl", "wb") as f:
    pickle.dump(
        {"index": index, "labels": dataset_labels, "image_paths": image_paths}, f
    )

print("Embeddings de ResNet18 guardados en embeddings_resnet18.pkl")
print("Ahora podes usar 'python app_etapa2.py' para probar ambos modelos")
