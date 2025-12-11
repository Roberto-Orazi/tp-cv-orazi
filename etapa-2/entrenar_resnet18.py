import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.dataset import DogDataset
from utils.transforms import get_train_transform, get_val_transform
from utils.models import get_resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando: {device}")

print("Cargando datasets...")
train_dataset = DogDataset(
    "../dataset/dogs.csv", "../dataset", "train", transform=get_train_transform()
)
val_dataset = DogDataset(
    "../dataset/dogs.csv", "../dataset", "valid", transform=get_val_transform()
)
test_dataset = DogDataset(
    "../dataset/dogs.csv", "../dataset", "test", transform=get_val_transform()
)

num_classes = train_dataset.num_classes
print(f"Numero de clases: {num_classes}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print("\nCreando modelo ResNet18...")
model = get_resnet18(num_classes, pretrained=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in tqdm(loader, desc="Entrenando"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Validando"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


num_epochs = 15
best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print(f"\nIniciando entrenamiento por {num_epochs} epochs...")

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "resnet18_best.pth")
        print(f"Mejor modelo guardado con Val Acc: {val_acc:.2f}%")

    scheduler.step()

print("\nGraficando resultados...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label="Train Loss")
ax1.plot(val_losses, label="Val Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.set_title("Loss durante entrenamiento")

ax2.plot(train_accs, label="Train Acc")
ax2.plot(val_accs, label="Val Acc")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.set_title("Accuracy durante entrenamiento")

plt.tight_layout()
plt.savefig("training_curves.png")
print("Curvas guardadas en training_curves.png")

print("\nCargando mejor modelo para evaluacion en test...")
model.load_state_dict(torch.load("resnet18_best.pth"))

print("Evaluando en conjunto de test...")
test_loss, test_acc = validate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

print("\nEntrenamiento completado!")
print(f"Mejor Val Acc: {best_val_acc:.2f}%")
print(f"Test Acc: {test_acc:.2f}%")
