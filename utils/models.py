import torch
import torch.nn as nn
from torchvision import models


def get_resnet18(num_classes, pretrained=True):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def get_resnet50(num_classes, pretrained=True):
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=None)

    model = nn.Sequential(*list(model.children())[:-1])
    return model


def get_feature_extractor(model_name="resnet50"):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model = nn.Sequential(*list(model.children())[:-1])
    return model
