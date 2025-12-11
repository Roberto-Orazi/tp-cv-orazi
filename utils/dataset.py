import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class DogDataset(Dataset):
    def __init__(self, csv_file, root_dir, dataset_type=None, transform=None):
        df = pd.read_csv(csv_file)

        if dataset_type:
            self.data = df[df["data set"] == dataset_type].reset_index(drop=True)
        else:
            self.data = df

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.label_to_idx = {
            label: idx for idx, label in enumerate(sorted(df["labels"].unique()))
        }
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.root_dir / self.data.iloc[idx]["filepaths"]
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["labels"]
        label_idx = self.label_to_idx[label]

        if self.transform:
            image = self.transform(image)

        return image, label_idx, str(img_path)
