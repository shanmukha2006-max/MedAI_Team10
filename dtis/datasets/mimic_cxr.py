import os, numpy as np, torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class CXRDataset(Dataset):
    def __init__(self, root, img_size=512, split="train"):
        self.root = root
        df = pd.read_csv(os.path.join(root, f"{split}_labels.csv"))
        self.ids = df["id"].tolist()
        self.labels = df.drop(columns=["id"]).values.astype("float32")
        self.img_dir = os.path.join(root, "images")
        self.img_size = img_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pid = self.ids[i]

        # Load & resize
        img = Image.open(os.path.join(self.img_dir, f"{pid}.png")).convert("RGB")
        img = img.resize((self.img_size, self.img_size))

        # Convert to tensor
        x = torch.tensor(np.array(img)).permute(2,0,1).float()/255.
        y = torch.tensor(self.labels[i])

        return x, y

