import os, numpy as np, torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SegDataset(Dataset):
    def __init__(self, root, split="train", img_size=224):
        self.root = root
        self.img_size = img_size
        self.X = np.load(os.path.join(root, f"{split}_images.npy"))
        self.Y = np.load(os.path.join(root, f"{split}_masks.npy"))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.tensor(self.X[i]).float()
        y = torch.tensor(self.Y[i]).long().squeeze(0)

        # ensure 3 channels
        if x.ndim == 3:
            pass
        else:
            x = x.repeat(3,1,1)

        # resize to match ViT expected input
        x = F.interpolate(x.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear", align_corners=False).squeeze(0)
        y = F.interpolate(y.unsqueeze(0).unsqueeze(0).float(), size=(self.img_size, self.img_size), mode="nearest").squeeze().long()


        return x, y
