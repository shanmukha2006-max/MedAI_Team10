import torch
import torch.nn as nn

class ClassifyHead(nn.Module):
    def __init__(self, in_dim, num_labels):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_labels)

    def forward(self, f):
        return self.fc(f)

import torch.nn.functional as F

class SegmentHead(nn.Module):
    def __init__(self, in_dim, out_ch=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 256, 4, stride=4),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, f):
        fmap = f.view(f.size(0), f.size(1), 1, 1).repeat(1,1,16,16)
        mask = self.up(fmap)

        # ✅ Force output resolution to 224x224
        mask = F.interpolate(mask, size=(224,224), mode="bilinear", align_corners=False)

        return mask


class ReportHead(nn.Module):
    def __init__(self, in_dim, vocab_size=8000, hidden=512):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab_size)

    def forward(self, f, max_len=64):
        h = self.proj(f).unsqueeze(1)
        outs, _ = self.gru(h.repeat(1, max_len, 1))
        logits = self.lm_head(outs)
        return logits
