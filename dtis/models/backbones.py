import torch.nn as nn
from torchvision.models import vit_b_16, convnext_base

class ImageBackbone(nn.Module):
    def __init__(self, name="vit_base_patch16_384"):
        super().__init__()
        if "vit" in name:
            m = vit_b_16(weights=None)
            self.out_dim = m.heads.head.in_features
            m.heads.head = nn.Identity()
            self.forward_impl = m
        elif "convnext" in name:
            m = convnext_base(weights=None)
            self.out_dim = m.classifier[2].in_features
            m.classifier[2] = nn.Identity()
            self.forward_impl = m
        else:
            raise ValueError("Unsupported backbone")

    def forward(self, x):
        return self.forward_impl(x)
