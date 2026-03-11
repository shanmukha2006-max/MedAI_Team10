import torch.nn as nn
from .backbones import ImageBackbone
from .heads import ClassifyHead, SegmentHead, ReportHead

class MultiTaskModel(nn.Module):
    def __init__(self, backbone="vit_base_patch16_384", num_labels=14, seg_out=2):
        super().__init__()
        self.backbone = ImageBackbone(backbone)
        D = self.backbone.out_dim
        self.cls = ClassifyHead(D, num_labels)
        self.seg = SegmentHead(D, seg_out)
        self.rep = ReportHead(D)

    def forward(self, x, max_len=64):
        f = self.backbone(x)
        logits = self.cls(f)
        mask = self.seg(f)
        rpt = self.rep(f, max_len=max_len)
        return {"logits": logits, "mask": mask, "rpt": rpt, "feat": f}
