import torch, torch.nn as nn
from monai.losses import DiceLoss

bce = nn.BCEWithLogitsLoss()
ce  = nn.CrossEntropyLoss()
dice = DiceLoss(to_onehot_y=True, softmax=True)

def multi_loss(outputs, y_cls=None, y_seg=None, y_rpt=None, w=(1.0,1.0,0.2)):
    Lc=Ls=Lr=0.0
    if y_cls is not None:
        Lc = bce(outputs["logits"], y_cls)
    if y_seg is not None:
        Ls = dice(outputs["mask"], y_seg.unsqueeze(1))
    if y_rpt is not None:
        B,T,V = outputs["rpt"].shape
        target = outputs["rpt"].detach().argmax(-1)
        Lr = ce(outputs["rpt"].reshape(B*T,V), target.reshape(B*T))
    return w[0]*Lc + w[1]*Ls + w[2]*Lr, (Lc, Ls, Lr)
