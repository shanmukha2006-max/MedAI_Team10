import torch, yaml
from dtis.models.multitask_model import MultiTaskModel
from dtis.datasets.mimic_cxr import CXRDataset
from dtis.datasets.msd import SegDataset
from dtis.utils.metrics import auroc_multilabel, dice_from_logits, efficiency_percent
from torch.utils.data import DataLoader

cfg = yaml.safe_load(open("configs/default.yaml"))
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MultiTaskModel().to(device)
model.load_state_dict(torch.load("ckpts/multitask.pt", map_location=device))
model.eval()

cxr_val = CXRDataset(cfg["data"]["cxr"]["path"], split="val")
seg_val = SegDataset(cfg["data"]["seg"]["path"], split="val")

dl1 = DataLoader(cxr_val, batch_size=cfg["data"]["batch_size"])
dl2 = DataLoader(seg_val, batch_size=cfg["data"]["batch_size"])

x,y = next(iter(dl1)); o = model(x.to(device))
A = auroc_multilabel(o["logits"], y.to(device))

xs,ys = next(iter(dl2)); os_ = model(xs.to(device))
D = dice_from_logits(os_["mask"], ys.to(device))

Eff = efficiency_percent(A, D, rouge_norm=0.5)
print(f"AUROC={A:.3f} Dice={D:.3f} Efficiency%={Eff:.2f}")
