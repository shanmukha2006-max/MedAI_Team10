import torch
from sklearn.metrics import roc_auc_score

def auroc_multilabel(logits, y_true):
    probs = torch.sigmoid(logits).detach().cpu()
    y = y_true.detach().cpu()
    try:
        return roc_auc_score(y, probs, average="macro")
    except:
        return 0.5

def dice_from_logits(logits, target):
    pred = logits.argmax(1)
    return (pred==target).float().mean().item()

def efficiency_percent(auroc, dice, rouge_norm=0.5, w=(0.5,0.4,0.1)):
    return 100*(w[0]*auroc + w[1]*dice + w[2]*rouge_norm)
