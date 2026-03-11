def threshold_probs(probs, thr=0.5):
    return [1 if p>thr else 0 for p in probs]
