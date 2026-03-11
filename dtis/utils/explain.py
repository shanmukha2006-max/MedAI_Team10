def fake_gradcam_heatmap(img_tensor):
    import torch
    B,C,H,W = img_tensor.shape
    return torch.rand(B,1,H,W)
