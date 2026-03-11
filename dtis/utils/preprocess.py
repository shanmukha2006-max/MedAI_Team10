import os, numpy as np
from .dicom_utils import load_dicom
from PIL import Image

def dicom_to_png_folder(dicom_dir, out_dir, size=512):
    os.makedirs(out_dir, exist_ok=True)
    for root,_,files in os.walk(dicom_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                arr = (load_dicom(os.path.join(root,f))*255).astype("uint8")
                Image.fromarray(arr).convert("L").resize((size,size)).save(
                    os.path.join(out_dir, f"{os.path.splitext(f)[0]}.png"))
