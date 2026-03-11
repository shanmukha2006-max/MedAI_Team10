import pydicom, numpy as np

def load_dicom(path, window=None):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype("float32")
    if window:
        center,width = window
        low,high = center - width/2, center + width/2
        img = (img - low)/(high-low)
    return np.clip(img,0,1)
