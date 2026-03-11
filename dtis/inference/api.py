import io
import torch
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile
import torch.nn.functional as F

from dtis.models.multitask_model import MultiTaskModel

app = FastAPI()

# Load Model
model = MultiTaskModel("vit_base_patch16_384", num_labels=14).cpu()
model.load_state_dict(torch.load("ckpts/multitask.pt", map_location="cpu"))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Step 1: Read image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Step 2: Resize to 224x224 (❗important fix)
    img = img.resize((224, 224))

    # Step 3: Convert to tensor
    x = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255
    x = x.unsqueeze(0)

    # Step 4: Model inference
    out = model(x)

    # Step 5: Extract outputs
    probs = out["probabilities"].cpu().numpy().tolist()

    return {"probabilities": probs}
