import torch
import torch.nn.functional as F
from PIL import Image

from core.config import (
    DEVICE, WEIGHTS_PATH,
    MIDPOINTS, AGE_LABELS
)
from core.model import ResNet34_LDL_CBAM
from utils.image_utils import transform

# 🔒 Load ONCE
model = ResNet34_LDL_CBAM().to(DEVICE)
ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
model.load_state_dict(
    ckpt["model_state"] if "model_state" in ckpt else ckpt
)
model.eval()

def predict(image: Image.Image, chronological_age: float):
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, dist_logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        dist = F.softmax(dist_logits, dim=1)[0].cpu().numpy()

    expected_age = float((dist * MIDPOINTS.numpy()).sum())
    deviation = round(expected_age - chronological_age, 2)

    return {
        "predicted_group": AGE_LABELS[int(probs.argmax())],
        "biological_age": round(expected_age, 2),
        "deviation_years": deviation,
        "class_probabilities": probs.tolist(),
        "distribution": dist.tolist()
    }
print("Model weights loaded from:", WEIGHTS_PATH)
print("First layer weight sum:",
      model.resnet.conv1.weight.sum().item())
