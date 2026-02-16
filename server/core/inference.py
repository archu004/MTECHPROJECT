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
def get_health_suggestion(deviation: float):
    """
    Returns food & lifestyle suggestions
    based on biological age deviation.
    """

    if deviation >= 5:
        return {
            "status": "Biological age is higher than expected",
            "risk_level": "Elevated",
            "food_suggestions": [
                "Increase antioxidant-rich foods (berries, spinach, nuts)",
                "Add omega-3 sources (flaxseeds, walnuts, fish)",
                "Reduce processed sugar and fried food",
                "Drink more water (2.5–3L daily)"
            ],
            "lifestyle_suggestions": [
                "Sleep 7–8 hours daily",
                "Regular cardio (30 mins/day)",
                "Stress reduction (meditation/yoga)"
            ]
        }

    elif deviation <= -5:
        return {
            "status": "Biological age is lower than expected",
            "risk_level": "Good",
            "food_suggestions": [
                "Maintain balanced diet",
                "Continue protein-rich meals",
                "Include fresh vegetables daily"
            ],
            "lifestyle_suggestions": [
                "Maintain current exercise routine",
                "Continue good sleep schedule"
            ]
        }

    else:
        return {
            "status": "Biological age is close to chronological age",
            "risk_level": "Normal",
            "food_suggestions": [
                "Balanced diet with fruits & vegetables",
                "Limit junk food",
                "Stay hydrated"
            ],
            "lifestyle_suggestions": [
                "Light exercise 4–5 days/week",
                "Maintain regular sleep"
            ]
        }


def predict(image: Image.Image, chronological_age: float):
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, dist_logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        dist = F.softmax(dist_logits, dim=1)[0].cpu().numpy()

    expected_age = float((dist * MIDPOINTS.numpy()).sum())
    deviation = round(expected_age - chronological_age, 2)

    suggestions = get_health_suggestion(deviation)

    return {
        "predicted_group": AGE_LABELS[int(probs.argmax())],
        "biological_age": round(expected_age, 2),
        "deviation_years": deviation,
        "health_analysis": suggestions,
        "class_probabilities": probs.tolist(),
        "distribution": dist.tolist()
    }
print("Model weights loaded from:", WEIGHTS_PATH)
print("First layer weight sum:",
      model.resnet.conv1.weight.sum().item())
