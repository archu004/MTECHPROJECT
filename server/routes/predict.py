from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
from core.inference import predict

router = APIRouter()

@router.post("/predict-age")
async def predict_age(
    file: UploadFile = File(...),
    chronological_age: float = Form(...)
):
    image = Image.open(file.file).convert("RGB")
    return predict(image, chronological_age)
