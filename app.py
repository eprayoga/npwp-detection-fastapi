from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
from utils.npwp_utils import extract_fields_from_yolo

app = FastAPI()
model = YOLO("models/npwp_detection/best.pt")

@app.post("/npwp/extract")
async def extract_npwp_info(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    results = model(img)

    extracted, model_output = extract_fields_from_yolo(img, results, model)

    return {
        "data": extracted,
        "result": model_output
    }
