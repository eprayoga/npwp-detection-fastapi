from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from utils.npwp_utils import extract_fields_from_yolo
from PIL import Image
import numpy as np
from ultralytics import YOLO
from io import BytesIO

app = FastAPI()

model_path = "models/npwp_detection/best.pt"
model = YOLO(model_path)


@app.post("/npwp/extract")
async def extract_npwp_info(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        img = Image.open(BytesIO(image_data)).convert("RGB")
        image_np = np.array(img)

        # YOLO inference
        results = model(image_np)

        # Extract fields + model raw output
        extracted_fields, detections = extract_fields_from_yolo(image_np, results, model)

        return JSONResponse({
            "status": "success",
            "data": extracted_fields,
            "detections": detections
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
