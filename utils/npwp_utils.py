import cv2
import numpy as np
import easyocr
from utils.ocr_preprocess import preprocess_for_ocr

reader = easyocr.Reader(['en'], gpu=False)

CLASSES = ["alamat", "nama", "npwp", "tanggal_terdaftar"]


def ocr_extract(image_bgr, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = image_bgr[y1:y2, x1:x2]

    # Preprocess
    proc = preprocess_for_ocr(roi)

    # OCR
    results = reader.readtext(proc)
    text = " ".join([res[1] for res in results])
    return text


def extract_fields_from_yolo(image_np, results, model):
    extracted = {cls: "" for cls in CLASSES}
    detection_output = []

    boxes = results[0].boxes
    if boxes is None:
        return extracted, detection_output

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()

        # Save detection info
        detection_output.append({
            "class": cls_name,
            "confidence": conf,
            "bbox": bbox
        })

        if cls_name in CLASSES:
            text = ocr_extract(image_np, bbox)
            extracted[cls_name] = text

    return extracted, detection_output
