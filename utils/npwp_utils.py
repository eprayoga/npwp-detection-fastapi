import cv2
import easyocr

reader = easyocr.Reader(['id'])

def extract_fields_from_yolo(img, results, model):
    extracted = {
        "alamat": "",
        "nama": "",
        "npwp": "",
        "tanggal_terdaftar": "",
    }

    model_output = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop area bbox
            crop = img[y1:y2, x1:x2]

            # OCR
            ocr_res = reader.readtext(crop, detail=0)
            text = " ".join(ocr_res).strip()

            # Mapping OCR ke field
            lower_label = label.lower()
            if lower_label in extracted:
                extracted[lower_label] = text

            # Simpan data bbox YOLO
            model_output.append({
                "label": lower_label,
                "confidence": float(box.conf),
                "bbox": [x1, y1, x2, y2],
                "ocr_text": text
            })

    return extracted, model_output
