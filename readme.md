# NPWP OCR API (FastAPI + YOLOv8)

API untuk ekstraksi informasi dari dokumen NPWP menggunakan model YOLOv8 dan OCR (EasyOCR).

## ✨ Fitur

- Inference model YOLOv8 custom (`best.pt`)
- Ekstrak 4 field NPWP:
  - `nama`
  - `alamat`
  - `npwp`
  - `tanggal_terdaftar`
- Response JSON hasil OCR
- Endpoint FastAPI
- Dependency management via `requirements.txt`

---

## 📂 Struktur Project (contoh)

```
project/
│── app.py
│── requirements.txt
│── .gitignore
│── README.md
│
├── models/
│   └── npwp_detection/
│       └── best.pt
│
├── utils/
│   └── npwp_utils.py
│   └── ocr_preprocess.py
│
└── venv/
```

---

## 🚀 Cara Menjalankan

### 1. Clone repo

```bash
git clone https://github.com/eprayoga/npwp-detection-fastapi
cd npwp-detection-fastapi
```

### 2. Buat Virtual Environment

```bash
python -m venv venv
```

### 3. Aktifkan Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Menjalankan API

```bash
uvicorn app:app --reload
```

API akan jalan di:

```
http://127.0.0.1:8000
```

---

## 📤 Endpoint

### **POST /npwp/extract**

**Body (multipart/form-data)**

- `file`: upload gambar NPWP

**Contoh Response**

```json
{
  "status": "success",
  "data": {
    "nama": "PT ABC Indonesia",
    "alamat": "Jl. Sudirman No. 10 Jakarta",
    "npwp": "12.345.678.9-012.xxx",
    "tanggal_terdaftar": "01-01-2020",
  },
  "detections": [
    {
      "class": "nama",
      "confidence": 0.92,
      "bbox": [x1, y1, x2, y2]
    },
    ...
  ]
}
```

---

## 🧠 Model Deteksi

Model disimpan di path:

```
models/npwp_detection/best.pt
```

Model menerima input gambar dan mengembalikan bounding box & label.

---

## 🧩 Utils

File utilitas OCR & parsing ada di:

```
utils/npwp_utils.py
utils/ocr_preprocess.py
```

---

## 📄 Lisensi

Project ini tidak memiliki lisensi khusus (bebas digunakan untuk kebutuhan dev).
