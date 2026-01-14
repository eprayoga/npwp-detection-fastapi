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
- Support virtual environment
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
│
└── venv/
```

---

## 🚀 Cara Menjalankan

### 1. Clone repo

```bash
git clone <repo-url>
cd project
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
  "nama": "PT ABC Indonesia",
  "alamat": "Jl. Sudirman No. 10 Jakarta",
  "npwp": "12.345.678.9-012.345",
  "tanggal_terdaftar": "01-01-2020",
  "model_output": [
    {
      "class": "nama",
      "bbox": [x1, y1, x2, y2],
      "conf": 0.92
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
```

---

## 📝 Catatan

- GPU tidak wajib, CPU akan tetap bisa inference (lebih lambat).
- Pastikan `python-multipart` terinstall untuk upload form data.

---

## 📄 Lisensi

Project ini tidak memiliki lisensi khusus (bebas digunakan untuk kebutuhan dev).

---

## 🤝 Kontribusi

Pull request dipersilakan. Open issue jika ada bug atau request fitur.

---

## 👨‍💻 Author

Dibuat oleh: **(isi nama kamu di sini)**
