import cv2
import numpy as np

def preprocess_for_ocr(image_bgr: np.ndarray):
    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Reduce Noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 10
    )

    # Morphology closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return morph
