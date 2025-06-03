import cv2
import numpy as np

def segmentate_dark_image(image):
    # 1. Aumento de brillo/contraste
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)

    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(bright) 

    # # 4. Suavizado
    blurred = cv2.GaussianBlur(clahe_img, (7, 7), 0)

    # # 5. Canny Edge Detection
    edges = cv2.Canny(blurred, 75, 175) 

    # # 6. Limpieza de bordes Canny
    kernel_close_canny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close_canny)
    edges_for_contours = edges_cleaned

    # 7. Encontrar y filtrar contornos
    contours, _ = cv2.findContours(edges_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    min_area = 50 
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            filtered_contours.append(cnt)

    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 1)

    pred_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(pred_mask, filtered_contours, -1, 255, thickness=-1)

    return pred_mask