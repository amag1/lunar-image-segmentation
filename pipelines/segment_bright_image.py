import cv2
import numpy as np
import matplotlib.pyplot as plt


def segment_bright_image(image):
    # Apply Blur

    blur = cv2.medianBlur(image, 7)

    # Apply Canny edge detection

    edges = cv2.Canny(blur, 20, 80)

    # Find contours

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 1)

    pred_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(pred_mask, contours, -1, 255, thickness=-1)

    kernel_mask = np.ones((5, 5), np.uint8)
    pred_mask_closed = cv2.morphologyEx(
        pred_mask, cv2.MORPH_CLOSE, kernel_mask, iterations=1
    )

    pred_mask_dilated = cv2.dilate(pred_mask_closed, kernel_mask, iterations=1)

    _, bin_mask = cv2.threshold(pred_mask_dilated, 180, 255, cv2.THRESH_BINARY)

    # find new contours
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crate final mask
    final_mask = np.zeros_like(pred_mask_dilated)
    cv2.drawContours(final_mask, contours, -1, 255, thickness=cv2.FILLED)

    return final_mask
