import numpy as np
import cv2


def fourier_remove_low_freq(image, mask):
    """
    Returns an image with low frequency components removed
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    radius = 50
    mask = np.ones((rows, cols), dtype=np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)

    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.uint8(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    return img_back


def opening(image, kernel_size=(5, 5), element=cv2.MORPH_ELLIPSE):
    """
    Applies morphological opening to the image
    """
    kernel = cv2.getStructuringElement(element, kernel_size)
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened_image


def closing(image, kernel_size=(5, 5), element=cv2.MORPH_ELLIPSE):
    """
    Applies morphological closing to the image
    """
    kernel = cv2.getStructuringElement(element, kernel_size)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed_image


def umbralize(image, threshold=50):
    """
    Applies binary thresholding to the image
    """
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def kmeans_segmentation(image, k=2):
    """
    Applies k-means segmentation to the image
    """
    Z = image.reshape((-1, 1))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    segmented_image = centers[labels.flatten()].reshape(image.shape).astype(np.uint8)
    return segmented_image


def segment_flat_image(image):
    """
    Returns a mask of an image
    """
    mask = np.zeros(image.shape, dtype=bool)
    mask = np.invert(mask)
    mask = fourier_remove_low_freq(image, mask)

    mask = kmeans_segmentation(mask, k=2)

    mask = opening(mask, kernel_size=(3, 3))
    mask = closing(mask, kernel_size=(3, 3))
    mask = umbralize(mask, threshold=10)

    return mask
