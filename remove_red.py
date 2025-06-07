import cv2


def remove_red(img_bgr, r_thresh=200, gb_thresh=50):
    """Pone en negro los píxeles predominantemente rojos de una imagen BGR."""
    b, g, r = cv2.split(img_bgr)
    red_mask = (r > r_thresh) & (g < gb_thresh) & (b < gb_thresh)
    img_bgr[red_mask] = (0, 0, 0)  # negro en BGR
    return img_bgr
