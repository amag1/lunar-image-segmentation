import cv2


def pixel_difference(image1, image2):
    """
    Calculate the magnitude of the pixel-wise difference between two images.
    :param image1: First image
    :param image2: Second image
    :return: number of pixels that differ
    """
    # Ensure both images are of the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")

    # Calculate the absolute difference
    imgDiff = cv2.absdiff(image1, image2)

    # Convert to grayscale
    imgDiffGray = cv2.cvtColor(imgDiff, cv2.COLOR_BGR2GRAY)
    # Threshold the difference image
    _, imgDiffThresh = cv2.threshold(imgDiffGray, 1, 255, cv2.THRESH_BINARY)
    # Count the number of non-zero pixels
    diff = cv2.countNonZero(imgDiffThresh)
    # Return the number of differing pixels

    return diff


def pixel_difference_percentage(image1, image2):
    """
    Calculate the percentage of pixels that differ between two images.
    :param image1: First image
    :param image2: Second image
    :return: percentage of pixels that differ
    """
    # Ensure both images are of the same size
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")

    # both images must be binary: only contain 0s and 255s
    if not (set(image1.flatten()) <= {0, 255}):
        # threshold the images to binary
        _, image1 = cv2.threshold(image1, 10, 255, cv2.THRESH_BINARY)
    if not (set(image2.flatten()) <= {0, 255}):
        # threshold the images to binary
        _, image2 = cv2.threshold(image2, 10, 255, cv2.THRESH_BINARY)

    # Calculate the absolute difference
    imgDiff = cv2.absdiff(image1, image2)

    # Threshold the difference image
    _, imgDiffThresh = cv2.threshold(imgDiff, 1, 255, cv2.THRESH_BINARY)

    # Count the number of non-zero pixels
    diff = cv2.countNonZero(imgDiffThresh)
    # Calculate the total number of pixels in the image
    total_pixels = imgDiffThresh.size

    # Calculate the percentage of differing pixels
    percentage_diff = (diff / total_pixels) * 100

    return percentage_diff
