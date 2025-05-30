import cv2


## Removes all full red pixels from an image and sets them to black
def remove_red(image):
    """
    Remove all full red pixels from an image and set them to black.
    :param image: The input image in RGB
    :return: The transformed image with red pixels removed
    """
    # Convert the image to BGR format
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create a mask for red pixels
    red_mask = (
        (bgr_image[:, :, 2] == 255)
        & (bgr_image[:, :, 1] == 0)
        & (bgr_image[:, :, 0] == 0)
    )

    # Set red pixels to black
    bgr_image[red_mask] = [0, 0, 0]

    # Convert back to RGB format
    transformed_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    return transformed_image
