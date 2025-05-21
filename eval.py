## Aplica la funcion de transformacion a una imagen y devuelve la loss
import cv2
from loss import pixel_difference_percentage


def eval(image_number, transform):
    """
    Apply the transformation to an image and return the loss.
    :param image_number: The number of the image to be transformed
    :param transform: The transformation function to be applied
    :return: The loss value
    """
    # Read the image
    image_path = f"./dataset/images/render/render{image_number}.png"
    mask_path = f"./dataset/images/ground/ground{image_number}.png"

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Apply the transformation
    transformed_image = transform(image)

    # Calculate the loss (for example, mean squared error)
    loss = pixel_difference_percentage(transformed_image, mask)

    return loss
