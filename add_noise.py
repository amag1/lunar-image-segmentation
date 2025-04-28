import cv2
import os

image_directory = "dataset/images/render"
MAX_NOISE = 10


## Add a random noise to the grayscale image
def add_noise(image):
    """
    Add random noise to an image.
    :param image: Input image
    :return: Image with added noise
    """
    # Generate random noise
    noise = cv2.randn(image.copy(), 0, MAX_NOISE)

    # Add noise to the image
    noisy_image = cv2.add(image, noise)

    return noisy_image


def main():
    # Loop through all images in the directory
    for filename in os.listdir(image_directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Read the image
            image_path = os.path.join(image_directory, filename)
            image = cv2.imread(image_path)

            # Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Add noise to the image
            noisy_image = add_noise(gray_image)

            # Save the noisy image
            noisy_image_path = os.path.join(image_directory, "noisy_" + filename)
            cv2.imwrite(noisy_image_path, noisy_image)
            print(f"Saved noisy image: {noisy_image_path}")


if __name__ == "__main__":
    main()
