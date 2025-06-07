from pipelines.segment_flat_image import segment_flat_image
from pipelines.segment_dark_image import segment_dark_image
from pipelines.segment_bright_image import segment_bright_image
from eval import eval

from concurrent.futures import ProcessPoolExecutor, as_completed

import os
import cv2
from tqdm import tqdm


target_dir = "dataset/images/render"
ground_dir = "dataset/images/ground"
result_dir = "dataset/images/resulting_masks"
methods = {
    "flat": segment_flat_image,
    "dark": segment_dark_image,
    "bright": segment_bright_image,
}


def process_image(code):
    # Apply all methods, check their results, and save the masks
    results = {}
    for method_name, method in methods.items():
        mask, loss = eval(code, method)
        results[method_name] = {"mask": mask, "loss": loss}

    # Save the best mask with code
    best_method = min(results, key=lambda x: results[x]["loss"])
    best_mask = results[best_method]["mask"]
    best_loss = round(results[best_method]["loss"])

    cv2.imwrite(
        os.path.join(result_dir, best_method, f"result{code}-loss{best_loss}.png"),
        best_mask,
    )


if __name__ == "__main__":
    for method in methods.keys():
        os.makedirs(os.path.join(result_dir, method), exist_ok=True)

    files = [f for f in os.listdir(target_dir) if f.endswith((".png", ".jpg"))]

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_image, f.split(".")[0][-4:]): f for f in files
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")
