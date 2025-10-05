import cv2
import numpy as np
import os
from glob import glob

def load_image(path, size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    return img

def load_data(root_path, size):
    images, masks = [], []
    image_paths = sorted(glob(os.path.join(root_path, "**", "*.png"), recursive=True))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in: {root_path}")

    for path in image_paths:
        if "_mask" in path:
            continue

        img = load_image(path, size)
        if img is None:
            continue

        mask_path = path.replace(".png", "_mask.png")
        if os.path.exists(mask_path):
            mask = load_image(mask_path, size)
            if mask is None:
                continue
        else:
            continue

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)
