import os
import random

import cv2
import numpy as np


def make_split(root, split, count, size, seed):
    random.seed(seed)
    np.random.seed(seed)
    images_dir = os.path.join(root, split, "images")
    masks_dir = os.path.join(root, split, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    for i in range(count):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)

        for _ in range(3):
            center = (random.randint(20, size - 20), random.randint(20, size - 20))
            radius = random.randint(10, 30)
            color = [random.randint(0, 255) for _ in range(3)]
            cv2.circle(image, center, radius, color, -1)
            if random.random() > 0.5:
                cv2.circle(mask, center, radius, 255, -1)

        name = f"{split}_{i:03d}.png"
        cv2.imwrite(os.path.join(images_dir, name), image)
        cv2.imwrite(os.path.join(masks_dir, name), mask)


def main():
    root = "data"
    make_split(root, "train", 10, 256, 1)
    make_split(root, "val", 4, 256, 2)
    make_split(root, "test", 4, 256, 3)
    print("dummy data created in ./data")


if __name__ == "__main__":
    main()
