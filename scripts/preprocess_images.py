import os
import sys

import cv2
from cv2.typing import MatLike

import scripts.shared
import shared.image

OUTPUT_DIR = ".data/images"


def main():
    if len(sys.argv) < 2:
        raise Exception("must provide a path to the images folder")

    images_dir = sys.argv[1]

    print(f"Reading '{images_dir}'...")
    filenames = os.listdir(images_dir)
    print(f"Found {len(filenames)} images")

    print("Extracting features...")
    all_descriptors: list[MatLike] = []

    sift = cv2.SIFT.create()
    next_step = 0

    for i, filename in enumerate(filenames):
        path = f"{images_dir}/{filename}"
        progress = 100 * (i / len(filenames))

        if progress >= next_step:
            print(f"Progress: {round(progress, 2)}%")

            while progress >= next_step:
                next_step += 0.01

        img = cv2.imread(path)
        assert img is not None, f"Failed to load {path}"
        img = shared.image.downscale(img)

        _, d = sift.detectAndCompute(img, None)
        if d is not None:
            all_descriptors.append(d)

    print("Progress: 100%")

    scripts.shared.preprocess(all_descriptors, filenames, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
