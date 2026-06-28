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

    sift = cv2.SIFT.create()

    filenames = os.listdir(images_dir)
    paths = [f"{images_dir}/{filename}" for filename in filenames]
    print(f"Found {len(paths)} images")

    print("Extracting features...")
    all_descriptors: list[MatLike] = []

    next_step = 0

    for i, path in enumerate(paths):
        progress = 100 * (i / len(paths))

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

    scripts.shared.preprocess(all_descriptors, filenames, output_dir=".data/images")


if __name__ == "__main__":
    main()
