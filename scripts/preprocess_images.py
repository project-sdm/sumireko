import os
import sys
from pathlib import Path

import cv2
from cv2.typing import MatLike

import scripts.shared
import shared.image
from scripts.shared import ProgressMeter

OUTPUT_DIR = Path(".data/images")


def main():
    if len(sys.argv) < 2:
        raise Exception("must provide a path to the images folder")

    images_dir = Path(sys.argv[1])

    print(f"Reading '{images_dir}'...")
    filenames = os.listdir(images_dir)
    print(f"Found {len(filenames)} images")

    print("Extracting features...")
    all_descriptors: list[MatLike] = []

    sift = cv2.SIFT.create()
    meter = ProgressMeter(0.0001)

    for i, filename in enumerate(filenames):
        meter.record(i / len(filenames))

        path = images_dir / filename

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert img is not None, f"Failed to load {path}"
        img = shared.image.downscale(img)

        _, d = sift.detectAndCompute(img, None)
        if d is not None:
            all_descriptors.append(d)

    meter.record(1)

    scripts.shared.preprocess(all_descriptors, filenames, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
