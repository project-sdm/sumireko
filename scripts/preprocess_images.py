import json
import math
import os
import sys

import cv2
import faiss
import numpy as np
from cv2.typing import MatLike

import shared.image

OUTPUT_DIR = ".data/images"
VBOW_LEN = 1000
KMEANS_ITER = 100


def main():
    if len(sys.argv) < 2:
        raise Exception("must provide a path to the images folder")

    images_dir = sys.argv[1]
    print(f"Reading '{images_dir}'...")

    sift = cv2.SIFT.create()

    filenames = os.listdir(images_dir)[:10]
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

    points = np.vstack(all_descriptors).astype(np.float32)

    print(f"Clustering... (k = {VBOW_LEN}, niter = {KMEANS_ITER})")
    kmeans = faiss.Kmeans(
        points.shape[1], VBOW_LEN, niter=KMEANS_ITER, gpu=True, verbose=True
    )
    kmeans.train(points)
    words = kmeans.centroids
    assert words is not None

    word_index = faiss.IndexFlatL2(words.shape[1])
    word_index.add(words)

    print("Computing histograms...")
    hists = []
    for desc in all_descriptors:
        _, labels = word_index.search(desc, 1)
        hists.append(np.bincount(labels.ravel(), minlength=VBOW_LEN))

    print("Building inverted index...")

    df = np.zeros(VBOW_LEN)

    for hist in hists:
        for word_id, tf in enumerate(hist):
            if tf > 0:
                df[word_id] += 1

    n = len(paths)
    index: list[list[tuple[int, float]]] = [[] for _ in range(VBOW_LEN)]
    lengths = np.zeros(n)

    def weight(word_id: int, tf: int) -> float:
        return math.log(1 + tf) * math.log((n + 1) / (df[word_id] + 1))

    for img_id, hist in enumerate(hists):
        for word_id, tf in enumerate(hist):
            if tf == 0:
                continue
            w = weight(word_id, tf)
            lengths[img_id] += w**2
            index[word_id].append((img_id, w))

    for img_id in range(n):
        lengths[img_id] = math.sqrt(lengths[img_id])

    print("Saving...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(f"{OUTPUT_DIR}/words.npy", words)
    np.save(f"{OUTPUT_DIR}/df.npy", df)
    np.save(f"{OUTPUT_DIR}/lengths.npy", lengths)

    faiss.write_index(word_index, f"{OUTPUT_DIR}/word_index.faiss")

    with open(f"{OUTPUT_DIR}/media_files.json", "w") as f:
        json.dump(filenames, f)

    with open(f"{OUTPUT_DIR}/index.json", "w") as f:
        json.dump(index, f)

    print("Done.")


if __name__ == "__main__":
    main()
