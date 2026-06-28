import json
import math
import os

import faiss
import numpy as np


def preprocess(
    all_descriptors: list[np.ndarray],
    filenames: list[str],
    output_dir: str,
    bow_len: int = 1000,
    kmeans_iter: int = 100,
):
    points = np.vstack(all_descriptors).astype(np.float32)

    print(f"Clustering... (k = {bow_len}, niter = {kmeans_iter})")
    kmeans = faiss.Kmeans(
        points.shape[1], bow_len, niter=kmeans_iter, gpu=True, verbose=True
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
        hists.append(np.bincount(labels.ravel(), minlength=bow_len))

    print("Building inverted index...")

    df = np.zeros(bow_len)

    for hist in hists:
        for word_id, tf in enumerate(hist):
            if tf > 0:
                df[word_id] += 1

    n = len(filenames)
    index: list[list[tuple[int, float]]] = [[] for _ in range(bow_len)]
    lengths = np.zeros(n)

    def weight(word_id: int, tf: int) -> float:
        return math.log(1 + tf) * math.log((n + 1) / (df[word_id] + 1))

    for audio_id, hist in enumerate(hists):
        for word_id, tf in enumerate(hist):
            if tf == 0:
                continue
            w = weight(word_id, tf)
            lengths[audio_id] += w**2
            index[word_id].append((audio_id, w))

    for audio_id in range(n):
        lengths[audio_id] = math.sqrt(lengths[audio_id])

    print("Saving...")
    os.makedirs(output_dir, exist_ok=True)

    np.save(f"{output_dir}/words.npy", words)
    np.save(f"{output_dir}/df.npy", df)
    np.save(f"{output_dir}/lengths.npy", lengths)

    faiss.write_index(word_index, f"{output_dir}/word_index.faiss")

    with open(f"{output_dir}/media_files.json", "w") as f:
        json.dump(filenames, f)

    with open(f"{output_dir}/index.json", "w") as f:
        json.dump(index, f)

    print("Done.")
