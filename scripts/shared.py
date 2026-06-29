import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

import shared


@dataclass
class ProgressMeter:
    step: float
    next_threshold: float = 0.0

    def record(self, progress: float):
        if progress >= self.next_threshold:
            print(f"Progress: {round(100 * progress, 2)}%")

            while progress >= self.next_threshold:
                self.next_threshold += self.step


def preprocess(
    all_descriptors: list[np.ndarray],
    filenames: list[str],
    output_dir: Path,
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
    hists: list[np.ndarray] = []

    for desc in all_descriptors:
        _, labels = word_index.search(desc, 1)
        hists.append(np.bincount(labels.ravel(), minlength=bow_len))

    print("Building inverted index...")

    df = np.zeros(bow_len, dtype=np.uint32)

    for hist in hists:
        for word_id, tf in enumerate(hist):
            if tf > 0:
                df[word_id] += 1

    n = len(filenames)
    index: list[list[tuple[int, float]]] = [[] for _ in range(bow_len)]
    lengths = np.zeros(n, dtype=np.float32)

    for audio_id, hist in enumerate(hists):
        for word_id, tf in enumerate(hist):
            if tf == 0:
                continue

            w = shared.weight(n, tf, df[word_id])
            lengths[audio_id] += w**2
            index[word_id].append((audio_id, w))

    for audio_id in range(n):
        lengths[audio_id] = math.sqrt(lengths[audio_id])

    print("Computing TF-IDF weighted histograms...")
    weighted_hists = np.zeros((n, bow_len), dtype=np.float32)

    for img_id, hist in enumerate(hists):
        for word_id, tf in enumerate(hist):
            if tf == 0:
                continue

            weighted_hists[img_id, word_id] = shared.weight(n, tf, df[word_id])

    print("Saving...")
    os.makedirs(output_dir, exist_ok=True)

    np.save(output_dir / "words.npy", words)
    np.save(output_dir / "df.npy", df)
    np.save(output_dir / "lengths.npy", lengths)
    np.save(output_dir / "histograms.npy", weighted_hists)

    faiss.write_index(word_index, str(output_dir / "word_index.faiss"))

    with open(output_dir / "media_files.json", "w") as f:
        json.dump(filenames, f)

    with open(output_dir / "index.json", "w") as f:
        json.dump(index, f)

    print("Done.")
