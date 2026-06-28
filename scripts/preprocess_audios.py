import json
import math
import os
import sys

import faiss
import librosa
import numpy as np

OUTPUT_DIR = ".data/audios"
ABOW_LEN = 1000
KMEANS_ITER = 100
PRE_EMPHASIS = 0.97


def main():
    if len(sys.argv) < 2:
        raise Exception("must provide a path to the audios folder")

    audios_dir = sys.argv[1]
    print(f"Reading '{audios_dir}'...")

    paths = [f"{audios_dir}/{filename}" for filename in os.listdir(audios_dir)]
    print(f"Found {len(paths)} audios")

    print("Extracting features...")
    all_descriptors: list[np.ndarray] = []

    next_step = 0

    for i, path in enumerate(paths):
        progress = 100 * (i / len(paths))

        if progress >= next_step:
            print(f"Progress: {round(progress, 2)}%")

            while progress >= next_step:
                next_step += 0.01

        try:
            audio, sr = librosa.load(path, sr=None)
        except Exception:
            print(f"failed to load {path}, skipping...", file=sys.stderr)
            continue

        audio = np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])

        d = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13,
            n_fft=int(0.025 * sr),
            win_length=int(0.025 * sr),
            hop_length=int(0.010 * sr),
            window="hamming",
            center=False,
        ).T

        if len(d) > 0:
            all_descriptors.append(d)

    print("Progress: 100%")

    points = np.vstack(all_descriptors).astype(np.float32)

    print(f"Clustering... (k = {ABOW_LEN}, niter = {KMEANS_ITER})")
    kmeans = faiss.Kmeans(
        points.shape[1], ABOW_LEN, niter=KMEANS_ITER, gpu=True, verbose=True
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
        hists.append(np.bincount(labels.ravel(), minlength=ABOW_LEN))

    print("Building inverted index...")

    df = np.zeros(ABOW_LEN)

    for hist in hists:
        for word_id, tf in enumerate(hist):
            if tf > 0:
                df[word_id] += 1

    n = len(paths)
    index: list[list[tuple[int, float]]] = [[] for _ in range(ABOW_LEN)]
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(f"{OUTPUT_DIR}/words.npy", words)
    np.save(f"{OUTPUT_DIR}/df.npy", df)
    np.save(f"{OUTPUT_DIR}/lengths.npy", lengths)

    faiss.write_index(word_index, f"{OUTPUT_DIR}/word_index.faiss")

    with open(f"{OUTPUT_DIR}/media_files.json", "w") as f:
        json.dump(paths, f)

    with open(f"{OUTPUT_DIR}/index.json", "w") as f:
        json.dump(index, f)

    print("Done.")


if __name__ == "__main__":
    main()
