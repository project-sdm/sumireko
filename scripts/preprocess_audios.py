import os
import sys
from pathlib import Path

import librosa
import numpy as np

import scripts.shared

PRE_EMPHASIS = 0.97
OUTPUT_DIR = Path(".data/audios")


def main():
    if len(sys.argv) < 2:
        raise Exception("must provide a path to the audios folder")

    audios_dir = Path(sys.argv[1])

    print(f"Reading '{audios_dir}'...")
    filenames = os.listdir(audios_dir)
    print(f"Found {len(filenames)} audios")

    print("Extracting features...")
    all_descriptors: list[np.ndarray] = []

    next_step = 0

    for i, filename in enumerate(filenames):
        path = audios_dir / filename
        progress = 100 * (i / len(filenames))

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

    scripts.shared.preprocess(all_descriptors, filenames, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
