import os
import sys
from pathlib import Path

import librosa
import numpy as np

import scripts.shared
from scripts.shared import ProgressMeter

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

    meter = ProgressMeter(0.0001)

    for i, filename in enumerate(filenames):
        meter.record(i / len(filenames))

        path = audios_dir / filename

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

    meter.record(1)

    scripts.shared.preprocess(all_descriptors, filenames, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
