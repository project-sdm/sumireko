import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np

from shared.text.index import weight_postings
from shared.text.spimi import spimi_index_construction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess text files into a TF-IDF index.",
    )
    _ = parser.add_argument("texts_dir", type=Path, help="Folder containing .txt files")
    _ = parser.add_argument("--output-dir", type=Path, default=Path(".data/texts"))
    _ = parser.add_argument(
        "--language",
        choices=["english", "spanish", "multilingual"],
        default="english",
    )
    _ = parser.add_argument("--merge-m", type=int, default=10)
    _ = parser.add_argument("--max-memory", type=int, default=2**20)
    return parser.parse_args()


def main():
    args = parse_args()

    language = cast(str, args.language)
    texts_dir = cast(Path, args.texts_dir)
    output_dir = cast(Path, args.output_dir)
    merge_m = cast(int, args.merge_m)
    max_memory = cast(int, args.max_memory)

    doc_filenames = os.listdir(texts_dir)
    doc_paths = [texts_dir / filename for filename in doc_filenames]

    print(f"Found {len(doc_paths)} text files.")

    os.makedirs(output_dir, exist_ok=True)

    with TemporaryDirectory(dir=output_dir) as tmp_dir:
        print("Constructing inverted index with SPIMI...")
        final_block_path = spimi_index_construction(
            doc_paths,
            language,
            m=merge_m,
            out_dir=Path(tmp_dir),
            max_memory=max_memory,
        )

        dict_path = output_dir / "index.dict"
        postings_path = output_dir / "index.postings"

        os.rename(final_block_path.with_suffix(".dict"), dict_path)
        os.rename(final_block_path.with_suffix(".postings"), postings_path)

    print("Weighing index with TF-IDF and calculating document lengths...")
    lengths = weight_postings(dict_path, postings_path, len(doc_paths))

    with open(output_dir / "files.json", "w") as f:
        json.dump(doc_filenames, f)

    np.save(output_dir / "lengths.npy", lengths)
    print("Done.")


if __name__ == "__main__":
    main()
