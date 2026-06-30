import argparse
from pathlib import Path
from pprint import pprint
from typing import cast

from tests.lib import bench


def parse_args() -> tuple[Path, str, int, int | None, int]:
    parser = argparse.ArgumentParser()

    _ = parser.add_argument("media_dir", type=str)
    _ = parser.add_argument("media_type", choices=["images", "audio"], type=str)
    _ = parser.add_argument("-i", "--n_iters", default=4, type=int)
    _ = parser.add_argument("-n", "--n_files", default=None, type=int)
    _ = parser.add_argument("-k", "--k", default=5, type=int)

    args = parser.parse_args()

    return (
        Path(cast(Path, args.media_dir)),
        cast(str, args.media_type),
        cast(int, args.n_iters),
        cast(int | None, args.n_files),
        cast(int, args.k),
    )


def main():
    media_dir, media_type, n_iters, n_files, k = parse_args()

    results = bench(media_dir, media_type, n_iters, n_files, k)
    pprint(results)


if __name__ == "__main__":
    main()
