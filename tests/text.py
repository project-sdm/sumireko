import argparse
from pprint import pprint
from typing import cast

from tests.lib import TextBenchParams, bench


def parse_args() -> TextBenchParams:
    parser = argparse.ArgumentParser()

    _ = parser.add_argument("query", type=str)
    _ = parser.add_argument("-i", "--n_iters", default=4, type=int)
    _ = parser.add_argument("-k", "--k", default=5, type=int)

    _ = parser.add_argument(
        "-l",
        "--language",
        choices=["english", "spanish", "multilingual"],
        default="english",
        type=str,
    )

    args = parser.parse_args()

    return TextBenchParams(
        cast(str, args.query),
        cast(int, args.n_iters),
        cast(int, args.k),
        cast(str, args.language),
    )


def main():
    params = parse_args()
    results = bench(params)

    pprint(results)


if __name__ == "__main__":
    main()
