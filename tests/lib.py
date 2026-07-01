import os
from enum import Enum
from pathlib import Path
from typing import override

import requests
from shared.utils import load_env_or

BASE_URL = load_env_or("TEST_API_BASE_URL", "http://localhost:8000")


class SearchResult:
    results: list[str]
    time_ms: float

    def __init__(self, obj):
        self.results = obj["results"]
        self.time_ms = obj["time_ms"]

    @override
    def __str__(self) -> str:
        return f"{self.time_ms} ms\t- {self.results}"


class SearchMode(str, Enum):
    native = "native"
    pg_brute = "pg-brute"
    pg_ivf = "pg-ivf"
    pg_hnsw = "pg-hnsw"


def request(path: Path, media_type: str, mode: str, k: int) -> SearchResult:
    with open(path, "rb") as f:
        req = requests.post(
            f"{BASE_URL}/{media_type}/search?mode={mode}&k={k}",
            files={"file": f},
        )

    if not req.ok:
        raise Exception(f"Request failed: {req.json()}")

    obj = req.json()
    return SearchResult(obj)


def run_test(
    media_dir: Path, media_type: str, modes: list[str], n_files: int | None, k: int
) -> dict[str, float]:
    filenames = os.listdir(media_dir)

    result = dict[str, float]()

    for i, filename in enumerate(filenames):
        if n_files is not None and i == n_files:
            break

        path = media_dir / filename

        print(f"file path: {path}")
        for mode in modes:
            res = request(path, media_type, mode, k)
            result[mode] = result.get(mode, 0.0) + res.time_ms

            print(f"[{mode}] {res}")

    n_files = n_files or len(filenames)
    return {k: (v / n_files) for k, v in result.items()}


def bench(
    media_dir: Path,
    media_type: str,
    n_iters: int,
    n_files: int | None,
    k: int,
) -> dict[str, float]:

    modes = [mode.value for mode in SearchMode]
    bench_res = dict[str, float]()

    for _ in range(n_iters):
        res = run_test(media_dir, media_type, modes, n_files, k)

        for mode in modes:
            bench_res[mode] = bench_res.get(mode, 0.0) + res[mode]

    return {k: (v / n_iters) for k, v in bench_res.items()}
