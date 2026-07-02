import os
from dataclasses import dataclass
from pathlib import Path
from random import shuffle
from typing import override

import requests
from shared.types import MediaSearchMode, TextSearchMode
from shared.utils import load_env_or

BASE_URL = load_env_or("TEST_API_BASE_URL", "http://localhost:8000")


@dataclass
class TextBenchParams:
    query: str
    n_iters: int
    k: int
    language: str


@dataclass
class MediaBenchParams:
    media_dir: Path
    media_type: str
    n_iters: int
    n_files: int | None
    k: int


class SearchResult:
    results: list[str]
    time_ms: float

    def __init__(self, obj):
        self.results = obj["results"]
        self.time_ms = obj["time_ms"]

    @override
    def __str__(self) -> str:
        return f"{self.time_ms} ms\t- {self.results}"


def request_text(query: str, mode: str, k: int, language: str) -> SearchResult:
    req = requests.get(
        f"{BASE_URL}/text/search?q={query}&mode={mode}&k={k}&language={language}"
    )

    if not req.ok:
        raise Exception(f"Request failed: {req.json()}")

    obj = req.json()
    return SearchResult(obj)


def request_media(path: Path, media_type: str, mode: str, k: int) -> SearchResult:
    with open(path, "rb") as f:
        req = requests.post(
            f"{BASE_URL}/{media_type}/search?mode={mode}&k={k}",
            files={"file": f},
        )

    if not req.ok:
        raise Exception(f"Request failed: {req.json()}")

    obj = req.json()
    return SearchResult(obj)


def run_text_test(
    query: str,
    modes: list[str],
    k: int,
    language: str,
):
    sanitized_query = query.strip().replace(" ", "+")
    result = dict[str, float]()

    print(f"original query: {query}")
    print(f"sanitized query: {sanitized_query}")
    for mode in modes:
        res = request_text(sanitized_query, mode, k, language)
        result[mode] = res.time_ms

        print(f"[{mode}] {res}")

    return result


def run_media_test(
    media_dir: Path,
    media_type: str,
    modes: list[str],
    n_files: int | None,
    k: int,
) -> dict[str, float]:
    filenames = os.listdir(media_dir)
    shuffle(filenames)
    filenames = filenames[:n_files] if n_files is not None else filenames

    result = dict[str, float]()

    for filename in filenames:
        path = media_dir / filename

        print(f"file path: {path}")
        for mode in modes:
            res = request_media(path, media_type, mode, k)
            result[mode] = result.get(mode, 0.0) + res.time_ms

            print(f"[{mode}] {res}")

    n_files = n_files or len(filenames)
    return {k: (v / n_files) for k, v in result.items()}


def bench(
    params: TextBenchParams | MediaBenchParams,
) -> dict[str, float]:

    match params:
        case TextBenchParams():
            modes = [mode.value for mode in TextSearchMode]
        case MediaBenchParams():
            modes = [mode.value for mode in MediaSearchMode]

    bench_res = dict[str, float]()

    for _ in range(params.n_iters):
        match params:
            case TextBenchParams():
                res = run_text_test(params.query, modes, params.k, params.language)
            case MediaBenchParams():
                res = run_media_test(
                    params.media_dir, params.media_type, modes, params.n_files, params.k
                )

        for mode in modes:
            bench_res[mode] = bench_res.get(mode, 0.0) + res[mode]

    return {k: (v / params.n_iters) for k, v in bench_res.items()}
