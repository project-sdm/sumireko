import time
from dataclasses import dataclass

import numpy as np
from psycopg import Connection

import shared
from app.common.state import PreprocessedMediaData
from shared.types import MediaSearchMode


@dataclass
class KnnResult:
    results: list[str]
    time_ms: float


def knn(
    descriptors: np.ndarray,
    data: PreprocessedMediaData,
    k: int | None,
) -> KnnResult:
    start = time.perf_counter()

    _, labels = data.word_index.search(descriptors, 1)
    q_hist = np.bincount(labels.ravel(), minlength=len(data.words))

    n = len(data.media_files)

    scores: dict[int, float] = {}

    for word_id, tf_query in enumerate(q_hist):
        if tf_query == 0:
            continue

        w_query = shared.weight(n, tf_query, data.df[word_id])

        for media_id, w_media in data.index[word_id]:
            scores[media_id] = scores.get(media_id, 0) + w_media * w_query

    for media_id in scores:
        scores[media_id] /= data.lengths[media_id]

    result = sorted(scores.items(), key=lambda tup: tup[1], reverse=True)
    elapsed_ms = (time.perf_counter() - start) * 1000

    top_files = [data.media_files[i] for i, _ in result[:k]]
    return KnnResult(results=top_files, time_ms=round(elapsed_ms, 2))


def knn_postgres(
    conn: Connection,
    table_name: str,
    descriptors: np.ndarray,
    data: PreprocessedMediaData,
    k: int,
    mode: MediaSearchMode,
) -> KnnResult:
    assert mode != MediaSearchMode.native

    COLUMNS = {
        MediaSearchMode.pg_brute: "histogram_brute",
        MediaSearchMode.pg_ivf: "histogram_ivf",
        MediaSearchMode.pg_hnsw: "histogram_hnsw",
    }
    column = COLUMNS[mode]

    start = time.perf_counter()
    hist = compute_query_histogram(descriptors, data)

    with conn.cursor() as cur:
        _ = cur.execute(
            f"select filename from {table_name} order by {column} <=> %s limit %s",
            (str(hist), k),
        )
        rows = cur.fetchall()
        elapsed_ms = (time.perf_counter() - start) * 1000

        return KnnResult(
            [row[0] for row in rows],
            round(elapsed_ms, 2),
        )


def compute_query_histogram(
    descriptors: np.ndarray,
    data: PreprocessedMediaData,
) -> list[float]:
    _, labels = data.word_index.search(descriptors, 1)
    q_hist = np.bincount(labels.ravel(), minlength=len(data.words))

    n = len(data.media_files)

    weighted = np.zeros(len(data.words), dtype=np.float32)

    for word_id, tf in enumerate(q_hist):
        if tf == 0:
            continue

        weighted[word_id] = shared.weight(n, tf, data.df[word_id])

    return weighted.tolist()
