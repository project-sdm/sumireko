import math
import time
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import psycopg
from cv2.typing import MatLike
from fastapi import APIRouter, HTTPException, Request, UploadFile
from psycopg import sql

import shared.image
from app.common.algos import knn
from app.common.state import AppState, PreprocessedData

image_router = APIRouter(prefix="/images", tags=["images"])


async def read_file_as_img(file: UploadFile) -> MatLike:
    q_contents = await file.read()
    q_nparr = np.frombuffer(q_contents, np.uint8)

    q_img = cv2.imdecode(q_nparr, cv2.IMREAD_GRAYSCALE)
    if q_img is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    return q_img


def compute_query_histogram(
    descriptors: np.ndarray, data: PreprocessedData
) -> list[float]:
    _, labels = data.word_index.search(descriptors, 1)
    q_hist = np.bincount(labels.ravel(), minlength=len(data.words))

    n = len(data.media_files)

    def weight(word_id: int, tf: int) -> float:
        return math.log(1 + tf) * math.log((n + 1) / (data.df[word_id] + 1))

    weighted = np.zeros(len(data.words), dtype=np.float32)
    for word_id, tf in enumerate(q_hist):
        if tf == 0:
            continue
        weighted[word_id] = weight(word_id, tf)

    return weighted.tolist()


@image_router.post("/search")
async def image_search(req: Request, file: UploadFile, k: int | None = 5):
    state = cast(AppState, req.app.state)
    data = state.image_data

    q_img = shared.image.downscale(await read_file_as_img(file))
    _, q_desc = state.sift.detectAndCompute(q_img, None)

    top_files = knn(q_desc, data, k)
    return {"results": top_files}


@dataclass
class PgImageResult:
    results: list[str]
    time_ms: float


@image_router.post("/search-pgvector")
async def image_search_pgvector(req: Request, file: UploadFile, k: int | None = 5):
    state = cast(AppState, req.app.state)
    data = state.image_data

    q_img = shared.image.downscale(await read_file_as_img(file))
    _, q_desc = state.sift.detectAndCompute(q_img, None)
    q_hist = compute_query_histogram(q_desc, data)

    results: dict[str, PgImageResult] = {}

    with psycopg.connect() as conn:
        with conn.cursor() as cur:
            INDICES = [
                ("brute", "histogram_brute"),
                ("ivfflat", "histogram_ivf"),
                ("hnsw", "histogram_hnsw"),
            ]

            for label, column in INDICES:
                start = time.perf_counter()
                _ = cur.execute(
                    sql.SQL(
                        f"select filename from images order by {column} <=> %s LIMIT %s"
                    ),
                    (str(q_hist), k),
                )
                rows = cur.fetchall()
                elapsed_ms = (time.perf_counter() - start) * 1000

                results[label] = PgImageResult(
                    [row[0] for row in rows],
                    round(elapsed_ms, 2),
                )

    return results
