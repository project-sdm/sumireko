import json
import logging
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import faiss
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import State

import common.image

PREPROCESSED_DIR = Path(".data/images")


@dataclass
class AppState(State):
    data: PreprocessedData


@dataclass
class PreprocessedData:
    sift: cv2.SIFT
    image_files: list[str]
    word_index: faiss.IndexFlatL2
    words: MatLike
    index: list[list[tuple[int, float]]]
    df: npt.NDArray[np.float32]
    lengths: npt.NDArray[np.float32]


logger = logging.getLogger("uvicorn.error")


def load_preprocessed() -> PreprocessedData:
    logger.info("[IMAGE] Loading preprocessed data...")

    words = np.load(f"{PREPROCESSED_DIR}/words.npy")
    df = np.load(f"{PREPROCESSED_DIR}/df.npy")
    lengths = np.load(f"{PREPROCESSED_DIR}/lengths.npy")

    word_index: faiss.IndexFlatL2 = faiss.read_index(
        f"{PREPROCESSED_DIR}/word_index.faiss"
    )

    with open(f"{PREPROCESSED_DIR}/image_files.json") as f:
        image_files = json.load(f)

    with open(f"{PREPROCESSED_DIR}/index.json") as f:
        index = json.load(f)

    logger.info(f"[IMAGE] Loaded {len(image_files)} images, {len(words)} visual words.")

    return PreprocessedData(
        sift=cv2.SIFT.create(),
        image_files=image_files,
        word_index=word_index,
        words=words,
        index=index,
        df=df,
        lengths=lengths,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.data = load_preprocessed()
    yield


app = FastAPI(lifespan=lifespan)

app.mount(
    "/fashion-dataset/images",
    StaticFiles(directory="fashion-dataset/images"),
    name="static",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/image-search")
async def image_search(req: Request, file: UploadFile):
    state = cast(AppState, req.app.state)
    data = state.data

    q_contents = await file.read()
    q_nparr = np.frombuffer(q_contents, np.uint8)

    q_img = cv2.imdecode(q_nparr, cv2.IMREAD_GRAYSCALE)
    if q_img is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    q_img = common.image.downscale(q_img)

    _, q_desc = data.sift.detectAndCompute(q_img, None)

    _, labels = data.word_index.search(q_desc, 1)
    q_hist = np.bincount(labels.ravel(), minlength=len(data.words))

    n = len(data.image_files)

    def weight(word_id: int, tf: int) -> float:
        return math.log(1 + tf) * math.log((n + 1) / (data.df[word_id] + 1))

    scores: dict[int, float] = {}
    query_len_sq = 0.0

    for word_id, tf_query in enumerate(q_hist):
        if tf_query == 0:
            continue

        w_query = weight(word_id, tf_query)
        query_len_sq += w_query**2

        for img_id, w_img in data.index[word_id]:
            scores[img_id] = scores.get(img_id, 0) + w_img * w_query

    query_length = math.sqrt(query_len_sq)

    for img_id in scores:
        scores[img_id] /= data.lengths[img_id] * query_length

    result = sorted(scores.items(), key=lambda tup: tup[1], reverse=True)
    top_files = [data.image_files[i] for i, _ in result[:5]]

    return {"results": [f"{path}" for path in top_files]}
