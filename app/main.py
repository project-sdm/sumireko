import logging
import math
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

import cv2
import faiss
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike
from fastapi import FastAPI, HTTPException, Request, UploadFile

IMAGES_DIR = "fashion-dataset/images"
VBOW_LEN = 500
KMEANS_ITER = 100


type Point = npt.NDArray[np.float32]


def histogram(index: faiss.IndexFlatL2, centroids: MatLike, descriptors: MatLike):
    _, labels = index.search(descriptors, 1)
    return np.bincount(labels.ravel(), minlength=len(centroids))


@dataclass
class PreprocessedData:
    sift: cv2.SIFT
    image_files: list[str]
    word_index: faiss.IndexFlatL2
    centroids: MatLike
    index: list[list[tuple[int, float]]]
    df: np.ndarray
    lengths: np.ndarray


def unwrap[T](val: T | None) -> T:
    assert val is not None
    return val


logger = logging.getLogger("uvicorn.error")


def preprocess() -> PreprocessedData:
    sift = cv2.SIFT.create()

    image_files = os.listdir(IMAGES_DIR)[:100]
    logger.info(f"[IMAGE] Listed {len(image_files)} images")

    logger.info("[IMAGE] Reading dataset images...")
    imgs = [unwrap(cv2.imread(f"{IMAGES_DIR}/{file}")) for file in image_files]

    n = len(image_files)

    logger.info("[IMAGE] Extracting features...")
    data = [sift.detectAndCompute(img, None) for img in imgs]

    points = np.vstack([desc for _, desc in data]).astype(np.float32)

    logger.info(f"[IMAGE] Clustering... (k = {VBOW_LEN}, niter = {KMEANS_ITER})")
    kmeans = faiss.Kmeans(
        points.shape[1], VBOW_LEN, niter=KMEANS_ITER, gpu=True, verbose=True
    )
    kmeans.train(points)
    centroids = kmeans.centroids
    assert centroids is not None

    word_index = faiss.IndexFlatL2(centroids.shape[1])
    word_index.add(centroids)

    logger.info("[IMAGE] Computing histograms...")
    hists = [histogram(word_index, centroids, desc) for _, desc in data]

    logger.info("[IMAGE] Building inverted index...")

    df = np.zeros(VBOW_LEN)

    for hist in hists:
        for word_id, tf in enumerate(hist):
            if tf > 0:
                df[word_id] += 1

    index: list[list[tuple[int, float]]] = [[] for _ in range(VBOW_LEN)]
    lengths = np.zeros(n)

    def weight(word_id: int, tf: int) -> float:
        return math.log(1 + tf) * math.log((n + 1) / (df[word_id] + 1))

    for img_id, hist in enumerate(hists):
        for word_id, tf in enumerate(hist):
            if tf == 0:
                continue

            w = weight(word_id, tf)
            lengths[img_id] += w**2
            index[word_id].append((img_id, w))

    for img_id in range(n):
        lengths[img_id] = math.sqrt(lengths[img_id])

    logger.info("[IMAGE] Done.")

    return PreprocessedData(
        sift=sift,
        image_files=image_files,
        word_index=word_index,
        centroids=centroids,
        index=index,
        df=df,
        lengths=lengths,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.data = preprocess()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/image-search")
async def sift(req: Request, file: UploadFile):
    data: PreprocessedData = req.app.state.data

    q_contents = await file.read()
    q_nparr = np.frombuffer(q_contents, np.uint8)

    q_img = cv2.imdecode(q_nparr, cv2.IMREAD_GRAYSCALE)
    if q_img is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    _, q_desc = data.sift.detectAndCompute(q_img, None)
    q_hist = histogram(data.word_index, data.centroids, q_desc)

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

    _, axes = plt.subplots(1, 6, figsize=(18, 4))

    q_img_color = cv2.imdecode(q_nparr, cv2.IMREAD_COLOR)
    axes[0].imshow(q_img_color)
    axes[0].set_title("Query")
    axes[0].axis("off")

    for i, filename in enumerate(top_files):
        img = cv2.imread(f"{IMAGES_DIR}/{filename}", cv2.IMREAD_COLOR_RGB)

        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"Top {i + 1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()

    return {"status": "ok"}
