import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import cv2
import faiss
import numpy as np
from cv2.typing import MatLike
from faiss import IndexFlatL2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from psycopg_pool import ConnectionPool

import app.postgres as postgres
from app.audio.router import audio_router
from app.common.logger import APP_LOGGER
from app.common.state import PreprocessedData
from app.images.router import image_router


def load_preprocessed(label: str, data_dir: Path) -> PreprocessedData:
    APP_LOGGER.info(f"[{label}] Loading preprocessed data...")

    words = cast(MatLike, np.load(data_dir / "words.npy"))
    df = cast(np.ndarray, np.load(data_dir / "df.npy"))
    lengths = cast(np.ndarray, np.load(data_dir / "lengths.npy"))

    word_index: IndexFlatL2 = cast(
        IndexFlatL2, faiss.read_index(str(data_dir / "word_index.faiss"))
    )

    with open(data_dir / "media_files.json") as f:
        media_files = cast(list[str], json.load(f))

    with open(data_dir / "index.json") as f:
        index = cast(list[list[tuple[int, float]]], json.load(f))

    APP_LOGGER.info(f"[{label}] Loaded {len(media_files)} items, {len(words)} words.")

    return PreprocessedData(
        media_files=media_files,
        word_index=word_index,
        words=words,
        index=index,
        df=df,
        lengths=lengths,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    IMAGES_PATH = Path(".data/images")
    AUDIOS_PATH = Path(".data/audios")

    postgres.init(IMAGES_PATH, "images")
    postgres.init(AUDIOS_PATH, "audios")

    app.state.sift = cv2.SIFT.create()
    app.state.image_data = load_preprocessed("IMAGE", IMAGES_PATH)
    app.state.audio_data = load_preprocessed("AUDIO", AUDIOS_PATH)

    app.state.db = ConnectionPool(min_size=2, max_size=20)

    yield


app = FastAPI(lifespan=lifespan)

app.mount(
    "/media/images",
    StaticFiles(directory="media/images"),
    name="images",
)

app.mount(
    "/media/audios",
    StaticFiles(directory="media/audios"),
    name="audios",
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


app.include_router(image_router)
app.include_router(audio_router)
