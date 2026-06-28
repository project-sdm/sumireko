import json
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.audio.router import audio_router
from app.common.logger import APP_LOGGER
from app.common.state import PreprocessedData
from app.images.router import image_router

PREPROCESSED_DIR = Path(".data")


def load_image_preprocessed() -> PreprocessedData:
    APP_LOGGER.info("[IMAGE] Loading preprocessed data...")
    IMAGE_DIR = f"{PREPROCESSED_DIR}/images"

    words = np.load(f"{IMAGE_DIR}/words.npy")
    df = np.load(f"{IMAGE_DIR}/df.npy")
    lengths = np.load(f"{IMAGE_DIR}/lengths.npy")

    word_index: faiss.IndexFlatL2 = faiss.read_index(f"{IMAGE_DIR}/word_index.faiss")

    with open(f"{IMAGE_DIR}/media_files.json") as f:
        media_files = json.load(f)

    with open(f"{IMAGE_DIR}/index.json") as f:
        index = json.load(f)

    APP_LOGGER.info(
        f"[IMAGE] Loaded {len(media_files)} images, {len(words)} visual words."
    )

    return PreprocessedData(
        media_files=media_files,
        word_index=word_index,
        words=words,
        index=index,
        df=df,
        lengths=lengths,
    )


def load_audio_preprocessed() -> PreprocessedData:
    APP_LOGGER.info("[AUDIO] Loading preprocessed data...")
    AUDIO_DIR = f"{PREPROCESSED_DIR}/audios"

    words = np.load(f"{AUDIO_DIR}/words.npy")
    df = np.load(f"{AUDIO_DIR}/df.npy")
    lengths = np.load(f"{AUDIO_DIR}/lengths.npy")

    word_index: faiss.IndexFlatL2 = faiss.read_index(f"{AUDIO_DIR}/word_index.faiss")

    with open(f"{AUDIO_DIR}/media_files.json") as f:
        media_files = json.load(f)

    with open(f"{AUDIO_DIR}/index.json") as f:
        index = json.load(f)

    APP_LOGGER.info(
        f"[AUDIO] Loaded {len(media_files)} audios, {len(words)} visual words."
    )

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
    app.state.sift = cv2.SIFT.create()
    app.state.image_data = load_image_preprocessed()
    app.state.audio_data = load_audio_preprocessed()
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
