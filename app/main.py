import json
import struct
from contextlib import asynccontextmanager
from enum import Enum
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
from app.common.state import PreprocessedMediaData, PreprocessedTextData
from app.images.router import image_router
from shared.text import DictEntry, DocId, PostingsEntry


class DataType(str, Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"


def load_preprocessed(
    kind: DataType, data_dir: Path
) -> PreprocessedMediaData | PreprocessedTextData:
    match kind:
        case DataType.TEXT:
            return load_text_preprocessed(kind, data_dir)
        case DataType.IMAGE | DataType.AUDIO:
            return load_media_preprocessed(kind, data_dir)


def load_text_preprocessed(kind: DataType, data_dir: Path) -> PreprocessedTextData:
    label = kind.value
    APP_LOGGER.info(f"[{label}] Loading preprocessed data...")

    lengths = cast(np.ndarray, np.load(data_dir / "lengths.npy"))

    with open(data_dir / "files.json") as f:
        files = cast(list[str], json.load(f))

    index = dict[str, list[tuple[int, float]]]()

    with (
        open(data_dir / "index.dict", "rb") as dict_file,
        open(data_dir / "index.postings", "rb") as postings_file,
    ):
        while raw := dict_file.read(DictEntry.PACK_SIZE):
            term_bytes, offset, length = tuple[bytes, int, int](
                struct.unpack(DictEntry.PACK_FMT, raw)
            )
            term = term_bytes.decode().rstrip("\x00")

            _ = postings_file.seek(offset)

            postings: list[tuple[int, float]] = []
            for _ in range(length):
                post_raw = postings_file.read(PostingsEntry.PACK_SIZE)
                doc_id, tf_weight = tuple[DocId, float](
                    struct.unpack(PostingsEntry.PACK_FMT, post_raw)
                )
                postings.append((doc_id, tf_weight))

            index[term] = postings

    APP_LOGGER.info(f"[{label}] Loaded {len(files)} documents, {len(index)} terms.")

    return PreprocessedTextData(files=files, index=index, lengths=lengths)


def load_media_preprocessed(kind: DataType, data_dir: Path) -> PreprocessedMediaData:
    label = kind.value
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

    return PreprocessedMediaData(
        media_files=media_files,
        word_index=word_index,
        words=words,
        index=index,
        df=df,
        lengths=lengths,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    TEXTS_PATH = Path(".data/texts")
    IMAGES_PATH = Path(".data/images")
    AUDIOS_PATH = Path(".data/audios")

    postgres.init(IMAGES_PATH, "images")
    postgres.init(AUDIOS_PATH, "audios")

    app.state.text_data = load_preprocessed(DataType.TEXT, TEXTS_PATH)
    app.state.image_data = load_preprocessed(DataType.IMAGE, IMAGES_PATH)
    app.state.audio_data = load_preprocessed(DataType.AUDIO, AUDIOS_PATH)

    app.state.sift = cv2.SIFT.create()
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
