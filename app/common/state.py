import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import cast

import cv2
import faiss
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike
from faiss import IndexFlatL2
from psycopg_pool import ConnectionPool
from starlette.datastructures import State

from app.common.logger import APP_LOGGER


@dataclass
class PreprocessedTextData:
    files: list[str]
    dict_path: Path
    postings_path: Path
    lengths: npt.NDArray[np.float32]


@dataclass
class PreprocessedMediaData:
    media_files: list[str]
    word_index: faiss.IndexFlatL2
    words: MatLike
    index: list[list[tuple[int, float]]]
    df: npt.NDArray[np.float32]
    lengths: npt.NDArray[np.float32]


@dataclass
class AppState(State):
    sift: cv2.SIFT
    text_data: PreprocessedTextData
    image_data: PreprocessedMediaData
    audio_data: PreprocessedMediaData
    db: ConnectionPool


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

    with open(data_dir / "files.json") as f:
        files = cast(list[str], json.load(f))

    lengths = cast(np.ndarray, np.load(data_dir / "lengths.npy"))

    APP_LOGGER.info(f"[{label}] Loaded {len(files)} documents.")

    return PreprocessedTextData(
        files=files,
        dict_path=data_dir / "index.dict",
        postings_path=data_dir / "index.postings",
        lengths=lengths,
    )


def load_media_preprocessed(kind: DataType, data_dir: Path) -> PreprocessedMediaData:
    label = kind.value
    APP_LOGGER.info(f"[{label}] Loading preprocessed data...")

    words = cast(MatLike, np.load(data_dir / "words.npy"))
    df = cast(np.ndarray, np.load(data_dir / "df.npy"))
    lengths = cast(np.ndarray, np.load(data_dir / "lengths.npy"))

    word_index = cast(IndexFlatL2, faiss.read_index(str(data_dir / "word_index.faiss")))

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


def load_app_state(TEXTS_PATH: Path, IMAGES_PATH: Path, AUDIOS_PATH: Path) -> State:
    st = State()

    st.text_data = load_preprocessed(DataType.TEXT, TEXTS_PATH)
    st.image_data = load_preprocessed(DataType.IMAGE, IMAGES_PATH)
    st.audio_data = load_preprocessed(DataType.AUDIO, AUDIOS_PATH)

    st.sift = cv2.SIFT.create()
    st.db = ConnectionPool(min_size=2, max_size=20)

    return st
