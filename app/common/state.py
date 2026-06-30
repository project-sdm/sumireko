from dataclasses import dataclass

import cv2
import faiss
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike
from psycopg_pool import ConnectionPool
from starlette.datastructures import State


@dataclass
class PreprocessedTextData:
    files: list[str]
    index: dict[str, list[tuple[int, float]]]
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
