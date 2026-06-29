from dataclasses import dataclass

import cv2
import faiss
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike
from psycopg_pool import ConnectionPool
from starlette.datastructures import State


@dataclass
class PreprocessedData:
    media_files: list[str]
    word_index: faiss.IndexFlatL2
    words: MatLike
    index: list[list[tuple[int, float]]]
    df: npt.NDArray[np.float32]
    lengths: npt.NDArray[np.float32]


@dataclass
class AppState(State):
    sift: cv2.SIFT
    image_data: PreprocessedData
    audio_data: PreprocessedData
    db: ConnectionPool
