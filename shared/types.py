from enum import Enum


class TextSearchMode(str, Enum):
    native = "native"
    pg = "pg"


class MediaSearchMode(str, Enum):
    native = "native"
    pg_brute = "pg-brute"
    pg_ivf = "pg-ivf"
    pg_hnsw = "pg-hnsw"
