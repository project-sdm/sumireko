import struct
from dataclasses import dataclass
from io import SEEK_CUR, SEEK_END, SEEK_SET, BufferedRandom, BufferedReader
from pathlib import Path
from typing import ClassVar

import numpy as np

import shared
from shared.text.processing import MAX_TERM_LEN

type DocId = int
type Posting = tuple[DocId, int]
type PostingsList = list[Posting]
type Dictionary = dict[str, PostingsList]


@dataclass
class DictEntry:
    PACK_FMT: ClassVar[str] = f"{MAX_TERM_LEN + 1}sNN"
    PACK_SIZE: ClassVar[int] = struct.calcsize(PACK_FMT)

    term: str
    offset: int
    len: int

    def pack(self) -> bytes:
        return struct.pack(self.PACK_FMT, self.term.encode(), self.offset, self.len)

    @classmethod
    def unpack(cls, data: bytes):
        term, offset, len = struct.unpack(cls.PACK_FMT, data)
        return cls(term=term.decode().rstrip("\x00"), offset=offset, len=len)


@dataclass
class PostingsEntry:
    PACK_FMT: ClassVar[str] = "Nf"
    PACK_SIZE: ClassVar[int] = struct.calcsize(PACK_FMT)

    doc_id: DocId
    value: float

    def pack(self) -> bytes:
        return struct.pack(self.PACK_FMT, self.doc_id, self.value)

    @classmethod
    def unpack(cls, data: bytes):
        doc_id, value = struct.unpack(cls.PACK_FMT, data)  # pyright: ignore[reportAny]
        return cls(doc_id=doc_id, value=value)  # pyright: ignore[reportAny]


class DictReader:
    file: BufferedReader
    entry_buf: DictEntry | None = None

    def __init__(self, file: BufferedReader):
        self.file = file

    def calc_size(self) -> int:
        prev_pos = self.file.tell()
        _ = self.file.seek(0, SEEK_END)
        size = self.file.tell()
        _ = self.file.seek(prev_pos, SEEK_SET)
        return size // DictEntry.PACK_SIZE

    def set_index(self, idx: int):
        _ = self.file.seek(idx * DictEntry.PACK_SIZE)

    def next(self) -> DictEntry | None:
        data = self.file.read(DictEntry.PACK_SIZE)
        if not data:
            return None

        return DictEntry.unpack(data)


class PostingsReader:
    file: BufferedRandom
    postings_len: int
    cur: int = 0

    def __init__(self, file: BufferedRandom, postings_len: int, offset: int):
        self.file = file
        self.postings_len = postings_len

        _ = self.file.seek(offset)

    def next(self) -> PostingsEntry | None:
        if self.cur >= self.postings_len:
            return None

        data = self.file.read(PostingsEntry.PACK_SIZE)
        self.cur += 1

        return PostingsEntry.unpack(data)

    def write(self, entry: PostingsEntry):
        assert self.cur < self.postings_len

        _ = self.file.write(entry.pack())
        self.cur += 1

    def step_back(self):
        assert self.cur > 0

        _ = self.file.seek(-PostingsEntry.PACK_SIZE, SEEK_CUR)
        self.cur -= 1


def make_block_path(base: Path, level: int, n: int) -> Path:
    return base / f"block_{level:02}_{n:02}"


def weight_postings(dict_path: Path, postings_path: Path, n_docs: int):
    with (
        open(dict_path, "rb") as dict_file,
        open(postings_path, "r+b") as postings_file,
    ):
        dict_reader = DictReader(dict_file)

        lengths_sq = np.zeros(n_docs, dtype=np.float32)

        while dict_entry := dict_reader.next():
            postings = PostingsReader(postings_file, dict_entry.len, dict_entry.offset)

            df = dict_entry.len

            while posting := postings.next():
                w = shared.weight(n_docs, posting.value, df)
                lengths_sq[posting.doc_id] += w**2

                postings.step_back()
                postings.write(PostingsEntry(doc_id=posting.doc_id, value=w))

    return np.sqrt(lengths_sq)
