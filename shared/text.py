import functools
import re
import struct
from collections import deque
from dataclasses import dataclass
from io import SEEK_CUR, SEEK_END, SEEK_SET, BufferedRandom, BufferedReader
from pathlib import Path
from typing import ClassVar, cast

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer

MAX_TERM_LEN = 23
TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9]+")

type DocId = int
type Posting = tuple[DocId, int]
type PostingsList = list[Posting]
type Dictionary = dict[str, PostingsList]


@dataclass
class Token:
    doc_id: int
    term: str


class TokenStream:
    doc_paths: list[Path]
    language: str
    stopwords: set[str]
    next_doc: int = 0
    cur_terms: deque[str] = deque()

    def __init__(self, doc_paths: list[Path], language: str):
        self.doc_paths = doc_paths
        self.language = language
        self.stopwords = _library_stopwords(language)

    def done(self) -> bool:
        return not self.cur_terms and self.next_doc >= len(self.doc_paths)

    def next(self) -> Token | None:
        term = self.cur_terms.popleft() if self.cur_terms else None

        while term is None:
            if self.next_doc >= len(self.doc_paths):
                return None

            text = self.doc_paths[self.next_doc].read_text()
            self.next_doc += 1

            self.cur_terms = deque(tokenize_text(text, language=self.language))
            term = self.cur_terms.popleft() if self.cur_terms else None

        assert len(term) <= MAX_TERM_LEN
        return Token(doc_id=self.next_doc - 1, term=term)


def get_stopwords(language: str) -> set[str]:
    if language == "spanish":
        return _library_stopwords("spanish")

    if language == "english":
        return _library_stopwords("english")

    if language == "multilingual":
        return _library_stopwords("spanish") | _library_stopwords("english")

    raise ValueError(f"unsupported language: {language}")


def tokenize_text(
    text: str,
    language: str = "spanish",
) -> list[str]:
    stopwords = get_stopwords(language)
    stemmer = _build_stemmer(language)

    tokens: list[str] = []

    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0)

        if token not in stopwords:
            token = cast(str, stemmer.stem(token))
            tokens.append(token)

    return tokens


@functools.lru_cache
def _library_stopwords(language: str) -> set[str]:
    try:
        return set(nltk_stopwords.words(language))
    except LookupError as ex:
        raise LookupError(
            "NLTK stopwords corpus is required. Install it with: "
            + "python -m nltk.downloader stopwords"
        ) from ex


@functools.lru_cache
def _build_stemmer(language: str) -> SnowballStemmer:
    return SnowballStemmer(language)


# TODO: move from here onwards to somewhere else


@dataclass
class DictEntry:
    PACK_FMT: ClassVar[str] = f"{MAX_TERM_LEN + 1}sii"
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


@dataclass
class PostingsEntry:
    PACK_FMT: ClassVar[str] = "if"
    PACK_SIZE: ClassVar[int] = struct.calcsize(PACK_FMT)

    doc_id: DocId
    tf: float

    def pack(self) -> bytes:
        return struct.pack(self.PACK_FMT, self.doc_id, self.tf)

    @classmethod
    def unpack(cls, data: bytes):
        doc_id, tf = struct.unpack(cls.PACK_FMT, data)  # pyright: ignore[reportAny]
        return cls(doc_id=doc_id, tf=tf)  # pyright: ignore[reportAny]


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
