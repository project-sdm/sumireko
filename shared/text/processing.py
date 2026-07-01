import functools
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer

TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9]+")
MAX_TERM_LEN = 23


@dataclass
class Token:
    doc_id: int
    term: str


class TokenStream:
    doc_paths: list[Path]
    language: str
    stopwords: set[str]
    next_doc: int
    cur_terms: deque[str]

    def __init__(self, doc_paths: list[Path], language: str):
        self.doc_paths = doc_paths
        self.language = language
        self.stopwords = _library_stopwords(language)
        self.next_doc = 0
        self.cur_terms = deque()

    def done(self) -> bool:
        return not self.cur_terms and self.next_doc >= len(self.doc_paths)

    def next(self) -> Token | None:
        term = self.cur_terms.popleft() if self.cur_terms else None

        while term is None:
            if self.next_doc >= len(self.doc_paths):
                return None

            text = self.doc_paths[self.next_doc].read_text(encoding="utf-8")
            self.next_doc += 1

            self.cur_terms = deque(tokenize_text(text, language=self.language))
            term = self.cur_terms.popleft() if self.cur_terms else None

        term_bytes = term.encode("utf-8")

        if len(term_bytes) > MAX_TERM_LEN:
            term_trunc = term_bytes[:MAX_TERM_LEN].decode("utf-8", errors="ignore")
            print(f"[WARNING] Truncating {term} to {term_trunc}")
            term = term_trunc

        return Token(doc_id=self.next_doc - 1, term=term)

    def next_batch(self, n: int) -> tuple[list[int], list[str]]:
        doc_ids: list[int] = []
        terms: list[str] = []

        for _ in range(n):
            token = self.next()
            if token is None:
                break
            doc_ids.append(token.doc_id)
            terms.append(token.term)

        return doc_ids, terms


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
    language: str = "english",
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


def parse_docs(paths: list[Path], language: str) -> TokenStream:
    return TokenStream(paths, language)
