import functools
import re
import string
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer

TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9]+")


@dataclass(frozen=True)
class TextDocument:
    source: str
    text: str

    @property
    def identifier(self) -> str:
        return self.source


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def yield_text_documents(paths: list[Path]) -> Iterator[TextDocument]:
    for path in paths:
        yield TextDocument(path.name, read_text_file(path))


def iter_text_documents(paths: list[Path]) -> list[TextDocument]:
    return list(yield_text_documents(paths))


def get_stopwords(language: str) -> set[str]:
    if language == "spanish":
        return _library_stopwords("spanish")

    if language == "english":
        return _library_stopwords("english")

    if language == "multilingual":
        return _library_stopwords("spanish") | _library_stopwords("english")

    raise ValueError(f"unsupported language: {language}")


def normalize_tokens(
    text: str,
    language: str = "spanish",
    min_token_len: int = 2,
    use_stemming: bool = True,
) -> list[str]:
    stopwords = get_stopwords(language)
    stemmer = _build_stemmer(language) if use_stemming else None

    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip(string.punctuation)

        if len(token) < min_token_len:
            continue

        if token in stopwords:
            continue

        if stemmer is not None:
            token = stemmer.stem(token)

        tokens.append(token)

    return tokens


def collection_term_counts(
    chunks: list[TextChunk],
    language: str = "spanish",
    min_token_len: int = 2,
    use_stemming: bool = True,
) -> Counter[str]:
    counts: Counter[str] = Counter()

    for chunk in chunks:
        counts.update(
            normalize_tokens(
                chunk.text,
                language=language,
                min_token_len=min_token_len,
                use_stemming=use_stemming,
            )
        )

    return counts


@functools.lru_cache
def _library_stopwords(language: str) -> set[str]:
    try:
        return set(nltk_stopwords.words(language))
    except LookupError as exc:
        raise LookupError(
            "NLTK stopwords corpus is required. Install it with: "
            "python -m nltk.downloader stopwords"
        ) from exc


def _stemmer_language(language: str) -> str:
    if language == "english":
        return "english"

    return "spanish"


@functools.lru_cache
def _build_stemmer(language: str) -> SnowballStemmer:
    return SnowballStemmer(_stemmer_language(language))
