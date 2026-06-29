import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer

TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9]+")


@dataclass(frozen=True)
class TextChunk:
    source: str
    ordinal: int
    text: str

    @property
    def identifier(self) -> str:
        return f"{self.source}#chunk_{self.ordinal}"


def split_paragraphs(content: str, min_chars: int = 1) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", content)]
    return [paragraph for paragraph in paragraphs if len(paragraph) >= min_chars]


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def iter_text_chunks(paths: list[Path], min_chars: int = 1) -> list[TextChunk]:
    chunks: list[TextChunk] = []

    for path in paths:
        paragraphs = split_paragraphs(read_text_file(path), min_chars=min_chars)

        for ordinal, paragraph in enumerate(paragraphs):
            chunks.append(TextChunk(path.name, ordinal, paragraph))

    return chunks


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


def _build_stemmer(language: str):
    return SnowballStemmer(_stemmer_language(language))
