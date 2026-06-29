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
