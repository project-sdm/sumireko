import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import shared
import shared.text
from scripts.shared import ProgressMeter

OUTPUT_DIR = Path(".data/texts")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess text files into TF-IDF chunks.")
    parser.add_argument("texts_dir", type=Path, help="Folder containing .txt files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--codebook-size", type=int, default=1000)
    parser.add_argument(
        "--language",
        choices=["spanish", "english", "multilingual"],
        default="spanish",
    )
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--min-token-len", type=int, default=2)
    parser.add_argument("--no-stemming", action="store_true")
    parser.add_argument("--block-size", type=int, default=5000)
    return parser.parse_args()


def build_codebook(
    chunks: list[shared.text.TextChunk],
    codebook_size: int,
    language: str,
    min_token_len: int,
    use_stemming: bool,
) -> list[str]:
    counts = shared.text.collection_term_counts(
        chunks,
        language=language,
        min_token_len=min_token_len,
        use_stemming=use_stemming,
    )

    return [term for term, _ in counts.most_common(codebook_size)]


def build_spimi_blocks(
    chunks: list[shared.text.TextChunk],
    word_to_id: dict[str, int],
    language: str,
    min_token_len: int,
    use_stemming: bool,
    block_size: int,
) -> list[list[list[tuple[int, int]]]]:
    blocks: list[list[list[tuple[int, int]]]] = []
    current = _empty_block(len(word_to_id))
    meter = ProgressMeter(0.0001)

    for chunk_id, chunk in enumerate(chunks):
        meter.record(chunk_id / len(chunks))
        tokens = shared.text.normalize_tokens(
            chunk.text,
            language=language,
            min_token_len=min_token_len,
            use_stemming=use_stemming,
        )
        frequencies = Counter(
            word_to_id[token] for token in tokens if token in word_to_id
        )

        for word_id, tf in frequencies.items():
            current[word_id].append((chunk_id, tf))

        if (chunk_id + 1) % block_size == 0:
            blocks.append(current)
            current = _empty_block(len(word_to_id))

    meter.record(1)

    if any(postings for postings in current):
        blocks.append(current)

    return blocks


def merge_blocks(
    blocks: list[list[list[tuple[int, int]]]],
    bow_len: int,
) -> list[list[tuple[int, int]]]:
    merged: list[list[tuple[int, int]]] = _empty_block(bow_len)

    for block in blocks:
        for word_id, postings in enumerate(block):
            merged[word_id].extend(postings)

    for postings in merged:
        postings.sort(key=lambda posting: posting[0])

    return merged


def _empty_block(bow_len: int) -> list[list[tuple[int, int]]]:
    return [[] for _ in range(bow_len)]
