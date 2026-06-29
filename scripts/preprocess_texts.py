import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import shared
import shared.text
from scripts.binary_index import (
    LexiconEntry,
    read_lexicon,
    read_raw_postings,
    write_lexicon,
    write_raw_postings,
)
from scripts.shared import ProgressMeter

OUTPUT_DIR = Path(".data/texts")


@dataclass(frozen=True)
class BlockFiles:
    postings_path: Path
    lexicon_path: Path


@dataclass
class JsonArrayWriter:
    path: Path
    first: bool = True

    def __post_init__(self):
        self.file = open(self.path, "w")
        self.file.write("[")

    def write(self, item):
        if not self.first:
            self.file.write(",")

        json.dump(item, self.file)
        self.first = False

    def close(self):
        self.file.write("]")
        self.file.close()

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
    parser.add_argument("--tmp-dir", type=Path)
    parser.add_argument("--keep-tmp", action="store_true")
    return parser.parse_args()


def build_codebook(
    chunks: Iterable[shared.text.TextChunk],
    codebook_size: int,
    language: str,
    min_token_len: int,
    use_stemming: bool,
) -> tuple[list[str], int]:
    counts: Counter[str] = Counter()
    chunk_count = 0

    for chunk in chunks:
        chunk_count += 1
        counts.update(
            shared.text.normalize_tokens(
                chunk.text,
                language=language,
                min_token_len=min_token_len,
                use_stemming=use_stemming,
            )
        )

    return [term for term, _ in counts.most_common(codebook_size)], chunk_count


def build_codebook_from_files(
    filenames: list[Path],
    codebook_size: int,
    language: str,
    min_chars: int,
    min_token_len: int,
    use_stemming: bool,
) -> tuple[list[str], int]:
    return build_codebook(
        shared.text.yield_text_chunks(filenames, min_chars=min_chars),
        codebook_size,
        language,
        min_token_len,
        use_stemming,
    )


def flush_raw_block(
    block: list[list[tuple[int, int]]],
    tmp_dir: Path,
    block_id: int,
) -> BlockFiles:
    os.makedirs(tmp_dir, exist_ok=True)

    postings_path = tmp_dir / f"block_{block_id:06d}.postings.bin"
    lexicon_path = tmp_dir / f"block_{block_id:06d}.lexicon.bin"
    entries: list[LexiconEntry] = []

    with open(postings_path, "wb") as postings_file:
        for word_id, postings in enumerate(block):
            if not postings:
                continue

            offset, posting_count = write_raw_postings(postings_file, postings)
            entries.append(LexiconEntry(word_id, offset, posting_count))

    with open(lexicon_path, "wb") as lexicon_file:
        write_lexicon(lexicon_file, entries)

    return BlockFiles(postings_path, lexicon_path)


def build_spimi_block_files(
    chunks: list[shared.text.TextChunk],
    word_to_id: dict[str, int],
    language: str,
    min_token_len: int,
    use_stemming: bool,
    block_size: int,
    tmp_dir: Path,
) -> list[BlockFiles]:
    block_files: list[BlockFiles] = []
    current = _empty_block(len(word_to_id))
    meter = ProgressMeter(0.0001)
    block_id = 0

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
            block_files.append(flush_raw_block(current, tmp_dir, block_id))
            current = _empty_block(len(word_to_id))
            block_id += 1

    meter.record(1)

    if any(postings for postings in current):
        block_files.append(flush_raw_block(current, tmp_dir, block_id))

    return block_files


def merge_block_files(
    block_files: list[BlockFiles],
    bow_len: int,
) -> list[list[tuple[int, int]]]:
    merged: list[list[tuple[int, int]]] = _empty_block(bow_len)

    for block in block_files:
        with open(block.lexicon_path, "rb") as lexicon_file:
            lexicon = read_lexicon(lexicon_file)

        with open(block.postings_path, "rb") as postings_file:
            for word_id, entry in lexicon.items():
                merged[word_id].extend(
                    read_raw_postings(
                        postings_file,
                        entry.offset,
                        entry.posting_count,
                    )
                )

    for postings in merged:
        postings.sort(key=lambda posting: posting[0])

    return merged


def compute_weighted_index(
    raw_index: list[list[tuple[int, int]]],
    chunk_count: int,
) -> tuple[list[list[tuple[int, float]]], np.ndarray, np.ndarray, np.ndarray]:
    bow_len = len(raw_index)
    df = np.zeros(bow_len, dtype=np.uint32)
    lengths = np.zeros(chunk_count, dtype=np.float32)
    weighted_hists = np.zeros((chunk_count, bow_len), dtype=np.float32)
    weighted_index: list[list[tuple[int, float]]] = [[] for _ in range(bow_len)]

    for word_id, postings in enumerate(raw_index):
        df[word_id] = len(postings)

    for word_id, postings in enumerate(raw_index):
        for chunk_id, tf in postings:
            weight = shared.weight(chunk_count, tf, int(df[word_id]))
            lengths[chunk_id] += weight**2
            weighted_hists[chunk_id, word_id] = weight
            weighted_index[word_id].append((chunk_id, weight))

    for chunk_id in range(chunk_count):
        lengths[chunk_id] = math.sqrt(lengths[chunk_id])

    return weighted_index, df, lengths, weighted_hists


def save_outputs(
    output_dir: Path,
    words: list[str],
    chunks: list[shared.text.TextChunk],
    index: list[list[tuple[int, float]]],
    df: np.ndarray,
    lengths: np.ndarray,
    weighted_hists: np.ndarray,
):
    os.makedirs(output_dir, exist_ok=True)

    np.save(output_dir / "words.npy", np.array(words, dtype=str))
    np.save(output_dir / "df.npy", df)
    np.save(output_dir / "lengths.npy", lengths)
    np.save(output_dir / "histograms.npy", weighted_hists)

    with open(output_dir / "media_files.json", "w") as f:
        json.dump([chunk.identifier for chunk in chunks], f)

    with open(output_dir / "index.json", "w") as f:
        json.dump(index, f)

    with open(output_dir / "chunks.json", "w") as f:
        json.dump(
            [
                {
                    "id": chunk.identifier,
                    "source": chunk.source,
                    "ordinal": chunk.ordinal,
                    "text": chunk.text,
                }
                for chunk in chunks
            ],
            f,
        )

    with open(output_dir / "word_to_id.json", "w") as f:
        json.dump({word: i for i, word in enumerate(words)}, f)


def _empty_block(bow_len: int) -> list[list[tuple[int, int]]]:
    return [[] for _ in range(bow_len)]


def main():
    args = parse_args()
    texts_dir: Path = args.texts_dir

    if not texts_dir.exists():
        raise Exception(f"texts folder does not exist: {texts_dir}")

    filenames = sorted(path for path in texts_dir.iterdir() if path.suffix == ".txt")
    if not filenames:
        raise Exception(f"no .txt files found in {texts_dir}")

    print(f"Reading '{texts_dir}'...")
    print(f"Found {len(filenames)} text files")

    chunks = shared.text.iter_text_chunks(filenames, min_chars=args.min_chars)
    if not chunks:
        raise Exception("no text chunks found")

    print(f"Found {len(chunks)} chunks")

    use_stemming = not args.no_stemming

    print(f"Building codebook... (k = {args.codebook_size})")
    words = build_codebook(
        chunks,
        args.codebook_size,
        args.language,
        args.min_token_len,
        use_stemming,
    )
    if not words:
        raise Exception("empty text codebook")

    word_to_id = {word: i for i, word in enumerate(words)}

    tmp_dir = args.tmp_dir or args.output_dir / "tmp"

    print("Building inverted index...")
    block_files = build_spimi_block_files(
        chunks,
        word_to_id,
        args.language,
        args.min_token_len,
        use_stemming,
        args.block_size,
        tmp_dir,
    )

    raw_index = merge_block_files(block_files, len(words))

    print("Computing TF-IDF weighted histograms...")
    index, df, lengths, weighted_hists = compute_weighted_index(raw_index, len(chunks))

    print("Saving...")
    save_outputs(args.output_dir, words, chunks, index, df, lengths, weighted_hists)

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()
