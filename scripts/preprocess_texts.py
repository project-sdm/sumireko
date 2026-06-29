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
    write_weighted_postings,
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
    parser = argparse.ArgumentParser(description="Preprocess text files into a TF-IDF index.")
    parser.add_argument("texts_dir", type=Path, help="Folder containing .txt files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--language",
        choices=["spanish", "english", "multilingual"],
        default="spanish",
    )
    parser.add_argument("--min-token-len", type=int, default=2)
    parser.add_argument("--no-stemming", action="store_true")
    parser.add_argument("--block-size", type=int, default=5000)
    parser.add_argument("--tmp-dir", type=Path)
    parser.add_argument("--keep-tmp", action="store_true")
    return parser.parse_args()


def build_vocabulary(
    documents: Iterable[shared.text.TextDocument],
    language: str,
    min_token_len: int,
    use_stemming: bool,
) -> tuple[list[str], int]:
    term_to_id: dict[str, int] = {}
    document_count = 0

    for document in documents:
        document_count += 1

        for term in shared.text.normalize_tokens(
            document.text,
            language=language,
            min_token_len=min_token_len,
            use_stemming=use_stemming,
        ):
            if term not in term_to_id:
                term_to_id[term] = len(term_to_id)

    return list(term_to_id.keys()), document_count


def build_vocabulary_from_files(
    filenames: list[Path],
    language: str,
    min_token_len: int,
    use_stemming: bool,
) -> tuple[list[str], int]:
    return build_vocabulary(
        shared.text.yield_text_documents(filenames),
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
    documents: Iterable[shared.text.TextDocument],
    document_count: int,
    term_to_id: dict[str, int],
    language: str,
    min_token_len: int,
    use_stemming: bool,
    block_size: int,
    tmp_dir: Path,
    documents_writer: JsonArrayWriter,
    media_files_writer: JsonArrayWriter,
) -> list[BlockFiles]:
    block_files: list[BlockFiles] = []
    current = _empty_block(len(term_to_id))
    meter = ProgressMeter(0.0001)
    block_id = 0

    for document_id, document in enumerate(documents):
        meter.record(document_id / document_count)
        media_files_writer.write(document.identifier)
        documents_writer.write(
            {
                "id": document.identifier,
                "source": document.source,
                "text": document.text,
            }
        )
        tokens = shared.text.normalize_tokens(
            document.text,
            language=language,
            min_token_len=min_token_len,
            use_stemming=use_stemming,
        )
        frequencies = Counter(
            term_to_id[token] for token in tokens if token in term_to_id
        )

        for word_id, tf in frequencies.items():
            current[word_id].append((document_id, tf))

        if (document_id + 1) % block_size == 0:
            block_files.append(flush_raw_block(current, tmp_dir, block_id))
            current = _empty_block(len(term_to_id))
            block_id += 1

    meter.record(1)

    if any(postings for postings in current):
        block_files.append(flush_raw_block(current, tmp_dir, block_id))

    return block_files


def merge_block_files(
    block_files: list[BlockFiles],
    bow_len: int,
) -> BlockFiles:
    raw_postings_path = block_files[0].postings_path.parent / "raw.postings.bin"
    raw_lexicon_path = block_files[0].lexicon_path.parent / "raw.lexicon.bin"
    block_lexicons = []

    for block in block_files:
        with open(block.lexicon_path, "rb") as lexicon_file:
            block_lexicons.append(read_lexicon(lexicon_file))

    entries: list[LexiconEntry] = []

    with open(raw_postings_path, "wb") as raw_postings_file:
        for word_id in range(bow_len):
            merged: list[tuple[int, int]] = []

            for block, lexicon in zip(block_files, block_lexicons):
                entry = lexicon.get(word_id)
                if entry is None:
                    continue

                with open(block.postings_path, "rb") as postings_file:
                    merged.extend(
                        read_raw_postings(
                            postings_file,
                            entry.offset,
                            entry.posting_count,
                        )
                    )

            if not merged:
                continue

            merged.sort(key=lambda posting: posting[0])
            offset, posting_count = write_raw_postings(raw_postings_file, merged)
            entries.append(LexiconEntry(word_id, offset, posting_count))

    with open(raw_lexicon_path, "wb") as raw_lexicon_file:
        write_lexicon(raw_lexicon_file, entries)

    return BlockFiles(raw_postings_path, raw_lexicon_path)


def compute_weighted_index_files(
    raw_files: BlockFiles,
    output_dir: Path,
    term_count: int,
    document_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    df = np.zeros(term_count, dtype=np.uint32)
    lengths = np.zeros(document_count, dtype=np.float32)

    postings_path = output_dir / "postings.bin"
    lexicon_path = output_dir / "lexicon.bin"
    entries: list[LexiconEntry] = []

    with open(raw_files.lexicon_path, "rb") as raw_lexicon_file:
        raw_lexicon = read_lexicon(raw_lexicon_file)

    with open(raw_files.postings_path, "rb") as raw_postings_file:
        with open(postings_path, "wb") as weighted_postings_file:
            for word_id in range(term_count):
                entry = raw_lexicon.get(word_id)
                if entry is None:
                    continue

                raw_postings = read_raw_postings(
                    raw_postings_file,
                    entry.offset,
                    entry.posting_count,
                )
                df[word_id] = len(raw_postings)
                weighted_postings: list[tuple[int, float]] = []

                for document_id, tf in raw_postings:
                    weight = shared.weight(document_count, tf, int(df[word_id]))
                    lengths[document_id] += weight**2
                    weighted_postings.append((document_id, weight))

                offset, posting_count = write_weighted_postings(
                    weighted_postings_file,
                    weighted_postings,
                )
                entries.append(LexiconEntry(word_id, offset, posting_count))

    with open(lexicon_path, "wb") as lexicon_file:
        write_lexicon(lexicon_file, entries)

    for document_id in range(document_count):
        lengths[document_id] = math.sqrt(lengths[document_id])

    return df, lengths


def save_outputs(
    output_dir: Path,
    terms: list[str],
    df: np.ndarray,
    lengths: np.ndarray,
):
    os.makedirs(output_dir, exist_ok=True)

    np.save(output_dir / "terms.npy", np.array(terms, dtype=str))
    np.save(output_dir / "df.npy", df)
    np.save(output_dir / "lengths.npy", lengths)

    with open(output_dir / "term_to_id.json", "w") as f:
        json.dump({term: i for i, term in enumerate(terms)}, f)


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

    use_stemming = not args.no_stemming

    print("Building vocabulary...")
    terms, document_count = build_vocabulary_from_files(
        filenames,
        args.language,
        args.min_token_len,
        use_stemming,
    )
    if document_count == 0:
        raise Exception("no text documents found")

    print(f"Found {document_count} documents")

    if not terms:
        raise Exception("empty text vocabulary")

    term_to_id = {term: i for i, term in enumerate(terms)}

    tmp_dir = args.tmp_dir or args.output_dir / "tmp"

    print("Building inverted index...")
    os.makedirs(args.output_dir, exist_ok=True)
    documents_writer = JsonArrayWriter(args.output_dir / "documents.json")
    media_files_writer = JsonArrayWriter(args.output_dir / "media_files.json")

    try:
        block_files = build_spimi_block_files(
            shared.text.yield_text_documents(filenames),
            document_count,
            term_to_id,
            args.language,
            args.min_token_len,
            use_stemming,
            args.block_size,
            tmp_dir,
            documents_writer,
            media_files_writer,
        )
    finally:
        documents_writer.close()
        media_files_writer.close()

    raw_files = merge_block_files(block_files, len(terms))

    print("Computing TF-IDF weighted index...")
    df, lengths = compute_weighted_index_files(
        raw_files,
        args.output_dir,
        len(terms),
        document_count,
    )

    print("Saving...")
    save_outputs(args.output_dir, terms, df, lengths)

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("Done.")


if __name__ == "__main__":
    main()
