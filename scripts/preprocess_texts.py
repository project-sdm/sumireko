import argparse
import heapq
import itertools
import json
import os
import shutil
from contextlib import ExitStack
from io import BufferedWriter
from pathlib import Path
from typing import cast

import numpy as np

import shared
from shared.text import (
    DictEntry,
    DictReader,
    DocId,
    PostingsEntry,
    PostingsReader,
    TokenStream,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess text files into a TF-IDF index.",
    )
    _ = parser.add_argument("texts_dir", type=Path, help="Folder containing .txt files")
    _ = parser.add_argument("--output-dir", type=Path, default=Path(".data/texts"))
    _ = parser.add_argument(
        "--language",
        choices=["english", "spanish", "multilingual"],
        default="english",
    )
    _ = parser.add_argument("--merge-m", type=int, default=10)
    _ = parser.add_argument("--max-memory", type=int, default=2**12)
    return parser.parse_args()


type Posting = tuple[DocId, int]
type PostingsList = list[Posting]
type Dictionary = dict[str, PostingsList]


def parse_docs(paths: list[Path], language: str) -> TokenStream:
    return TokenStream(paths, language)


def add_to_dictionary(dictionary: Dictionary, term: str) -> PostingsList:
    dictionary[term] = []
    return dictionary[term]


def get_posting_list(dictionary: Dictionary, term: str) -> PostingsList:
    return dictionary[term]


def add_to_postings_list(postings_list: PostingsList, doc_id: DocId):
    last = len(postings_list) - 1

    if len(postings_list) > 0 and postings_list[last][0] == doc_id:
        postings_list[last] = (doc_id, postings_list[last][1] + 1)
    else:
        postings_list.append((doc_id, 1))


def sort_terms(dictionary: Dictionary) -> list[str]:
    return sorted(dictionary.keys())


class PostingsWriter:
    file: BufferedWriter

    def __init__(self, file: BufferedWriter):
        self.file = file

    def write_posting(self, doc_id: int, tf: int):
        entry = PostingsEntry(doc_id=doc_id, value=tf)
        _ = self.file.write(entry.pack())


def write_block_to_disk(
    sorted_terms: list[str],
    dictionary: Dictionary,
    block_path: Path,
):
    postings_path = block_path.with_suffix(".postings")
    dict_path = block_path.with_suffix(".dict")

    with open(postings_path, "wb") as postings_file, open(dict_path, "wb") as dict_file:
        for term in sorted_terms:
            postings_list = get_posting_list(dictionary, term)

            offset = postings_file.tell()

            for doc_id, tf in postings_list:
                posting = PostingsEntry(doc_id=doc_id, value=tf)
                _ = postings_file.write(posting.pack())

            dict_entry = DictEntry(term=term, offset=offset, len=len(postings_list))
            _ = dict_file.write(dict_entry.pack())


def spimi_invert(token_stream: TokenStream, block_path: Path, max_memory: int):
    dictionary: Dictionary = {}
    bytes_used = 0

    while bytes_used < max_memory and (token := token_stream.next()):
        if token.term not in dictionary:
            postings_list = add_to_dictionary(dictionary, token.term)
            bytes_used += 16 + len(token.term)
        else:
            postings_list = get_posting_list(dictionary, token.term)

        add_to_postings_list(postings_list, token.doc_id)
        bytes_used += 56

    sorted_terms = sort_terms(dictionary)
    write_block_to_disk(sorted_terms, dictionary, block_path)


def make_block_path(base: Path, level: int, n: int) -> Path:
    return base / f"block_{level:02}_{n:02}"


def merge_blocks_pass(base: Path, level: int, n: int, blocks: tuple[int, ...]):
    print(f"Merging range {level}-{n} ({len(blocks)} blocks)...")

    paths = [make_block_path(base, level, b) for b in blocks]
    out_path = make_block_path(base, level + 1, n)

    with (
        ExitStack() as stack,
        open(out_path.with_suffix(".dict"), "wb") as out_dict,
        open(out_path.with_suffix(".postings"), "wb") as out_postings,
    ):
        dict_files = [
            stack.enter_context(open(path.with_suffix(".dict"), "rb")) for path in paths
        ]
        postings_files = [
            stack.enter_context(open(path.with_suffix(".postings"), "r+b"))
            for path in paths
        ]

        dict_readers = [DictReader(file) for file in dict_files]
        q1: list[tuple[str, int, DictEntry]] = []

        for i, reader in enumerate(dict_readers):
            if entry := reader.next():
                q1.append((entry.term, i, entry))

        heapq.heapify(q1)

        while len(q1) > 0:
            term, i, entry = heapq.heappop(q1)

            if next_entry := dict_readers[i].next():
                heapq.heappush(q1, (next_entry.term, i, next_entry))

            merge_items = [(i, entry)]

            # Consider other local dicts on the same term
            while len(q1) > 0 and q1[0][0] == term:
                _, i, entry = heapq.heappop(q1)

                if next_entry := dict_readers[i].next():
                    heapq.heappush(q1, (next_entry.term, i, next_entry))

                merge_items.append((i, entry))

            posting_readers = [
                PostingsReader(postings_files[i], entry.len, entry.offset)
                for i, entry in merge_items
            ]

            q2: list[tuple[DocId, int, PostingsEntry]] = []

            for i, reader in enumerate(posting_readers):
                if entry := reader.next():
                    q2.append((entry.doc_id, i, entry))

            heapq.heapify(q2)

            offset = out_postings.tell()
            total_postings = 0

            while len(q2) > 0:
                doc_id, i, entry = heapq.heappop(q2)
                total_tf = entry.value

                if entry := posting_readers[i].next():
                    heapq.heappush(q2, (entry.doc_id, i, entry))

                # Consider other postings with same doc_id
                while len(q2) > 0 and q2[0][0] == doc_id:
                    _, i, entry_extra = heapq.heappop(q2)

                    if entry := posting_readers[i].next():
                        heapq.heappush(q2, (entry.doc_id, i, entry))

                    total_tf += entry_extra.value

                out_entry = PostingsEntry(doc_id=doc_id, value=total_tf)
                _ = out_postings.write(out_entry.pack())
                total_postings += 1

            out_dict_entry = DictEntry(term=term, offset=offset, len=total_postings)
            _ = out_dict.write(out_dict_entry.pack())


def merge_blocks(base_dir: Path, level: int, n_blocks: int, m: int) -> int:
    while n_blocks > 1:
        batches = itertools.batched(range(n_blocks), m)
        n_blocks = 0

        for i, batch in enumerate(batches):
            merge_blocks_pass(base_dir, level, i, batch)
            n_blocks += 1

        level += 1

    return level


def spimi_index_construction(
    doc_paths: list[Path],
    language: str,
    m: int,
    out_dir: Path,
    max_memory: int,
) -> Path:
    n = 0
    token_stream = parse_docs(doc_paths, language)

    while not token_stream.done():
        print(f"Processing block {n}")

        block_path = make_block_path(out_dir, 0, n)
        spimi_invert(token_stream, block_path, max_memory)
        n += 1

    final_level = merge_blocks(out_dir, 0, n, m=m)
    return make_block_path(out_dir, final_level, 0)


def main():
    args = parse_args()

    language = cast(str, args.language)
    texts_dir = cast(Path, args.texts_dir)
    output_dir = cast(Path, args.output_dir)
    merge_m = cast(int, args.merge_m)
    max_memory = cast(int, args.max_memory)

    doc_filenames = os.listdir(texts_dir)
    doc_paths = [texts_dir / filename for filename in doc_filenames]

    print(f"Found {len(doc_paths)} text files.")

    tmp_dir = output_dir / "tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    print("Constructing inverted index with SPIMI...")
    final_block_path = spimi_index_construction(
        doc_paths,
        language,
        m=merge_m,
        out_dir=tmp_dir,
        max_memory=max_memory,
    )

    dict_path = output_dir / "index.dict"
    postings_path = output_dir / "index.postings"

    os.rename(final_block_path.with_suffix(".dict"), dict_path)
    os.rename(final_block_path.with_suffix(".postings"), postings_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("Weighing index with TF-IDF and calculating document lengths...")
    with (
        open(dict_path, "rb") as dict_file,
        open(postings_path, "r+b") as postings_file,
    ):
        dict_reader = DictReader(dict_file)
        n = len(doc_paths)

        lengths = np.zeros(n)

        while dict_entry := dict_reader.next():
            postings = PostingsReader(postings_file, dict_entry.len, dict_entry.offset)

            df = dict_entry.len

            while posting := postings.next():
                w = shared.weight(n, posting.value, df)
                lengths[posting.doc_id] += w**2

                postings.step_back()
                postings.write(PostingsEntry(doc_id=posting.doc_id, value=w))

        lengths = np.sqrt(lengths)

    with open(output_dir / "files.json", "w") as f:
        json.dump(doc_filenames, f)

    np.save(output_dir / "lengths.npy", lengths)
    print("Done.")


if __name__ == "__main__":
    main()
