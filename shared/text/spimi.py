import heapq
import itertools
from contextlib import ExitStack
from pathlib import Path

import spimi_cpp
from shared.text.index import (
    DictEntry,
    DictReader,
    DocId,
    PostingsEntry,
    PostingsReader,
    make_block_path,
)
from shared.text.processing import parse_docs


def _merge_blocks_partial(base: Path, level: int, n: int, blocks: tuple[int, ...]):
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


def _merge_blocks(base_dir: Path, level: int, n_blocks: int, m: int) -> int:
    while n_blocks > 1:
        batches = itertools.batched(range(n_blocks), m)
        n_blocks = 0

        for i, batch in enumerate(batches):
            _merge_blocks_partial(base_dir, level, i, batch)
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
        print(f"Processing block {n}...")

        block_path = make_block_path(out_dir, 0, n)
        spimi_cpp.spimi_invert(token_stream, str(block_path), max_memory)
        n += 1

    final_level = _merge_blocks(out_dir, 0, n, m=m)
    return make_block_path(out_dir, final_level, 0)
