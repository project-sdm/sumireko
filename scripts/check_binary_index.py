import math
import tempfile
from pathlib import Path

from scripts.binary_index import (
    LexiconEntry,
    read_lexicon,
    read_raw_postings,
    read_weighted_postings,
    write_lexicon,
    write_raw_postings,
    write_weighted_postings,
)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        raw_path = tmp_dir / "raw.postings.bin"
        raw_postings = [(0, 2), (5, 1), (9, 4)]
        with open(raw_path, "wb") as f:
            raw_offset, raw_count = write_raw_postings(f, raw_postings)

        with open(raw_path, "rb") as f:
            assert read_raw_postings(f, raw_offset, raw_count) == raw_postings

        weighted_path = tmp_dir / "weighted.postings.bin"
        weighted_postings = [(1, 0.25), (4, 1.5), (7, 3.75)]
        with open(weighted_path, "wb") as f:
            weighted_offset, weighted_count = write_weighted_postings(
                f, weighted_postings
            )

        with open(weighted_path, "rb") as f:
            actual = read_weighted_postings(f, weighted_offset, weighted_count)
            assert len(actual) == len(weighted_postings)

            for (actual_chunk, actual_weight), (expected_chunk, expected_weight) in zip(
                actual, weighted_postings
            ):
                assert actual_chunk == expected_chunk
                assert math.isclose(actual_weight, expected_weight, rel_tol=1e-6)

        lexicon_path = tmp_dir / "lexicon.bin"
        entries = [
            LexiconEntry(0, raw_offset, raw_count),
            LexiconEntry(3, weighted_offset, weighted_count),
        ]
        with open(lexicon_path, "wb") as f:
            write_lexicon(f, entries)

        with open(lexicon_path, "rb") as f:
            assert read_lexicon(f) == {entry.word_id: entry for entry in entries}

    print("Done.")


if __name__ == "__main__":
    main()
