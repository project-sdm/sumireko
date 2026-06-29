import struct
from collections.abc import BinaryIO, Sequence

UINT32_MAX = 2**32 - 1
UINT64_MAX = 2**64 - 1

RAW_POSTING = struct.Struct("<II")
WEIGHTED_POSTING = struct.Struct("<If")
LEXICON_ENTRY = struct.Struct("<IQI")


def _check_uint32(value: int, label: str):
    if value < 0 or value > UINT32_MAX:
        raise ValueError(f"{label} must fit uint32")


def _check_uint64(value: int, label: str):
    if value < 0 or value > UINT64_MAX:
        raise ValueError(f"{label} must fit uint64")


def write_raw_postings(
    file: BinaryIO,
    postings: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    offset = file.tell()
    _check_uint64(offset, "offset")

    for chunk_id, tf in postings:
        _check_uint32(chunk_id, "chunk_id")
        _check_uint32(tf, "tf")
        file.write(RAW_POSTING.pack(chunk_id, tf))

    _check_uint32(len(postings), "posting_count")
    return offset, len(postings)
