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


def read_raw_postings(
    file: BinaryIO,
    offset: int,
    posting_count: int,
) -> list[tuple[int, int]]:
    _check_uint64(offset, "offset")
    _check_uint32(posting_count, "posting_count")

    file.seek(offset)
    postings: list[tuple[int, int]] = []

    for _ in range(posting_count):
        raw = file.read(RAW_POSTING.size)
        if len(raw) != RAW_POSTING.size:
            raise EOFError("unexpected end of raw postings file")

        postings.append(RAW_POSTING.unpack(raw))

    return postings


def write_weighted_postings(
    file: BinaryIO,
    postings: Sequence[tuple[int, float]],
) -> tuple[int, int]:
    offset = file.tell()
    _check_uint64(offset, "offset")

    for chunk_id, weight in postings:
        _check_uint32(chunk_id, "chunk_id")
        file.write(WEIGHTED_POSTING.pack(chunk_id, float(weight)))

    _check_uint32(len(postings), "posting_count")
    return offset, len(postings)


def read_weighted_postings(
    file: BinaryIO,
    offset: int,
    posting_count: int,
) -> list[tuple[int, float]]:
    _check_uint64(offset, "offset")
    _check_uint32(posting_count, "posting_count")

    file.seek(offset)
    postings: list[tuple[int, float]] = []

    for _ in range(posting_count):
        raw = file.read(WEIGHTED_POSTING.size)
        if len(raw) != WEIGHTED_POSTING.size:
            raise EOFError("unexpected end of weighted postings file")

        postings.append(WEIGHTED_POSTING.unpack(raw))

    return postings
