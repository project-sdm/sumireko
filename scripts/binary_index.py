import struct

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
