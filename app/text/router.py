import time
from collections import Counter
from typing import cast

from fastapi import APIRouter, FastAPI, Request

import shared.text
from app.common.algos import KnnResult
from app.common.state import AppState
from shared.text import DictEntry, DictReader, DocId, PostingsReader

text_router = APIRouter(prefix="/text", tags=["text"])


def find_dict_entry(
    dict_reader: DictReader, dict_size: int, term: str
) -> DictEntry | None:
    low = 0
    high = dict_size - 1

    while low < high:
        mid = (low + high) // 2

        dict_reader.set_index(mid)
        entry = dict_reader.next()
        assert entry is not None

        if entry.term == term:
            return entry

        if term < entry.term:
            high = mid - 1
        else:
            low = mid + 1

    return None


@text_router.get("/search")
async def text_search(req: Request, q: str, k: int = 10, language: str = "english"):
    app = cast(FastAPI, req.app)
    state = cast(AppState, app.state)
    data = state.text_data

    start = time.perf_counter()
    tokens = Counter(shared.text.tokenize_text(q, language=language))
    print(tokens)

    n = len(data.files)

    with open(data.dict_path, "rb") as dict_file, open(
        data.postings_path, "r+b"
    ) as postings_file:
        dict_reader = DictReader(dict_file)
        dict_size = dict_reader.calc_size()

        scores: dict[DocId, float] = {}

        for term, tf_query in tokens.items():
            term_entry = find_dict_entry(dict_reader, dict_size, term)
            if term_entry is None:
                continue

            postings = PostingsReader(postings_file, term_entry.len, term_entry.offset)
            w_query = shared.weight(n, tf_query, term_entry.len)

            while posting := postings.next():
                if not posting.doc_id in scores:
                    scores[posting.doc_id] = 0

                w_doc = posting.value
                scores[posting.doc_id] += w_doc * w_query

    for img_id in scores:
        scores[img_id] /= data.lengths[img_id]

    result = sorted(scores.items(), key=lambda tup: tup[1], reverse=True)
    top_files = [data.files[i] for i, _ in result[:k]]

    elapsed_ms = (time.perf_counter() - start) * 1000
    return KnnResult(results=top_files, time_ms=round(elapsed_ms, 2))
