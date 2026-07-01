import time
from collections import Counter
from typing import Literal, cast

from fastapi import APIRouter, FastAPI, Query, Request

import shared
from app.common.algos import KnnResult, TextSearchMode
from app.common.state import AppState
from shared.text.index import DictEntry, DictReader, DocId, PostingsReader
from shared.text.processing import tokenize_text

text_router = APIRouter(prefix="/text", tags=["text"])


def find_dict_entry(
    dict_reader: DictReader, dict_size: int, term: str
) -> DictEntry | None:
    low = 0
    high = dict_size - 1

    while low <= high:
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
async def text_search(
    req: Request,
    q: str = Query(min_length=1),
    k: int = Query(10, ge=1),
    language: Literal["english", "spanish", "multilingual"] = "english",
    mode: TextSearchMode = TextSearchMode.native,
):
    app = cast(FastAPI, req.app)
    state = cast(AppState, app.state)

    if mode == "pg":
        start = time.perf_counter()

        with state.db.connection() as conn:
            with conn.cursor() as cur:
                _ = cur.execute(
                    """
                    with q as (select plainto_tsquery(%s, %s) as query)
                    select filename from texts cross join q
                    where content_tsv @@ q.query
                    order by ts_rank(content_tsv, q.query) desc limit %s
                    """,
                    (language, q, k),
                )
                rows = cur.fetchall()

        elapsed_ms = (time.perf_counter() - start) * 1000
        return KnnResult(
            results=[row[0] for row in rows],
            time_ms=round(elapsed_ms, 2),
        )

    data = state.text_data

    start = time.perf_counter()
    tokens = Counter(tokenize_text(q, language=language))

    n = len(data.files)

    with (
        open(data.dict_path, "rb") as dict_file,
        open(data.postings_path, "r+b") as postings_file,
    ):
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
                if posting.doc_id not in scores:
                    scores[posting.doc_id] = 0

                w_doc = posting.value
                scores[posting.doc_id] += w_doc * w_query

    for img_id in scores:
        scores[img_id] /= data.lengths[img_id]

    result = sorted(scores.items(), key=lambda tup: tup[1], reverse=True)
    top_files = [data.files[i] for i, _ in result[:k]]

    elapsed_ms = (time.perf_counter() - start) * 1000
    return KnnResult(results=top_files, time_ms=round(elapsed_ms, 2))
