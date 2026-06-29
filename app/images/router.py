from typing import cast

import cv2
import numpy as np
from cv2.typing import MatLike
from fastapi import APIRouter, HTTPException, Request, UploadFile

import app.common.algos as algos
from app.common.state import AppState

image_router = APIRouter(prefix="/images", tags=["images"])


async def extract_descriptors(state: AppState, file: UploadFile) -> MatLike:
    q_contents = await file.read()
    q_nparr = np.frombuffer(q_contents, np.uint8)

    q_img = cv2.imdecode(q_nparr, cv2.IMREAD_GRAYSCALE)
    if q_img is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    _, q_desc = state.sift.detectAndCompute(q_img, None)

    return q_desc


@image_router.post("/search")
async def image_search(req: Request, file: UploadFile, k: int | None = 5):
    state = cast(AppState, req.app.state)
    q_desc = await extract_descriptors(state, file)

    return algos.knn(q_desc, state.image_data, k)


@image_router.post("/search-pgvector")
async def image_search_pgvector(req: Request, file: UploadFile, k: int = 5):
    state = cast(AppState, req.app.state)
    q_desc = await extract_descriptors(state, file)
    q_hist = algos.compute_query_histogram(q_desc, state.image_data)

    with state.db.connection() as conn:
        return algos.knn_postgres(conn, "images", q_hist, k)
