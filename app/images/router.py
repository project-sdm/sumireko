from typing import cast

import cv2
import numpy as np
from cv2.typing import MatLike
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, UploadFile

import app.common.algos as algos
import shared.image
from app.common.algos import MediaSearchMode
from app.common.state import AppState

image_router = APIRouter(prefix="/images", tags=["images"])


async def extract_descriptors(state: AppState, file: UploadFile) -> MatLike:
    q_contents = await file.read()
    q_nparr = np.frombuffer(q_contents, np.uint8)

    q_img = cv2.imdecode(q_nparr, cv2.IMREAD_GRAYSCALE)
    if q_img is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    q_img = shared.image.downscale(q_img)

    _, q_desc = state.sift.detectAndCompute(q_img, None)

    if q_desc is None or len(q_desc) == 0:
        raise HTTPException(status_code=400, detail="No features detected in image")

    return q_desc


@image_router.post("/search")
async def image_search(
    req: Request,
    file: UploadFile,
    k: int = Query(10, ge=1),
    mode: MediaSearchMode = MediaSearchMode.native,
):
    app = cast(FastAPI, req.app)
    state = cast(AppState, app.state)
    q_desc = await extract_descriptors(state, file)

    if mode == MediaSearchMode.native:
        return algos.knn(q_desc, state.image_data, k)

    with state.db.connection() as conn:
        return algos.knn_postgres(conn, "images", q_desc, state.image_data, k, mode)
