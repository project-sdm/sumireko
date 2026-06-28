from typing import cast

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request, UploadFile

import shared.image
from app.common.algos import knn
from app.common.state import AppState

image_router = APIRouter(prefix="/images", tags=["images"])


@image_router.post("/search")
async def image_search(req: Request, file: UploadFile, k: int | None = 5):
    state = cast(AppState, req.app.state)
    data = state.image_data

    q_contents = await file.read()
    q_nparr = np.frombuffer(q_contents, np.uint8)

    q_img = cv2.imdecode(q_nparr, cv2.IMREAD_GRAYSCALE)
    if q_img is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    q_img = shared.image.downscale(q_img)

    _, q_desc = state.sift.detectAndCompute(q_img, None)

    top_files = knn(q_desc, data, k)

    return {"results": [f"{path}" for path in top_files]}
