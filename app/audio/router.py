from io import BytesIO
from typing import cast

import librosa
import numpy as np
from cv2.typing import MatLike
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, UploadFile

import app.common.algos as algos
from app.common.state import AppState
from shared.types import MediaSearchMode

audio_router = APIRouter(prefix="/audio", tags=["audio"])

PRE_EMPHASIS = 0.97


async def extract_descriptors(file: UploadFile) -> MatLike:
    try:
        q_contents = await file.read()
        q_audio, sr = librosa.load(BytesIO(q_contents), sr=None)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read audio")

    if len(q_audio) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    q_audio = np.append(q_audio[0], q_audio[1:] - PRE_EMPHASIS * q_audio[:-1])

    try:
        q_desc = librosa.feature.mfcc(
            y=q_audio,
            sr=sr,
            n_mfcc=13,
            n_fft=int(0.025 * sr),
            win_length=int(0.025 * sr),
            hop_length=int(0.010 * sr),
            window="hamming",
            center=False,
        ).T
    except Exception:
        raise HTTPException(status_code=400, detail="Audio too short to extract MFCCs")

    if len(q_desc) == 0:
        raise HTTPException(status_code=400, detail="Could not extract MFCCs")

    return q_desc


@audio_router.post("/search")
async def audio_search(
    req: Request,
    file: UploadFile,
    k: int = Query(10, ge=1),
    mode: MediaSearchMode = MediaSearchMode.native,
):
    app = cast(FastAPI, req.app)
    state = cast(AppState, app.state)
    q_desc = await extract_descriptors(file)

    if mode == MediaSearchMode.native:
        return algos.knn(q_desc, state.audio_data, k)

    with state.db.connection() as conn:
        return algos.knn_postgres(conn, "audios", q_desc, state.audio_data, k, mode)
