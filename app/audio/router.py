from io import BytesIO
from typing import cast

import librosa
import numpy as np
from cv2.typing import MatLike
from fastapi import APIRouter, HTTPException, Request, UploadFile

import app.common.algos as algos
from app.common.algos import SearchMode
from app.common.state import AppState

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

    if len(q_desc) == 0:
        raise HTTPException(status_code=400, detail="Could not extract MFCCs")

    return q_desc


@audio_router.post("/search")
async def audio_search(
    req: Request,
    file: UploadFile,
    k: int = 5,
    mode: SearchMode = SearchMode.native,
):
    state = cast(AppState, req.app.state)
    q_desc = await extract_descriptors(file)

    if mode == SearchMode.native:
        return algos.knn(q_desc, state.audio_data, k)

    with state.db.connection() as conn:
        return algos.knn_postgres(conn, "audios", q_desc, state.audio_data, k, mode)
