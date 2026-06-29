from io import BytesIO
from typing import cast

import librosa
import numpy as np
from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.common.algos import knn
from app.common.state import AppState

audio_router = APIRouter(prefix="/audio", tags=["audio"])

PRE_EMPHASIS = 0.97


@audio_router.post("/search")
async def audio_search(req: Request, file: UploadFile, k: int | None = 5):
    state = cast(AppState, req.app.state)
    data = state.audio_data

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

    top_files = knn(q_desc, data, k)
    return {"results": top_files}
