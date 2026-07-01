from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import app.postgres as postgres
from app.audio.router import audio_router
from app.common.state import load_app_state
from app.images.router import image_router
from app.text.router import text_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    TEXTS_PATH = Path(".data/texts")
    IMAGES_PATH = Path(".data/images")
    AUDIOS_PATH = Path(".data/audios")

    postgres.init_text(Path("media/texts"), "texts")
    postgres.init(IMAGES_PATH, "images")
    postgres.init(AUDIOS_PATH, "audios")

    app.state = load_app_state(TEXTS_PATH, IMAGES_PATH, AUDIOS_PATH)

    yield


app = FastAPI(lifespan=lifespan)

app.mount(
    "/media/images",
    StaticFiles(directory="media/images"),
    name="images",
)

app.mount(
    "/media/audios",
    StaticFiles(directory="media/audios"),
    name="audios",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health():
    return {"status": "ok"}


app.include_router(text_router)
app.include_router(image_router)
app.include_router(audio_router)
