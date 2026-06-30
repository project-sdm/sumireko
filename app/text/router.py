from fastapi import APIRouter, Request

from app.common.algos import SearchMode

text_router = APIRouter(prefix="/text", tags=["text"])


@text_router.get("/search")
async def text_search(
    req: Request,
    q: str,
    k: int = 5,
    mode: SearchMode = SearchMode.native,
):
    pass
